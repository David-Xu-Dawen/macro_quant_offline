"""Black-Litterman 组合优化。

金融逻辑
--------
LightGBM 给出的是“相对强弱观点”，不能直接当权重。
Black-Litterman 把主观观点 (P, Q, Omega) 与市场先验均衡收益结合，
得到后验期望收益，再做带约束的均值-方差优化，从而：
- 预测强的资产上调权重；
- 同时受协方差与权重上限约束，避免 All-in 单资产。

先验 π 这里用风险平价隐含均衡收益近似：
    π = δ * Σ * w_prior
其中 w_prior 取风险平价权重（比市值权重更适合跨市场多资产）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import (
    BL_CONFIDENCE_LOOKBACK,
    BL_CONFIDENCE_MAX,
    BL_CONFIDENCE_MIN,
    BL_CONFIDENCE_MODE,
    BL_RISK_AVERSION,
    BL_TAU,
    BL_VIEW_CONFIDENCE,
    CVAR_ALPHA,
    CVAR_LOOKBACK,
    CVAR_RISK_AVERSION,
    OPTIMIZER_MODE,
    TRADE_UNIVERSE,
    VIEW_DECAY_HALF_LIFE_DAYS,
    VIEW_DECAY_LOOKBACK,
    WEIGHT_MAX,
    WEIGHT_MIN,
)
from portfolio_optimizers import cvar_weights, historical_cvar


def estimate_cov(returns: pd.DataFrame, lookback: int = 120) -> pd.DataFrame:
    """用近期日收益估计协方差，并年化。加入对角线收缩增强数值稳定。"""
    r = returns.tail(lookback).dropna(axis=1, how="any")
    if r.shape[1] < 2:
        r = returns.tail(lookback).fillna(0.0)
    cov = r.cov().values * 252
    shrink = 0.2
    diag = np.diag(np.diag(cov))
    cov = (1 - shrink) * cov + shrink * diag
    return pd.DataFrame(cov, index=r.columns, columns=r.columns)


def risk_parity_weights(cov: pd.DataFrame) -> pd.Series:
    """简化风险平价：目标各资产风险贡献相等。"""
    assets = list(cov.columns)
    sigma = cov.values
    n = len(assets)

    def obj(w):
        w = np.maximum(w, 1e-8)
        w = w / w.sum()
        port_var = w @ sigma @ w
        mrc = sigma @ w
        rc = w * mrc
        target = port_var / n
        return np.sum((rc - target) ** 2)

    x0 = np.ones(n) / n
    bounds = [(1e-6, 1.0)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(obj, x0, bounds=bounds, constraints=cons, method="SLSQP")
    w = res.x if res.success else x0
    w = np.maximum(w, 0)
    w = w / w.sum()
    return pd.Series(w, index=assets, name="w_prior")


def black_litterman_posterior(
    cov: pd.DataFrame,
    w_prior: pd.Series,
    views: pd.Series,
    tau: float = BL_TAU,
    delta: float = BL_RISK_AVERSION,
    view_confidence: float | pd.Series = BL_VIEW_CONFIDENCE,
) -> pd.Series:
    """标准 BL 后验期望收益 μ*。

    观点设置：
    - P = I（绝对观点：对每个有信号的资产直接给期望超额收益）
    - Q = 模型预测得分映射后的观点收益
    - Omega = diag(1/confidence * P Σ P')，置信越高，观点越硬
    """
    assets = list(cov.columns)
    sigma = cov.loc[assets, assets].values
    w = w_prior.reindex(assets).fillna(0).values.reshape(-1, 1)
    pi = (delta * sigma @ w).reshape(-1)  # 均衡隐含收益

    q = views.reindex(assets).astype(float)
    # 将得分标准化到收益量级：用先验波动的一定比例缩放
    vol = np.sqrt(np.diag(sigma))
    q_z = (q - q.mean()) / (q.std() + 1e-8)
    q_vec = (q_z * vol * 0.5).values.reshape(-1, 1)  # 半个标准差量级的主动观点

    p = np.eye(len(assets))
    tau_sigma = tau * sigma
    if isinstance(view_confidence, pd.Series):
        confidence = (
            view_confidence.reindex(assets)
            .fillna(BL_VIEW_CONFIDENCE)
            .clip(BL_CONFIDENCE_MIN, BL_CONFIDENCE_MAX)
            .to_numpy()
        )
    else:
        confidence = np.full(len(assets), max(float(view_confidence), 1e-6))
    omega = np.diag(np.diag(p @ tau_sigma @ p.T) / confidence)

    # μ* = π + τΣ P' (P τΣ P' + Ω)^-1 (Q - Pπ)
    mid = np.linalg.inv(p @ tau_sigma @ p.T + omega)
    mu = pi.reshape(-1, 1) + tau_sigma @ p.T @ mid @ (q_vec - p @ pi.reshape(-1, 1))
    return pd.Series(mu.reshape(-1), index=assets, name="mu_bl")


def build_view_confidence(
    scores: pd.Series,
    *,
    disagreement: pd.Series | None = None,
    oof_history: pd.DataFrame | None = None,
) -> pd.Series:
    """用模型分歧、历史 OOF 质量和截面清晰度构造逐资产置信度。"""
    assets = list(scores.index)
    score = scores.astype(float)
    centered = (score - score.median()).abs()
    clarity = centered / (centered.max() + 1e-8)

    if disagreement is None:
        disagreement_conf = pd.Series(BL_VIEW_CONFIDENCE, index=assets)
    else:
        disagreement_conf = 1.0 - disagreement.reindex(assets).fillna(1.0).clip(0.0, 1.0)

    track = pd.Series(BL_VIEW_CONFIDENCE, index=assets, dtype=float)
    if oof_history is not None and not oof_history.empty:
        hist = oof_history.dropna(subset=["oof_pred", "y_ret"]).copy()
        hist = hist.sort_values("date")
        if BL_CONFIDENCE_LOOKBACK > 0:
            dates = hist["date"].drop_duplicates().tail(BL_CONFIDENCE_LOOKBACK)
            hist = hist[hist["date"].isin(dates)]
        for asset, group in hist.groupby("asset"):
            if asset not in track.index or len(group) < 20:
                continue
            ic = group["oof_pred"].corr(group["y_ret"], method="spearman")
            if pd.notna(ic):
                track.loc[asset] = float(np.clip((ic + 1.0) / 2.0, 0.0, 1.0))

    confidence = 0.45 * disagreement_conf + 0.35 * track + 0.20 * clarity
    return confidence.clip(BL_CONFIDENCE_MIN, BL_CONFIDENCE_MAX).rename("view_confidence")


def decay_view_history(
    history: list[tuple[pd.Timestamp, pd.Series]],
    asof_date: pd.Timestamp,
) -> pd.Series:
    """在再平衡点对近期观点指数加权；不增加交易频率。"""
    recent = history[-VIEW_DECAY_LOOKBACK:]
    if not recent:
        return pd.Series(dtype=float)
    weighted: list[pd.Series] = []
    weights: list[float] = []
    for dt, scores in recent:
        age = max((pd.Timestamp(asof_date) - pd.Timestamp(dt)).days, 0)
        weight = float(0.5 ** (age / max(VIEW_DECAY_HALF_LIFE_DAYS, 1e-6)))
        weighted.append(scores.astype(float) * weight)
        weights.append(weight)
    return pd.concat(weighted, axis=1).sum(axis=1) / max(sum(weights), 1e-12)


def mean_variance_weights(
    mu: pd.Series,
    cov: pd.DataFrame,
    delta: float = BL_RISK_AVERSION,
    w_min: float = WEIGHT_MIN,
    w_max: float = WEIGHT_MAX,
) -> pd.Series:
    """带盒约束的均值-方差：max w'μ - 0.5 δ w'Σw, s.t. sum w=1, w_min≤w≤w_max。"""
    assets = list(mu.index)
    m = mu.values
    sigma = cov.loc[assets, assets].values
    n = len(assets)

    def neg_util(w):
        return -(w @ m - 0.5 * delta * w @ sigma @ w)

    x0 = np.ones(n) / n
    bounds = [(w_min, w_max)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(neg_util, x0, method="SLSQP", bounds=bounds, constraints=cons)
    w = res.x if res.success else x0
    # 若上界过紧导致不可行，做投影归一
    w = np.clip(w, w_min, w_max)
    if w.sum() <= 0:
        w = np.ones(n) / n
    else:
        w = w / w.sum()
    return pd.Series(w, index=assets, name="weight")


def scores_to_weights(
    scores: pd.Series,
    returns_history: pd.DataFrame,
    universe: list[str] | None = None,
    *,
    disagreement: pd.Series | None = None,
    oof_history: pd.DataFrame | None = None,
    confidence_mode: str = BL_CONFIDENCE_MODE,
    optimizer_mode: str = OPTIMIZER_MODE,
    weight_max: float = WEIGHT_MAX,
    weight_min: float = WEIGHT_MIN,
    cvar_risk_aversion: float = CVAR_RISK_AVERSION,
    bl_risk_aversion: float = BL_RISK_AVERSION,
) -> pd.DataFrame:
    """把某一日的 LightGBM 得分 -> BL 权重。

    LambdaRank 的分值尺度没有绝对含义，因此先看横截面第一名与第二名的
    标准化间距。间距小代表模型没有明确观点，最终权重向风险平价先验收缩；
    只有头部区分度足够大时才充分采用 BL 主动权重。
    """
    universe = universe or [a for a in TRADE_UNIVERSE if a in returns_history.columns]
    rets = returns_history[universe].dropna(how="all")
    cov = estimate_cov(rets)
    # 协方差可能因缺失剔除部分资产，权重宇宙与其对齐
    assets = list(cov.columns)
    w_prior = risk_parity_weights(cov)
    aligned_scores = scores.reindex(assets).astype(float)
    aligned_scores = aligned_scores.fillna(aligned_scores.median())
    ordered = np.sort(aligned_scores.to_numpy())
    dispersion = float(np.nanstd(ordered))
    top_gap = float(ordered[-1] - ordered[-2]) if len(ordered) >= 2 else 0.0
    signal_confidence = float(np.clip(top_gap / (2.0 * dispersion + 1e-8), 0.0, 1.0))

    if confidence_mode == "dynamic":
        view_confidence = build_view_confidence(
            aligned_scores,
            disagreement=disagreement,
            oof_history=oof_history,
        )
    else:
        view_confidence = pd.Series(BL_VIEW_CONFIDENCE, index=assets)

    mu = black_litterman_posterior(cov, w_prior, aligned_scores, view_confidence=view_confidence)
    optimizer_used = "mean_variance"
    optimizer_status = "success"
    cvar_95 = np.nan
    w_active: pd.Series | None = None
    if optimizer_mode in {"cvar", "auto"}:
        cvar_result = cvar_weights(
            mu,
            rets.tail(CVAR_LOOKBACK),
            alpha=CVAR_ALPHA,
            risk_aversion=cvar_risk_aversion,
            w_min=weight_min,
            w_max=weight_max,
        )
        w_active = cvar_result.weights
        optimizer_status = cvar_result.status
        if w_active is not None:
            optimizer_used = "cvar"
            cvar_95 = cvar_result.cvar_95 if cvar_result.cvar_95 is not None else np.nan
    if w_active is None:
        try:
            w_active = mean_variance_weights(
                mu,
                cov,
                delta=bl_risk_aversion,
                w_min=weight_min,
                w_max=weight_max,
            )
            optimizer_used = "mean_variance"
            if optimizer_mode in {"cvar", "auto"}:
                optimizer_status = f"fallback:{optimizer_status}"
        except Exception as exc:
            w_active = w_prior.copy()
            optimizer_used = "risk_parity"
            optimizer_status = f"fallback:{type(exc).__name__}"

    if confidence_mode == "dynamic":
        # 动态 Ω 已完成观点软化，不再二次按 signal_confidence 收缩。
        w = w_active
    else:
        w = signal_confidence * w_active + (1.0 - signal_confidence) * w_prior
    w = w / w.sum()
    if not pd.notna(cvar_95):
        cvar_95 = historical_cvar((rets.tail(CVAR_LOOKBACK) * w).sum(axis=1), alpha=CVAR_ALPHA)
    out = pd.DataFrame(
        {
            "score": aligned_scores,
            "w_prior_risk_parity": w_prior.reindex(assets),
            "mu_bl": mu.reindex(assets),
            "w_active_bl": w_active.reindex(assets),
            "signal_confidence": signal_confidence,
            "view_confidence": view_confidence.reindex(assets),
            "optimizer_used": optimizer_used,
            "optimizer_status": optimizer_status,
            "cvar_95": cvar_95,
            "weight": w.reindex(assets),
        }
    )
    return out


def allocate_over_time(
    pred_panel: pd.DataFrame,
    close: pd.DataFrame,
    rebalance_every: int = 20,
    score_col: str = "oof_pred",
    confidence_mode: str = BL_CONFIDENCE_MODE,
    optimizer_mode: str = OPTIMIZER_MODE,
    weight_max: float = WEIGHT_MAX,
    weight_min: float = WEIGHT_MIN,
    cvar_risk_aversion: float = CVAR_RISK_AVERSION,
    bl_risk_aversion: float = BL_RISK_AVERSION,
) -> pd.DataFrame:
    """按 rebalance_every 个交易日再平衡，生成历史权重轨迹。"""
    returns = close.pct_change()
    dates = sorted(pred_panel["date"].unique())
    # 从有足够协方差估计样本之后开始
    start_i = max(120, rebalance_every)
    picked_dates = dates[start_i::rebalance_every]

    rows = []
    view_history: list[tuple[pd.Timestamp, pd.Series]] = []
    for dt in picked_dates:
        day = pred_panel[pred_panel["date"] == dt]
        if day.empty:
            continue
        scores = day.set_index("asset")[score_col]
        view_history.append((pd.Timestamp(dt), scores))
        scores = decay_view_history(view_history, pd.Timestamp(dt))
        disagreement = (
            day.set_index("asset")["model_disagreement"]
            if "model_disagreement" in day.columns
            else None
        )
        hist = returns.loc[:dt].iloc[:-1]  # 不含当日，避免微弱泄漏
        oof_hist = pred_panel[pred_panel["date"] < dt]
        try:
            alloc = scores_to_weights(
                scores,
                hist,
                disagreement=disagreement,
                oof_history=oof_hist,
                confidence_mode=confidence_mode,
                optimizer_mode=optimizer_mode,
                weight_max=weight_max,
                weight_min=weight_min,
                cvar_risk_aversion=cvar_risk_aversion,
                bl_risk_aversion=bl_risk_aversion,
            )
        except Exception:
            continue
        alloc = alloc.reset_index().rename(columns={"index": "asset"})
        alloc["date"] = dt
        rows.append(alloc)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
