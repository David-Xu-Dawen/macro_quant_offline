"""严格 walk-forward 滚动回测。

流程（每个再平衡日 T）
1. 仅用标签已实现的历史样本重训 LightGBM（防未来函数）
2. 用 T 日特征生成各资产得分
3. 得分输入 Black-Litterman 得到目标权重
4. 持有至下一再平衡日，按日盯市；换手扣交易成本
5. 同步记录等权、国债现金基准净值

输出：日净值、权重轨迹、再平衡日志、绩效摘要（含最大回撤/换手/夏普）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from black_litterman import decay_view_history, scores_to_weights
from config import (
    BL_CONFIDENCE_MODE,
    BL_RISK_AVERSION,
    BENCHMARK,
    BT_COST_BPS,
    BT_MIN_TRAIN_DAYS,
    BT_RETRAIN_EACH_REBALANCE,
    CORR_THRESHOLD,
    CVAR_RISK_AVERSION,
    FORWARD_DAYS,
    LABEL_MODE,
    OPTIMIZER_MODE,
    TRADE_UNIVERSE,
    WEIGHT_MAX,
    WEIGHT_MIN,
)
from model_lgbm import fit_expanding_window, predict_score_details


@dataclass
class RollingBacktestResult:
    nav: pd.DataFrame
    weights: pd.DataFrame
    rebalance_log: pd.DataFrame
    metrics: dict


def _max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def _ann_factor(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return np.nan
    years = (index[-1] - index[0]).days / 365.25
    return years


def summarize_nav(nav_df: pd.DataFrame, turnover: pd.Series | None = None) -> dict:
    """从日净值序列计算绩效指标。"""
    out = {}
    for col in ["strategy", "equal_weight", "cash"]:
        if col not in nav_df.columns:
            continue
        s = nav_df[col].dropna()
        rets = s.pct_change().dropna()
        years = _ann_factor(s.index)
        total = float(s.iloc[-1] / s.iloc[0] - 1.0) if len(s) > 1 else np.nan
        ann = float((1 + total) ** (1 / years) - 1) if years and years > 0 else np.nan
        vol = float(rets.std() * np.sqrt(252)) if len(rets) else np.nan
        sharpe = float(ann / vol) if vol and vol > 0 else np.nan
        out[f"{col}_total_return"] = total
        out[f"{col}_ann_return"] = ann
        out[f"{col}_ann_vol"] = vol
        out[f"{col}_sharpe"] = sharpe
        out[f"{col}_max_drawdown"] = _max_drawdown(s)

    if "strategy" in nav_df.columns and "cash" in nav_df.columns:
        out["excess_ann_vs_cash"] = out.get("strategy_ann_return", np.nan) - out.get("cash_ann_return", np.nan)
    if "strategy" in nav_df.columns and "equal_weight" in nav_df.columns:
        out["excess_ann_vs_ew"] = out.get("strategy_ann_return", np.nan) - out.get("equal_weight_ann_return", np.nan)
    if turnover is not None and len(turnover):
        out["avg_turnover"] = float(turnover.mean())
        out["avg_annual_turnover"] = float(turnover.mean() * (252 / FORWARD_DAYS))
    out["n_days"] = int(len(nav_df))
    out["start"] = str(nav_df.index.min().date()) if len(nav_df) else None
    out["end"] = str(nav_df.index.max().date()) if len(nav_df) else None
    return out


def run_rolling_backtest(
    panel: pd.DataFrame,
    close: pd.DataFrame,
    rebalance_every: int = FORWARD_DAYS,
    forward_days: int = FORWARD_DAYS,
    label_mode: str = LABEL_MODE,
    corr_threshold: float = CORR_THRESHOLD,
    min_train_days: int = BT_MIN_TRAIN_DAYS,
    cost_bps: float = BT_COST_BPS,
    retrain: bool = BT_RETRAIN_EACH_REBALANCE,
    oof_pred: pd.DataFrame | None = None,
    universe: list[str] | None = None,
    confidence_mode: str = BL_CONFIDENCE_MODE,
    optimizer_mode: str = OPTIMIZER_MODE,
    weight_max: float = WEIGHT_MAX,
    weight_min: float = WEIGHT_MIN,
    cvar_risk_aversion: float = CVAR_RISK_AVERSION,
    bl_risk_aversion: float = BL_RISK_AVERSION,
    exclude_features: tuple[str, ...] = (),
) -> RollingBacktestResult:
    """执行滚动回测主循环。"""
    universe = universe or [a for a in TRADE_UNIVERSE if a in close.columns]
    close = close[universe].dropna(how="all")
    dates = list(close.index)
    returns = close.pct_change()

    # 再平衡日程：至少攒够训练窗口后再开始
    start_i = min_train_days + forward_days
    rebalance_idx = list(range(start_i, len(dates) - 1, rebalance_every))
    if not rebalance_idx:
        raise RuntimeError("样本不足以启动滚动回测，请缩短 BT_MIN_TRAIN_DAYS 或增加历史数据")

    weight_rows = []
    reb_logs = []
    prev_w = pd.Series(0.0, index=universe)
    # 日权重矩阵（用于盯市）
    w_daily = pd.DataFrame(0.0, index=close.index, columns=universe)
    view_history: list[tuple[pd.Timestamp, pd.Series]] = []

    for k, i in enumerate(rebalance_idx):
        dt = dates[i]
        end_i = min(i + rebalance_every, len(dates) - 1)
        hold_dates = dates[i:end_i]  # 持有区间 [T, T_next)，最后一天开仓日前不重复计

        scores = None
        disagreement = None
        n_feats = 0
        if retrain:
            model, feats = fit_expanding_window(
                panel=panel,
                asof_date=pd.Timestamp(dt),
                forward_days=forward_days,
                mode=label_mode,
                corr_threshold=corr_threshold,
                min_train_days=min_train_days,
                exclude_features=exclude_features,
            )
            if model is None:
                continue
            day_panel = panel[panel["date"] == dt]
            if day_panel.empty:
                continue
            details = predict_score_details(model, day_panel, feats, mode=label_mode)
            details.index = day_panel["asset"].values
            scores = details["score"]
            disagreement = details["model_disagreement"]
            scores = scores.reindex(universe)
            disagreement = disagreement.reindex(universe)
            n_feats = len(feats)
        else:
            if oof_pred is None:
                raise ValueError("retrain=False 时必须提供 oof_pred")
            day = oof_pred[oof_pred["date"] == dt]
            if day.empty:
                continue
            scores = day.set_index("asset")["oof_pred"].reindex(universe)
            if "model_disagreement" in day.columns:
                disagreement = day.set_index("asset")["model_disagreement"].reindex(universe)
            n_feats = 0

        if scores is None or scores.dropna().empty:
            continue

        # ---- 观点衰减 + BL 权重（协方差只用 T 之前收益）----
        view_history.append((pd.Timestamp(dt), scores))
        decayed_scores = decay_view_history(view_history, pd.Timestamp(dt)).reindex(universe)
        hist = returns.loc[:dt].iloc[:-1]
        cutoff = dates[max(0, i - forward_days)]
        oof_hist = (
            oof_pred[oof_pred["date"] <= cutoff]
            if oof_pred is not None and not oof_pred.empty
            else None
        )
        try:
            alloc = scores_to_weights(
                decayed_scores,
                hist,
                universe=universe,
                disagreement=disagreement,
                oof_history=oof_hist,
                confidence_mode=confidence_mode,
                optimizer_mode=optimizer_mode,
                weight_max=weight_max,
                weight_min=weight_min,
                cvar_risk_aversion=cvar_risk_aversion,
                bl_risk_aversion=bl_risk_aversion,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip {pd.Timestamp(dt).date()}] BL failed: {exc}")
            continue

        w = alloc["weight"].reindex(universe).fillna(0.0)
        if w.sum() <= 0:
            continue
        w = w / w.sum()

        # 换手：相对上期权重的 L1/2
        turnover = float((w - prev_w.reindex(universe).fillna(0.0)).abs().sum() / 2.0)
        cost = turnover * (cost_bps / 10000.0)

        for d in hold_dates:
            w_daily.loc[d] = w.values

        for asset in universe:
            weight_rows.append(
                {
                    "date": dt,
                    "asset": asset,
                    "score": float(scores.get(asset, np.nan)),
                    "decayed_score": float(decayed_scores.get(asset, np.nan)),
                    "weight": float(w.get(asset, 0.0)),
                    "signal_confidence": float(alloc["signal_confidence"].iloc[0]),
                    "view_confidence": float(alloc["view_confidence"].get(asset, np.nan)),
                    "optimizer_used": str(alloc["optimizer_used"].iloc[0]),
                    "optimizer_status": str(alloc["optimizer_status"].iloc[0]),
                    "cvar_95": float(alloc["cvar_95"].iloc[0]),
                    "turnover": turnover,
                    "cost": cost,
                }
            )
        reb_logs.append(
            {
                "date": dt,
                "hold_end": dates[end_i],
                "turnover": turnover,
                "cost": cost,
                "n_features": n_feats,
                "top_asset": w.idxmax(),
                "top_weight": float(w.max()),
                "signal_confidence": float(alloc["signal_confidence"].iloc[0]),
                "mean_view_confidence": float(alloc["view_confidence"].mean()),
                "optimizer_used": str(alloc["optimizer_used"].iloc[0]),
                "optimizer_status": str(alloc["optimizer_status"].iloc[0]),
                "cvar_95": float(alloc["cvar_95"].iloc[0]),
            }
        )
        prev_w = w
        if (k + 1) % 10 == 0 or k == 0:
            print(f"  rebalance {k+1}/{len(rebalance_idx)} @ {pd.Timestamp(dt).date()} "
                  f"top={w.idxmax()}({w.max():.1%}) turnover={turnover:.1%}")

    weights = pd.DataFrame(weight_rows)
    reb_log = pd.DataFrame(reb_logs)
    if weights.empty:
        raise RuntimeError("滚动回测未生成任何有效再平衡，请检查数据/训练窗口")

    # ---- 日收益盯市 ----
    # 策略：昨日权重 * 今日资产收益；再平衡日额外扣一次成本
    asset_ret = returns.fillna(0.0)
    # 用昨收权重：shift(1)；首日权重在再平衡当日开盘后生效，简化为当日收盘权重从次日起计
    w_lag = w_daily.shift(1).fillna(0.0)
    port_ret = (w_lag * asset_ret).sum(axis=1)

    cost_by_date = reb_log.set_index("date")["cost"] if len(reb_log) else pd.Series(dtype=float)
    # 成本在再平衡日收盘执行（从当日收益扣除）
    port_ret = port_ret.copy()
    for dt, c in cost_by_date.items():
        if dt in port_ret.index:
            port_ret.loc[dt] = port_ret.loc[dt] - c

    ew_ret = asset_ret.mean(axis=1)
    cash_ret = asset_ret[BENCHMARK] if BENCHMARK in asset_ret.columns else pd.Series(0.0, index=asset_ret.index)

    # 仅从首次有效持仓开始计净值
    first_dt = reb_log["date"].min()
    mask = port_ret.index >= first_dt
    nav = pd.DataFrame(
        {
            "strategy": (1 + port_ret.loc[mask]).cumprod(),
            "equal_weight": (1 + ew_ret.loc[mask]).cumprod(),
            "cash": (1 + cash_ret.loc[mask]).cumprod(),
            "port_ret": port_ret.loc[mask],
            "ew_ret": ew_ret.loc[mask],
            "cash_ret": cash_ret.loc[mask],
        }
    )
    nav.index.name = "date"

    metrics = summarize_nav(nav, turnover=reb_log["turnover"] if len(reb_log) else None)
    metrics["cost_bps"] = cost_bps
    metrics["rebalance_every"] = rebalance_every
    metrics["retrain"] = retrain
    metrics["n_rebalances"] = int(len(reb_log))
    metrics["portfolio_cvar_95"] = float(reb_log["cvar_95"].mean())
    metrics["mean_view_confidence"] = float(reb_log["mean_view_confidence"].mean())
    metrics["cvar_optimizer_usage"] = float((reb_log["optimizer_used"] == "cvar").mean())
    metrics["optimizer_fallback_rate"] = float(
        reb_log["optimizer_status"].astype(str).str.startswith("fallback").mean()
    )
    metrics["confidence_mode"] = confidence_mode
    metrics["optimizer_mode"] = optimizer_mode
    metrics["weight_max"] = weight_max
    metrics["cvar_risk_aversion"] = cvar_risk_aversion
    metrics["bl_risk_aversion"] = bl_risk_aversion
    return RollingBacktestResult(nav=nav, weights=weights, rebalance_log=reb_log, metrics=metrics)
