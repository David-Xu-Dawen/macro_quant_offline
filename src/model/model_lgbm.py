"""LightGBM 时序训练：回归基线 + 横截面 LambdaRank。

排序模型的 query 是交易日，同一天的资产属于一个 group。所有时序切分均在
训练和验证间留出 FORWARD_DAYS 隔离带，避免 20 日远期标签重叠。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, ndcg_score, r2_score
from sklearn.model_selection import TimeSeriesSplit

from config import (
    DART_PARAMS,
    EARLY_STOPPING_ROUNDS,
    FORWARD_DAYS,
    LABEL_MODE,
    LGBM_BOOSTING_TYPE,
    LGBM_PARAMS,
    LGBM_RANKER_PARAMS,
    N_TS_SPLITS,
    RANK_BLEND_WEIGHT,
    RANK_TRAIN_DATE_STRIDE,
)
from macro_features import (
    categorical_feature_columns,
    drop_highly_correlated_features,
    feature_columns,
)


@dataclass
class TrainResult:
    model: lgb.LGBMModel
    features: list[str]
    corr_kept: pd.DataFrame
    oof_pred: pd.DataFrame
    importance: pd.DataFrame
    cv_metrics: pd.DataFrame


@dataclass
class RankBlendModel:
    """以固定权重融合 LambdaRank 与收益回归的横截面排名。"""

    ranker: lgb.LGBMRanker
    regressor: lgb.LGBMRegressor
    rank_weight: float = RANK_BLEND_WEIGHT


RANKER_CANDIDATES = [
    {"num_leaves": 7, "min_child_samples": 100, "reg_lambda": 3.0},
    {"num_leaves": 15, "min_child_samples": 80, "reg_lambda": 2.0},
    {"num_leaves": 31, "min_child_samples": 120, "reg_lambda": 4.0},
]
ACTIVE_BOOSTING_TYPE = LGBM_BOOSTING_TYPE


def set_boosting_type(boosting_type: str) -> None:
    """为一次实验切换 GBDT/DART，不修改全局配置文件。"""
    if boosting_type not in {"gbdt", "dart"}:
        raise ValueError(f"不支持 boosting_type={boosting_type}")
    global ACTIVE_BOOSTING_TYPE
    ACTIVE_BOOSTING_TYPE = boosting_type


def _make_model(mode: str, overrides: dict | None = None) -> lgb.LGBMModel:
    if mode == "ranking":
        params = dict(LGBM_RANKER_PARAMS)
        params["boosting_type"] = ACTIVE_BOOSTING_TYPE
        if ACTIVE_BOOSTING_TYPE == "dart":
            params.update(DART_PARAMS)
        if overrides:
            params.update(overrides)
        return lgb.LGBMRanker(**params)

    params = dict(LGBM_PARAMS)
    params["boosting_type"] = ACTIVE_BOOSTING_TYPE
    if ACTIVE_BOOSTING_TYPE == "dart":
        params.update(DART_PARAMS)
    params.setdefault("verbosity", -1)
    if overrides:
        params.update(overrides)
    if mode == "classification":
        return lgb.LGBMClassifier(objective="binary", **params)
    return lgb.LGBMRegressor(objective="regression", **params)


def _date_splits(unique_dates: np.ndarray, n_splits: int, gap: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding TimeSeriesSplit with a label-overlap purge gap."""
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    return list(splitter.split(unique_dates))


def _select_features(
    train: pd.DataFrame,
    all_features: list[str],
    corr_threshold: float,
) -> tuple[list[str], pd.DataFrame]:
    categorical = [c for c in categorical_feature_columns(train) if c in all_features]
    continuous = [c for c in all_features if c not in categorical]
    usable = train[continuous].dropna(axis=1, how="all")
    kept_cont, corr = drop_highly_correlated_features(usable, threshold=corr_threshold)
    kept_cont = [
        c
        for c in kept_cont
        if pd.notna(train[c].std(skipna=True)) and train[c].std(skipna=True) > 0
    ]
    return kept_cont + categorical, corr


def _sort_group(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    ordered = df.sort_values(["date", "asset"]).copy()
    groups = ordered.groupby("date", sort=False).size().astype(int).tolist()
    return ordered, groups


def _fit(
    model: lgb.LGBMModel,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    mode: str,
) -> lgb.LGBMModel:
    categorical = [c for c in categorical_feature_columns(train) if c in features]
    callbacks = (
        []
        if ACTIVE_BOOSTING_TYPE == "dart"
        else [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
    )
    if mode == "ranking":
        # 20 日标签逐日高度重叠。按固定步长抽取训练 query，验证仍覆盖全部日期。
        train_dates = np.array(sorted(train["date"].unique()))
        sampled_dates = set(train_dates[::RANK_TRAIN_DATE_STRIDE])
        train = train[train["date"].isin(sampled_dates)]
        tr, tr_group = _sort_group(train)
        va, va_group = _sort_group(valid)
        model.fit(
            tr[features],
            tr["y"].astype(int),
            group=tr_group,
            eval_set=[(va[features], va["y"].astype(int))],
            eval_group=[va_group],
            eval_at=[1, 3, 5],
            categorical_feature=categorical,
            callbacks=callbacks,
        )
    else:
        model.fit(
            train[features],
            train["y"],
            eval_set=[(valid[features], valid["y"])],
            categorical_feature=categorical,
            callbacks=callbacks,
        )
    return model


def _predict(model: lgb.LGBMModel, df: pd.DataFrame, features: list[str], mode: str) -> np.ndarray:
    if mode == "classification":
        return model.predict_proba(df[features])[:, 1]
    return model.predict(df[features])


def rank_metrics(
    df: pd.DataFrame,
    pred: np.ndarray,
    target_col: str = "y_ret",
    forward_days: int | None = None,
) -> dict[str, float]:
    """计算与资产选择目标对齐的日期内 OOF 指标。"""
    if target_col not in df.columns:
        raise KeyError(f"指标目标列不存在: {target_col}")
    metric_horizon = forward_days
    if metric_horizon is None and "forward_days" in df.columns:
        horizons = df["forward_days"].dropna()
        metric_horizon = int(horizons.iloc[0]) if not horizons.empty else None
    metric_horizon = metric_horizon or FORWARD_DAYS

    tmp = df[["date", "asset", target_col]].copy().rename(columns={target_col: "target"})
    tmp["pred"] = np.asarray(pred)
    daily_ic: list[float] = []
    daily_ndcg: list[float] = []
    top1_hit: list[float] = []
    top3_hit: list[float] = []
    top1_excess: list[float] = []
    top3_excess: list[float] = []

    for _, g in tmp.groupby("date", sort=False):
        g = g.dropna(subset=["pred", "target"])
        if len(g) < 3 or g["pred"].nunique() < 2 or g["target"].nunique() < 2:
            continue
        ic = g["pred"].corr(g["target"], method="spearman")
        if pd.notna(ic):
            daily_ic.append(float(ic))

        # NDCG 要求 relevance 非负；日期内真实收益转为 0..n-1 等级。
        relevance = g["target"].rank(method="first").to_numpy() - 1
        daily_ndcg.append(float(ndcg_score([relevance], [g["pred"].to_numpy()], k=min(3, len(g)))))

        predicted = g.sort_values("pred", ascending=False)
        actual = g.sort_values("target", ascending=False)
        top1_hit.append(float(predicted.iloc[0]["asset"] == actual.iloc[0]["asset"]))
        actual_top3 = set(actual.head(3)["asset"])
        top3_hit.append(float(len(set(predicted.head(3)["asset"]) & actual_top3) / min(3, len(g))))
        equal_weight = float(g["target"].mean())
        top1_excess.append(float(predicted.head(1)["target"].mean() - equal_weight))
        top3_excess.append(float(predicted.head(3)["target"].mean() - equal_weight))

    ic_arr = np.asarray(daily_ic, dtype=float)
    return {
        "rank_ic": float(np.nanmean(ic_arr)) if len(ic_arr) else np.nan,
        "rank_ic_std": float(np.nanstd(ic_arr, ddof=1)) if len(ic_arr) > 1 else np.nan,
        "icir": (
            float(np.nanmean(ic_arr) / np.nanstd(ic_arr, ddof=1) * np.sqrt(252 / metric_horizon))
            if len(ic_arr) > 1 and np.nanstd(ic_arr, ddof=1) > 0
            else np.nan
        ),
        "ndcg_at_3": float(np.nanmean(daily_ndcg)) if daily_ndcg else np.nan,
        "top1_hit_rate": float(np.nanmean(top1_hit)) if top1_hit else np.nan,
        "top3_overlap": float(np.nanmean(top3_hit)) if top3_hit else np.nan,
        "top1_excess": float(np.nanmean(top1_excess)) if top1_excess else np.nan,
        "top3_excess": float(np.nanmean(top3_excess)) if top3_excess else np.nan,
        "n_metric_days": int(len(daily_ic)),
    }


def _tune_ranker(
    train: pd.DataFrame,
    features: list[str],
    gap: int,
) -> dict:
    """只在外层训练集内部做一次尾部时序验证，按 Rank IC 选稳健参数。"""
    dates = np.array(sorted(train["date"].unique()))
    if len(dates) < max(180, gap * 4):
        return RANKER_CANDIDATES[1]
    split = int(len(dates) * 0.8)
    inner_train_end = max(1, split - gap)
    tr_dates = set(dates[:inner_train_end])
    va_dates = set(dates[split:])
    tr = train[train["date"].isin(tr_dates)]
    va = train[train["date"].isin(va_dates)]
    if tr.empty or va.empty:
        return RANKER_CANDIDATES[1]

    best_params = RANKER_CANDIDATES[1]
    best_ic = -np.inf
    for candidate in RANKER_CANDIDATES:
        model = _fit(_make_model("ranking", candidate), tr, va, features, "ranking")
        pred = _predict(model, va, features, "ranking")
        ic = rank_metrics(va, pred)["rank_ic"]
        score = ic if pd.notna(ic) else -np.inf
        if score > best_ic:
            best_ic = score
            best_params = candidate
    return best_params


def train_with_timeseries_cv(
    panel: pd.DataFrame,
    mode: str = LABEL_MODE,
    n_splits: int = N_TS_SPLITS,
    corr_threshold: float = 0.8,
    purge_gap: int = FORWARD_DAYS,
    exclude_features: tuple[str, ...] = (),
) -> TrainResult:
    """按日期做 purged OOF 训练；早期无 OOF 区域保持 NaN，绝不回填泄漏预测。"""
    panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)
    excluded = set(exclude_features)
    feats_all = [c for c in feature_columns(panel) if c not in excluded]
    unique_dates = np.array(sorted(panel["date"].unique()))
    oof = np.full(len(panel), np.nan)
    fold_rows: list[dict] = []
    chosen: list[tuple] = []
    last_kept = feats_all
    last_corr = pd.DataFrame()

    for fold, (tr_idx, va_idx) in enumerate(
        _date_splits(unique_dates, n_splits=n_splits, gap=purge_gap), start=1
    ):
        tr_dates = set(unique_dates[tr_idx])
        va_dates = set(unique_dates[va_idx])
        tr_mask = panel["date"].isin(tr_dates)
        va_mask = panel["date"].isin(va_dates)
        tr = panel.loc[tr_mask].copy()
        va = panel.loc[va_mask].copy()
        kept, corr = _select_features(tr, feats_all, corr_threshold)
        last_kept, last_corr = kept, corr

        params = _tune_ranker(tr, kept, purge_gap) if mode == "ranking" else {}
        if params:
            chosen.append(tuple(sorted(params.items())))
        model = _fit(_make_model(mode, params), tr, va, kept, mode)
        pred = _predict(model, va, kept, mode)
        oof[np.flatnonzero(va_mask.to_numpy())] = pred

        metrics = rank_metrics(va, pred)
        metrics.update(
            {
                "fold": fold,
                "n_train_days": len(tr_dates),
                "n_valid_days": len(va_dates),
                "purge_gap_days": purge_gap,
                "n_features": len(kept),
                "train_end": str(max(tr_dates).date()),
                "valid_start": str(min(va_dates).date()),
            }
        )
        if mode == "regression":
            metrics["rmse"] = float(np.sqrt(mean_squared_error(va["y"], pred)))
            metrics["r2"] = float(r2_score(va["y"], pred))
        fold_rows.append(metrics)
        print(
            f"[CV fold {fold}] RankIC={metrics['rank_ic']:.4f} "
            f"NDCG@3={metrics['ndcg_at_3']:.4f} features={len(kept)} "
            f"gap={purge_gap}"
        )

    # 最终模型的参数取外层折中最常被选中的组合；特征仅在训练段拟合。
    final_params = dict(Counter(chosen).most_common(1)[0][0]) if chosen else {}
    split = int(len(unique_dates) * 0.85)
    tr_end = max(1, split - purge_gap)
    tr_dates = set(unique_dates[:tr_end])
    va_dates = set(unique_dates[split:])
    final_train = panel[panel["date"].isin(tr_dates)]
    final_valid = panel[panel["date"].isin(va_dates)]
    last_kept, last_corr = _select_features(final_train, feats_all, corr_threshold)
    final = _fit(_make_model(mode, final_params), final_train, final_valid, last_kept, mode)

    importance = (
        pd.DataFrame({"feature": last_kept, "importance": final.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    output_cols = [
        "date", "asset", "cn_name", "y", "y_ret", "y_excess",
        "y_holding_vol", "y_risk_adjusted", "y_target",
        "y_rank_pct", "y_rank_relevance", "forward_days", "label_target",
    ]
    oof_df = panel[[c for c in output_cols if c in panel.columns]].copy()
    oof_df["oof_pred"] = oof
    return TrainResult(
        model=final,
        features=last_kept,
        corr_kept=last_corr,
        oof_pred=oof_df,
        importance=importance,
        cv_metrics=pd.DataFrame(fold_rows),
    )


def predict_scores(
    model: lgb.LGBMModel | RankBlendModel,
    panel: pd.DataFrame,
    features: list[str],
    mode: str = LABEL_MODE,
) -> pd.Series:
    return predict_score_details(model, panel, features, mode=mode)["score"]


def predict_score_details(
    model: lgb.LGBMModel | RankBlendModel,
    panel: pd.DataFrame,
    features: list[str],
    mode: str = LABEL_MODE,
) -> pd.DataFrame:
    """返回融合得分及 rank/reg 分歧，供动态 BL 置信度使用。"""
    if isinstance(model, RankBlendModel):
        rank_raw = pd.Series(model.ranker.predict(panel[features]), index=panel.index)
        reg_raw = pd.Series(model.regressor.predict(panel[features]), index=panel.index)
        # 实盘预测截面只有一个日期；百分位化消除两个模型输出尺度差异。
        rank_pct = rank_raw.rank(pct=True, method="average")
        reg_pct = reg_raw.rank(pct=True, method="average")
        blend = model.rank_weight * rank_pct + (1.0 - model.rank_weight) * reg_pct
        return pd.DataFrame(
            {
                "score": blend,
                "rank_score": rank_pct,
                "reg_score": reg_pct,
                "model_disagreement": (rank_pct - reg_pct).abs(),
            }
        )
    score = pd.Series(_predict(model, panel, features, mode), index=panel.index)
    return pd.DataFrame(
        {
            "score": score,
            "rank_score": score,
            "reg_score": score,
            "model_disagreement": 0.0,
        }
    )


def fit_expanding_window(
    panel: pd.DataFrame,
    asof_date: pd.Timestamp,
    forward_days: int,
    mode: str = LABEL_MODE,
    corr_threshold: float = 0.8,
    min_train_days: int = 252,
    exclude_features: tuple[str, ...] = (),
) -> tuple[lgb.LGBMModel | RankBlendModel, list[str]] | tuple[None, list[str]]:
    """在再平衡日严格重训；只使用远期标签已完全实现的样本。"""
    panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)
    all_dates = np.array(sorted(panel["date"].unique()))
    pos = int(np.searchsorted(all_dates, pd.Timestamp(asof_date).to_datetime64(), side="right") - 1)
    if pos < forward_days + min_train_days:
        return None, []

    last_label_date = all_dates[pos - forward_days]
    pool = panel[panel["date"] <= last_label_date].copy()
    dates = np.array(sorted(pool["date"].unique()))
    if len(dates) < min_train_days:
        return None, []

    excluded = set(exclude_features)
    all_features = [c for c in feature_columns(panel) if c not in excluded]
    features, _ = _select_features(pool, all_features, corr_threshold)
    split = int(len(dates) * 0.85)
    tr_end = max(1, split - forward_days)
    tr = pool[pool["date"].isin(set(dates[:tr_end]))]
    va = pool[pool["date"].isin(set(dates[split:]))]
    if tr.empty or va.empty:
        return None, []

    if mode == "ranking":
        params = _tune_ranker(pool, features, forward_days)
        ranker = _fit(_make_model("ranking", params), tr, va, features, "ranking")

        # 回归模型与 Ranker 的误差结构互补。只融合两者的日期内排名，
        # 所以最终信号仍然是“相对强弱”，不是绝对收益预测。
        reg_tr = tr.copy()
        reg_va = va.copy()
        regression_target = "y_target" if "y_target" in pool.columns else "y_ret"
        reg_tr["y"] = reg_tr[regression_target]
        reg_va["y"] = reg_va[regression_target]
        regressor = _fit(
            _make_model("regression"),
            reg_tr,
            reg_va,
            features,
            "regression",
        )
        return RankBlendModel(
            ranker=ranker,
            regressor=regressor,
            rank_weight=RANK_BLEND_WEIGHT,
        ), features

    model = _fit(_make_model(mode), tr, va, features, mode)
    return model, features
