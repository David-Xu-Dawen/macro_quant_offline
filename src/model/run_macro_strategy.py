"""宏观资产配置主流程：特征 -> 标签 -> 时序CV LightGBM -> Black-Litterman -> 滚动回测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from black_litterman import allocate_over_time
from config import (
    AGGRESSION_PROFILES,
    BL_CONFIDENCE_MODE,
    BT_COST_BPS,
    BT_MIN_TRAIN_DAYS,
    BT_RETRAIN_EACH_REBALANCE,
    CORR_THRESHOLD,
    DEFAULT_AGGRESSION,
    FORWARD_DAYS,
    LABEL_MODE,
    LGBM_BOOSTING_TYPE,
    MODEL_DIR,
    OPTIMIZER_MODE,
    OUTPUT_DIR,
    PANEL_DIR,
    START_DATE,
    TRADE_UNIVERSE,
    WEIGHT_MIN,
    aggression_output_dir,
    resolve_aggression,
)
from macro_features import (
    add_labels,
    build_panel_features,
    drop_highly_correlated_features,
    feature_columns,
    load_price_panel,
)
from model_lgbm import rank_metrics, set_boosting_type, train_with_timeseries_cv
from rolling_backtest import run_rolling_backtest


def run_pipeline(
    start: str = START_DATE,
    forward_days: int = FORWARD_DAYS,
    label_mode: str = LABEL_MODE,
    corr_threshold: float = CORR_THRESHOLD,
    skip_rolling: bool = False,
    retrain: bool = BT_RETRAIN_EACH_REBALANCE,
    boosting_type: str = LGBM_BOOSTING_TYPE,
    confidence_mode: str = BL_CONFIDENCE_MODE,
    optimizer_mode: str | None = None,
    experiment_name: str = "",
    aggression: str = DEFAULT_AGGRESSION,
    label_target: str = "return",
    include_asset_id: bool = True,
):
    agg_key, profile = resolve_aggression(aggression)
    opt_mode = optimizer_mode or profile["optimizer_mode"]
    weight_max = float(profile["weight_max"])
    cvar_risk_aversion = float(profile["cvar_risk_aversion"])
    bl_risk_aversion = float(profile["bl_risk_aversion"])

    if experiment_name:
        output_dir = OUTPUT_DIR / experiment_name
    else:
        output_dir = aggression_output_dir(agg_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    set_boosting_type(boosting_type)

    print(f"\nAggression={agg_key} ({profile['label']}) | optimizer={opt_mode} | "
          f"w_max={weight_max} | cvar_ra={cvar_risk_aversion} | bl_ra={bl_risk_aversion}")
    print(f"Output -> {output_dir}")

    # ============================================================
    # 第一步：特征工程
    # ============================================================
    print("\n[1/5] Feature engineering ...")
    close = load_price_panel(TRADE_UNIVERSE, start=start)
    panel = build_panel_features(close, TRADE_UNIVERSE)
    excluded_features = () if include_asset_id else ("asset_id",)
    feat_cols = [c for c in feature_columns(panel) if c not in excluded_features]

    sample_X = panel[feat_cols].dropna()
    kept_report, corr_kept = drop_highly_correlated_features(sample_X, threshold=corr_threshold)
    corr_full = sample_X.corr()
    corr_full.to_csv(output_dir / "feature_corr_full.csv")
    corr_kept.to_csv(output_dir / "feature_corr_kept.csv")
    pd.Series(kept_report, name="feature").to_csv(output_dir / "features_after_corr_filter.csv", index=False)
    print(f"  assets={list(close.columns)}")
    print(f"  raw features={len(feat_cols)} | after |ρ|>{corr_threshold} filter (report)={len(kept_report)}")
    print(f"  dropped={sorted(set(feat_cols) - set(kept_report))}")

    # ============================================================
    # 第二步：标签定义
    # ============================================================
    print("\n[2/5] Labeling ...")
    panel = add_labels(
        panel,
        close,
        forward_days=forward_days,
        mode=label_mode,
        target=label_target,
    )
    panel_path = (
        output_dir / "macro_lgbm_panel.csv"
        if experiment_name
        else PANEL_DIR / "macro_lgbm_panel.csv"
    )
    panel.to_csv(panel_path, index=False)
    try:
        panel.to_parquet(panel_path.with_suffix(".parquet"), index=False)
    except Exception:
        pass
    print(
        f"  mode={label_mode} | target={label_target} | forward={forward_days}d | rows={len(panel)} | "
        f"{panel['date'].min().date()} ~ {panel['date'].max().date()}"
    )

    # ============================================================
    # 第三步：LightGBM + TimeSeriesSplit
    # ============================================================
    print("\n[3/5] LightGBM time-series CV ...")
    result = train_with_timeseries_cv(
        panel,
        mode=label_mode,
        corr_threshold=corr_threshold,
        purge_gap=forward_days,
        exclude_features=excluded_features,
    )
    result.cv_metrics.to_csv(output_dir / "cv_metrics.csv", index=False)
    result.importance.to_csv(output_dir / "feature_importance.csv", index=False)
    result.oof_pred.to_csv(output_dir / "oof_predictions_ranker.csv", index=False)
    model_path = (
        output_dir / "macro_lgbm_bl.joblib"
        if experiment_name
        else MODEL_DIR / "macro_lgbm_bl.joblib"
    )
    joblib.dump(
        {"model": result.model, "features": result.features, "label_mode": label_mode},
        model_path,
    )
    print("\n=== CV metrics ===")
    print(result.cv_metrics.to_string(index=False))
    print("\n=== Feature Importance (Top 15) ===")
    print(result.importance.head(15).to_string(index=False))

    # 与旧的“绝对收益回归”同一套 purged folds 对照。两者都用横截面指标
    # 评价，避免拿 Rank IC 与回归 R² 做无意义的直接比较。
    if label_mode == "ranking":
        print("\n=== Purged regression baseline ===")
        regression_panel = panel.copy()
        regression_panel["y"] = regression_panel["y_target"]
        baseline = train_with_timeseries_cv(
            regression_panel,
            mode="regression",
            corr_threshold=corr_threshold,
            purge_gap=forward_days,
            exclude_features=excluded_features,
        )
        baseline.cv_metrics.to_csv(output_dir / "cv_metrics_regression_baseline.csv", index=False)
        baseline.oof_pred.to_csv(output_dir / "oof_predictions_regression_baseline.csv", index=False)

        # 固定 25% Ranker + 75% 回归的日期内百分位融合。权重是预先固定的保守值，
        # 不在最终回测上搜索，避免把回测噪声优化成“好结果”。
        rank_raw = result.oof_pred["oof_pred"]
        reg_raw = baseline.oof_pred["oof_pred"]
        rank_pct = rank_raw.groupby(result.oof_pred["date"]).rank(pct=True, method="average")
        reg_pct = reg_raw.groupby(result.oof_pred["date"]).rank(pct=True, method="average")
        ensemble_pred = 0.25 * rank_pct + 0.75 * reg_pct
        result.oof_pred["rank_pred_pct"] = rank_pct
        result.oof_pred["reg_pred_pct"] = reg_pct
        result.oof_pred["model_disagreement"] = (rank_pct - reg_pct).abs()
        result.oof_pred["oof_pred"] = ensemble_pred
        result.oof_pred.to_csv(output_dir / "oof_predictions.csv", index=False)

        valid = result.oof_pred["oof_pred"].notna()
        ensemble_overall = rank_metrics(
            result.oof_pred.loc[valid],
            result.oof_pred.loc[valid, "oof_pred"].to_numpy(),
            forward_days=forward_days,
        )
        ensemble_folds = []
        valid_dates = sorted(result.oof_pred.loc[valid, "date"].unique())
        cursor = 0
        for _, fold_row in result.cv_metrics.iterrows():
            n_fold_dates = int(fold_row["n_valid_days"])
            fold_dates = valid_dates[cursor : cursor + n_fold_dates]
            cursor += n_fold_dates
            fold_data = result.oof_pred[result.oof_pred["date"].isin(fold_dates)]
            fold_metric = rank_metrics(
                fold_data,
                fold_data["oof_pred"].to_numpy(),
                forward_days=forward_days,
            )
            fold_metric["fold"] = int(fold_row["fold"])
            ensemble_folds.append(fold_metric)
        pd.DataFrame(ensemble_folds).to_csv(output_dir / "cv_metrics_rank_ensemble.csv", index=False)

        compare_cols = [
            "rank_ic",
            "icir",
            "ndcg_at_3",
            "top1_hit_rate",
            "top3_overlap",
            "top1_excess",
            "top3_excess",
        ]
        comparison = pd.DataFrame(
            [
                {
                    "model": "lambdarank",
                    **{c: result.cv_metrics[c].mean() for c in compare_cols},
                },
                {
                    "model": "regression",
                    **{c: baseline.cv_metrics[c].mean() for c in compare_cols},
                    "r2": baseline.cv_metrics["r2"].mean(),
                },
                {
                    "model": "rank_ensemble_25_75",
                    **{c: ensemble_overall[c] for c in compare_cols},
                },
            ]
        )
        comparison.to_csv(output_dir / "ranker_vs_regression.csv", index=False)
        print(comparison.to_string(index=False))
        joblib.dump(
            {
                "ranker": result.model,
                "regressor": baseline.model,
                "features": result.features,
                "label_mode": "ranking",
                "rank_weight": 0.25,
            },
            model_path,
        )
    else:
        result.oof_pred.to_csv(output_dir / "oof_predictions.csv", index=False)

    # ============================================================
    # 第四步：OOF 信号 -> BL 权重（快速诊断）
    # ============================================================
    print("\n[4/5] Black-Litterman allocation (OOF diagnostic) ...")
    weights = allocate_over_time(
        result.oof_pred.dropna(subset=["oof_pred"]),
        close,
        rebalance_every=forward_days,
        score_col="oof_pred",
        confidence_mode=confidence_mode,
        optimizer_mode=opt_mode,
        weight_max=weight_max,
        weight_min=WEIGHT_MIN,
        cvar_risk_aversion=cvar_risk_aversion,
        bl_risk_aversion=bl_risk_aversion,
    )
    weights.to_csv(output_dir / "bl_weights.csv", index=False)
    if not weights.empty:
        latest_dt = weights["date"].max()
        latest = weights[weights["date"] == latest_dt].sort_values("weight", ascending=False)
        latest.to_csv(output_dir / "bl_weights_latest.csv", index=False)
        print(f"\n=== Latest BL weights @ {pd.Timestamp(latest_dt).date()} ===")
        print(latest[["asset", "score", "w_prior_risk_parity", "mu_bl", "weight"]].to_string(index=False))

    # ============================================================
    # 第五步：严格 walk-forward 滚动回测
    # 每个再平衡日重训 -> BL 权重 -> 日净值 / 回撤 / 换手
    # ============================================================
    bt = None
    if not skip_rolling:
        print("\n[5/5] Walk-forward rolling backtest ...")
        print(
            f"  retrain={retrain} | rebalance_every={forward_days} | "
            f"min_train_days={BT_MIN_TRAIN_DAYS} | cost_bps={BT_COST_BPS}"
        )
        bt = run_rolling_backtest(
            panel=panel,
            close=close,
            rebalance_every=forward_days,
            forward_days=forward_days,
            label_mode=label_mode,
            corr_threshold=corr_threshold,
            min_train_days=BT_MIN_TRAIN_DAYS,
            cost_bps=BT_COST_BPS,
            retrain=retrain,
            oof_pred=result.oof_pred,
            confidence_mode=confidence_mode,
            optimizer_mode=opt_mode,
            weight_max=weight_max,
            weight_min=WEIGHT_MIN,
            cvar_risk_aversion=cvar_risk_aversion,
            bl_risk_aversion=bl_risk_aversion,
            exclude_features=excluded_features,
        )
        bt.nav.to_csv(output_dir / "rolling_nav.csv")
        bt.weights.to_csv(output_dir / "rolling_weights.csv", index=False)
        bt.rebalance_log.to_csv(output_dir / "rolling_rebalance_log.csv", index=False)
        pd.Series(bt.metrics).to_csv(output_dir / "rolling_metrics.csv", header=["value"])
        (output_dir / "rolling_metrics.json").write_text(
            json.dumps(bt.metrics, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print("\n=== Rolling backtest metrics ===")
        m = bt.metrics
        def fmt(key, as_pct=False):
            v = m.get(key)
            if v is None:
                return "n/a"
            if as_pct:
                return f"{v:.2%}"
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        print(f"  period: {m.get('start')} ~ {m.get('end')} | rebalances={m.get('n_rebalances')}")
        print(f"  strategy: ann={fmt('strategy_ann_return', True)} vol={fmt('strategy_ann_vol', True)} "
              f"sharpe={fmt('strategy_sharpe')} mdd={fmt('strategy_max_drawdown', True)}")
        print(f"  equal-weight: ann={fmt('equal_weight_ann_return', True)} mdd={fmt('equal_weight_max_drawdown', True)}")
        print(f"  cash: ann={fmt('cash_ann_return', True)}")
        print(f"  excess vs cash={fmt('excess_ann_vs_cash', True)} | vs EW={fmt('excess_ann_vs_ew', True)}")
        print(f"  avg turnover/rebalance={fmt('avg_turnover', True)} | cost_bps={m.get('cost_bps')}")

    meta = {
        "start": start,
        "forward_days": forward_days,
        "label_mode": label_mode,
        "label_target": label_target,
        "include_asset_id": include_asset_id,
        "corr_threshold": corr_threshold,
        "universe": TRADE_UNIVERSE,
        "n_features_final": len(result.features),
        "features_final": result.features,
        "boosting_type": boosting_type,
        "confidence_mode": confidence_mode,
        "optimizer_mode": opt_mode,
        "aggression": agg_key,
        "aggression_label": profile["label"],
        "weight_max": weight_max,
        "cvar_risk_aversion": cvar_risk_aversion,
        "bl_risk_aversion": bl_risk_aversion,
        "experiment_name": experiment_name,
        "rolling_retrain": retrain,
        "rolling_metrics": bt.metrics if bt is not None else None,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nArtifacts saved under: {output_dir}")
    return result, weights, bt


def main():
    parser = argparse.ArgumentParser(description="宏观大类资产 LightGBM + Black-Litterman 框架")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--forward-days", type=int, default=FORWARD_DAYS)
    parser.add_argument(
        "--label-mode",
        choices=["ranking", "regression", "classification"],
        default=LABEL_MODE,
    )
    parser.add_argument("--corr-threshold", type=float, default=CORR_THRESHOLD)
    parser.add_argument(
        "--label-target",
        choices=["return", "risk_adjusted"],
        default="return",
        help="排序/回归目标：未来收益或未来风险调整超额收益",
    )
    parser.add_argument(
        "--exclude-asset-id",
        action="store_true",
        help="从特征中移除 asset_id，检查模型是否依赖资产身份记忆",
    )
    parser.add_argument("--boosting-type", choices=["gbdt", "dart"], default=LGBM_BOOSTING_TYPE)
    parser.add_argument(
        "--confidence-mode",
        choices=["scalar", "dynamic"],
        default=BL_CONFIDENCE_MODE,
    )
    parser.add_argument(
        "--optimizer-mode",
        choices=["mean_variance", "cvar", "auto"],
        default=None,
        help="覆盖档位默认优化器；默认跟随 --aggression",
    )
    parser.add_argument(
        "--aggression",
        choices=list(AGGRESSION_PROFILES.keys()),
        default=DEFAULT_AGGRESSION,
        help="激进程度：conservative/balanced/aggressive（稳健/均衡/进取）",
    )
    parser.add_argument(
        "--experiment-name",
        default="",
        help="非空时写入 output/<name>，用于 A/B 而不覆盖生产结果",
    )
    parser.add_argument("--skip-rolling", action="store_true", help="跳过滚动回测（仅 CV+OOF）")
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="滚动回测不重训，改用 OOF 得分（更快，但不算严格 walk-forward）",
    )
    args = parser.parse_args()
    run_pipeline(
        start=args.start,
        forward_days=args.forward_days,
        label_mode=args.label_mode,
        corr_threshold=args.corr_threshold,
        skip_rolling=args.skip_rolling,
        retrain=not args.no_retrain,
        boosting_type=args.boosting_type,
        confidence_mode=args.confidence_mode,
        optimizer_mode=args.optimizer_mode,
        experiment_name=args.experiment_name,
        aggression=args.aggression,
        label_target=args.label_target,
        include_asset_id=not args.exclude_asset_id,
    )


if __name__ == "__main__":
    main()
