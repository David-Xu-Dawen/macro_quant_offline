#!/usr/bin/env python3
"""基于已有 OOF 信号，快速生成各激进档位的组合回测结果。

不重训 LightGBM，只重跑 Black-Litterman / 滚动净值。
均衡档沿用 output/；其余写入 output/aggression_{key}/。

用法:
  cd "model prediction" && python3 run_aggression_profiles.py
  python3 run_aggression_profiles.py --only conservative,aggressive
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from black_litterman import allocate_over_time
from config import (
    AGGRESSION_PROFILES,
    BL_CONFIDENCE_MODE,
    BT_COST_BPS,
    BT_MIN_TRAIN_DAYS,
    DEFAULT_AGGRESSION,
    FORWARD_DAYS,
    LABEL_MODE,
    OUTPUT_DIR,
    PANEL_DIR,
    START_DATE,
    TRADE_UNIVERSE,
    WEIGHT_MIN,
    aggression_output_dir,
    resolve_aggression,
)
from macro_features import load_price_panel
from rolling_backtest import run_rolling_backtest

SHARED_FILES = [
    "oof_predictions.csv",
    "oof_predictions_ranker.csv",
    "oof_predictions_regression_baseline.csv",
    "feature_importance.csv",
    "cv_metrics.csv",
    "cv_metrics_regression_baseline.csv",
    "cv_metrics_rank_ensemble.csv",
    "ranker_vs_regression.csv",
    "features_after_corr_filter.csv",
]


def _copy_shared(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in SHARED_FILES:
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)


def run_one(key: str, *, base_dir: Path, close: pd.DataFrame, panel: pd.DataFrame, oof: pd.DataFrame) -> dict:
    agg_key, profile = resolve_aggression(key)
    out_dir = aggression_output_dir(agg_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_shared(base_dir, out_dir)

    opt_mode = profile["optimizer_mode"]
    weight_max = float(profile["weight_max"])
    cvar_ra = float(profile["cvar_risk_aversion"])
    bl_ra = float(profile["bl_risk_aversion"])

    print(f"\n=== {profile['label']} ({agg_key}) ===")
    print(f"  optimizer={opt_mode} w_max={weight_max} cvar_ra={cvar_ra} bl_ra={bl_ra}")
    print(f"  -> {out_dir}")

    weights = allocate_over_time(
        oof.dropna(subset=["oof_pred"]),
        close,
        rebalance_every=FORWARD_DAYS,
        score_col="oof_pred",
        confidence_mode=BL_CONFIDENCE_MODE,
        optimizer_mode=opt_mode,
        weight_max=weight_max,
        weight_min=WEIGHT_MIN,
        cvar_risk_aversion=cvar_ra,
        bl_risk_aversion=bl_ra,
    )
    weights.to_csv(out_dir / "bl_weights.csv", index=False)
    if not weights.empty:
        latest_dt = weights["date"].max()
        latest = weights[weights["date"] == latest_dt].sort_values("weight", ascending=False)
        latest.to_csv(out_dir / "bl_weights_latest.csv", index=False)

    bt = run_rolling_backtest(
        panel=panel,
        close=close,
        rebalance_every=FORWARD_DAYS,
        forward_days=FORWARD_DAYS,
        label_mode=LABEL_MODE,
        min_train_days=BT_MIN_TRAIN_DAYS,
        cost_bps=BT_COST_BPS,
        retrain=False,
        oof_pred=oof,
        confidence_mode=BL_CONFIDENCE_MODE,
        optimizer_mode=opt_mode,
        weight_max=weight_max,
        weight_min=WEIGHT_MIN,
        cvar_risk_aversion=cvar_ra,
        bl_risk_aversion=bl_ra,
    )
    bt.nav.to_csv(out_dir / "rolling_nav.csv")
    bt.weights.to_csv(out_dir / "rolling_weights.csv", index=False)
    bt.rebalance_log.to_csv(out_dir / "rolling_rebalance_log.csv", index=False)
    pd.Series(bt.metrics).to_csv(out_dir / "rolling_metrics.csv", header=["value"])
    (out_dir / "rolling_metrics.json").write_text(
        json.dumps(bt.metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    base_meta = {}
    base_meta_path = base_dir / "run_meta.json"
    if base_meta_path.exists():
        base_meta = json.loads(base_meta_path.read_text(encoding="utf-8"))

    meta = {
        **{k: base_meta.get(k) for k in ("start", "forward_days", "label_mode", "corr_threshold",
                                         "universe", "n_features_final", "features_final",
                                         "boosting_type", "confidence_mode")},
        "optimizer_mode": opt_mode,
        "aggression": agg_key,
        "aggression_label": profile["label"],
        "weight_max": weight_max,
        "cvar_risk_aversion": cvar_ra,
        "bl_risk_aversion": bl_ra,
        "rolling_retrain": False,
        "signal_source": "oof_no_retrain",
        "rolling_metrics": bt.metrics,
        "note": "组合层档位回测：复用生产 OOF 信号，不重训 LightGBM。",
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # 画图
    from plot_results import main as plot_main
    import sys

    argv_backup = sys.argv
    try:
        if agg_key == DEFAULT_AGGRESSION:
            sys.argv = ["plot_results.py"]
        else:
            sys.argv = ["plot_results.py", "--experiment-name", f"aggression_{agg_key}"]
        plot_main()
    finally:
        sys.argv = argv_backup

    m = bt.metrics
    print(
        f"  ann={m.get('strategy_ann_return', 0):.2%} "
        f"sharpe={m.get('strategy_sharpe', 0):.2f} "
        f"mdd={m.get('strategy_max_drawdown', 0):.2%} "
        f"vs_ew={m.get('excess_ann_vs_ew', 0):.2%}"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="生成各激进档位组合回测")
    parser.add_argument(
        "--only",
        default="",
        help="逗号分隔档位 key；默认除均衡外全部（均衡沿用现有 output）",
    )
    parser.add_argument(
        "--include-balanced",
        action="store_true",
        help="也重算均衡档（会覆盖 output/ 组合结果，但仍用 OOF 无重训）",
    )
    args = parser.parse_args()

    base_dir = OUTPUT_DIR
    oof_path = base_dir / "oof_predictions.csv"
    panel_path = PANEL_DIR / "macro_lgbm_panel.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"缺少 {oof_path}，请先跑完整 run_macro_strategy / run_all")
    if not panel_path.exists():
        raise FileNotFoundError(f"缺少 {panel_path}")

    oof = pd.read_csv(oof_path, parse_dates=["date"])
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    close = load_price_panel(TRADE_UNIVERSE, start=START_DATE)

    if args.only.strip():
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
    else:
        keys = [k for k in AGGRESSION_PROFILES if k != DEFAULT_AGGRESSION]
        if args.include_balanced:
            keys = [DEFAULT_AGGRESSION, *keys]

    # 给均衡档补写 aggression 元数据（不重算）
    if DEFAULT_AGGRESSION not in keys:
        bal_meta_path = base_dir / "run_meta.json"
        if bal_meta_path.exists():
            meta = json.loads(bal_meta_path.read_text(encoding="utf-8"))
            _, profile = resolve_aggression(DEFAULT_AGGRESSION)
            meta.setdefault("aggression", DEFAULT_AGGRESSION)
            meta.setdefault("aggression_label", profile["label"])
            meta.setdefault("weight_max", profile["weight_max"])
            meta.setdefault("cvar_risk_aversion", profile["cvar_risk_aversion"])
            meta.setdefault("bl_risk_aversion", profile["bl_risk_aversion"])
            bal_meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

    for key in keys:
        run_one(key, base_dir=base_dir, close=close, panel=panel, oof=oof)

    print("\nDone.")


if __name__ == "__main__":
    main()
