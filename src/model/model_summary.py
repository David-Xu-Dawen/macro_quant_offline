"""Pure model-output summary helpers, independent of the web application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    AGGRESSION_PROFILES,
    DEFAULT_AGGRESSION,
    aggression_output_dir,
    resolve_aggression,
)

ASSET_CN = {
    "sse50": "上证50",
    "csi300": "沪深300",
    "csi500": "中证500",
    "csi1000": "中证1000",
    "bond_gov": "中债国债",
    "bond_corp": "中债企业债",
    "csi_cb": "中证转债",
    "crude_sc": "原油",
    "gold_au": "沪金",
    "spx": "标普500",
    "hsi": "恒生指数",
}

FIGURE_META = [
    {"file": "01_nav_curve.png", "title": "净值曲线", "desc": "策略 vs 等权 vs 国债"},
    {"file": "02_drawdown.png", "title": "策略回撤", "desc": "策略最大回撤轨迹"},
]


def _pct(value: float | None, digits: int = 1) -> str | None:
    if value is None or (isinstance(value, float) and not pd.notna(value)):
        return None
    return f"{100 * float(value):.{digits}f}%"


def _num(value: float | None, digits: int = 2) -> float | None:
    if value is None or (isinstance(value, float) and not pd.notna(value)):
        return None
    return round(float(value), digits)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def list_profiles() -> list[dict[str, Any]]:
    rows = []
    for key, profile in AGGRESSION_PROFILES.items():
        out = aggression_output_dir(key)
        rows.append(
            {
                "key": key,
                "label": profile["label"],
                "weight_max": profile["weight_max"],
                "cvar_risk_aversion": profile["cvar_risk_aversion"],
                "bl_risk_aversion": profile["bl_risk_aversion"],
                "optimizer_mode": profile["optimizer_mode"],
                "ready": (out / "rolling_metrics.json").exists(),
                "is_default": key == DEFAULT_AGGRESSION,
            }
        )
    return rows


def figure_path(name: str, aggression: str | None = None) -> Path:
    key, _ = resolve_aggression(aggression)
    safe = Path(name).name
    if not safe.endswith(".png"):
        raise ValueError("仅支持 png")
    return aggression_output_dir(key) / "figures" / safe


def summarize(aggression: str | None = None) -> dict[str, Any]:
    key, profile = resolve_aggression(aggression)
    out_dir = aggression_output_dir(key)
    metrics_path = out_dir / "rolling_metrics.json"
    if not out_dir.exists():
        raise FileNotFoundError(f"档位「{profile['label']}」结果目录不存在: {out_dir}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"档位「{profile['label']}」缺少 rolling_metrics.json")

    meta = _load_json(out_dir / "run_meta.json")
    metrics = _load_json(metrics_path) or meta.get("rolling_metrics") or {}
    weights = _read_csv(out_dir / "bl_weights_latest.csv")
    feat_imp = _read_csv(out_dir / "feature_importance.csv")
    model_cmp = _read_csv(out_dir / "ranker_vs_regression.csv")
    cv = _read_csv(out_dir / "cv_metrics.csv")

    latest_weights: list[dict[str, Any]] = []
    as_of = None
    if not weights.empty:
        w = weights.copy()
        if "date" in w.columns:
            as_of = str(pd.to_datetime(w["date"]).max().date())
        for _, row in w.sort_values("weight", ascending=False).iterrows():
            asset = str(row.get("asset", ""))
            latest_weights.append(
                {
                    "asset": asset,
                    "asset_cn": ASSET_CN.get(asset, asset),
                    "weight": _num(row.get("weight"), 4),
                    "score": _num(row.get("score"), 3),
                    "mu_bl": _num(row.get("mu_bl"), 4),
                    "w_active_bl": _num(row.get("w_active_bl"), 4),
                    "view_confidence": _num(row.get("view_confidence"), 3),
                    "optimizer_used": str(row.get("optimizer_used", "")),
                }
            )

    top_features = []
    if not feat_imp.empty and {"feature", "importance"}.issubset(feat_imp.columns):
        top_features = [
            {
                "feature": str(row["feature"]),
                "importance": _num(row["importance"], 1),
            }
            for _, row in feat_imp.sort_values("importance", ascending=False)
            .head(12)
            .iterrows()
        ]

    model_comparison = []
    for _, row in model_cmp.iterrows():
        model_comparison.append(
            {
                "model": str(row.get("model", "")),
                "rank_ic": _num(row.get("rank_ic"), 3),
                "icir": _num(row.get("icir"), 3),
                "ndcg_at_3": _num(row.get("ndcg_at_3"), 3),
                "top1_hit_rate": _num(row.get("top1_hit_rate"), 3),
                "top3_overlap": _num(row.get("top3_overlap"), 3),
            }
        )

    cv_folds = []
    for _, row in cv.iterrows():
        cv_folds.append(
            {
                "fold": (
                    int(row["fold"])
                    if "fold" in row and pd.notna(row["fold"])
                    else None
                ),
                "rank_ic": _num(row.get("rank_ic"), 3),
                "icir": _num(row.get("icir"), 3),
                "ndcg_at_3": _num(row.get("ndcg_at_3"), 3),
                "top1_hit_rate": _num(row.get("top1_hit_rate"), 3),
                "train_end": str(row.get("train_end", "")),
                "valid_start": str(row.get("valid_start", "")),
            }
        )

    figures = []
    for item in FIGURE_META:
        if (out_dir / "figures" / item["file"]).exists():
            figures.append(
                {
                    **item,
                    "url": f"/model-prediction/figures/{item['file']}?aggression={key}",
                }
            )

    retrain = meta.get("rolling_retrain")
    signal_note = "严格 walk-forward 重训" if retrain else "复用 OOF 信号（无重训，仅改组合层）"
    universe = meta.get("universe") or []
    return {
        "as_of": as_of,
        "aggression": key,
        "aggression_label": profile["label"],
        "aggression_profile": {
            "weight_max": profile["weight_max"],
            "cvar_risk_aversion": profile["cvar_risk_aversion"],
            "bl_risk_aversion": profile["bl_risk_aversion"],
            "optimizer_mode": profile["optimizer_mode"],
        },
        "profiles": list_profiles(),
        "forward_days": meta.get("forward_days"),
        "label_mode": meta.get("label_mode"),
        "corr_threshold": meta.get("corr_threshold"),
        "n_features_final": meta.get("n_features_final"),
        "universe": [{"id": asset, "name": ASSET_CN.get(asset, asset)} for asset in universe],
        "metrics": {
            "ann_return": _num(metrics.get("strategy_ann_return"), 4),
            "ann_vol": _num(metrics.get("strategy_ann_vol"), 4),
            "sharpe": _num(metrics.get("strategy_sharpe"), 3),
            "max_drawdown": _num(metrics.get("strategy_max_drawdown"), 4),
            "total_return": _num(metrics.get("strategy_total_return"), 4),
            "excess_ann_vs_ew": _num(metrics.get("excess_ann_vs_ew"), 4),
            "excess_ann_vs_cash": _num(metrics.get("excess_ann_vs_cash"), 4),
            "avg_turnover": _num(metrics.get("avg_turnover"), 3),
            "avg_annual_turnover": _num(metrics.get("avg_annual_turnover"), 2),
            "n_rebalances": metrics.get("n_rebalances"),
            "cost_bps": metrics.get("cost_bps"),
            "rebalance_every": metrics.get("rebalance_every"),
            "start": metrics.get("start"),
            "end": metrics.get("end"),
            "equal_weight_ann_return": _num(metrics.get("equal_weight_ann_return"), 4),
            "equal_weight_sharpe": _num(metrics.get("equal_weight_sharpe"), 3),
            "cash_ann_return": _num(metrics.get("cash_ann_return"), 4),
            "portfolio_cvar_95": _num(metrics.get("portfolio_cvar_95"), 4),
            "mean_view_confidence": _num(metrics.get("mean_view_confidence"), 3),
            "cvar_optimizer_usage": _num(metrics.get("cvar_optimizer_usage"), 3),
            "optimizer_fallback_rate": _num(metrics.get("optimizer_fallback_rate"), 3),
        },
        "metrics_display": {
            "年化": _pct(metrics.get("strategy_ann_return")),
            "波动": _pct(metrics.get("strategy_ann_vol")),
            "Sharpe": _num(metrics.get("strategy_sharpe"), 2),
            "最大回撤": _pct(metrics.get("strategy_max_drawdown")),
            "相对等权超额年化": _pct(metrics.get("excess_ann_vs_ew"), 2),
            "平均换手": _pct(metrics.get("avg_turnover")),
            "95% CVaR": _pct(metrics.get("portfolio_cvar_95")),
            "平均观点置信度": _pct(metrics.get("mean_view_confidence")),
        },
        "latest_weights": latest_weights,
        "top_features": top_features,
        "model_comparison": model_comparison,
        "cv_folds": cv_folds,
        "figures": figures,
        "note": (
            f"激进程度={profile['label']}（上限{profile['weight_max']:.0%}，"
            f"CVaR厌恶{profile['cvar_risk_aversion']}，优化器={profile['optimizer_mode']}）；"
            "档位只改组合约束与风险厌恶，不改预测信号。"
            f"标签={meta.get('label_mode')}，{signal_note}；"
            f"前瞻 {meta.get('forward_days')} 交易日，成本 {metrics.get('cost_bps')}bps，"
            f"每 {metrics.get('rebalance_every')} 日再平衡。"
        ),
    }
