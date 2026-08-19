"""绘制滚动回测与模型结果图，输出到 output/figures/。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 中文字体回退
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 160
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

from config import OUTPUT_DIR

OUT = OUTPUT_DIR
FIG = OUT / "figures"

ASSET_CN = {
    "sse50": "上证50",
    "csi300": "沪深300",
    "csi500": "中证500",
    "csi1000": "中证1000",
    "bond_gov": "国债",
    "bond_corp": "企债",
    "csi_cb": "转债",
    "crude_sc": "原油",
    "gold_au": "沪金",
    "spx": "标普500",
}


def _load(out_dir: Path = OUT):
    nav = pd.read_csv(out_dir / "rolling_nav.csv", parse_dates=["date"]).set_index("date")
    weights = pd.read_csv(out_dir / "rolling_weights.csv", parse_dates=["date"])
    reb = pd.read_csv(out_dir / "rolling_rebalance_log.csv", parse_dates=["date"])
    imp = pd.read_csv(out_dir / "feature_importance.csv")
    metrics = json.loads((out_dir / "rolling_metrics.json").read_text(encoding="utf-8"))
    return nav, weights, reb, imp, metrics


def plot_nav(nav: pd.DataFrame, metrics: dict, path: Path):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(nav.index, nav["strategy"], label="策略 (LGBM+BL)", lw=1.8, color="#1f4e79")
    ax.plot(nav.index, nav["equal_weight"], label="等权", lw=1.4, color="#7a7a7a", alpha=0.9)
    ax.plot(nav.index, nav["cash"], label="国债(现金)", lw=1.4, color="#2e7d32", alpha=0.85)
    ax.axhline(1.0, color="#bbbbbb", lw=0.8, ls="--")
    ax.set_title("滚动回测净值曲线（Walk-forward，含交易成本）", fontsize=13, pad=10)
    ax.set_ylabel("累计净值")
    ax.set_xlabel("日期")
    ax.legend(frameon=False, loc="upper left")
    txt = (
        f"策略年化 {metrics['strategy_ann_return']:.1%}  "
        f"夏普 {metrics['strategy_sharpe']:.2f}  "
        f"最大回撤 {metrics['strategy_max_drawdown']:.1%}\n"
        f"等权年化 {metrics['equal_weight_ann_return']:.1%}  |  "
        f"相对等权超额 {metrics['excess_ann_vs_ew']:.1%}"
    )
    ax.text(0.01, 0.02, txt, transform=ax.transAxes, fontsize=9, color="#333333", va="bottom")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown(nav: pd.DataFrame, path: Path):
    dd = nav["strategy"] / nav["strategy"].cummax() - 1.0
    dd_ew = nav["equal_weight"] / nav["equal_weight"].cummax() - 1.0
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.fill_between(dd.index, dd.values, 0, color="#1f4e79", alpha=0.35, label="策略回撤")
    ax.plot(dd_ew.index, dd_ew.values, color="#7a7a7a", lw=1.0, label="等权回撤")
    ax.set_title("回撤曲线", fontsize=13, pad=10)
    ax.set_ylabel("回撤")
    ax.set_xlabel("日期")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_weights(weights: pd.DataFrame, path: Path):
    wp = weights.pivot_table(index="date", columns="asset", values="weight", aggfunc="last").sort_index()
    # 按平均权重排序堆叠，图例更清晰
    order = wp.mean().sort_values(ascending=False).index.tolist()
    wp = wp[order]
    colors = plt.cm.tab20(np.linspace(0, 1, len(order)))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.stackplot(
        wp.index,
        *[wp[c].fillna(0).values for c in order],
        labels=[ASSET_CN.get(c, c) for c in order],
        colors=colors,
        alpha=0.92,
    )
    ax.set_ylim(0, 1)
    ax.set_title("再平衡权重轨迹（堆叠）", fontsize=13, pad=10)
    ax.set_ylabel("权重")
    ax.set_xlabel("再平衡日期")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, path: Path):
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["feature"], top["importance"], color="#1f4e79", alpha=0.85)
    ax.set_title("LightGBM 特征重要性 Top 12", fontsize=13, pad=10)
    ax.set_xlabel("Importance (split count)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_turnover(reb: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.bar(reb["date"], reb["turnover"], width=18, color="#c45c26", alpha=0.8, label="单次换手")
    ax.axhline(reb["turnover"].mean(), color="#1f4e79", ls="--", lw=1.2, label=f"均值 {reb['turnover'].mean():.1%}")
    ax.set_title("再平衡换手率", fontsize=13, pad=10)
    ax.set_ylabel("换手 (单边 L1/2)")
    ax.set_xlabel("日期")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_latest_weights(weights: pd.DataFrame, path: Path):
    latest = weights[weights["date"] == weights["date"].max()].copy()
    latest["name"] = latest["asset"].map(lambda x: ASSET_CN.get(x, x))
    latest = latest.sort_values("weight", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(latest["name"], latest["weight"], color="#1f4e79", alpha=0.85)
    ax.set_title(f"最新一期 BL 权重 @ {pd.Timestamp(latest['date'].iloc[0]).date()}", fontsize=13, pad=10)
    ax.set_xlabel("权重")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    for y, v in enumerate(latest["weight"]):
        ax.text(v + 0.005, y, f"{v:.1%}", va="center", fontsize=8, color="#333")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_rank_diagnostics(oof: pd.DataFrame, path: Path):
    """展示日期内 Rank IC 的稳定性，而不是只看单个平均值。"""
    valid = oof.dropna(subset=["oof_pred", "y_ret"]).copy()
    daily_ic = valid.groupby("date").apply(
        lambda g: g["oof_pred"].corr(g["y_ret"], method="spearman"),
        include_groups=False,
    ).dropna()
    rolling_ic = daily_ic.rolling(60, min_periods=20).mean()
    expanding_ic = daily_ic.expanding(60).mean()

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(daily_ic.index, rolling_ic, color="#1f4e79", lw=1.3, label="60日滚动 Rank IC")
    axes[0].axhline(0, color="#999999", lw=0.8, ls="--")
    axes[0].set_title("OOF 横截面 Rank IC 稳定性", fontsize=13, pad=10)
    axes[0].set_ylabel("Spearman Rank IC")
    axes[0].legend(frameon=False)

    axes[1].plot(
        daily_ic.index,
        expanding_ic,
        color="#2e7d32",
        lw=1.4,
        label=f"扩展平均 IC（期末 {daily_ic.mean():.3f}）",
    )
    axes[1].axhline(0, color="#999999", lw=0.8, ls="--")
    axes[1].set_ylabel("累计平均 Rank IC")
    axes[1].set_xlabel("日期")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(comparison: pd.DataFrame, path: Path):
    """比较 LambdaRank、回归排名与固定权重融合信号。"""
    labels = {
        "lambdarank": "LambdaRank",
        "regression": "收益回归",
        "rank_ensemble_25_75": "排名融合",
    }
    names = [labels.get(x, x) for x in comparison["model"]]
    metrics = [
        ("rank_ic", "Rank IC"),
        ("ndcg_at_3", "NDCG@3"),
        ("top1_hit_rate", "Top-1 命中率"),
        ("top3_excess", "Top-3 期均超额"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = ["#7a7a7a", "#c45c26", "#1f4e79"]
    for ax, (col, title) in zip(axes.ravel(), metrics):
        vals = comparison[col].to_numpy()
        if col in {"top1_hit_rate", "top3_excess"}:
            vals = vals * 100
            suffix = "%"
        else:
            suffix = ""
        ax.bar(names, vals, color=colors[: len(names)], alpha=0.88)
        ax.set_title(title)
        ax.set_ylabel(title + (f" ({suffix})" if suffix else ""))
        ax.tick_params(axis="x", rotation=10)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}{suffix}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("模型相对强弱预测质量对比（Purged OOF）", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(nav, weights, reb, imp, metrics, path: Path):
    """六宫格总览。"""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(nav.index, nav["strategy"], label="策略", lw=1.6, color="#1f4e79")
    ax1.plot(nav.index, nav["equal_weight"], label="等权", lw=1.2, color="#7a7a7a")
    ax1.plot(nav.index, nav["cash"], label="国债", lw=1.2, color="#2e7d32")
    ax1.set_title("净值 · 回撤 · 权重 · 重要性 · 换手 总览", fontsize=14, pad=8)
    ax1.set_ylabel("净值")
    ax1.legend(frameon=False, ncol=3, loc="upper left")

    ax2 = fig.add_subplot(gs[1, 0])
    dd = nav["strategy"] / nav["strategy"].cummax() - 1.0
    ax2.fill_between(dd.index, dd.values, 0, color="#1f4e79", alpha=0.35)
    ax2.set_title("策略回撤")
    ax2.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")

    ax3 = fig.add_subplot(gs[1, 1])
    top = imp.head(8).iloc[::-1]
    ax3.barh(top["feature"], top["importance"], color="#1f4e79", alpha=0.85)
    ax3.set_title("特征重要性 Top8")

    ax4 = fig.add_subplot(gs[2, 0])
    wp = weights.pivot_table(index="date", columns="asset", values="weight", aggfunc="last")
    order = wp.mean().sort_values(ascending=False).index.tolist()
    ax4.stackplot(wp.index, *[wp[c].fillna(0).values for c in order], colors=plt.cm.tab20(np.linspace(0, 1, len(order))), alpha=0.9)
    ax4.set_ylim(0, 1)
    ax4.set_title("权重堆叠")
    ax4.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.bar(reb["date"], reb["turnover"], width=18, color="#c45c26", alpha=0.8)
    ax5.axhline(reb["turnover"].mean(), color="#1f4e79", ls="--", lw=1)
    ax5.set_title(f"换手（均值 {reb['turnover'].mean():.1%}）")
    ax5.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")

    fig.suptitle(
        f"Walk-forward 回测  {metrics['start']} ~ {metrics['end']}  |  "
        f"年化 {metrics['strategy_ann_return']:.1%}  MDD {metrics['strategy_max_drawdown']:.1%}  "
        f"Sharpe {metrics['strategy_sharpe']:.2f}",
        fontsize=11,
        y=0.995,
        color="#333",
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="")
    args = parser.parse_args()
    out_dir = OUT / args.experiment_name if args.experiment_name else OUT
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    nav, weights, reb, imp, metrics = _load(out_dir)

    plot_nav(nav, metrics, fig_dir / "01_nav_curve.png")
    plot_drawdown(nav, fig_dir / "02_drawdown.png")
    plot_weights(weights, fig_dir / "03_weights_stack.png")
    plot_importance(imp, fig_dir / "04_feature_importance.png")
    plot_turnover(reb, fig_dir / "05_turnover.png")
    plot_latest_weights(weights, fig_dir / "06_latest_weights.png")
    oof_path = out_dir / "oof_predictions.csv"
    compare_path = out_dir / "ranker_vs_regression.csv"
    if oof_path.exists():
        oof = pd.read_csv(oof_path, parse_dates=["date"])
        plot_rank_diagnostics(oof, fig_dir / "07_rank_ic_diagnostics.png")
    if compare_path.exists():
        comparison = pd.read_csv(compare_path)
        plot_model_comparison(comparison, fig_dir / "08_model_comparison.png")
    plot_dashboard(nav, weights, reb, imp, metrics, fig_dir / "00_dashboard.png")

    print("Saved figures:")
    for p in sorted(fig_dir.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
