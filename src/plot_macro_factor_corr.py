#!/usr/bin/env python3
"""
六个宏观因子月度相关性热力图。

因子来源在 output/factors/；结果写到 output/corr/。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

from panel_config import heatmap_factors, load_panel_config
from paths import CORR_DIR, EXCHANGE_DIR, GROWTH_DIR, INFLATION_DIR, CREDIT_DIR, POLITICS_DIR, RATE_DIR, ensure_output_dirs

FACTOR_LABELS = [
    "增长因子",
    "通胀因子",
    "利率因子",
    "信用因子",
    "汇率因子",
    "地缘因子",
]

FACTOR_PATHS = {
    "增长因子": (GROWTH_DIR / "growth_factor.csv", "date", "raw_growth_factor"),
    "通胀因子": (INFLATION_DIR / "inflation_factor.csv", "date", "inflation_factor"),
    "利率因子": (RATE_DIR / "rate_factor.csv", "date", "rate_factor"),
    "信用因子": (CREDIT_DIR / "credit_factor.csv", "date", "credit_factor"),
    "地缘因子": (POLITICS_DIR / "geo_factor.csv", "date", "geo_factor"),
}

OUTPUT_PANEL = CORR_DIR / "macro_factor_monthly.csv"
OUTPUT_CORR = CORR_DIR / "macro_factor_corr.csv"
OUTPUT_JSON = CORR_DIR / "macro_factor_corr.json"
OUTPUT_PNG = CORR_DIR / "macro_factor_corr_heatmap.png"
SAMPLE_START = "2021-01"


def setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_monthly_factor(path: Path, date_col: str, value_col: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=[date_col])
    monthly = df.groupby(pd.to_datetime(df[date_col]).dt.to_period("M"))[value_col].last()
    return monthly.dropna()


def load_dxy_monthly_level() -> pd.Series:
    dxy = pd.read_csv(EXCHANGE_DIR / "dxy_yahoo.csv", parse_dates=["Date"])
    dxy["date"] = pd.to_datetime(dxy["Date"], utc=True).dt.tz_convert(None)
    monthly = dxy.set_index("date")["close"].resample("ME").last()
    monthly.index = monthly.index.to_period("M")
    return monthly.dropna()


def build_factor_panel() -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    for label, (path, date_col, value_col) in FACTOR_PATHS.items():
        series[label] = load_monthly_factor(path, date_col, value_col)
    series["汇率因子"] = load_dxy_monthly_level()
    return pd.DataFrame(series)


def cell_facecolor(val: float, is_diag: bool) -> tuple[float, float, float]:
    if is_diag:
        return to_rgb("#e53935")
    if np.isnan(val):
        return (1.0, 1.0, 1.0)
    v = float(np.clip(val, -1.0, 1.0))
    if v > 0:
        t = v
        return (1.0, 1.0 - 0.55 * t, 1.0 - 0.55 * t)
    if v < 0:
        t = abs(v)
        return (1.0 - 0.55 * t, 1.0, 1.0 - 0.45 * t)
    return (1.0, 1.0, 1.0)


def plot_corr_table(corr: pd.DataFrame, start: str, end: str, n_months: int) -> None:
    setup_chinese_font()
    labels = corr.columns.tolist()
    n = len(labels)
    data = corr.values

    table_data = [[""] + labels]
    for i, row_label in enumerate(labels):
        row = [row_label] + [f"{data[i, j]:.2f}" if not np.isnan(data[i, j]) else "—" for j in range(n)]
        table_data.append(row)

    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    ax.axis("off")
    fig.suptitle(
        f"样本区间：{start} ~ {end}（{n_months} 个月）",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    table = ax.table(cellText=table_data, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.15, 2.0)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#000000")
        cell.set_linewidth(1.2)
        if row == 0 or col == 0:
            cell.set_facecolor("#ffffff")
            if row > 0 or col > 0:
                cell.get_text().set_fontweight("bold")
            continue
        val = data[row - 1, col - 1]
        is_diag = row == col
        cell.set_facecolor(cell_facecolor(val, is_diag))
        if is_diag:
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.get_text().set_color("#111111")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_corr_payload(panel: pd.DataFrame, start: str, end: str, labels: list[str]) -> dict:
    subset = panel.loc[start:end].dropna(how="any")
    if len(subset) < 3:
        raise ValueError(f"有效样本不足: {start} ~ {end} 仅 {len(subset)} 个月")
    corr = subset.corr(method="pearson")
    common = panel.dropna(how="any")
    series = {
        label: [None if pd.isna(v) else round(float(v), 6) for v in common[label].tolist()]
        for label in labels
    }
    return {
        "labels": labels,
        "months": common.index.astype(str).tolist(),
        "series": series,
        "start": str(subset.index.min()),
        "end": str(subset.index.max()),
        "n_months": len(subset),
        "corr": [[round(v, 4) for v in row] for row in corr.values.tolist()],
    }


def main() -> None:
    ensure_output_dirs()
    cfg = load_panel_config()
    hm = cfg["heatmap"]
    labels = heatmap_factors(cfg)
    sample_start = hm.get("lf_start") or SAMPLE_START
    panel = build_factor_panel()
    missing = [f for f in labels if f not in panel.columns]
    if missing:
        raise ValueError("热力图缺少因子列: " + "、".join(missing))
    panel = panel[labels]
    if sample_start:
        panel = panel[panel.index >= pd.Period(sample_start, freq="M")]
    panel.index = panel.index.astype(str)

    panel_out = panel.copy()
    panel_out.reset_index(names="ym").to_csv(OUTPUT_PANEL, index=False, encoding="utf-8-sig", float_format="%.6f")

    common = panel.dropna(how="any")
    start = str(common.index.min()) if len(common) else "—"
    end = str(common.index.max()) if len(common) else "—"
    min_months = int(hm.get("min_months") or 12)
    corr = common.corr(method="pearson", min_periods=min_months)
    corr.to_csv(OUTPUT_CORR, encoding="utf-8-sig", float_format="%.4f")

    payload = build_corr_payload(panel, start, end, labels)
    payload["default_start"] = payload["start"]
    payload["default_end"] = payload["end"]
    payload["fixed_start"] = sample_start
    payload["panel_config"] = {"heatmap": hm, "alerts": cfg["alerts"]}
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_corr_table(corr, start, end, len(common))

    print("宏观因子月度面板:", OUTPUT_PANEL)
    print("相关性矩阵:", OUTPUT_CORR)
    print("网页默认 JSON:", OUTPUT_JSON)
    print("热力图:", OUTPUT_PNG)
    print(f"共同样本: {len(common)} 个月 ({start} ~ {end})")
    print("\n相关系数矩阵:")
    print(corr.round(2).to_string())


if __name__ == "__main__":
    main()
