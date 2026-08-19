#!/usr/bin/env python3
"""
六个高频拟合宏观因子的周频相关性热力图。

输出:
  macro_hf_factor_weekly.csv
  macro_hf_factor_corr.csv
  macro_hf_factor_corr.json
  macro_hf_factor_corr_heatmap.png
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

ROOT = Path(__file__).parent

FACTOR_LABELS = [
    "增长因子",
    "通胀因子",
    "利率因子",
    "信用因子",
    "汇率因子",
    "地缘因子",
]

OUTPUT_PANEL = ROOT / "macro_hf_factor_weekly.csv"
OUTPUT_CORR = ROOT / "macro_hf_factor_corr.csv"
OUTPUT_JSON = ROOT / "macro_hf_factor_corr.json"
OUTPUT_PNG = ROOT / "macro_hf_factor_corr_heatmap.png"
SAMPLE_START = "2021-01-01"
WEEK_FREQ = "W-FRI"


def setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def read_series(path: Path, date_col: str, value_col: str, scale: float = 1.0) -> pd.Series:
    df = pd.read_csv(path, parse_dates=[date_col])
    series = df.sort_values(date_col).set_index(date_col)[value_col].astype(float) * scale
    return series.dropna()


def weekly_last(series: pd.Series) -> pd.Series:
    return series.resample(WEEK_FREQ).last().dropna()


def weekly_sum(series: pd.Series) -> pd.Series:
    return series.resample(WEEK_FREQ).sum(min_count=1).dropna()


def weekly_mean(series: pd.Series) -> pd.Series:
    return series.resample(WEEK_FREQ).mean().dropna()


def weekly_log_yoy(series: pd.Series) -> pd.Series:
    weekly = series.resample(WEEK_FREQ).last().dropna()
    return (np.log(weekly / weekly.shift(52)) * 100).dropna()


def build_hf_panel() -> pd.DataFrame:
    growth = weekly_last(
        read_series(ROOT / "growth" / "hf_growth_factor_synthetic.csv", "date", "hf_yoy")
    )
    inflation = weekly_last(
        read_series(ROOT / "inflasion" / "hf_inflation_weekly.csv", "date", "hf_yoy_pct")
    )
    rate = weekly_last(
        read_series(ROOT / "interest rate" / "hf_rate_factor_daily.csv", "日期", "hf_level")
    )
    # 信用高频：财富差去趋势取反后的指数水平（与利率 hf_level 一样取周末值）
    credit = weekly_last(
        read_series(ROOT / "credit" / "hf_credit_factor_daily.csv", "日期", "hf_credit_factor")
    )

    dxy = pd.read_csv(ROOT / "exchange" / "dxy_yahoo.csv", parse_dates=["Date"])
    dxy["date"] = pd.to_datetime(dxy["Date"], utc=True).dt.tz_convert(None)
    exchange = weekly_last(dxy.sort_values("date").set_index("date")["close"].astype(float))

    politics = weekly_last(
        read_series(ROOT / "politics" / "hf_geo_factor_synthetic.csv", "date", "hf_geo_factor")
    )

    panel = pd.DataFrame(
        {
            "增长因子": growth,
            "通胀因子": inflation,
            "利率因子": rate,
            "信用因子": credit,
            "汇率因子": exchange,
            "地缘因子": politics,
        }
    )
    return panel


def build_alert_panel() -> pd.DataFrame:
    """与因子暴露一致的周度环比/变化，用于纯静态页面的波动警报。"""
    growth = weekly_sum(
        read_series(ROOT / "growth" / "hf_growth_factor_synthetic.csv", "date", "hf_mom_pct")
    )
    inflation = weekly_last(
        read_series(ROOT / "inflasion" / "hf_inflation_weekly.csv", "date", "hf_wow")
    )
    rate = weekly_sum(
        read_series(ROOT / "interest rate" / "hf_rate_factor_daily.csv", "日期", "hf_mom_pct")
    )
    credit = weekly_sum(
        read_series(ROOT / "credit" / "hf_credit_factor_daily.csv", "日期", "hf_mom_pct")
    )
    dxy = pd.read_csv(ROOT / "exchange" / "dxy_yahoo.csv", parse_dates=["Date"])
    dxy["date"] = pd.to_datetime(dxy["Date"], utc=True).dt.tz_convert(None)
    dxy_weekly = weekly_last(dxy.sort_values("date").set_index("date")["close"].astype(float))
    exchange = np.log(dxy_weekly / dxy_weekly.shift(1)) * 100
    geo_weekly = weekly_last(
        read_series(ROOT / "politics" / "hf_geo_factor_synthetic.csv", "date", "hf_geo_factor")
    )
    politics = np.log(geo_weekly / geo_weekly.shift(1)) * 100
    return pd.DataFrame(
        {
            "增长因子": growth,
            "通胀因子": inflation,
            "利率因子": rate,
            "信用因子": credit,
            "汇率因子": exchange,
            "地缘因子": politics,
        }
    ).dropna(how="any")


def build_corr_payload(panel: pd.DataFrame, start: str, end: str, labels: list[str]) -> dict:
    subset = panel.loc[start:end].dropna(how="any")
    if len(subset) < 12:
        raise ValueError(f"有效样本不足: {start} ~ {end} 仅 {len(subset)} 周")
    corr = subset.corr(method="pearson")
    common = panel.dropna(how="any")
    periods = common.index.strftime("%Y-%m-%d").tolist()
    series = {
        label: [None if pd.isna(v) else round(float(v), 6) for v in common[label].tolist()]
        for label in labels
    }
    return {
        "labels": labels,
        "periods": periods,
        "weeks": periods,
        "series": series,
        "start": subset.index.min().strftime("%Y-%m-%d"),
        "end": subset.index.max().strftime("%Y-%m-%d"),
        "n_periods": len(subset),
        "n_weeks": len(subset),
        "freq": "W-FRI",
        "corr": [[round(v, 4) for v in row] for row in corr.values.tolist()],
    }


def cell_facecolor(val: float, is_diag: bool) -> tuple[float, float, float]:
    if is_diag:
        return to_rgb("#e53935")
    if np.isnan(val):
        return (1.0, 1.0, 1.0)
    v = float(np.clip(val, -1.0, 1.0))
    if v > 0:
        return (1.0, 1.0 - 0.55 * v, 1.0 - 0.55 * v)
    if v < 0:
        t = abs(v)
        return (1.0 - 0.55 * t, 1.0, 1.0 - 0.45 * t)
    return (1.0, 1.0, 1.0)


def format_corr_value(val: float) -> str:
    clean = 0.0 if abs(val) < 0.005 else val
    return f"{clean:.2f}"


def plot_corr_table(corr: pd.DataFrame, start: str, end: str, n_weeks: int) -> None:
    setup_chinese_font()
    labels = corr.columns.tolist()
    data = corr.values

    table_data = [[""] + labels]
    for i, row_label in enumerate(labels):
        table_data.append([row_label] + [format_corr_value(data[i, j]) for j in range(len(labels))])

    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    ax.axis("off")
    fig.suptitle(f"高频周频样本：{start} ~ {end}（{n_weeks} 周）", fontsize=12, fontweight="bold", y=0.98)
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

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    cfg = load_panel_config()
    hm = cfg["heatmap"]
    labels = heatmap_factors(cfg)
    sample_start = hm.get("hf_start") or SAMPLE_START
    panel = build_hf_panel()
    missing = [f for f in labels if f not in panel.columns]
    if missing:
        raise ValueError("周频热力图缺少因子列: " + "、".join(missing))
    panel = panel[labels]
    panel = panel[panel.index >= pd.Timestamp(sample_start)]
    common = panel.dropna(how="any")
    start = common.index.min().strftime("%Y-%m-%d")
    end = common.index.max().strftime("%Y-%m-%d")

    panel_out = panel.copy()
    panel_out.index = panel_out.index.strftime("%Y-%m-%d")
    panel_out.reset_index(names="week").to_csv(OUTPUT_PANEL, index=False, encoding="utf-8-sig", float_format="%.6f")

    corr = common.corr(method="pearson")
    corr.to_csv(OUTPUT_CORR, encoding="utf-8-sig", float_format="%.4f")

    payload = build_corr_payload(panel, start, end, labels)
    alert_panel = build_alert_panel()
    alert_cols = [f for f in labels if f in alert_panel.columns]
    payload["alert_weeks"] = alert_panel.index.strftime("%Y-%m-%d").tolist()
    payload["alert_series"] = {
        label: [round(float(v), 6) for v in alert_panel[label].tolist()]
        for label in alert_cols
    }
    payload["default_start"] = payload["start"]
    payload["default_end"] = payload["end"]
    payload["fixed_start"] = sample_start
    payload["panel_config"] = {"heatmap": hm, "alerts": cfg["alerts"]}
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_corr_table(corr, start, end, len(common))

    print("高频周频面板:", OUTPUT_PANEL)
    print("高频相关性矩阵:", OUTPUT_CORR)
    print("高频网页 JSON:", OUTPUT_JSON)
    print("高频热力图:", OUTPUT_PNG)
    print(f"共同样本: {len(common)} 周 ({start} ~ {end})")
    print("\n高频相关系数矩阵:")
    print(corr.map(format_corr_value).to_string())


if __name__ == "__main__":
    main()
