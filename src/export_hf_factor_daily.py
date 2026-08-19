#!/usr/bin/env python3
"""把全部高频因子导出成日频 Excel / CSV。

水平：增长/流动性为累计净值，通胀为累计净值，利率为国债净价取负，
信用为去趋势后的指数，汇率为美元指数，地缘为金油拟合 GPR。
通胀因子看板里按周计算；日频表用同一套猪肉/布油/螺纹钢权重和滞后期
做成日收益，并把水平对齐到最近一个周五的周频净值。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paths import (
    ROOT,
    CREDIT_DIR,
    EXCHANGE_DIR,
    GROWTH_DIR,
    HF_DIR,
    INFLATION_DIR,
    MOBILITY_DIR,
    POLITICS_DIR,
    RATE_DIR,
    ensure_output_dirs,
)

OUT_CSV = HF_DIR / "hf_factor_daily.csv"
OUT_CHANGE_CSV = HF_DIR / "hf_factor_daily_change.csv"
OUT_XLSX = HF_DIR / "hf_factor_daily.xlsx"
TRADING_DAYS_MONTH = 21

LEVEL_COLS = ["增长因子", "通胀因子", "利率因子", "信用因子", "汇率因子", "地缘因子", "流动性因子"]


def _to_date_index(values, dates) -> pd.Series:
    idx = pd.to_datetime(dates, errors="coerce", utc=True)
    idx = idx.dt.tz_convert(None).dt.normalize()
    series = pd.Series(pd.to_numeric(values, errors="coerce").to_numpy(), index=idx)
    series = series.dropna()
    series = series.groupby(level=0).last().sort_index()
    return series


def _read(path: Path, date_col: str, value_col: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    series = _to_date_index(frame[value_col], frame[date_col])
    series.name = value_col
    return series


def _daily_inflation() -> pd.Series:
    """优先用猪肉/布油/螺纹钢日收益合成；没有日频原料时用周频净值按日填上。"""
    weekly = _read(INFLATION_DIR / "hf_inflation_weekly.csv", "date", "hf_nav")
    meta_path = INFLATION_DIR / "hf_regression_results.json"
    try:
        from load_wind_data import DEFAULT_DATA, load_wind_data

        df = load_wind_data(DEFAULT_DATA)
        needed = ["猪肉", "布伦特原油", "螺纹钢"]
        if all(c in df.columns and df[c].notna().any() for c in needed):
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            weights = meta.get("weights") or {"pork": 0.2, "brent": 0.4, "rebar": 0.4}
            lags = meta.get("lags_months") or {"pork": 0, "brent": 1, "rebar": 3}
            daily = pd.DataFrame(
                {
                    "pork": df["猪肉"],
                    "brent": df["布伦特原油"],
                    "rebar": df["螺纹钢"],
                }
            ).apply(pd.to_numeric, errors="coerce")
            daily.index = pd.to_datetime(daily.index, errors="coerce", utc=True)
            daily.index = daily.index.tz_convert(None).normalize()
            daily = daily.groupby(level=0).last().dropna(how="any").sort_index()
            ret = np.log(daily / daily.shift(1)) * 100
            parts = [
                weights[c] * ret[c].shift(int(lags[c]) * TRADING_DAYS_MONTH)
                for c in ("pork", "brent", "rebar")
            ]
            mom = pd.concat(parts, axis=1).sum(axis=1, min_count=3).dropna()
            nav = np.exp((mom / 100.0).cumsum())
            common = nav.index.intersection(weekly.index)
            if len(common):
                last = common.max()
                if nav.loc[last] and pd.notna(weekly.loc[last]):
                    nav = nav * (float(weekly.loc[last]) / float(nav.loc[last]))
            nav.name = "通胀因子"
            return nav
    except Exception:
        pass
    daily = weekly.resample("B").ffill()
    daily.name = "通胀因子"
    return daily


def build_level_panel() -> pd.DataFrame:
    growth = _read(GROWTH_DIR / "hf_growth_factor_synthetic.csv", "date", "hf_growth_factor")
    inflation = _daily_inflation()
    rate = _read(RATE_DIR / "hf_rate_factor_daily.csv", "日期", "hf_level")
    credit = _read(CREDIT_DIR / "hf_credit_factor_daily.csv", "日期", "hf_credit_factor")
    fx = _read(EXCHANGE_DIR / "dxy_yahoo.csv", "Date", "close")
    geo = _read(POLITICS_DIR / "hf_geo_factor_synthetic.csv", "date", "hf_geo_factor")
    liquidity = _read(MOBILITY_DIR / "hf_mobility_factor_synthetic.csv", "date", "hf_mobility_factor")
    panel = pd.DataFrame(
        {
            "增长因子": growth,
            "通胀因子": inflation,
            "利率因子": rate,
            "信用因子": credit,
            "汇率因子": fx,
            "地缘因子": geo,
            "流动性因子": liquidity,
        }
    )
    panel.index.name = "date"
    return panel.sort_index()[LEVEL_COLS]


def build_change_panel(level: pd.DataFrame) -> pd.DataFrame:
    change = pd.DataFrame(index=level.index)
    try:
        change["增长因子"] = _read(
            GROWTH_DIR / "hf_growth_factor_synthetic.csv", "date", "hf_mom_pct"
        )
    except FileNotFoundError:
        change["增长因子"] = np.log(level["增长因子"] / level["增长因子"].shift(1)) * 100
    try:
        change["利率因子"] = _read(
            RATE_DIR / "hf_rate_factor_daily.csv", "日期", "hf_mom_pct"
        )
    except FileNotFoundError:
        change["利率因子"] = np.log(level["利率因子"].abs() / level["利率因子"].abs().shift(1)) * 100
    try:
        change["信用因子"] = _read(
            CREDIT_DIR / "hf_credit_factor_daily.csv", "日期", "hf_mom_pct"
        )
    except FileNotFoundError:
        change["信用因子"] = level["信用因子"].pct_change() * 100
    try:
        change["流动性因子"] = _read(
            MOBILITY_DIR / "hf_mobility_factor_synthetic.csv", "date", "hf_mom_pct"
        )
    except FileNotFoundError:
        change["流动性因子"] = np.log(level["流动性因子"] / level["流动性因子"].shift(1)) * 100

    infl = level["通胀因子"]
    change["通胀因子"] = np.log(infl / infl.shift(1)) * 100
    fx = level["汇率因子"]
    change["汇率因子"] = np.log(fx / fx.shift(1)) * 100
    geo = level["地缘因子"]
    change["地缘因子"] = np.log(geo.abs() / geo.abs().shift(1)) * 100
    change.index.name = "date"
    return change.sort_index()[LEVEL_COLS]


def _format_dates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out


def _notes() -> pd.DataFrame:
    rows = [
        ("增长因子", "水平=累计净值（起点约 100）；日变化=成分日收益按固定权重加权，单位 %"),
        ("通胀因子", "看板按周计算。日频用猪肉/布油/螺纹钢同一套权重和滞后期做日收益；水平对齐到最近周五的周频净值。没有日频原料时，该周内沿用周五净值"),
        ("利率因子", "水平=国债净价指数取负；日变化=对应对数变化，单位 %（涨表示利率因子走强，对应债价下跌）"),
        ("信用因子", "水平=信用利差财富差去趋势后的指数；日变化=该指数日变化，单位 %"),
        ("汇率因子", "水平=美元指数收盘；日变化=对数日变化，单位 %"),
        ("地缘因子", "水平=黄金+原油拟合的 GPR；日变化=对数日变化，单位 %"),
        ("流动性因子", "水平=累计净值（起点约 100）；日变化=大小盘 PE 日收益加权，单位 %"),
        ("空值", "某因子尚未开始、或当天休市，格子会空着。美元指数历史最长，所以表会从 2000 年写起"),
    ]
    return pd.DataFrame(rows, columns=["项目", "说明"])


def export() -> None:
    ensure_output_dirs()
    level = build_level_panel()
    change = build_change_panel(level)
    level_out = _format_dates(level)
    change_out = _format_dates(change)
    level_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")
    change_out.to_csv(OUT_CHANGE_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        level_out.to_excel(writer, sheet_name="水平", index=False)
        change_out.to_excel(writer, sheet_name="日变化", index=False)
        _notes().to_excel(writer, sheet_name="说明", index=False)
    print(f"  → {OUT_XLSX.relative_to(ROOT)}（工作表：水平、日变化、说明）")
    print(f"  → {OUT_CSV.relative_to(ROOT)}")
    print(f"  → {OUT_CHANGE_CSV.relative_to(ROOT)}")
    print(f"    区间 {level.index.min().date()} ~ {level.index.max().date()}，{len(level)} 日")


if __name__ == "__main__":
    export()
