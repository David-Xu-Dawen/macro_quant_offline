#!/usr/bin/env python3
"""用 Wind data.xlsx（或 data.csv）离线更新低频/高频因子与资产面板。

用法:
  python3 update_from_xlsx.py
  python3 update_from_xlsx.py --data data.xlsx
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter

from load_wind_data import (
    DEFAULT_DATA,
    additional_asset_path,
    expected_wind_name,
    load_additional_assets,
    load_wind_data,
    pretty_asset_name,
)
from panel_config import sync_asset_factor_mask
from paths import (
    ROOT,
    COMBINED_CLOSE,
    CREDIT_DIR,
    EXCHANGE_DIR,
    EXPOSURE_DIR,
    GROWTH_DIR,
    INFLATION_DIR,
    MOBILITY_DIR,
    POLITICS_DIR,
    RATE_DIR,
    ensure_output_dirs,
)

warnings.filterwarnings("ignore")

GROWTH_WEIGHTS = {
    "pmi_yoy_diff_filled": 0.579829835,
    "fai_yoy_filled": 0.069358574,
    "retail_yoy_filled": 0.246186048,
    "trade_yoy_weighted_filled": 0.104625543,
}

HP_LAMBDA_MONTHLY = 129600
HP_LAMBDA_DAILY = HP_LAMBDA_MONTHLY * (21**4)
HF_INDEX_VOL = 0.05
TRADING_DAYS_YEAR = 252
TRADING_DAYS_MONTH = 21
WEEKS_PER_MONTH = 4
WEEKS_YEAR = 52
NAV_BASE = 100.0
LAG_RANGE = range(0, 4)

FACTOR_REQUIRED = [
    "pmi",
    "fai_yoy",
    "retail_yoy",
    "export_yoy",
    "import_yoy",
    "cpi_yoy",
    "ppi_yoy",
    "国债10Y",
    "中票AA_3Y",
    "国开债_3Y",
    "m2_yoy",
    "sf_yoy",
    "美元指数",
    "gpr",
    "国债净价",
    "企债财富_3_5",
    "国开财富_3_5",
    "猪肉",
    "布伦特原油",
    "螺纹钢",
    "恒生指数",
    "申万房地产",
    "沪金",
]


def _require_columns(df: pd.DataFrame, names: list[str]) -> None:
    missing = []
    for name in names:
        if name not in df.columns or not df[name].notna().any():
            missing.append(f"{name}（Wind 中请导出「{expected_wind_name(name)}」）")
    if missing:
        have = "、".join(df.columns) if len(df.columns) else "无"
        raise ValueError(
            "Excel 缺少必要列:\n  - "
            + "\n  - ".join(missing)
            + f"\n已识别列: {have}\n请在 Wind 补上这些指标后重新导出，覆盖 data1.xlsx"
        )


def _to_naive_datetime(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce", utc=True)
    return out.dt.tz_convert(None)


def _write_merged(path: Path, incoming: pd.DataFrame, date_col: str, value_cols: list[str]) -> None:
    """保留新数据起点之前的旧历史，起点之后整段替换。

    不能逐日 outer-merge：旧源与 Wind 的交易日/时间戳不同，会让累计 NAV、
    同比和价格水平在相邻日交错，制造假收益。
    """
    incoming = incoming.copy()
    incoming[date_col] = _to_naive_datetime(incoming[date_col])
    incoming = incoming.dropna(subset=[date_col]).drop_duplicates(date_col, keep="last")
    if incoming.empty:
        raise ValueError(f"{path.name}: 没有可写入的新数据")
    incoming = incoming[[date_col] + [c for c in value_cols if c in incoming.columns]]
    start = incoming[date_col].min()

    if path.exists():
        existing = pd.read_csv(path)
        if date_col in existing.columns:
            existing[date_col] = _to_naive_datetime(existing[date_col])
            existing = existing.dropna(subset=[date_col])
            existing = existing[existing[date_col] < start]
        else:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    out = pd.concat([existing, incoming], ignore_index=True, sort=False)
    out = out.drop_duplicates(date_col, keep="last").sort_values(date_col).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"  → {path.relative_to(ROOT)} ({len(out)} 行；{start.date()} 起整段替换)")


def _write_monthly(path: Path, incoming: pd.DataFrame, value_cols: list[str]) -> None:
    """按月份而非具体日拼接，确保每月只有一条记录。"""
    incoming = incoming.copy()
    incoming["date"] = _to_naive_datetime(incoming["date"])
    incoming = incoming.dropna(subset=["date"])
    incoming["_month"] = incoming["date"].dt.to_period("M")
    incoming = incoming.drop_duplicates("_month", keep="last")
    if incoming.empty:
        raise ValueError(f"{path.name}: 没有可写入的月度数据")
    start_month = incoming["_month"].min()
    incoming = incoming[["date", "_month"] + [c for c in value_cols if c in incoming.columns]]

    if path.exists():
        existing = pd.read_csv(path)
        if "date" in existing.columns:
            existing["date"] = _to_naive_datetime(existing["date"])
            existing = existing.dropna(subset=["date"])
            existing["_month"] = existing["date"].dt.to_period("M")
            existing = existing[existing["_month"] < start_month]
            existing = existing.drop_duplicates("_month", keep="last")
        else:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    out = pd.concat([existing, incoming], ignore_index=True, sort=False)
    out = out.drop_duplicates("_month", keep="last").sort_values("_month")
    out = out.drop(columns=["_month"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"  → {path.relative_to(ROOT)} ({len(out)} 月；{start_month} 起按月替换)")


def _rebase_level_to_history(
    path: Path,
    incoming: pd.DataFrame,
    date_col: str,
    level_col: str,
) -> pd.DataFrame:
    """只调整累计序列的基数，让新旧接缝连续；收益率不受常数缩放影响。"""
    if not path.exists() or level_col not in incoming.columns:
        return incoming
    old = pd.read_csv(path)
    if date_col not in old.columns or level_col not in old.columns:
        return incoming

    out = incoming.copy()
    out[date_col] = _to_naive_datetime(out[date_col])
    old[date_col] = _to_naive_datetime(old[date_col])
    start = out[date_col].dropna().min()
    old_level = pd.to_numeric(
        old.loc[old[date_col] < start, level_col], errors="coerce"
    ).dropna()
    new_level = pd.to_numeric(out[level_col], errors="coerce").dropna()
    if old_level.empty or new_level.empty or new_level.iloc[0] == 0:
        return out

    scale = float(old_level.iloc[-1] / new_level.iloc[0])
    out[level_col] = pd.to_numeric(out[level_col], errors="coerce") * scale
    print(f"  {path.name}: {level_col} 接缝缩放 {scale:.6f}")
    return out


def _recompute_log_change_with_history(
    path: Path,
    incoming: pd.DataFrame,
    date_col: str,
    level_col: str,
    change_col: str,
    periods: int,
) -> pd.DataFrame:
    """用接缝前历史补足首年同比，避免 Wind 起始后的前 52 周/252 日为空。"""
    if not path.exists():
        return incoming
    old = pd.read_csv(path)
    if date_col not in old.columns or level_col not in old.columns:
        return incoming

    out = incoming.copy()
    out[date_col] = _to_naive_datetime(out[date_col])
    old[date_col] = _to_naive_datetime(old[date_col])
    start = out[date_col].dropna().min()
    history = old.loc[old[date_col] < start, [date_col, level_col]]
    levels = pd.concat(
        [history, out[[date_col, level_col]]],
        ignore_index=True,
    )
    levels[level_col] = pd.to_numeric(levels[level_col], errors="coerce")
    levels = levels.dropna().drop_duplicates(date_col, keep="last").sort_values(date_col)
    levels[change_col] = np.log(levels[level_col] / levels[level_col].shift(periods)) * 100
    change_by_date = levels.set_index(date_col)[change_col]
    out[change_col] = out[date_col].map(change_by_date)
    return out


def monthly_last(series: pd.Series) -> pd.Series:
    s = series.dropna().sort_index()
    if s.empty:
        return s
    return s.resample("ME").last().dropna()


def fill_jan_feb(series: pd.Series) -> pd.Series:
    """1/2 月缺失时用邻月均值填充（与增长管线一致）。"""
    out = series.copy()
    idx = out.dropna().index
    if idx.empty:
        return out
    years = sorted({ts.year for ts in idx} | {ts.year for ts in out.index})
    for year in years:
        dec = pd.Timestamp(year - 1, 12, 31)
        jan = pd.Timestamp(year, 1, 31)
        feb = pd.Timestamp(year, 2, 28)
        # align to month-end index if present
        def _pick(ts: pd.Timestamp) -> pd.Timestamp | None:
            cands = [i for i in out.index if i.year == ts.year and i.month == ts.month]
            return cands[-1] if cands else None

        jan_i, feb_i, mar_i = _pick(jan), _pick(feb), _pick(pd.Timestamp(year, 3, 31))
        dec_i = _pick(dec)
        mar_v = out.get(mar_i) if mar_i is not None else np.nan
        dec_v = out.get(dec_i) if dec_i is not None else np.nan
        if jan_i is not None and pd.isna(out.get(jan_i)) and pd.notna(dec_v) and pd.notna(mar_v):
            out.loc[jan_i] = (dec_v + mar_v) / 2
        if feb_i is not None and pd.isna(out.get(feb_i)):
            jan_v = out.get(jan_i) if jan_i is not None else np.nan
            if pd.notna(jan_v) and pd.notna(mar_v):
                out.loc[feb_i] = (jan_v + mar_v) / 2
    return out


def normalized_weights(betas: dict[str, float]) -> dict[str, float]:
    s = sum(abs(v) for v in betas.values())
    return {k: v / s for k, v in betas.items()} if s else betas


def fit_ols(y: pd.Series, X: pd.DataFrame):
    common = pd.concat([y, X], axis=1).dropna()
    if len(common) < 24:
        return None
    model = sm.OLS(common.iloc[:, 0], sm.add_constant(common.iloc[:, 1:])).fit()
    return {"model": model, "bic": model.bic, "n": len(common)}


def search_joint_lags(y: pd.Series, X: pd.DataFrame, assets: list[str]):
    best, best_lags = None, None
    for lags in itertools.product(LAG_RANGE, repeat=len(assets)):
        X_lag = pd.DataFrame({c: X[c].shift(l) for c, l in zip(assets, lags)}, index=X.index)
        res = fit_ols(y, X_lag)
        if res and (best is None or res["bic"] < best["bic"]):
            best, best_lags = res, dict(zip(assets, lags))
    if best is None:
        raise RuntimeError("回归样本不足，无法估计高频系数")
    return best, best_lags


# ── LF builders ───────────────────────────────────────────────────

def build_lf_growth(df: pd.DataFrame) -> None:
    pmi_m = monthly_last(df["pmi"])
    pmi_yoy_diff = pmi_m - pmi_m.shift(12)
    fai = monthly_last(df["fai_yoy"])
    retail = monthly_last(df["retail_yoy"])
    trade = (monthly_last(df["export_yoy"]) + monthly_last(df["import_yoy"])) / 2

    idx = pmi_yoy_diff.dropna().index.union(fai.dropna().index).union(retail.dropna().index).union(trade.dropna().index)
    idx = pd.DatetimeIndex(sorted(idx))
    panel = pd.DataFrame(index=idx)
    panel["pmi_yoy_diff"] = pmi_yoy_diff.reindex(idx)
    panel["fai_yoy"] = fai.reindex(idx)
    panel["retail_yoy"] = retail.reindex(idx)
    panel["trade_yoy_weighted"] = trade.reindex(idx)
    panel["pmi_yoy_diff_filled"] = panel["pmi_yoy_diff"]
    panel["trade_yoy_weighted_filled"] = panel["trade_yoy_weighted"]
    panel["fai_yoy_filled"] = fill_jan_feb(panel["fai_yoy"])
    panel["retail_yoy_filled"] = fill_jan_feb(panel["retail_yoy"])
    growth_cols = list(GROWTH_WEIGHTS)
    available_weight = sum(panel[c].notna() * GROWTH_WEIGHTS[c] for c in growth_cols)
    weighted_sum = sum(panel[c].fillna(0) * GROWTH_WEIGHTS[c] for c in growth_cols)
    available_count = panel[growth_cols].notna().sum(axis=1)
    # Wind 从 2021 年开始，PMI 同比差要到 2022 年才有。首年用其余至少
    # 两项指标按可用权重重新归一化，保证固定的 2021 至今矩阵有完整样本。
    panel["raw_growth_factor"] = (
        (weighted_sum / available_weight)
        .where(available_count >= 2)
        .bfill(limit=1)
    )
    valid = panel["raw_growth_factor"].dropna()
    if len(valid) >= 24:
        _, trend = hpfilter(valid.values, lamb=HP_LAMBDA_MONTHLY)
        panel["growth_factor_hp"] = pd.Series(trend, index=valid.index)
    out = panel.reset_index().rename(columns={"index": "date"})
    _write_monthly(
        GROWTH_DIR / "growth_factor.csv",
        out,
        list(out.columns.drop("date")),
    )


def build_lf_inflation(df: pd.DataFrame) -> None:
    cpi = monthly_last(df["cpi_yoy"])
    ppi = monthly_last(df["ppi_yoy"])
    panel = pd.concat([cpi.rename("cpi_yoy"), ppi.rename("ppi_yoy")], axis=1).dropna(how="all")
    # 波动率倒数加权；样本不足时等权
    vol_c = panel["cpi_yoy"].rolling(12, min_periods=6).std()
    vol_p = panel["ppi_yoy"].rolling(12, min_periods=6).std()
    inv = 1 / vol_c + 1 / vol_p
    w_c = (1 / vol_c) / inv
    w_p = (1 / vol_p) / inv
    factor = w_c * panel["cpi_yoy"] + w_p * panel["ppi_yoy"]
    factor = factor.fillna(0.5 * panel["cpi_yoy"] + 0.5 * panel["ppi_yoy"])
    out = pd.DataFrame(
        {
            "date": panel.index,
            "cpi_yoy": panel["cpi_yoy"].values,
            "ppi_yoy": panel["ppi_yoy"].values,
            "inflation_factor": factor.values,
            "cpi_yoy_official": panel["cpi_yoy"].values,
            "ppi_yoy_official": panel["ppi_yoy"].values,
        }
    )
    _write_monthly(
        INFLATION_DIR / "inflation_factor.csv",
        out,
        ["cpi_yoy", "ppi_yoy", "inflation_factor", "cpi_yoy_official", "ppi_yoy_official"],
    )


def build_lf_rate(df: pd.DataFrame) -> None:
    y = monthly_last(df["国债10Y"])
    out = pd.DataFrame({"date": y.index, "yield_10y": y.values, "rate_factor": y.values})
    _write_monthly(RATE_DIR / "rate_factor.csv", out, ["yield_10y", "rate_factor"])
    daily = df["国债10Y"].dropna()
    d_out = pd.DataFrame({"日期": daily.index, "yield_10y": daily.values})
    _write_merged(RATE_DIR / "cn10y_yield_daily.csv", d_out, "日期", ["yield_10y"])


def build_lf_credit(df: pd.DataFrame) -> None:
    aa = monthly_last(df["中票AA_3Y"])
    cdb = monthly_last(df["国开债_3Y"])
    spread = aa - cdb
    out = pd.DataFrame(
        {
            "date": spread.index,
            "spread_bp": (spread * 100).values,
            "credit_factor": spread.values,
        }
    )
    _write_monthly(CREDIT_DIR / "credit_factor.csv", out, ["spread_bp", "credit_factor"])


def build_lf_mobility(df: pd.DataFrame) -> None:
    m2 = monthly_last(df["m2_yoy"])
    sf = monthly_last(df["sf_yoy"])
    mob = m2 - sf
    out = pd.DataFrame(
        {"date": mob.index, "m2_yoy": m2.reindex(mob.index).values, "sf_yoy": sf.reindex(mob.index).values, "mobility_factor": mob.values}
    )
    _write_monthly(MOBILITY_DIR / "mobility_factor.csv", out, ["m2_yoy", "sf_yoy", "mobility_factor"])


def build_dxy(df: pd.DataFrame) -> None:
    s = df["美元指数"].dropna()
    out = pd.DataFrame(
        {
            "Date": s.index,
            "open": s.values,
            "high": s.values,
            "low": s.values,
            "close": s.values,
            "volume": 0,
        }
    )
    _write_merged(EXCHANGE_DIR / "dxy_yahoo.csv", out, "Date", ["open", "high", "low", "close", "volume"])


def build_lf_geo(df: pd.DataFrame) -> None:
    if "gpr" not in df.columns or not df["gpr"].notna().any():
        raise ValueError("缺少「全球:地缘政治风险指数」，无法更新低频地缘因子")
    gpr = monthly_last(df["gpr"])
    out = pd.DataFrame({"date": gpr.index, "gpr": gpr.values, "geo_factor": gpr.values})
    _write_monthly(POLITICS_DIR / "geo_factor.csv", out, ["gpr", "geo_factor"])


# ── HF builders ───────────────────────────────────────────────────

def build_hf_rate(df: pd.DataFrame) -> None:
    index = df["国债净价"].dropna()
    yield_d = df["国债10Y"].reindex(index.index).ffill()
    index_neg = -index
    neg_log_mom = -np.log(index / index.shift(1)) * 100
    out = pd.DataFrame(
        {
            "日期": index.index,
            "index_net": index.values,
            "index_neg": index_neg.values,
            "neg_log_mom_pct": neg_log_mom.values,
            "hf_level": index_neg.values,
            "hf_fitted": np.nan,
            "hf_mom_pct": neg_log_mom.values,
            "yield_10y": yield_d.values,
        }
    )
    _write_merged(
        RATE_DIR / "hf_rate_factor_daily.csv",
        out,
        "日期",
        ["index_net", "index_neg", "neg_log_mom_pct", "hf_level", "hf_fitted", "hf_mom_pct", "yield_10y"],
    )
    idx_out = pd.DataFrame({"日期": index.index, "index_net": index.values})
    _write_merged(RATE_DIR / "cn_gov_bond_index_daily.csv", idx_out, "日期", ["index_net"])


def build_hf_credit(df: pd.DataFrame) -> None:
    corp = df["企债财富_3_5"].dropna()
    cdb = df["国开财富_3_5"].reindex(corp.index).ffill()
    common = pd.concat([corp.rename("corp"), cdb.rename("cdb")], axis=1).dropna()
    wealth_diff = common["corp"] - common["cdb"]
    cycle, _ = hpfilter(wealth_diff.values, lamb=HP_LAMBDA_DAILY)
    cycle = pd.Series(cycle, index=wealth_diff.index)
    hf_raw = -cycle
    std = float(hf_raw.std())
    hf_index = 1.0 + (hf_raw - hf_raw.mean()) / std * HF_INDEX_VOL
    hf_mom = hf_index.pct_change() * 100
    out = pd.DataFrame(
        {
            "日期": common.index,
            "corp_wealth": common["corp"].values,
            "cdb_wealth": common["cdb"].values,
            "wealth_ratio": (common["corp"] / common["cdb"]).values,
            "wealth_diff": wealth_diff.values,
            "wealth_diff_cycle": cycle.values,
            "hf_raw": hf_raw.values,
            "hf_credit_factor": hf_index.values,
            "hf_mom_pct": hf_mom.values,
        }
    )
    _write_merged(
        CREDIT_DIR / "hf_credit_factor_daily.csv",
        out,
        "日期",
        [
            "corp_wealth",
            "cdb_wealth",
            "wealth_ratio",
            "wealth_diff",
            "wealth_diff_cycle",
            "hf_raw",
            "hf_credit_factor",
            "hf_mom_pct",
        ],
    )


def build_hf_inflation(df: pd.DataFrame) -> None:
    meta_path = INFLATION_DIR / "hf_regression_results.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    weights = meta.get("weights") or {"pork": 0.2, "brent": 0.4, "rebar": 0.4}
    lags = meta.get("lags_months") or {"pork": 0, "brent": 1, "rebar": 3}

    weekly = (
        pd.DataFrame(
            {
                "pork": df["猪肉"],
                "brent": df["布伦特原油"],
                "rebar": df["螺纹钢"],
            }
        )
        .dropna(how="all")
        .resample("W-FRI")
        .last()
        .ffill()
        .dropna(how="any")
    )
    wow = np.log(weekly / weekly.shift(1)) * 100
    parts = [weights[c] * wow[c].shift(int(lags[c]) * WEEKS_PER_MONTH) for c in ("pork", "brent", "rebar")]
    hf_wow = pd.concat(parts, axis=1).sum(axis=1, min_count=len(parts))
    valid = hf_wow.dropna()
    # hf_wow 是对数收益率，累计时必须用 exp(cumsum)，不能按简单收益 cumprod。
    hf_nav = np.exp((valid / 100.0).cumsum())
    hf_nav = hf_nav.reindex(weekly.index)
    hf_yoy = np.log(hf_nav / hf_nav.shift(WEEKS_YEAR)) * 100
    out = weekly.copy()
    out["hf_wow"] = hf_wow
    out["hf_nav"] = hf_nav
    out["hf_yoy_pct"] = hf_yoy
    out = out.reset_index().rename(columns={"index": "date"})
    out = _rebase_level_to_history(
        INFLATION_DIR / "hf_inflation_weekly.csv",
        out,
        "date",
        "hf_nav",
    )
    out = _recompute_log_change_with_history(
        INFLATION_DIR / "hf_inflation_weekly.csv",
        out,
        "date",
        "hf_nav",
        "hf_yoy_pct",
        WEEKS_YEAR,
    )
    _write_merged(
        INFLATION_DIR / "hf_inflation_weekly.csv",
        out,
        "date",
        ["pork", "brent", "rebar", "hf_wow", "hf_nav", "hf_yoy_pct"],
    )
    # commodities 日/周缓存
    _write_merged(
        INFLATION_DIR / "commodities.csv",
        weekly.reset_index().rename(columns={"index": "date"}),
        "date",
        ["pork", "brent", "rebar"],
    )


def build_hf_growth(df: pd.DataFrame) -> None:
    if "CAD" in df.columns and df["CAD"].notna().any():
        # CAD 是新浪外盘代码中的 LME 三个月铜，不是加拿大元。
        # 使用旧版固定参数，保持增长因子的历史定义和符号可比。
        assets = ["恒生指数", "CAD", "申万房地产"]
        lags = {"恒生指数": 3, "CAD": 0, "申万房地产": 1}
        weights = {
            "恒生指数": -0.08095842563026655,
            "CAD": 0.7717700560579429,
            "申万房地产": -0.14727151831179047,
        }
        print("  增长 HF: 使用旧版 LME 铜固定权重/滞后")
    else:
        assets = ["恒生指数", "南华沪铜", "申万房地产"]
        lags = None
        weights = None
    daily = df[assets].dropna(how="any")
    if lags is None or weights is None:
        # 没有 LME 铜时才用南华沪铜替代并动态估计。
        monthly = daily.resample("ME").last()
        X = np.log(monthly / monthly.shift(12)) * 100
        X.index = X.index.to_period("M")

        growth = pd.read_csv(GROWTH_DIR / "growth_factor.csv", parse_dates=["date"])
        y = growth.set_index(growth["date"].dt.to_period("M"))["raw_growth_factor"].dropna()
        idx = X.dropna(how="any").index.intersection(y.index)
        try:
            best, lags = search_joint_lags(y.loc[idx], X.loc[idx, assets], assets)
            model = best["model"]
            betas = {c: float(model.params[c]) for c in assets}
            weights = normalized_weights(betas)
            print(f"  增长 HF 重估: lags={lags}, weights={ {k: round(v,3) for k,v in weights.items()} }")
        except Exception as exc:
            print(f"  增长 HF 重估失败（{exc}），改用等权 lag0")
            lags = {c: 0 for c in assets}
            weights = {c: 1 / len(assets) for c in assets}

    daily_ret = np.log(daily / daily.shift(1)) * 100
    parts = [weights[c] * daily_ret[c].shift(int(lags[c]) * TRADING_DAYS_MONTH) for c in assets]
    hf_mom = pd.concat(parts, axis=1).sum(axis=1, min_count=len(parts))
    valid = hf_mom.dropna()
    hf_nav = NAV_BASE * np.exp((valid / 100.0).cumsum())
    hf_yoy = np.log(hf_nav / hf_nav.shift(TRADING_DAYS_YEAR)) * 100
    out = pd.DataFrame(
        {
            "date": daily.index,
            "hf_mom_pct": hf_mom.reindex(daily.index).values,
            "hf_growth_factor": hf_nav.reindex(daily.index).values,
            "hf_yoy": hf_yoy.reindex(daily.index).values,
        }
    )
    out = _rebase_level_to_history(
        GROWTH_DIR / "hf_growth_factor_synthetic.csv",
        out,
        "date",
        "hf_growth_factor",
    )
    out = _recompute_log_change_with_history(
        GROWTH_DIR / "hf_growth_factor_synthetic.csv",
        out,
        "date",
        "hf_growth_factor",
        "hf_yoy",
        TRADING_DAYS_YEAR,
    )
    _write_merged(
        GROWTH_DIR / "hf_growth_factor_synthetic.csv",
        out,
        "date",
        ["hf_mom_pct", "hf_growth_factor", "hf_yoy"],
    )
    _write_merged(
        GROWTH_DIR / "growth_high_freq_daily.csv",
        daily.reset_index().rename(columns={"index": "date"}),
        "date",
        assets,
    )


def build_hf_mobility(df: pd.DataFrame) -> None:
    meta_path = MOBILITY_DIR / "hf_regression_results.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    weights = meta.get("weights") or {"申万大盘市盈率": -0.64, "申万小盘市盈率": 0.36}
    lags = meta.get("lags_months") or {"申万大盘市盈率": 0, "申万小盘市盈率": 0}
    assets = ["申万大盘市盈率", "申万小盘市盈率"]
    daily = df[assets].dropna(how="any")
    daily_ret = np.log(daily / daily.shift(1)) * 100
    parts = [weights[c] * daily_ret[c].shift(int(lags[c]) * TRADING_DAYS_MONTH) for c in assets]
    hf_mom = pd.concat(parts, axis=1).sum(axis=1, min_count=len(parts))
    valid = hf_mom.dropna()
    hf_nav = NAV_BASE * np.exp((valid / 100.0).cumsum())
    hf_yoy = np.log(hf_nav / hf_nav.shift(TRADING_DAYS_YEAR)) * 100
    out = pd.DataFrame(
        {
            "date": daily.index,
            "hf_mom_pct": hf_mom.reindex(daily.index).values,
            "hf_mobility_factor": hf_nav.reindex(daily.index).values,
            "hf_yoy": hf_yoy.reindex(daily.index).values,
        }
    )
    out = _rebase_level_to_history(
        MOBILITY_DIR / "hf_mobility_factor_synthetic.csv",
        out,
        "date",
        "hf_mobility_factor",
    )
    out = _recompute_log_change_with_history(
        MOBILITY_DIR / "hf_mobility_factor_synthetic.csv",
        out,
        "date",
        "hf_mobility_factor",
        "hf_yoy",
        TRADING_DAYS_YEAR,
    )
    _write_merged(
        MOBILITY_DIR / "hf_mobility_factor_synthetic.csv",
        out,
        "date",
        ["hf_mom_pct", "hf_mobility_factor", "hf_yoy"],
    )
    _write_merged(
        MOBILITY_DIR / "mobility_high_freq_daily.csv",
        daily.reset_index().rename(columns={"index": "date"}),
        "date",
        assets,
    )


def build_hf_geo(df: pd.DataFrame) -> None:
    assets = ["沪金", "布伦特原油"]
    missing = [c for c in assets if c not in df.columns or not df[c].notna().any()]
    if missing:
        raise ValueError(f"缺少高频地缘代理: {', '.join(missing)}")
    if "gpr" not in df.columns or not df["gpr"].notna().any():
        raise ValueError("缺少「全球:地缘政治风险指数」，无法估计黄金/石油权重")

    xlsx_start = df.index.min()
    daily = df[assets].copy()
    asset_path = COMBINED_CLOSE
    if asset_path.exists():
        history = pd.read_csv(asset_path)
        history["date"] = _to_naive_datetime(history["date"])
        keep_cols = [c for c in assets if c in history.columns]
        if keep_cols:
            history = history.set_index("date")[keep_cols]
            history = history[history.index < xlsx_start]
            daily = pd.concat([history, daily], axis=0)
    daily = daily.apply(pd.to_numeric, errors="coerce")
    daily = daily[~daily.index.duplicated(keep="last")].sort_index().ffill().dropna(how="any")

    y = monthly_last(df["gpr"])
    y.index = y.index.to_period("M")
    X = daily.resample("ME").last()
    X.index = X.index.to_period("M")
    idx = X.dropna(how="any").index.intersection(y.dropna().index)
    lags = {"沪金": 3, "布伦特原油": 0}
    betas = {"沪金": 0.10918047909126594, "布伦特原油": 1.2941936309647657}
    intercept = -27.28954659639882
    try:
        best, lags = search_joint_lags(y.loc[idx], X.loc[idx, assets], assets)
        model = best["model"]
        intercept = float(model.params.get("const", 0.0))
        betas = {c: float(model.params[c]) for c in assets}
        print(
            "  地缘 HF: 金/油价格水平拟合 GPR 水平 "
            f"R²={model.rsquared:.3f} lags={lags} "
            f"betas={{ {', '.join(f'{k}:{v:.4f}' for k, v in betas.items())} }}"
        )
        meta = {
            "lags_months": lags,
            "weights": normalized_weights(betas),
            "betas": betas,
            "intercept": intercept,
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "bic": float(best["bic"]),
            "n_obs": int(best["n"]),
            "assets": assets,
            "y_definition": "全球地缘政治风险指数月末水平",
            "x_definition": "沪金、布伦特原油月末绝对价格",
        }
        fitted = model.fittedvalues
        monthly_fit = pd.DataFrame(
            {
                "ym": fitted.index.astype(str),
                "gpr_actual": y.loc[fitted.index].values,
                "gpr_fitted": fitted.values,
            }
        )
        monthly_fit.to_csv(
            POLITICS_DIR / "geo_fit_monthly.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
        )
    except Exception as exc:
        print(f"  地缘 HF 重估失败（{exc}），改用旧水平拟合系数")
        meta = {
            "lags_months": lags,
            "weights": normalized_weights(betas),
            "betas": betas,
            "intercept": intercept,
            "assets": assets,
            "fallback": str(exc),
            "y_definition": "全球地缘政治风险指数月末水平",
            "x_definition": "沪金、布伦特原油月末绝对价格",
        }

    level = pd.Series(intercept, index=daily.index, dtype=float)
    for col in assets:
        level = level + betas[col] * daily[col].shift(int(lags[col]) * TRADING_DAYS_MONTH)
    level = level[level.index >= xlsx_start]
    current = daily.loc[daily.index >= xlsx_start]
    out = pd.DataFrame({"date": level.index, "hf_geo_factor": level.values})
    geo_path = POLITICS_DIR / "hf_geo_factor_synthetic.csv"
    geo_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(geo_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"  → {geo_path.relative_to(ROOT)} ({len(out)} 行；金/油绝对价格线性拟合 GPR 水平)")
    _write_merged(
        POLITICS_DIR / "geo_high_freq_daily.csv",
        current.reset_index().rename(columns={"index": "date"}),
        "date",
        assets,
    )
    meta_path = POLITICS_DIR / "hf_regression_results.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def build_assets(df: pd.DataFrame) -> None:
    cols = [
        "上证50",
        "沪深300",
        "中证500",
        "中证1000",
        "恒生指数",
        "中债国债",
        "中债企业债",
        "中证转债",
        "布伦特原油",
        "沪金",
        "标普500",
        "美元兑人民币",
    ]
    panel = df[[c for c in cols if c in df.columns]].copy().dropna(how="all")
    missing_assets = [c for c in cols if c not in panel.columns]
    if missing_assets:
        print(f"  资产面板缺列，沿用历史: {', '.join(missing_assets)}")
    extra = load_additional_assets()
    extra_names = list(extra.columns) if not extra.empty else []
    path = COMBINED_CLOSE
    # Wind 财富指数与旧 chinabond 序列量纲常不一致。按日 merge 会交错假跳价；
    # 正确做法：xlsx 起点后整段替换，并在接缝处按旧序列水平缩放，保留 Wind 收益率。
    xlsx_start = panel.index.min()
    extra_start = extra.index.min() if extra_names else None
    old = None
    if path.exists():
        old = pd.read_csv(path)
        old["date"] = _to_naive_datetime(old["date"])
        old = old.dropna(subset=["date"]).set_index("date").sort_index()

    if old is not None:
        keep = old[old.index < pd.Timestamp(xlsx_start)].copy()
        for c in panel.columns:
            if c not in keep.columns:
                keep[c] = pd.NA
            old_hist = keep[c].dropna()
            new_hist = panel[c].dropna()
            if not old_hist.empty and not new_hist.empty and abs(float(new_hist.iloc[0])) > 1e-12:
                scale = float(old_hist.iloc[-1]) / float(new_hist.iloc[0])
                if abs(scale - 1.0) > 0.02:
                    panel[c] = panel[c] * scale
                    print(f"    缩放 {c}: ×{scale:.4f}（接缝对齐）")
        for c in missing_assets:
            if c in old.columns:
                panel[c] = old[c].reindex(panel.index)
        keep_core = keep.reindex(columns=cols)
        panel_core = panel.reindex(columns=cols)
        keep_core = keep_core.reset_index()[["date"] + cols]
        incoming = panel_core.reset_index().rename(columns={"index": "date"})
        out = pd.concat([keep_core, incoming], ignore_index=True)
        out = out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
        if missing_assets:
            old_full = old.reset_index().rename(columns={"index": "date"})
            for c in missing_assets:
                if c in old_full.columns:
                    restored = old_full[["date", c]].dropna(subset=[c])
                    out = out.drop(columns=[c], errors="ignore").merge(restored, on="date", how="left")
        # additional_asset 文件在的话，额外资产以这份表为准；删掉的旧列不再留在暴露图上。
        extra_keys = set(extra_names) | {pretty_asset_name(c) for c in extra_names}
        old_extras = [
            c
            for c in old.columns
            if c not in cols
            and c not in extra_keys
            and pretty_asset_name(c) not in extra_keys
        ]
        if additional_asset_path() is not None:
            if old_extras:
                print("  已按 additional_asset 去掉旧额外资产: " + "、".join(old_extras))
        elif old_extras:
            hist = old.reset_index().rename(columns={"index": "date"})[["date"] + old_extras]
            out = out.merge(hist, on="date", how="left")
    else:
        out = panel.reset_index().rename(columns={"index": "date"})

    if extra_names:
        extra_panel = extra.copy()
        if old is not None and extra_start is not None:
            keep_extra = old[old.index < pd.Timestamp(extra_start)]
            for c in extra_names:
                old_col = c if c in keep_extra.columns else next(
                    (
                        oc
                        for oc in keep_extra.columns
                        if pretty_asset_name(oc) == c or pretty_asset_name(oc) == pretty_asset_name(c)
                    ),
                    None,
                )
                old_hist = keep_extra[old_col].dropna() if old_col else pd.Series(dtype=float)
                new_hist = extra_panel[c].dropna()
                if not old_hist.empty and not new_hist.empty and abs(float(new_hist.iloc[0])) > 1e-12:
                    scale = float(old_hist.iloc[-1]) / float(new_hist.iloc[0])
                    if abs(scale - 1.0) > 0.02:
                        extra_panel[c] = extra_panel[c] * scale
                        print(f"    缩放 {c}: ×{scale:.4f}（额外资产接缝对齐）")
        extra_in = extra_panel.reset_index().rename(columns={"index": "date"})
        extra_in["date"] = _to_naive_datetime(extra_in["date"])
        out["date"] = _to_naive_datetime(out["date"])
        out = out.merge(extra_in, on="date", how="outer", suffixes=("", "_new"))
        for c in extra_names:
            new_col = f"{c}_new"
            if new_col in out.columns:
                out[c] = out[new_col].combine_first(out[c]) if c in out.columns else out[new_col]
                out = out.drop(columns=[new_col])
        out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")
    extra_msg = f"；另含额外资产 {', '.join(extra_names)}" if extra_names else ""
    print(f"  → {path.relative_to(ROOT)} ({len(out)} 行；{xlsx_start.date()} 起用 Wind 整段替换{extra_msg})")
    asset_cols = [c for c in out.columns if str(c).strip() and str(c).strip().lower() != "date"]
    sync_asset_factor_mask(asset_cols)

    raw_dir = EXPOSURE_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in ("沪金", "布伦特原油"):
        s = df[name].dropna()
        out = pd.DataFrame({"date": s.index, "close": s.values, "asset": name, "source": "data.xlsx"})
        raw_path = raw_dir / f"{name}.csv"
        if raw_path.exists():
            old = pd.read_csv(raw_path)
            old["date"] = _to_naive_datetime(old["date"])
            keep = old[old["date"] < s.index.min()].copy()
            cols_keep = [c for c in ("date", "close", "asset", "source") if c in keep.columns]
            if cols_keep and not keep.empty and "close" in keep.columns and abs(float(s.iloc[0])) > 1e-12:
                scale = float(pd.to_numeric(keep["close"], errors="coerce").dropna().iloc[-1]) / float(s.iloc[0])
                if abs(scale - 1.0) > 0.02:
                    out["close"] = out["close"] * scale
            keep = keep[cols_keep] if cols_keep else keep
            out = pd.concat([keep, out], ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
        out.to_csv(raw_path, index=False, encoding="utf-8-sig", float_format="%.6f")
        print(f"  → {raw_path.relative_to(ROOT)} ({len(out)} 行)")


def run_all(data_path: Path) -> None:
    ensure_output_dirs()
    print(f"读取 {data_path}")
    df = load_wind_data(data_path)
    print(f"样本 {df.index.min().date()} ~ {df.index.max().date()}，{len(df)} 日，{len(df.columns)} 列")
    _require_columns(df, FACTOR_REQUIRED)

    print("\n[LF]")
    build_lf_growth(df)
    build_lf_inflation(df)
    build_lf_rate(df)
    build_lf_credit(df)
    build_lf_mobility(df)
    build_dxy(df)
    build_lf_geo(df)

    print("\n[HF]")
    build_hf_rate(df)
    build_hf_credit(df)
    build_hf_inflation(df)
    build_hf_growth(df)
    build_hf_mobility(df)
    build_hf_geo(df)

    print("\n[Assets]")
    build_assets(df)
    print("\nWind 本地数据因子更新完成。")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="从 Wind data.xlsx 更新全部因子")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Wind data.xlsx / data.csv 路径")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_all(args.data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
