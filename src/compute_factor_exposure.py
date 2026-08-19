#!/usr/bin/env python3
"""基于高频环比收益的滚动 Bootstrap + Lasso 估计资产宏观因子暴露。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler

from paths import (
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
from panel_config import (
    build_factor_mask,
    exposure_factor_columns,
    factors_for_asset,
    load_panel_config,
)

ASSET_CLOSE_CSV = COMBINED_CLOSE
WEEK_FREQ = "W-FRI"

FACTOR_LABELS = ["增长因子", "通胀因子", "利率因子", "信用因子", "汇率因子", "地缘因子", "流动性因子"]
NON_CREDIT_FACTORS = ["增长因子", "通胀因子", "利率因子", "汇率因子", "地缘因子", "流动性因子"]
BOND_ASSETS = {"中债国债", "中债企业债", "中证转债"}
BOND_NAME_MARKERS = ("债", "转债", "城投", "政金债", "信用债", "利率债")

ROLLING_WINDOW_WEEKS = 260
SAMPLE_LENGTH_WEEKS = 104
BOOTSTRAP_SAMPLES = 3000
RANDOM_SEED = 42
ALPHA_SCALE = 0.5

OUTPUT_CSV = EXPOSURE_DIR / "factor_exposure_latest.csv"
OUTPUT_JSON = EXPOSURE_DIR / "factor_exposure_latest.json"
OUTPUT_PNG = EXPOSURE_DIR / "factor_exposure_latest.png"
OUTPUT_PANEL = EXPOSURE_DIR / "factor_exposure_weekly_panel.csv"
OUTPUT_CUSTOM_CSV = EXPOSURE_DIR / "factor_exposure_custom.csv"
OUTPUT_CUSTOM_JSON = EXPOSURE_DIR / "factor_exposure_custom.json"

DATE_COLUMNS = ("date", "日期", "Date", "时间", "datetime")
PRICE_COLUMNS = ("close", "price", "收盘", "收盘价", "净值", "value", "nav", "Close", "Price")


def setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {str(c).strip().lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def load_custom_price_series(path: Path) -> pd.Series:
    """读用户给的资产时间序列。两列即可：日期 + 收盘价/净值。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到资产文件: {path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        raw = pd.read_excel(path, header=None)
        raw = raw.dropna(how="all")
        if raw.empty:
            raise ValueError("资产文件是空的")
        raw = raw.reset_index(drop=True)
        frame = raw.copy()
        frame.columns = [str(c).strip() if pd.notna(c) else "" for c in raw.iloc[0].tolist()]
        frame = frame.iloc[1:].reset_index(drop=True)
    elif suffix in {".csv", ".txt"}:
        frame = None
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "gb18030", "gbk"):
            try:
                frame = pd.read_csv(path, encoding=encoding)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
        if frame is None:
            raise ValueError(f"无法识别文件编码: {path}") from last_error
    else:
        raise ValueError("资产文件请用 csv 或 xlsx，至少两列：日期、收盘价")

    frame = frame.dropna(axis=1, how="all")
    if frame.empty or frame.shape[1] < 2:
        raise ValueError("资产文件至少需要一列日期、一列价格")

    date_col = _pick_column(frame.columns.tolist(), DATE_COLUMNS)
    if date_col is None:
        date_col = frame.columns[0]
    price_col = _pick_column(frame.columns.tolist(), PRICE_COLUMNS)
    if price_col is None or price_col == date_col:
        numeric_cols = [
            c
            for c in frame.columns
            if c != date_col and pd.to_numeric(frame[c], errors="coerce").notna().any()
        ]
        if not numeric_cols:
            raise ValueError("找不到价格列。请把价格列命名为 close / 收盘价 / 净值")
        price_col = numeric_cols[0]

    dates = pd.to_datetime(frame[date_col], errors="coerce")
    if dates.isna().mean() > 0.5:
        serial = pd.to_numeric(frame[date_col], errors="coerce")
        dates = pd.Timestamp("1899-12-30") + pd.to_timedelta(serial, unit="D")
    prices = pd.to_numeric(frame[price_col], errors="coerce")
    series = pd.Series(prices.to_numpy(), index=dates).dropna()
    series = series[series > 0]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        raise ValueError("价格序列为空，或全是 0 / 空值")
    series.name = path.stem
    return series


def prices_to_weekly_returns(prices: pd.Series) -> pd.Series:
    weekly = prices.resample(WEEK_FREQ).last().dropna()
    returns = np.log(weekly / weekly.shift(1)) * 100
    return returns.dropna()


def load_asset_weekly_returns() -> pd.DataFrame:
    prices = pd.read_csv(ASSET_CLOSE_CSV, parse_dates=["date"]).sort_values("date").set_index("date")
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.mask(prices <= 0)
    prices = prices.ffill()
    weekly_price = prices.resample(WEEK_FREQ).last()
    returns = np.log(weekly_price / weekly_price.shift(1)) * 100
    return returns.dropna(how="all")


def read_series(path: Path, date_col: str, value_col: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=[date_col])
    return df.sort_values(date_col).set_index(date_col)[value_col].astype(float).dropna()


def weekly_sum(series: pd.Series) -> pd.Series:
    return series.resample(WEEK_FREQ).sum(min_count=1).dropna()


def load_macro_weekly_mom() -> pd.DataFrame:
    """宏观高频因子的周度环比收益/变化。"""
    growth = weekly_sum(
        read_series(GROWTH_DIR / "hf_growth_factor_synthetic.csv", "date", "hf_mom_pct")
    )
    inflation = read_series(INFLATION_DIR / "hf_inflation_weekly.csv", "date", "hf_wow")
    inflation = inflation.resample(WEEK_FREQ).last().dropna()
    rate = weekly_sum(
        read_series(RATE_DIR / "hf_rate_factor_daily.csv", "日期", "hf_mom_pct")
    )
    credit = weekly_sum(
        read_series(CREDIT_DIR / "hf_credit_factor_daily.csv", "日期", "hf_mom_pct")
    )

    dxy = pd.read_csv(EXCHANGE_DIR / "dxy_yahoo.csv", parse_dates=["Date"])
    dxy["date"] = pd.to_datetime(dxy["Date"], utc=True).dt.tz_convert(None)
    dxy_weekly = dxy.sort_values("date").set_index("date")["close"].astype(float).resample(WEEK_FREQ).last()
    exchange = np.log(dxy_weekly / dxy_weekly.shift(1)) * 100

    geo_level = read_series(POLITICS_DIR / "hf_geo_factor_synthetic.csv", "date", "hf_geo_factor")
    geo_weekly = geo_level.resample(WEEK_FREQ).last()
    politics = np.log(geo_weekly / geo_weekly.shift(1)) * 100
    mobility = weekly_sum(
        read_series(MOBILITY_DIR / "hf_mobility_factor_synthetic.csv", "date", "hf_mom_pct")
    )

    panel = pd.DataFrame(
        {
            "增长因子": growth,
            "通胀因子": inflation,
            "利率因子": rate,
            "信用因子": credit,
            "汇率因子": exchange,
            "地缘因子": politics,
            "流动性因子": mobility,
        }
    )
    return panel[FACTOR_LABELS].dropna(how="all")


def allowed_factors(asset: str, as_bond: bool = False, cfg: dict | None = None) -> list[str]:
    return factors_for_asset(asset, cfg, as_bond=as_bond)


def standardized_lasso_coefficients(
    y: pd.Series,
    x: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
    sample_length: int,
    alpha_scale: float,
) -> tuple[pd.Series, float]:
    common = pd.concat([y.rename("asset_return"), x], axis=1).dropna()
    if len(common) < sample_length:
        return pd.Series(0.0, index=x.columns), 0.0

    y_full = common["asset_return"].to_numpy(dtype=float)
    x_full = common[x.columns].to_numpy(dtype=float)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_std = x_scaler.fit_transform(x_full)
    y_std = y_scaler.fit_transform(y_full.reshape(-1, 1)).ravel()

    if np.isclose(np.std(y_std), 0):
        return pd.Series(0.0, index=x.columns), 0.0

    cv = min(5, len(common))
    lasso_cv = LassoCV(cv=cv, random_state=RANDOM_SEED, max_iter=20000).fit(x_std, y_std)
    alpha = float(lasso_cv.alpha_) * alpha_scale

    max_start = len(common) - sample_length
    coefs = np.zeros((n_bootstrap, len(x.columns)))
    intercepts = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        start = int(rng.integers(0, max_start + 1))
        sl = slice(start, start + sample_length)
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=20000)
        model.fit(x_std[sl], y_std[sl])
        coefs[i] = model.coef_
        intercepts[i] = model.intercept_

    coef_median = np.median(coefs, axis=0)
    intercept_median = float(np.median(intercepts))
    y_hat = intercept_median + x_std @ coef_median
    ss_res = float(np.sum((y_std - y_hat) ** 2))
    ss_tot = float(np.sum((y_std - np.mean(y_std)) ** 2))
    r_squared = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return pd.Series(coef_median, index=x.columns), r_squared


def ols_r_squared(y: pd.Series, x: pd.DataFrame) -> float:
    """同一窗口、同一批因子的普通最小二乘 R²，作为解释力上限。"""
    common = pd.concat([y.rename("asset_return"), x], axis=1).dropna()
    if len(common) <= x.shape[1] + 2:
        return 0.0
    y_full = common["asset_return"].to_numpy(dtype=float)
    x_full = common[x.columns].to_numpy(dtype=float)
    if np.isclose(np.std(y_full), 0):
        return 0.0
    score = float(LinearRegression().fit(x_full, y_full).score(x_full, y_full))
    if not np.isfinite(score):
        return 0.0
    return max(0.0, score)


def select_exposure_window(
    common_index: pd.DatetimeIndex,
    rolling_window: int,
    sample_length: int,
    end_date: str | None = None,
) -> tuple[pd.DatetimeIndex, int]:
    if len(common_index) < sample_length:
        raise RuntimeError(
            f"共同样本不足，至少需要 {sample_length} 周重叠，当前只有 {len(common_index)} 周。"
            "请把资产历史加长（建议 3–5 年日收盘价）。"
        )
    if rolling_window > len(common_index):
        print(
            f"提示: 请求滚动窗口 {rolling_window} 周超过可用共同样本 "
            f"{len(common_index)} 周，已自动下调为 {len(common_index)} 周。"
        )
        rolling_window = len(common_index)
    if rolling_window < sample_length:
        raise RuntimeError(
            f"滚动窗口 {rolling_window} 周小于样本长度 {sample_length} 周"
        )
    if end_date is None:
        return common_index[-rolling_window:], rolling_window

    end_ts = pd.Timestamp(end_date)
    valid_ends = common_index[common_index <= end_ts]
    if valid_ends.empty:
        raise ValueError(f"结束日期 {end_date} 早于可用样本起点")
    end_ts = valid_ends[-1]
    end_pos = common_index.get_loc(end_ts)
    start_pos = end_pos - rolling_window + 1
    if start_pos < 0:
        raise ValueError(
            f"结束周 {end_ts.strftime('%Y-%m-%d')} 之前不足 {rolling_window} 周有效样本"
        )
    return common_index[start_pos : end_pos + 1], rolling_window


def exposure_meta(
    window_index: pd.DatetimeIndex,
    rolling_window: int,
    sample_length: int,
    n_bootstrap: int,
    alpha_scale: float,
    extra: dict | None = None,
    cfg: dict | None = None,
) -> dict:
    meta = {
        "as_of": window_index[-1].strftime("%Y-%m-%d"),
        "window_start": window_index[0].strftime("%Y-%m-%d"),
        "window_end": window_index[-1].strftime("%Y-%m-%d"),
        "frequency": "weekly",
        "y_definition": "大类资产周度对数收益率",
        "x_definition": "宏观高频因子周度环比收益/变化",
        "rolling_window_weeks": rolling_window,
        "sample_length_weeks": sample_length,
        "bootstrap_samples": n_bootstrap,
        "alpha_scale": alpha_scale,
        "method": "standardized Lasso coefficients, weekly rolling contiguous-block bootstrap median",
        "credit_factor_rule": (
            "信用因子仅进入债券类资产回归，其余资产默认置 0；asset_factor_mask 可按资产覆盖"
            if (cfg or {}).get("exposure", {}).get("credit_only_for_bonds", True)
            else "信用因子对全部资产开放；asset_factor_mask 可按资产关掉"
        ),
        "geo_factor_rule": "沪金、布伦特原油默认参与地缘因子暴露回归（可用 asset_factor_mask 或 asset_exclude_factors 关掉）",
        "bond_assets": list((cfg or {}).get("exposure", {}).get("bond_assets") or sorted(BOND_ASSETS)),
        "r_squared_definition": "同一窗口、同一批因子的普通最小二乘 R²；系数仍来自 Bootstrap + Lasso 中位数",
    }
    if extra:
        meta.update(extra)
    return meta


def compute_custom_exposure(
    price_path: Path,
    asset_name: str | None = None,
    as_bond: bool = False,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    rolling_window: int = ROLLING_WINDOW_WEEKS,
    sample_length: int = SAMPLE_LENGTH_WEEKS,
    seed: int = RANDOM_SEED,
    alpha_scale: float = ALPHA_SCALE,
    end_date: str | None = None,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    cfg = cfg or load_panel_config()
    factor_cols = exposure_factor_columns(cfg)
    prices = load_custom_price_series(price_path)
    name = (asset_name or "").strip() or str(prices.name)
    weekly = prices_to_weekly_returns(prices).rename(name)
    macro = load_macro_weekly_mom()
    common_index = weekly.dropna().index.intersection(macro.dropna(how="any").index).sort_values()
    window_index, rolling_window = select_exposure_window(
        common_index, rolling_window, sample_length, end_date
    )
    asset_window = weekly.loc[window_index]
    macro_window = macro.loc[window_index]
    factors = allowed_factors(name, as_bond=as_bond, cfg=cfg)
    rng = np.random.default_rng(seed)
    coefs, r2 = standardized_lasso_coefficients(
        asset_window,
        macro_window[factors],
        rng=rng,
        n_bootstrap=n_bootstrap,
        sample_length=sample_length,
        alpha_scale=alpha_scale,
    )
    exposure = pd.DataFrame(0.0, index=[name], columns=factor_cols)
    exposure.loc[name, factors] = coefs
    r_squared = pd.Series(
        {name: ols_r_squared(asset_window, macro_window[factors])},
        name="R方",
    )
    meta = exposure_meta(
        window_index,
        rolling_window,
        sample_length,
        n_bootstrap,
        alpha_scale,
        extra={
            "custom_asset": name,
            "custom_file": str(price_path),
            "price_start": prices.index.min().strftime("%Y-%m-%d"),
            "price_end": prices.index.max().strftime("%Y-%m-%d"),
            "n_price_rows": int(len(prices)),
            "n_overlap_weeks": int(len(common_index)),
            "as_bond": bool(as_bond),
            "factor_mask": build_factor_mask([name], cfg, as_bond={name: bool(as_bond)}),
            "asset_factor_mask": cfg["exposure"].get("asset_factor_mask") or {},
        },
        cfg=cfg,
    )
    return exposure, r_squared, meta


def compute_latest_exposure(
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    rolling_window: int = ROLLING_WINDOW_WEEKS,
    sample_length: int = SAMPLE_LENGTH_WEEKS,
    seed: int = RANDOM_SEED,
    alpha_scale: float = ALPHA_SCALE,
    end_date: str | None = None,
    write_panel: bool = True,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    cfg = cfg or load_panel_config()
    factor_cols = exposure_factor_columns(cfg)
    asset_returns = load_asset_weekly_returns()
    macro = load_macro_weekly_mom()
    common_index = asset_returns.index.intersection(macro.dropna(how="any").index).sort_values()
    window_index, rolling_window = select_exposure_window(
        common_index, rolling_window, sample_length, end_date
    )

    macro_window = macro.loc[window_index]
    asset_window = asset_returns.loc[window_index]
    if write_panel:
        panel_out = pd.concat(
            {
                "asset_return": asset_window,
                "macro_factor": macro_window,
            },
            axis=1,
        )
        panel_out.to_csv(OUTPUT_PANEL, encoding="utf-8-sig", float_format="%.6f")
    rng = np.random.default_rng(seed)

    exposure = pd.DataFrame(0.0, index=asset_window.columns, columns=factor_cols)
    r_squared = pd.Series(0.0, index=asset_window.columns, name="R方")
    for asset in asset_window.columns:
        factors = allowed_factors(asset, cfg=cfg)
        coefs, _r2 = standardized_lasso_coefficients(
            asset_window[asset],
            macro_window[factors],
            rng=rng,
            n_bootstrap=n_bootstrap,
            sample_length=sample_length,
            alpha_scale=alpha_scale,
        )
        exposure.loc[asset, factors] = coefs
        r_squared.loc[asset] = ols_r_squared(asset_window[asset], macro_window[factors])

    core_assets = [
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
    extra_assets = [a for a in exposure.index.tolist() if a not in core_assets]
    meta = exposure_meta(
        window_index,
        rolling_window,
        sample_length,
        n_bootstrap,
        alpha_scale,
        extra={
            "additional_assets": extra_assets,
            "used_factors": factor_cols,
            "exclude_factors": list(cfg["exposure"].get("exclude_factors") or []),
            "include_factors": list(cfg["exposure"].get("include_factors") or []),
            "asset_exclude_factors": cfg["exposure"].get("asset_exclude_factors") or {},
            "asset_factor_mask": cfg["exposure"].get("asset_factor_mask") or {},
            "factor_mask": build_factor_mask(exposure.index.tolist(), cfg),
        },
        cfg=cfg,
    )
    return exposure, r_squared, meta


def list_available_weeks(min_window: int = ROLLING_WINDOW_WEEKS) -> list[str]:
    asset_returns = load_asset_weekly_returns()
    macro = load_macro_weekly_mom()
    common_index = asset_returns.index.intersection(macro.dropna(how="any").index).sort_values()
    if len(common_index) < min_window:
        return []
    return common_index[min_window - 1 :].strftime("%Y-%m-%d").tolist()


def _json_number(value) -> float:
    if value is None or pd.isna(value):
        return 0.0
    number = float(value)
    if not np.isfinite(number):
        return 0.0
    return round(number, 6)


def build_exposure_payload(
    exposure: pd.DataFrame, r_squared: pd.Series, meta: dict
) -> dict:
    return {
        **meta,
        "assets": exposure.index.tolist(),
        "factors": exposure.columns.tolist(),
        "matrix": {
            asset: {factor: _json_number(exposure.loc[asset, factor]) for factor in exposure.columns}
            for asset in exposure.index
        },
        "r_squared": {asset: _json_number(r_squared.loc[asset]) for asset in exposure.index},
    }


def save_json(exposure: pd.DataFrame, r_squared: pd.Series, meta: dict) -> None:
    payload = build_exposure_payload(exposure, r_squared, meta)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_exposure(exposure: pd.DataFrame, meta: dict) -> None:
    setup_chinese_font()
    max_abs = max(0.1, float(np.nanmax(np.abs(exposure.values))))
    n_assets = max(1, len(exposure.index))
    fig, ax = plt.subplots(figsize=(10.5, max(7.5, 0.42 * n_assets + 2.2)), dpi=150)
    im = ax.imshow(exposure.values, cmap="RdYlGn_r", norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs))

    ax.set_xticks(np.arange(len(exposure.columns)))
    ax.set_yticks(np.arange(len(exposure.index)))
    ax.set_xticklabels(exposure.columns)
    ax.set_yticklabels(exposure.index)
    ax.set_title(
        f"资产-宏观因子暴露矩阵（{meta['window_start']} ~ {meta['window_end']}）",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    mask = meta.get("factor_mask") or {}
    for i in range(exposure.shape[0]):
        for j in range(exposure.shape[1]):
            asset = str(exposure.index[i])
            factor = str(exposure.columns[j])
            used = True
            if asset in mask and factor in mask[asset]:
                used = int(mask[asset][factor]) != 0
            if used:
                ax.text(
                    j, i, f"{exposure.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=8, color="#111827",
                )
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#94a3b8")

    ax.tick_params(axis="x", rotation=35)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("标准化暴露系数")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_custom_result(exposure: pd.DataFrame, r_squared: pd.Series, meta: dict) -> None:
    output = exposure.copy()
    output["R方"] = r_squared
    output.to_csv(OUTPUT_CUSTOM_CSV, encoding="utf-8-sig", float_format="%.6f")
    payload = build_exposure_payload(exposure, r_squared, meta)
    OUTPUT_CUSTOM_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    parser = argparse.ArgumentParser(description="计算资产对宏观因子的暴露矩阵")
    parser.add_argument("--bootstrap", type=int, default=None, help="Bootstrap 次数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--rolling-window-weeks", type=int, default=None, help="滚动窗口周数")
    parser.add_argument("--sample-length-weeks", type=int, default=None, help="每次重采样的连续周数")
    parser.add_argument(
        "--alpha-scale",
        type=float,
        default=None,
        help="LassoCV alpha 的缩放系数；小于 1 表示惩罚更弱",
    )
    parser.add_argument("--end-date", type=str, default=None, help="结束周 YYYY-MM-DD，默认读 panel_config.json")
    parser.add_argument(
        "--custom-asset",
        type=Path,
        default=None,
        help="你自己的资产价格文件（csv/xlsx，两列：日期、收盘价）。只算这一条，不覆盖看板里的 12 资产矩阵",
    )
    parser.add_argument("--name", type=str, default="", help="自定义资产显示名，默认用文件名")
    parser.add_argument(
        "--bond",
        action="store_true",
        help="把该资产当债券：回归里会加入信用因子。股票/商品不要加这个开关",
    )
    args = parser.parse_args()
    cfg = load_panel_config()
    exp = cfg["exposure"]
    n_bootstrap = args.bootstrap if args.bootstrap is not None else int(exp["bootstrap_samples"])
    rolling_window = (
        args.rolling_window_weeks if args.rolling_window_weeks is not None else int(exp["rolling_window_weeks"])
    )
    sample_length = (
        args.sample_length_weeks if args.sample_length_weeks is not None else int(exp["sample_length_weeks"])
    )
    seed = args.seed if args.seed is not None else int(exp["random_seed"])
    alpha_scale = args.alpha_scale if args.alpha_scale is not None else float(exp["alpha_scale"])
    end_date = args.end_date or exp.get("end_date") or None

    if args.custom_asset is not None:
        exposure, r_squared, meta = compute_custom_exposure(
            args.custom_asset,
            asset_name=args.name,
            as_bond=args.bond,
            n_bootstrap=n_bootstrap,
            rolling_window=rolling_window,
            sample_length=sample_length,
            seed=seed,
            alpha_scale=alpha_scale,
            end_date=end_date,
            cfg=cfg,
        )
        save_custom_result(exposure, r_squared, meta)
        print(f"资产: {exposure.index[0]}")
        print(f"价格区间: {meta['price_start']} ~ {meta['price_end']}（{meta['n_price_rows']} 个点）")
        print(
            f"暴露窗口: {meta['window_start']} ~ {meta['window_end']}, "
            f"重叠 {meta['n_overlap_weeks']} 周, bootstrap={meta['bootstrap_samples']}"
        )
        print(exposure.round(2).to_string())
        print(f"R方: {float(r_squared.iloc[0]):.3f}")
        print("已写入:", OUTPUT_CUSTOM_CSV)
        print("JSON:", OUTPUT_CUSTOM_JSON)
        print("这不会改网页上原来的 12 个资产暴露。")
        return

    exposure, r_squared, meta = compute_latest_exposure(
        n_bootstrap=n_bootstrap,
        rolling_window=rolling_window,
        sample_length=sample_length,
        seed=seed,
        alpha_scale=alpha_scale,
        end_date=end_date,
        cfg=cfg,
    )
    output = exposure.copy()
    output["R方"] = r_squared
    output.to_csv(OUTPUT_CSV, encoding="utf-8-sig", float_format="%.6f")
    save_json(exposure, r_squared, meta)
    plot_exposure(exposure, meta)

    print("因子暴露矩阵:", OUTPUT_CSV)
    print("网页 JSON:", OUTPUT_JSON)
    print("热力图:", OUTPUT_PNG)
    print(
        f"周频窗口: {meta['window_start']} ~ {meta['window_end']}, "
        f"bootstrap={meta['bootstrap_samples']}, alpha_scale={meta['alpha_scale']}"
    )
    print(exposure.round(2).to_string())
    print("R方:")
    print(r_squared.round(3).to_string())


if __name__ == "__main__":
    main()
