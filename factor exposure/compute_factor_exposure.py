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
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent
ASSET_CLOSE_CSV = OUTPUT_DIR / "data" / "combined_close.csv"
WEEK_FREQ = "W-FRI"

FACTOR_LABELS = ["增长因子", "通胀因子", "利率因子", "信用因子", "汇率因子", "地缘因子", "流动性因子"]
NON_CREDIT_FACTORS = ["增长因子", "通胀因子", "利率因子", "汇率因子", "地缘因子", "流动性因子"]
BOND_ASSETS = {"中债国债", "中债企业债", "中证转债"}
GEO_SOURCE_ASSETS = {"布伦特原油", "沪金"}

ROLLING_WINDOW_WEEKS = 260
SAMPLE_LENGTH_WEEKS = 104
BOOTSTRAP_SAMPLES = 3000
RANDOM_SEED = 42
ALPHA_SCALE = 0.5

OUTPUT_CSV = OUTPUT_DIR / "factor_exposure_latest.csv"
OUTPUT_JSON = OUTPUT_DIR / "factor_exposure_latest.json"
OUTPUT_PNG = OUTPUT_DIR / "factor_exposure_latest.png"
OUTPUT_PANEL = OUTPUT_DIR / "factor_exposure_weekly_panel.csv"


def setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_asset_weekly_returns() -> pd.DataFrame:
    prices = pd.read_csv(ASSET_CLOSE_CSV, parse_dates=["date"]).sort_values("date").set_index("date")
    prices = prices.apply(pd.to_numeric, errors="coerce").ffill()
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
        read_series(ROOT / "growth" / "hf_growth_factor_synthetic.csv", "date", "hf_mom_pct")
    )
    inflation = read_series(ROOT / "inflasion" / "hf_inflation_weekly.csv", "date", "hf_wow")
    inflation = inflation.resample(WEEK_FREQ).last().dropna()
    rate = weekly_sum(
        read_series(ROOT / "interest rate" / "hf_rate_factor_daily.csv", "日期", "hf_mom_pct")
    )
    credit = weekly_sum(
        read_series(ROOT / "credit" / "hf_credit_factor_daily.csv", "日期", "hf_mom_pct")
    )

    dxy = pd.read_csv(ROOT / "exchange" / "dxy_yahoo.csv", parse_dates=["Date"])
    dxy["date"] = pd.to_datetime(dxy["Date"], utc=True).dt.tz_convert(None)
    dxy_weekly = dxy.sort_values("date").set_index("date")["close"].astype(float).resample(WEEK_FREQ).last()
    exchange = np.log(dxy_weekly / dxy_weekly.shift(1)) * 100

    geo_level = read_series(ROOT / "politics" / "hf_geo_factor_synthetic.csv", "date", "hf_geo_factor")
    geo_weekly = geo_level.resample(WEEK_FREQ).last()
    politics = np.log(geo_weekly / geo_weekly.shift(1)) * 100
    mobility = weekly_sum(
        read_series(ROOT / "mobility" / "hf_mobility_factor_synthetic.csv", "date", "hf_mom_pct")
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


def allowed_factors(asset: str) -> list[str]:
    factors = FACTOR_LABELS if asset in BOND_ASSETS else NON_CREDIT_FACTORS
    # 地缘因子本身由布伦特和沪金合成，不能再用它解释这两个资产；
    # 否则会把“资产解释自己”的机械相关误报为高暴露和高 R²。
    if asset in GEO_SOURCE_ASSETS:
        factors = [factor for factor in factors if factor != "地缘因子"]
    return factors


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


def compute_latest_exposure(
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    rolling_window: int = ROLLING_WINDOW_WEEKS,
    sample_length: int = SAMPLE_LENGTH_WEEKS,
    seed: int = RANDOM_SEED,
    alpha_scale: float = ALPHA_SCALE,
    end_date: str | None = None,
    write_panel: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    asset_returns = load_asset_weekly_returns()
    macro = load_macro_weekly_mom()
    common_index = asset_returns.index.intersection(macro.dropna(how="any").index).sort_values()
    if len(common_index) < sample_length:
        raise RuntimeError(
            f"共同样本不足，至少需要 sample_length={sample_length} 周，当前 {len(common_index)} 周"
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
        window_index = common_index[-rolling_window:]
    else:
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
        window_index = common_index[start_pos : end_pos + 1]

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

    exposure = pd.DataFrame(0.0, index=asset_window.columns, columns=FACTOR_LABELS)
    r_squared = pd.Series(0.0, index=asset_window.columns, name="R方")
    for asset in asset_window.columns:
        factors = allowed_factors(asset)
        coefs, r2 = standardized_lasso_coefficients(
            asset_window[asset],
            macro_window[factors],
            rng=rng,
            n_bootstrap=n_bootstrap,
            sample_length=sample_length,
            alpha_scale=alpha_scale,
        )
        exposure.loc[asset, factors] = coefs
        r_squared.loc[asset] = r2

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
        "credit_factor_rule": "信用因子仅进入债券类资产（中债国债、中债企业债、中证转债）回归，其余资产信用因子暴露置 0",
        "geo_factor_rule": "地缘因子由布伦特原油和沪金合成，因此不进入这两个资产自身的暴露回归",
        "bond_assets": sorted(BOND_ASSETS),
        "r_squared_definition": (
            "104 周连续区间 Bootstrap 所得标准化 Lasso 中位系数，"
            "在完整滚动窗口上的解释度；这是稳定性调整后的 R²，不等同于完整窗口直接拟合的 OLS R²"
        ),
    }
    return exposure, r_squared, meta


def list_available_weeks(min_window: int = ROLLING_WINDOW_WEEKS) -> list[str]:
    asset_returns = load_asset_weekly_returns()
    macro = load_macro_weekly_mom()
    common_index = asset_returns.index.intersection(macro.dropna(how="any").index).sort_values()
    if len(common_index) < min_window:
        return []
    return common_index[min_window - 1 :].strftime("%Y-%m-%d").tolist()


def build_exposure_payload(
    exposure: pd.DataFrame, r_squared: pd.Series, meta: dict
) -> dict:
    return {
        **meta,
        "assets": exposure.index.tolist(),
        "factors": exposure.columns.tolist(),
        "matrix": {
            asset: {factor: round(float(exposure.loc[asset, factor]), 6) for factor in exposure.columns}
            for asset in exposure.index
        },
        "r_squared": {asset: round(float(r_squared.loc[asset]), 6) for asset in exposure.index},
    }


def save_json(exposure: pd.DataFrame, r_squared: pd.Series, meta: dict) -> None:
    payload = build_exposure_payload(exposure, r_squared, meta)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_exposure(exposure: pd.DataFrame, meta: dict) -> None:
    setup_chinese_font()
    max_abs = max(0.1, float(np.nanmax(np.abs(exposure.values))))
    fig, ax = plt.subplots(figsize=(10.5, 7.5), dpi=150)
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

    for i in range(exposure.shape[0]):
        for j in range(exposure.shape[1]):
            val = exposure.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="#111827")

    ax.tick_params(axis="x", rotation=35)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("标准化暴露系数")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="计算资产对宏观因子的暴露矩阵")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_SAMPLES, help="Bootstrap 次数")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子")
    parser.add_argument("--rolling-window-weeks", type=int, default=ROLLING_WINDOW_WEEKS, help="滚动窗口周数")
    parser.add_argument("--sample-length-weeks", type=int, default=SAMPLE_LENGTH_WEEKS, help="每次重采样的连续周数")
    parser.add_argument(
        "--alpha-scale",
        type=float,
        default=ALPHA_SCALE,
        help="LassoCV alpha 的缩放系数；小于 1 表示惩罚更弱，默认 0.5",
    )
    parser.add_argument("--end-date", type=str, default=None, help="结束周 YYYY-MM-DD，默认取最新可用周")
    args = parser.parse_args()

    exposure, r_squared, meta = compute_latest_exposure(
        n_bootstrap=args.bootstrap,
        rolling_window=args.rolling_window_weeks,
        sample_length=args.sample_length_weeks,
        seed=args.seed,
        alpha_scale=args.alpha_scale,
        end_date=args.end_date,
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


if __name__ == "__main__":
    main()
