"""宏观大类资产配置 — 特征工程。

金融逻辑概要
------------
大类配置的 Alpha 很少来自单资产价格本身，而更多来自：
1) 资产自身的动量/趋势/波动状态；
2) 跨资产相对价值与宏观状态（信用、通胀、风险偏好）。
因此特征分两层：asset-specific + shared macro regime。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ASSETS,
    ASSET_CN_NAME,
    BENCHMARK,
    CORR_THRESHOLD,
    FORWARD_DAYS,
    MA_WINDOWS,
    MOMENTUM_WINDOWS,
    RAW_DIR,
    START_DATE,
    TRADE_UNIVERSE,
    VOL_WINDOWS,
)

CATEGORICAL_FEATURES = ["asset_id", "asset_class_id"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_MACRO_FILE = PROJECT_ROOT / "macro_hf_factor_weekly.csv"
LIQUIDITY_FILE = PROJECT_ROOT / "mobility" / "hf_mobility_factor_synthetic.csv"


def load_price_panel(
    universe: list[str] | None = None,
    raw_dir: Path = RAW_DIR,
    start: str = START_DATE,
) -> pd.DataFrame:
    """读取 raw CSV，拼成宽表 close（index=date, columns=asset）。

    日历对齐逻辑：
    - 以 A 股+国债交易日为锚（中国市场可交易日）；
    - 外盘/商品用 ffill 对齐，避免因时区休市造成错位样本。
    """
    universe = universe or TRADE_UNIVERSE
    frames = []
    for asset in universe:
        path = raw_dir / f"{asset}.csv"
        if not path.exists():
            raise FileNotFoundError(f"缺少本地模型数据: {path}，请先运行 prepare_local_raw_data.py")
        df = pd.read_csv(path, parse_dates=["date"])[["date", "close"]].assign(asset=asset)
        frames.append(df)

    long_df = pd.concat(frames, ignore_index=True)
    close = long_df.pivot_table(index="date", columns="asset", values="close", aggfunc="last").sort_index()

    # 锚定中国市场交易日
    anchor_cols = [c for c in ["csi300", "bond_gov", "sse50"] if c in close.columns]
    if anchor_cols:
        mask = close[anchor_cols].notna().all(axis=1)
        close = close.loc[mask]
    close = close.ffill()
    close = close.loc[close.index >= pd.Timestamp(start), universe]
    return close


def _asset_style_features(close: pd.Series, prefix: str = "") -> pd.DataFrame:
    """单资产动量 / 趋势 / 波动特征。

    - ret_N: 中期动量（趋势延续 vs 反转的核心输入）
    - ma_dev: 价格相对均线偏离，度量过热/超卖
    - vol: 实现波动，用于风险状态与后续组合优化
    """
    r = close.pct_change()
    out = pd.DataFrame(index=close.index)

    for w in MOMENTUM_WINDOWS:
        out[f"{prefix}ret_{w}d"] = close.pct_change(w)

    for w in MA_WINDOWS:
        ma = close.rolling(w).mean()
        out[f"{prefix}ma_dev_{w}d"] = close / ma - 1.0

    for w in VOL_WINDOWS:
        out[f"{prefix}vol_{w}d"] = r.rolling(w).std() * np.sqrt(252)

    out[f"{prefix}mom_20_60"] = out.get(f"{prefix}ret_20d", close.pct_change(20)) - out.get(
        f"{prefix}ret_60d", close.pct_change(60)
    )
    out[f"{prefix}dd_60d"] = close / close.rolling(60).max() - 1.0
    # 风险调整动量比裸收益更适合跨资产比较：债券和原油的波动量级差异很大。
    annual_vol = r.rolling(20).std() * np.sqrt(252)
    out[f"{prefix}risk_adj_mom_20d"] = close.pct_change(20) / annual_vol.replace(0, np.nan)
    out[f"{prefix}up_ratio_20d"] = (r > 0).rolling(20).mean()
    downside = r.where(r < 0, 0.0)
    out[f"{prefix}downside_vol_20d"] = downside.rolling(20).std() * np.sqrt(252)
    out[f"{prefix}vol_ratio_10_60"] = (
        r.rolling(10).std() / r.rolling(60).std().replace(0, np.nan)
    )
    out[f"{prefix}ema_trend_20_60"] = (
        close.ewm(span=20, adjust=False).mean()
        / close.ewm(span=60, adjust=False).mean()
        - 1.0
    )
    out[f"{prefix}return_skew_60d"] = r.rolling(60).skew()
    out[f"{prefix}return_autocorr_20d"] = r.rolling(20).corr(r.shift(1))
    return out


def load_external_macro_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """读取项目已生成的宏观状态，并严格按可得时点向后填充。

    周频市场型因子只 ffill；流动性底层为月度发布数据，额外滞后约一个月，
    避免把尚未公布的数据放进当期特征。
    """
    out = pd.DataFrame(index=index)
    if HF_MACRO_FILE.exists():
        hf = pd.read_csv(HF_MACRO_FILE, parse_dates=["week"]).set_index("week").sort_index()
        rename = {
            "增长因子": "ext_growth",
            "通胀因子": "ext_inflation",
            "利率因子": "ext_rate",
            "信用因子": "ext_credit",
            "汇率因子": "ext_fx",
            "地缘因子": "ext_geo",
        }
        hf = hf.rename(columns=rename)[list(rename.values())]
        aligned = hf.reindex(index.union(hf.index)).sort_index().ffill().reindex(index)
        out = out.join(aligned)

    if LIQUIDITY_FILE.exists():
        liq = (
            pd.read_csv(LIQUIDITY_FILE, parse_dates=["date"])
            .set_index("date")["hf_mobility_factor"]
            .astype(float)
            .sort_index()
        )
        aligned = liq.reindex(index.union(liq.index)).sort_index().ffill().reindex(index)
        out["ext_liquidity"] = aligned.shift(21)

    base_cols = list(out.columns)
    for col in base_cols:
        out[f"{col}_chg5"] = out[col].diff(5)
        out[f"{col}_chg20"] = out[col].diff(20)
        out[f"{col}_chg60"] = out[col].diff(60)
        out[f"{col}_vol20"] = out[col].diff().rolling(20).std()
        rolling_mean = out[col].rolling(252, min_periods=126).mean()
        rolling_std = out[col].rolling(252, min_periods=126).std().replace(0, np.nan)
        out[f"{col}_z252"] = (out[col] - rolling_mean) / rolling_std
    return out


def build_macro_features(close: pd.DataFrame) -> pd.DataFrame:
    """构建跨资产宏观状态因子（所有资产共享同一套 regime 特征）。

    1) 信用利差代理：企业债相对国债的累计超额/收益差
       - 利差走阔 → 信用风险偏好下降，利空权益/转债，利多利率债
    2) 通胀预期代理：原油相对黄金的强弱
       - 油强金弱 → 再通胀交易；油弱金强 → 避险/滞胀担忧
    3) 外盘情绪：标普500 动量与波动
       - 全球 risk-on/off 的领先/同步信号，影响 A 股风险溢价
    4) 股债相对：沪深300 对国债的 20 日超额
       - 直接刻画国内 risk appetite
    """
    macro = pd.DataFrame(index=close.index)

    if {"bond_corp", "bond_gov"}.issubset(close.columns):
        # 用相对收益差近似信用条件变化（指数层面，非收益率曲线 bp）
        corp_ret = close["bond_corp"].pct_change(20)
        gov_ret = close["bond_gov"].pct_change(20)
        macro["credit_spread_proxy_20d"] = corp_ret - gov_ret
        macro["credit_spread_proxy_60d"] = close["bond_corp"].pct_change(60) - close["bond_gov"].pct_change(60)

    if {"crude_sc", "gold_au"}.issubset(close.columns):
        oil_gold = close["crude_sc"] / close["gold_au"]
        macro["inflation_proxy_oil_gold"] = oil_gold.pct_change(20)
        macro["inflation_proxy_level_z"] = (oil_gold - oil_gold.rolling(60).mean()) / oil_gold.rolling(60).std()
        macro["gold_ret_20d"] = close["gold_au"].pct_change(20)
        macro["oil_ret_20d"] = close["crude_sc"].pct_change(20)

    if "spx" in close.columns:
        spx = close["spx"]
        macro["spx_ret_20d"] = spx.pct_change(20)
        macro["spx_ret_60d"] = spx.pct_change(60)
        macro["spx_vol_20d"] = spx.pct_change().rolling(20).std() * np.sqrt(252)
        macro["spx_ma_dev_20d"] = spx / spx.rolling(20).mean() - 1.0

    if {"csi300", "bond_gov"}.issubset(close.columns):
        macro["equity_bond_spread_20d"] = close["csi300"].pct_change(20) - close["bond_gov"].pct_change(20)
        macro["equity_bond_spread_60d"] = close["csi300"].pct_change(60) - close["bond_gov"].pct_change(60)

    if "csi_cb" in close.columns and "csi300" in close.columns:
        # 转债相对权益：偏债/偏股属性切换
        macro["cb_equity_spread_20d"] = close["csi_cb"].pct_change(20) - close["csi300"].pct_change(20)

    macro = macro.join(load_external_macro_features(close.index), how="left")
    return macro


def build_panel_features(close: pd.DataFrame, universe: list[str] | None = None) -> pd.DataFrame:
    """生成 (date, asset) 面板特征：自身技术面 + 宏观 regime + 相对强弱。"""
    universe = universe or [c for c in TRADE_UNIVERSE if c in close.columns]
    macro = build_macro_features(close)
    # 等权市场动量，用于相对强弱（横截面排序的关键输入）
    mkt_ret_20 = close[universe].pct_change(20).mean(axis=1)
    mkt_ret_60 = close[universe].pct_change(60).mean(axis=1)

    rows = []
    asset_to_id = {asset: idx for idx, asset in enumerate(universe)}
    classes = sorted({ASSETS.get(asset, {}).get("asset_class", "other") for asset in universe})
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    for asset in universe:
        style = _asset_style_features(close[asset], prefix="")
        feat = style.join(macro, how="left")
        feat["asset"] = asset
        feat["asset_id"] = asset_to_id[asset]
        asset_class = ASSETS.get(asset, {}).get("asset_class", "other")
        feat["asset_class_id"] = class_to_id[asset_class]
        feat["cn_name"] = ASSET_CN_NAME.get(asset, asset)
        feat["close"] = close[asset]
        feat["rel_strength_20d"] = style["ret_20d"] - mkt_ret_20
        feat["rel_strength_60d"] = style["ret_60d"] - mkt_ret_60
        if BENCHMARK in close.columns:
            feat["excess_vs_cash_20d"] = style["ret_20d"] - close[BENCHMARK].pct_change(20)
        # Regime × asset-class interaction：同一个 risk-on 信号对股票、债券方向不同。
        is_equity = float(asset_class.startswith("equity"))
        is_bond = float(asset_class == "bond")
        is_commodity = float(asset_class == "commodity")
        is_convertible = float(asset_class == "convertible")
        if "equity_bond_spread_20d" in feat:
            feat["regime_equity_exposure"] = feat["equity_bond_spread_20d"] * (is_equity - is_bond)
        if "inflation_proxy_oil_gold" in feat:
            feat["regime_inflation_exposure"] = feat["inflation_proxy_oil_gold"] * is_commodity
        if "spx_ret_20d" in feat:
            feat["regime_global_risk_exposure"] = feat["spx_ret_20d"] * is_equity
        if "ext_growth_chg20" in feat:
            feat["regime_growth_exposure"] = feat["ext_growth_chg20"] * (is_equity - is_bond)
        if "ext_rate_chg20" in feat:
            feat["regime_rate_exposure"] = feat["ext_rate_chg20"] * is_bond
        if "ext_credit_chg20" in feat:
            feat["regime_credit_exposure"] = feat["ext_credit_chg20"] * is_bond
        if "ext_liquidity_chg20" in feat:
            feat["regime_liquidity_exposure"] = feat["ext_liquidity_chg20"] * is_equity
        if "ext_geo_chg20" in feat:
            feat["regime_geo_exposure"] = feat["ext_geo_chg20"] * (is_commodity + is_equity)

        # 中期变化和历史分位状态 × 资产类别。共享宏观列在同一交易日对所有资产
        # 完全相同，显式交互后才具有稳定的横截面辨识度。
        macro_exposure = {
            "growth": is_equity - is_bond,
            "inflation": is_commodity - is_bond,
            "rate": is_bond - is_equity,
            "credit": is_equity + is_convertible - is_bond,
            "liquidity": is_equity + is_convertible - is_bond,
            "geo": is_commodity - is_equity,
        }
        for macro_name, exposure in macro_exposure.items():
            for suffix in ("chg60", "z252"):
                source = f"ext_{macro_name}_{suffix}"
                if source in feat:
                    feat[f"regime_{macro_name}_{suffix}_exposure"] = feat[source] * exposure

        # 资产对主要宏观锚的 beta/相关性会随 regime 变化，也是区分横截面的信息。
        asset_ret = close[asset].pct_change()
        for anchor in ["csi300", "bond_gov", "gold_au", "crude_sc", "spx"]:
            if anchor not in close.columns or anchor == asset:
                continue
            anchor_ret = close[anchor].pct_change()
            feat[f"corr60_{anchor}"] = asset_ret.rolling(60).corr(anchor_ret)
            anchor_var = anchor_ret.rolling(60).var().replace(0, np.nan)
            feat[f"beta60_{anchor}"] = asset_ret.rolling(60).cov(anchor_ret) / anchor_var
        rows.append(feat.reset_index().rename(columns={"index": "date"}))

    panel = pd.concat(rows, ignore_index=True)
    if "date" not in panel.columns:
        panel = panel.rename(columns={panel.columns[0]: "date"})
    panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)

    # 日期内横截面标准化：让模型直接看到“该资产相对同日其它资产有多强”。
    cross_sectional = [
        "ret_5d",
        "ret_20d",
        "ret_60d",
        "ret_120d",
        "vol_20d",
        "vol_ratio_10_60",
        "risk_adj_mom_20d",
        "ma_dev_20d",
        "ma_dev_120d",
        "dd_60d",
        "ema_trend_20_60",
    ]
    cross_sectional_new: dict[str, pd.Series] = {}
    for col in cross_sectional:
        if col not in panel.columns:
            continue
        grouped = panel.groupby("date")[col]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        cross_sectional_new[f"{col}_cs_z"] = (panel[col] - mean) / std
        cross_sectional_new[f"{col}_cs_rank"] = grouped.rank(pct=True, method="average")
    if cross_sectional_new:
        panel = pd.concat([panel, pd.DataFrame(cross_sectional_new, index=panel.index)], axis=1)
    return panel


def drop_highly_correlated_features(
    X: pd.DataFrame,
    threshold: float = CORR_THRESHOLD,
) -> tuple[list[str], pd.DataFrame]:
    """剔除高度相关冗余特征，缓解多重共线性与树模型的分裂稀释。

    算法（贪心）：
    1. 计算 |corr| 矩阵；
    2. 按「与其它特征平均绝对相关」从高到低检查；
    3. 若某特征与任一已保留特征 |ρ|>threshold，则丢弃。

    注意：必须只在训练集上拟合该选择，再应用到验证/测试，避免信息泄漏。
    """
    corr = X.corr().abs()
    # 平均相关度高的特征更可能是“信息重复中心”，优先审查
    avg_corr = corr.mean().sort_values(ascending=False)
    kept: list[str] = []
    dropped: list[str] = []

    for col in avg_corr.index:
        if not kept:
            kept.append(col)
            continue
        if corr.loc[col, kept].max() >= threshold:
            dropped.append(col)
        else:
            kept.append(col)

    return kept, corr.loc[kept, kept]


def add_labels(
    panel: pd.DataFrame,
    close: pd.DataFrame,
    forward_days: int = FORWARD_DAYS,
    mode: str = "regression",
    target: str = "return",
) -> pd.DataFrame:
    """标签定义。

    target:
        return = 未来 N 日收益；
        risk_adjusted = 未来超额收益 / 同期实现波动，便于跨资产比较。
    ranking:
        y = 同一日期内目标值的整数相关性等级，供 LambdaRank 使用。
    regression:
        y = 未来 N 日简单收益率。模型学习相对强弱的连续得分。
    classification:
        y = 1{未来 N 日收益 > 国债未来 N 日收益}，即是否跑赢现金/利率基准。
    """
    if target not in {"return", "risk_adjusted"}:
        raise ValueError(f"不支持 label target={target}")

    out = panel.copy()
    fwd_ret = close.shift(-forward_days) / close - 1.0
    fwd_long = fwd_ret.stack().rename("y_ret").reset_index()
    fwd_long.columns = ["date", "asset", "y_ret"]
    out = out.merge(fwd_long, on=["date", "asset"], how="left")

    if BENCHMARK in fwd_ret.columns:
        bench = fwd_ret[BENCHMARK].rename("bench_fwd")
        out = out.merge(bench, left_on="date", right_index=True, how="left")
        out["y_excess"] = out["y_ret"] - out["bench_fwd"]
        out = out.drop(columns=["bench_fwd"])
    else:
        out["y_excess"] = out["y_ret"]

    # t 日之后 N 个交易日的实现波动。它只参与标签构造，不进入特征，
    # 因而不会把未来信息泄漏给模型。转成持有期波动后构造类 Sharpe 目标。
    daily_ret = close.pct_change()
    fwd_vol = daily_ret.shift(-1).iloc[::-1].rolling(forward_days).std().iloc[::-1]
    holding_vol = fwd_vol * np.sqrt(forward_days)
    fwd_vol_long = holding_vol.stack().rename("y_holding_vol").reset_index()
    fwd_vol_long.columns = ["date", "asset", "y_holding_vol"]
    out = out.merge(fwd_vol_long, on=["date", "asset"], how="left")
    out["y_risk_adjusted"] = (
        out["y_excess"] / out["y_holding_vol"].replace(0, np.nan)
    ).clip(-10.0, 10.0)
    out["y_target"] = out["y_ret"] if target == "return" else out["y_risk_adjusted"]

    out["y_cls"] = (out["y_excess"] > 0).astype(int)
    # LightGBM LambdaRank 要求非负整数 relevance；最强资产得到最大等级。
    out["y_rank_pct"] = out.groupby("date")["y_target"].rank(pct=True, method="average")
    out["y_rank_relevance"] = (
        out.groupby("date")["y_target"].rank(ascending=True, method="first") - 1
    ).astype("Int64")
    out["forward_days"] = forward_days
    out["label_mode"] = mode
    out["label_target"] = target
    if mode == "ranking":
        out["y"] = out["y_rank_relevance"]
    elif mode == "regression":
        out["y"] = out["y_target"]
    else:
        out["y"] = out["y_cls"]
    return out.dropna(subset=["y"]).reset_index(drop=True)


def feature_columns(panel: pd.DataFrame) -> list[str]:
    meta = {
        "date",
        "asset",
        "cn_name",
        "close",
        "y",
        "y_ret",
        "y_excess",
        "y_holding_vol",
        "y_risk_adjusted",
        "y_target",
        "y_cls",
        "y_rank_pct",
        "y_rank_relevance",
        "forward_days",
        "label_mode",
        "label_target",
    }
    cols = [c for c in panel.columns if c not in meta]
    return [c for c in cols if pd.api.types.is_numeric_dtype(panel[c])]


def categorical_feature_columns(panel: pd.DataFrame) -> list[str]:
    """返回模型类别变量；这些列不参与相关系数剔除。"""
    return [c for c in CATEGORICAL_FEATURES if c in panel.columns]
