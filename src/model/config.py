"""宏观大类资产配置框架 — 全局参数。"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from paths import (  # noqa: E402
    MODEL_MODELS_DIR,
    MODEL_PANEL_DIR,
    MODEL_RAW_DIR,
    MODEL_RUN_DIR,
)

RAW_DIR = MODEL_RAW_DIR
PANEL_DIR = MODEL_PANEL_DIR
MODEL_DIR = MODEL_MODELS_DIR
OUTPUT_DIR = MODEL_RUN_DIR

START_DATE = "2015-01-01"
END_DATE = None

# -------- 预测与标签 --------
# 金融逻辑：中期配置常用 20 个交易日（约 1 个月）作为持有期，
# 比日频噪声更稳，又比季度再平衡更灵敏。
FORWARD_DAYS = 20

# 现金/基准：国债指数近似无风险资产，用于「是否跑赢现金」分类标签
BENCHMARK = "bond_gov"

# 标签模式:
# "ranking"（日期内相对强弱排序，推荐）| "regression"（未来收益）|
# "classification"（是否跑赢基准）
LABEL_MODE = "ranking"

# -------- 特征工程 --------
MOMENTUM_WINDOWS = (5, 10, 20, 60, 120)
VOL_WINDOWS = (10, 20, 60)
MA_WINDOWS = (20, 60, 120)
CORR_THRESHOLD = 0.8  # |ρ| 超过该值视为冗余

# -------- 交易宇宙（用户指定大类）--------
TRADE_UNIVERSE = [
    "sse50",
    "csi300",
    "csi500",
    "csi1000",
    "bond_gov",
    "bond_corp",
    "csi_cb",
    "crude_sc",
    "gold_au",
    "spx",
]

ASSET_CN_NAME = {
    "sse50": "上证50",
    "csi300": "沪深300",
    "csi500": "中证500",
    "csi1000": "中证1000",
    "bond_gov": "中债国债(国债指数)",
    "bond_corp": "中债企业债(企债指数)",
    "csi_cb": "中证转债",
    "crude_sc": "原油",
    "gold_au": "沪金",
    "spx": "标普500",
    "hsi": "恒生指数",
    "cbond_comp": "中债新综合",
}

# 本地模型资产元数据；价格统一由 combined_close.csv 提供。
ASSETS = {
    "sse50": {"cn_name": "上证50", "asset_class": "equity_cn"},
    "csi300": {"cn_name": "沪深300", "asset_class": "equity_cn"},
    "csi500": {"cn_name": "中证500", "asset_class": "equity_cn"},
    "csi1000": {"cn_name": "中证1000", "asset_class": "equity_cn"},
    "hsi": {"cn_name": "恒生指数", "asset_class": "equity_hk"},
    "bond_gov": {"cn_name": "中债国债", "asset_class": "bond"},
    "bond_corp": {"cn_name": "中债企业债", "asset_class": "bond"},
    "cbond_comp": {"cn_name": "中债新综合指数", "asset_class": "bond"},
    "csi_cb": {"cn_name": "中证转债", "asset_class": "convertible"},
    "crude_sc": {"cn_name": "布伦特原油", "asset_class": "commodity"},
    "gold_au": {"cn_name": "沪金", "asset_class": "commodity"},
    "spx": {"cn_name": "标普500", "asset_class": "equity_us"},
}

# -------- LightGBM --------
LGBM_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}
LGBM_RANKER_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "n_estimators": 800,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 80,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.2,
    "reg_lambda": 2.0,
    # 线性 gain 避免默认指数 gain 让第一名标签权重过度膨胀。
    "label_gain": list(range(11)),
    "lambdarank_truncation_level": 3,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}
# 候选模型配置。默认仍保留 GBDT 基线；DART 通过 CLI/A-B 显式启用。
LGBM_BOOSTING_TYPE = "gbdt"
DART_PARAMS = {
    "boosting_type": "dart",
    "drop_rate": 0.1,
    "max_drop": 30,
    "skip_drop": 0.5,
}
RANK_BLEND_WEIGHT = 0.25
# 20 日标签高度重叠；排序模型每 5 个交易日取一个训练截面，降低样本冗余。
RANK_TRAIN_DATE_STRIDE = 5
N_TS_SPLITS = 5
EARLY_STOPPING_ROUNDS = 80

# -------- Black-Litterman --------
# tau: 对均衡先验不确定性的缩放；越小越信任先验市值/风险平价权重
BL_TAU = 0.05
# 观点置信度缩放：把模型得分映射为 Omega 对角线
BL_VIEW_CONFIDENCE = 0.25
BL_CONFIDENCE_MODE = "dynamic"  # scalar | dynamic
BL_CONFIDENCE_MIN = 0.05
BL_CONFIDENCE_MAX = 0.80
BL_CONFIDENCE_LOOKBACK = 252
# 每次再平衡聚合近期观点；20 日半衰期意味着上次观点权重约减半。
VIEW_DECAY_HALF_LIFE_DAYS = 20.0
VIEW_DECAY_LOOKBACK = 3
# 风险厌恶系数（均值方差优化）
BL_RISK_AVERSION = 2.5
# 权重约束
WEIGHT_MIN = 0.0
WEIGHT_MAX = 0.35

# -------- 尾部风险优化 --------
OPTIMIZER_MODE = "auto"  # mean_variance | cvar | auto
CVAR_ALPHA = 0.95
CVAR_RISK_AVERSION = 4.0
CVAR_LOOKBACK = 252

# -------- 激进程度档位（只改组合层，不改 LightGBM 信号）--------
DEFAULT_AGGRESSION = "balanced"
AGGRESSION_PROFILES = {
    "conservative": {
        "label": "稳健",
        "weight_max": 0.30,
        "cvar_risk_aversion": 6.0,
        "bl_risk_aversion": 3.5,
        "optimizer_mode": "cvar",
    },
    "balanced": {
        "label": "均衡",
        "weight_max": 0.35,
        "cvar_risk_aversion": 4.0,
        "bl_risk_aversion": 2.5,
        "optimizer_mode": "auto",
    },
    "aggressive": {
        "label": "进取",
        "weight_max": 0.45,
        "cvar_risk_aversion": 2.0,
        "bl_risk_aversion": 1.8,
        "optimizer_mode": "auto",
    },
}


def resolve_aggression(key: str | None = None) -> tuple[str, dict]:
    """返回 (profile_key, profile_dict)。"""
    k = (key or DEFAULT_AGGRESSION).strip().lower()
    aliases = {
        "稳健": "conservative",
        "均衡": "balanced",
        "进取": "aggressive",
        "conservative": "conservative",
        "balanced": "balanced",
        "aggressive": "aggressive",
    }
    k = aliases.get(k, k)
    if k not in AGGRESSION_PROFILES:
        raise ValueError(
            f"未知激进档位: {key}，可选: {', '.join(AGGRESSION_PROFILES)}"
        )
    return k, AGGRESSION_PROFILES[k]


def aggression_output_dir(key: str | None = None) -> "Path":
    from pathlib import Path

    k, _ = resolve_aggression(key)
    if k == DEFAULT_AGGRESSION:
        return Path(OUTPUT_DIR)
    return Path(OUTPUT_DIR) / f"aggression_{k}"


# -------- 滚动回测 --------
# 最少训练交易日数：太短则协方差/模型不稳定
BT_MIN_TRAIN_DAYS = 504  # ~2 年
# 单边交易成本（bp），换手成本 = turnover * cost_bps / 10000
BT_COST_BPS = 10.0
# True: 每个再平衡日重训；False: 仅用预计算 OOF（更快但不算严格 walk-forward）
BT_RETRAIN_EACH_REBALANCE = True
