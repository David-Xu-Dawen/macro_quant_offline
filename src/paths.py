"""仓库目录约定。所有脚本从这里取路径，不要再写死文件夹名。"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
WEB_DIR = ROOT / "web"
SCRIPTS_DIR = ROOT / "scripts"

FACTORS_DIR = OUTPUT_DIR / "factors"
GROWTH_DIR = FACTORS_DIR / "growth"
INFLATION_DIR = FACTORS_DIR / "inflation"
CREDIT_DIR = FACTORS_DIR / "credit"
RATE_DIR = FACTORS_DIR / "interest_rate"
EXCHANGE_DIR = FACTORS_DIR / "exchange"
POLITICS_DIR = FACTORS_DIR / "politics"
MOBILITY_DIR = FACTORS_DIR / "mobility"

CORR_DIR = OUTPUT_DIR / "corr"
HF_DIR = OUTPUT_DIR / "hf"
EXPOSURE_DIR = OUTPUT_DIR / "exposure"

MODEL_SRC = SRC_DIR / "model"
MODEL_OUT = OUTPUT_DIR / "model"
MODEL_RAW_DIR = MODEL_OUT / "data" / "raw"
MODEL_PANEL_DIR = MODEL_OUT / "data" / "panel"
MODEL_MODELS_DIR = MODEL_OUT / "models"
MODEL_RUN_DIR = MODEL_OUT / "output"

CONFIG_PATH = CONFIG_DIR / "panel_config.json"
COMBINED_CLOSE = EXPOSURE_DIR / "combined_close.csv"

WIND_CANDIDATES = (
    DATA_DIR / "data1.xlsx",
    DATA_DIR / "data_new.xlsx",
    DATA_DIR / "data.xlsx",
    DATA_DIR / "data.csv",
    ROOT / "data1.xlsx",
    ROOT / "data_new.xlsx",
    ROOT / "data.xlsx",
    ROOT / "data.csv",
)


def default_wind_data() -> Path:
    for path in WIND_CANDIDATES:
        if path.exists():
            return path
    return DATA_DIR / "data1.xlsx"


def additional_asset_path() -> Path | None:
    for folder in (DATA_DIR, ROOT):
        for name in ("additional_asset.xlsx", "additional_asset.xlsm", "additional_asset.csv"):
            path = folder / name
            if path.exists():
                return path
    return None


def ensure_output_dirs() -> None:
    for folder in (
        CONFIG_DIR,
        DATA_DIR,
        GROWTH_DIR,
        INFLATION_DIR,
        CREDIT_DIR,
        RATE_DIR,
        EXCHANGE_DIR,
        POLITICS_DIR,
        MOBILITY_DIR,
        CORR_DIR,
        HF_DIR,
        EXPOSURE_DIR,
        MODEL_RAW_DIR,
        MODEL_PANEL_DIR,
        MODEL_MODELS_DIR,
        MODEL_RUN_DIR,
        WEB_DIR,
    ):
        folder.mkdir(parents=True, exist_ok=True)
