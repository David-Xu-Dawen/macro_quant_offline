#!/usr/bin/env python3
"""读取根目录 panel_config.json。没有文件或缺字段时用下面的默认值。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "panel_config.json"

HEATMAP_UNIVERSE = ["增长因子", "通胀因子", "利率因子", "信用因子", "汇率因子", "地缘因子"]
EXPOSURE_UNIVERSE = [
    "增长因子",
    "通胀因子",
    "利率因子",
    "信用因子",
    "汇率因子",
    "地缘因子",
    "流动性因子",
]

DEFAULTS: dict = {
    "heatmap": {
        "lf_start": "2021-01",
        "hf_start": "2021-01-01",
        "factors": list(HEATMAP_UNIVERSE),
        "include_factors": [],
        "exclude_factors": [],
        "rolling_corr_months": 12,
        "rolling_corr_weeks": 52,
        "min_months": 12,
        "min_weeks": 52,
    },
    "alerts": {
        "vol_window_weeks": 13,
        "shock_z": 2.0,
        "shock_lookback_weeks": 52,
        "high_percentile": 85,
        "watch_percentile": 70,
        "forecast_high": 75,
        "forecast_mid": 55,
    },
    "exposure": {
        "rolling_window_weeks": 260,
        "sample_length_weeks": 104,
        "bootstrap_samples": 3000,
        "alpha_scale": 0.5,
        "random_seed": 42,
        "end_date": None,
        "factors": list(EXPOSURE_UNIVERSE),
        "include_factors": [],
        "exclude_factors": [],
        "credit_only_for_bonds": True,
        "bond_assets": ["中债国债", "中债企业债", "中证转债"],
        "bond_name_markers": ["债", "转债", "城投", "政金债", "信用债", "利率债"],
        "asset_exclude_factors": {},
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_panel_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else CONFIG_PATH
    data: dict = {}
    if cfg_path.exists():
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data = raw
        else:
            print(f"  提示: {cfg_path.name} 不是对象，已改用默认参数")
    else:
        print(f"  提示: 找不到 {cfg_path.name}，已改用默认参数")
    return _deep_merge(DEFAULTS, data)


def resolve_factor_list(
    universe: list[str],
    factors: list | None,
    include_factors: list | None,
    exclude_factors: list | None,
) -> list[str]:
    requested = [str(f).strip() for f in (factors or universe) if str(f).strip()]
    unknown = [f for f in requested if f not in universe]
    base = [f for f in requested if f in universe]
    if unknown:
        print("  忽略未知因子: " + "、".join(unknown))
    include = [str(f).strip() for f in (include_factors or []) if str(f).strip()]
    if include:
        allow = set(include)
        extra = [f for f in include if f not in universe]
        if extra:
            print("  include_factors 里有未知因子: " + "、".join(extra))
        base = [f for f in base if f in allow]
        for name in include:
            if name in universe and name not in base:
                base.append(name)
    exclude = {str(f).strip() for f in (exclude_factors or []) if str(f).strip()}
    base = [f for f in base if f not in exclude]
    if not base:
        raise ValueError("因子名单空了。请检查 panel_config.json 里的 factors / include_factors / exclude_factors")
    return base


def heatmap_factors(cfg: dict | None = None) -> list[str]:
    cfg = cfg or load_panel_config()
    hm = cfg["heatmap"]
    return resolve_factor_list(
        HEATMAP_UNIVERSE,
        hm.get("factors"),
        hm.get("include_factors"),
        hm.get("exclude_factors"),
    )


def exposure_factor_columns(cfg: dict | None = None) -> list[str]:
    cfg = cfg or load_panel_config()
    exp = cfg["exposure"]
    return resolve_factor_list(
        EXPOSURE_UNIVERSE,
        exp.get("factors"),
        exp.get("include_factors"),
        exp.get("exclude_factors"),
    )


def is_bond_asset(asset: str, cfg: dict | None = None) -> bool:
    cfg = cfg or load_panel_config()
    exp = cfg["exposure"]
    if asset in set(exp.get("bond_assets") or []):
        return True
    return any(mark in asset for mark in (exp.get("bond_name_markers") or []))


def factors_for_asset(asset: str, cfg: dict | None = None, *, as_bond: bool = False) -> list[str]:
    cfg = cfg or load_panel_config()
    columns = exposure_factor_columns(cfg)
    exp = cfg["exposure"]
    treat_as_bond = as_bond or is_bond_asset(asset, cfg)
    if exp.get("credit_only_for_bonds", True) and not treat_as_bond:
        columns = [f for f in columns if f != "信用因子"]
    extra = exp.get("asset_exclude_factors") or {}
    drop = {str(f).strip() for f in extra.get(asset, []) if str(f).strip()}
    if drop:
        columns = [f for f in columns if f not in drop]
    if not columns:
        raise ValueError(f"{asset} 排除完之后没有可回归的因子，请检查 asset_exclude_factors")
    return columns


def summarize_config(cfg: dict | None = None) -> str:
    cfg = cfg or load_panel_config()
    hm = heatmap_factors(cfg)
    exp = exposure_factor_columns(cfg)
    e = cfg["exposure"]
    a = cfg["alerts"]
    return (
        f"热力图 {cfg['heatmap']['lf_start']} 起 · {len(hm)} 个因子（{'、'.join(hm)}）；"
        f"暴露窗口 {e['rolling_window_weeks']} 周 / 样本 {e['sample_length_weeks']} 周 / "
        f"bootstrap {e['bootstrap_samples']} · {len(exp)} 个因子（{'、'.join(exp)}）；"
        f"警报 {a['vol_window_weeks']} 周波动 / {a['shock_z']}σ"
    )
