#!/usr/bin/env python3
"""读取 config/panel_config.json。没有文件或缺字段时用下面的默认值。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from paths import CONFIG_PATH, ROOT

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
        "asset_factor_mask": {},
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


def _mask_flag(value) -> bool | None:
    """把 1/0、true/false 转成开关；空值表示这一格没写。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _lookup_asset_value(mapping: dict | None, asset: str):
    """精确匹配资产名；否则用最长的包含/前缀键（例如 中际旭创 → 中际旭创(300308)）。"""
    if not mapping or not asset:
        return None
    if asset in mapping:
        return mapping[asset]
    ranked: list[tuple[int, object]] = []
    for key, value in mapping.items():
        name = str(key).strip()
        if not name:
            continue
        if name == asset or asset.startswith(name) or name.startswith(asset):
            ranked.append((1000 + len(name), value))
        elif len(name) >= 2 and name in asset:
            ranked.append((len(name), value))
    if not ranked:
        return None
    ranked.sort(key=lambda item: -item[0])
    return ranked[0][1]


def _parse_mask_overrides(row, universe: list[str]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    if isinstance(row, dict):
        items = row.items()
    elif isinstance(row, (list, tuple)):
        items = ((universe[i], value) for i, value in enumerate(row) if i < len(universe))
    else:
        return out
    for key, value in items:
        flag = _mask_flag(value)
        if flag is None:
            continue
        name = str(key).strip()
        if name:
            out[name] = flag
    return out


def factors_for_asset(asset: str, cfg: dict | None = None, *, as_bond: bool = False) -> list[str]:
    cfg = cfg or load_panel_config()
    universe = exposure_factor_columns(cfg)
    columns = list(universe)
    exp = cfg["exposure"]
    treat_as_bond = as_bond or is_bond_asset(asset, cfg)
    if exp.get("credit_only_for_bonds", True) and not treat_as_bond:
        columns = [f for f in columns if f != "信用因子"]
    extra = exp.get("asset_exclude_factors") or {}
    drop_raw = _lookup_asset_value(extra, asset)
    drop = {str(f).strip() for f in (drop_raw or []) if str(f).strip()}
    if drop:
        columns = [f for f in columns if f not in drop]
    selected = set(columns)
    mask_row = _lookup_asset_value(exp.get("asset_factor_mask") or {}, asset)
    overrides = _parse_mask_overrides(mask_row, universe)
    unknown = [name for name in overrides if name not in universe]
    if unknown:
        print(f"  {asset} 的 asset_factor_mask 忽略未知/未入列因子: " + "、".join(unknown))
    for name, on in overrides.items():
        if name not in universe:
            continue
        if on:
            selected.add(name)
        else:
            selected.discard(name)
    columns = [f for f in universe if f in selected]
    if not columns:
        raise ValueError(
            f"{asset} 排除完之后没有可回归的因子，请检查 asset_factor_mask / asset_exclude_factors"
        )
    return columns


def build_factor_mask(
    assets: list[str],
    cfg: dict | None = None,
    *,
    as_bond: dict[str, bool] | None = None,
) -> dict[str, dict[str, int]]:
    """实际进入回归的 0/1 矩阵，和暴露表的行列对齐。"""
    cfg = cfg or load_panel_config()
    columns = exposure_factor_columns(cfg)
    bond_flags = as_bond or {}
    mask: dict[str, dict[str, int]] = {}
    for asset in assets:
        used = set(factors_for_asset(asset, cfg, as_bond=bool(bond_flags.get(asset, False))))
        mask[str(asset)] = {factor: (1 if factor in used else 0) for factor in columns}
    return mask


def _lookup_asset_key(mapping: dict | None, asset: str) -> str | None:
    """找出 mapping 里对应这个资产的键，供改名时把旧开关带过去。"""
    hit = _lookup_asset_value(mapping, asset)
    if hit is None or not mapping:
        return None
    if asset in mapping:
        return asset
    ranked: list[tuple[int, str]] = []
    for key in mapping:
        name = str(key).strip()
        if not name:
            continue
        if name == asset or asset.startswith(name) or name.startswith(asset):
            ranked.append((1000 + len(name), name))
        elif len(name) >= 2 and name in asset:
            ranked.append((len(name), name))
    if not ranked:
        return None
    ranked.sort(key=lambda item: -item[0])
    return ranked[0][1]


def default_mask_row(asset: str, cfg: dict | None = None) -> dict[str, int]:
    """新资产的默认 0/1：非债券信用因子为 0，其余为 1。不读已有 mask。"""
    cfg = copy.deepcopy(cfg or load_panel_config())
    cfg.setdefault("exposure", {})["asset_factor_mask"] = {}
    used = set(factors_for_asset(asset, cfg))
    return {factor: (1 if factor in used else 0) for factor in exposure_factor_columns(cfg)}


def _merge_mask_row(asset: str, existing, cfg: dict) -> dict[str, int]:
    defaults = default_mask_row(asset, cfg)
    factors = list(defaults)
    overrides = _parse_mask_overrides(existing, factors)
    return {factor: (1 if overrides.get(factor, bool(defaults[factor])) else 0) for factor in factors}


def _format_asset_factor_mask(mask: dict[str, dict[str, int]], factors: list[str]) -> str:
    if not mask:
        return "{}"
    names = list(mask)
    width = max(len(f'"{name}":') for name in names)
    lines = ["{"]
    for i, asset in enumerate(names):
        row = mask[asset]
        inner = ", ".join(f'"{factor}": {int(row.get(factor, 0))}' for factor in factors)
        pad = " " * (width - len(f'"{asset}":') + 1)
        comma = "," if i < len(names) - 1 else ""
        lines.append(f'      "{asset}":{pad}{{{inner}}}{comma}')
    lines.append("    }")
    return "\n".join(lines)


def _dump_panel_config(raw: dict, path: Path, mask: dict[str, dict[str, int]], factors: list[str]) -> None:
    sentinel = "__ASSET_FACTOR_MASK_BLOCK__"
    exposure = raw.setdefault("exposure", {})
    exposure["asset_factor_mask"] = sentinel
    dumped = json.dumps(raw, ensure_ascii=False, indent=2)
    dumped = dumped.replace(f'"{sentinel}"', _format_asset_factor_mask(mask, factors))
    exposure["asset_factor_mask"] = mask
    path.write_text(dumped + "\n", encoding="utf-8")


def sync_asset_factor_mask(assets: list[str], path: str | Path | None = None) -> dict[str, dict[str, int]]:
    """让 panel_config.json 的 asset_factor_mask 与当前资产名单一致。

    多出来的资产按默认规则补一行；Excel 里已经没有的从矩阵删掉。
    仍在名单里的资产保留原来的 0/1，不覆盖手工开关。
    """
    cfg_path = Path(path) if path else CONFIG_PATH
    names = [str(a).strip() for a in assets if str(a).strip() and str(a).strip().lower() != "date"]
    if not names:
        raise ValueError("资产名单为空，无法同步 asset_factor_mask")

    raw: dict = {}
    if cfg_path.exists():
        loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw = loaded
    cfg = load_panel_config(cfg_path)
    factors = exposure_factor_columns(cfg)
    old_mask = dict((cfg.get("exposure") or {}).get("asset_factor_mask") or {})
    if isinstance(raw.get("exposure"), dict) and isinstance(raw["exposure"].get("asset_factor_mask"), dict):
        old_mask = dict(raw["exposure"]["asset_factor_mask"])

    used_keys: set[str] = set()
    new_mask: dict[str, dict[str, int]] = {}
    added: list[str] = []
    renamed: list[str] = []
    for asset in names:
        remaining = {key: value for key, value in old_mask.items() if key not in used_keys}
        key = _lookup_asset_key(remaining, asset)
        if key is None:
            new_mask[asset] = default_mask_row(asset, cfg)
            added.append(asset)
            continue
        used_keys.add(key)
        new_mask[asset] = _merge_mask_row(asset, old_mask[key], cfg)
        if key != asset:
            renamed.append(f"{key} → {asset}")

    removed = [key for key in old_mask if key not in used_keys]
    already_aligned = list(old_mask) == names and all(
        isinstance(old_mask.get(asset), dict)
        and all(_mask_flag(old_mask[asset].get(factor)) == bool(new_mask[asset][factor]) for factor in factors)
        for asset in names
    )
    if already_aligned:
        print(f"  asset_factor_mask 已与 {len(names)} 个资产对齐，无需改写")
        return new_mask

    raw.setdefault("exposure", {})
    raw["exposure"]["asset_factor_mask"] = new_mask
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    _dump_panel_config(raw, cfg_path, new_mask, factors)

    bits: list[str] = [f"共 {len(names)} 个资产"]
    if added:
        bits.append("新增 " + "、".join(added))
    if removed:
        bits.append("移除 " + "、".join(str(x) for x in removed))
    if renamed:
        bits.append("改名 " + "、".join(renamed))
    try:
        shown = cfg_path.resolve().relative_to(ROOT)
    except ValueError:
        shown = cfg_path
    print(f"  → {shown}（asset_factor_mask {'；'.join(bits)}）")
    return new_mask


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
