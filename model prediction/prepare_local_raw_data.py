#!/usr/bin/env python3
"""Build model raw asset CSVs from the repository's local close-price panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import ASSETS, RAW_DIR, TRADE_UNIVERSE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "factor exposure" / "data" / "combined_close.csv"
RAW_COLUMNS = [
    "asset",
    "cn_name",
    "asset_class",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

# The factor-exposure panel uses display names while the model uses stable IDs.
SOURCE_COLUMNS = {
    "sse50": "上证50",
    "csi300": "沪深300",
    "csi500": "中证500",
    "csi1000": "中证1000",
    "bond_gov": "中债国债",
    "bond_corp": "中债企业债",
    "csi_cb": "中证转债",
    "crude_sc": "布伦特原油",
    "gold_au": "沪金",
    "spx": "标普500",
}


def prepare_local_raw_data(
    source: Path = DEFAULT_SOURCE,
    raw_dir: Path = RAW_DIR,
) -> list[dict[str, object]]:
    """Convert the local wide close panel to one legacy-compatible CSV per asset."""
    if not source.exists():
        raise FileNotFoundError(f"本地资产价格面板不存在: {source}")

    panel = pd.read_csv(source)
    if "date" not in panel.columns:
        raise ValueError(f"{source} 缺少 date 列")

    missing_mappings = [asset for asset in TRADE_UNIVERSE if asset not in SOURCE_COLUMNS]
    if missing_mappings:
        raise ValueError(f"本地价格列映射缺失: {', '.join(missing_mappings)}")

    missing_columns = [
        SOURCE_COLUMNS[asset]
        for asset in TRADE_UNIVERSE
        if SOURCE_COLUMNS[asset] not in panel.columns
    ]
    if missing_columns:
        raise ValueError(f"{source} 缺少模型资产列: {', '.join(missing_columns)}")

    dates = pd.to_datetime(panel["date"], errors="coerce")
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []

    for asset in TRADE_UNIVERSE:
        source_column = SOURCE_COLUMNS[asset]
        close = pd.to_numeric(panel[source_column], errors="coerce")
        meta = ASSETS[asset]
        out = pd.DataFrame({"date": dates, "close": close}).dropna(
            subset=["date", "close"]
        )
        out = out.sort_values("date").drop_duplicates("date", keep="last")
        if out.empty:
            raise ValueError(f"{source_column} 没有可用的本地价格")

        for column in ("open", "high", "low"):
            out[column] = out["close"]
        out["volume"] = pd.NA
        out.insert(0, "asset_class", meta["asset_class"])
        out.insert(0, "cn_name", meta["cn_name"])
        out.insert(0, "asset", asset)
        out = out[RAW_COLUMNS]
        out.to_csv(raw_dir / f"{asset}.csv", index=False, date_format="%Y-%m-%d")

        summary.append(
            {
                "asset": asset,
                "source_column": source_column,
                "rows": len(out),
                "start": str(out["date"].min().date()),
                "end": str(out["date"].max().date()),
            }
        )

    pd.DataFrame(summary).to_csv(raw_dir / "_local_prepare_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 factor exposure 本地价格面板准备模型 raw 数据"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    summary = prepare_local_raw_data(args.source, args.raw_dir)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\n已离线生成 {len(summary)} 个模型资产文件: {args.raw_dir}")


if __name__ == "__main__":
    main()
