#!/usr/bin/env python3
"""读取 Wind 导出的 GB18030/UTF-8 data.csv。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data.csv"

COLUMN_MAP: dict[str, str] = {
    "上证50指数": "上证50",
    "沪深300指数": "沪深300",
    "中证500指数": "中证500",
    "中证1000指数:收盘价": "中证1000",
    "恒生指数": "恒生指数",
    "中证转债:收盘价(前复权)": "中证转债",
    "标普500:收盘价(前复权)": "标普500",
    "中间价:美元兑人民币": "美元兑人民币",
    "美元指数": "美元指数",
    "SHFE黄金:收盘价": "沪金",
    "ICE布油:收盘价": "布伦特原油",
    "SHFE螺纹钢:收盘价": "螺纹钢",
    "房地产(申万):收盘价(前复权)": "申万房地产",
    "中国:M2:同比": "m2_yoy",
    "中国:社会融资规模存量:同比": "sf_yoy",
    "市盈率:申万大盘指数": "申万大盘市盈率",
    "市盈率:申万小盘指数": "申万小盘市盈率",
    "南华沪铜指数": "南华沪铜",
    "期货收盘价(电子盘):LME3个月铜": "CAD",
    "中债-国债总财富(总值)指数:收盘价(前复权)": "中债国债",
    "中债-企业债总财富(总值)指数:收盘价(前复权)": "中债企业债",
    "中债-国债总净价(总值)指数:收盘价(前复权)": "国债净价",
    "中债-国开行债券总财富(3-5年)指数:收盘价(前复权)": "国开财富_3_5",
    "中债-企业债总财富(3-5年)指数:收盘价(前复权)": "企债财富_3_5",
    "中债国开债到期收益率:3年": "国开债_3Y",
    "中债中短期票据到期收益率(AA):3年": "中票AA_3Y",
    "中债国债到期收益率:10年": "国债10Y",
    "中国:平均批发价:猪肉": "猪肉",
    "中国:制造业PMI": "pmi",
    "中国:固定资产投资完成额:累计同比": "fai_yoy",
    "中国:社会消费品零售总额:当月同比(1-2月合并)": "retail_yoy",
    "中国:出口金额:当月同比": "export_yoy",
    "中国:进口金额:当月同比": "import_yoy",
    "中国:CPI:当月同比": "cpi_yoy",
    "中国:PPI:当月同比": "ppi_yoy",
}

META_LABELS = {"指标名称", "频率", "单位", "指标ID", "来源"}


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, header=None, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"无法识别 CSV 编码: {path}")


def load_wind_data(path: str | Path = DEFAULT_DATA) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 Wind 数据文件: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("唯一支持的输入是根目录 data.csv")

    raw = _read_csv(path)
    if raw.empty:
        raise ValueError(f"空文件: {path}")
    header = [str(c).strip() if pd.notna(c) else "" for c in raw.iloc[0].tolist()]
    data = raw.iloc[1:].copy()
    data.columns = header
    first_col = data.iloc[:, 0].astype(str)
    data = data.loc[
        ~(first_col.isin(META_LABELS) | first_col.str.startswith("数据来源"))
    ].copy()
    date_col = header[0]
    data["date"] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    out = pd.DataFrame({"date": data["date"].to_numpy()})
    for wind_name, short_name in COLUMN_MAP.items():
        if wind_name in data.columns:
            out[short_name] = pd.to_numeric(data[wind_name], errors="coerce").to_numpy()
    return out.drop_duplicates("date", keep="last").set_index("date").sort_index()


if __name__ == "__main__":
    frame = load_wind_data()
    print(f"rows={len(frame)} cols={len(frame.columns)}")
    print(f"range={frame.index.min().date()} ~ {frame.index.max().date()}")
