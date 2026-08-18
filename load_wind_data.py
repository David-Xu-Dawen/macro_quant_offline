#!/usr/bin/env python3
"""读取 Wind 导出的 Excel（Windows 纯 .xlsx）或 CSV。"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA1 = ROOT / "data1.xlsx"
DEFAULT_NEW = ROOT / "data_new.xlsx"
DEFAULT_XLSX = ROOT / "data.xlsx"
DEFAULT_CSV = ROOT / "data.csv"
if DEFAULT_DATA1.exists():
    DEFAULT_DATA = DEFAULT_DATA1
elif DEFAULT_NEW.exists():
    DEFAULT_DATA = DEFAULT_NEW
elif DEFAULT_XLSX.exists():
    DEFAULT_DATA = DEFAULT_XLSX
else:
    DEFAULT_DATA = DEFAULT_CSV

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
    "全球:地缘政治风险指数": "gpr",
    "全球:地缘政治风险指数(参考十家报纸)": "gpr",
}

META_LABELS = {"指标名称", "频率", "单位", "指标ID", "来源"}
WIND_CORNER_LABELS = {"wind", "万得", "wind资讯", "wind信息"}
# 同比类指标偶尔真的是 0.0；价格、指数、PMI、收益率等出现 0 一律当空值。
ALLOW_TRUE_ZERO = {
    "m2_yoy",
    "sf_yoy",
    "fai_yoy",
    "retail_yoy",
    "export_yoy",
    "import_yoy",
    "cpi_yoy",
    "ppi_yoy",
}
EXCEL_EPOCH = pd.Timestamp("1899-12-30")
EXCEL_SUFFIXES = {".xlsx", ".xlsm"}


def _norm_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\u3000", " ").strip()


def _wind_numeric(values, *, allow_true_zero: bool) -> pd.Series:
    """Wind 常把空格子导出成 0；空值必须回到 NaN，否则月末取值会拿到假 0。"""
    series = pd.to_numeric(values, errors="coerce")
    zeros = series.eq(0)
    if not zeros.any():
        return series
    if not allow_true_zero:
        return series.mask(zeros)
    observed = int(series.notna().sum())
    # 日频表上把空月份填成 0 时，0 会占绝大多数；真的 0.0 很少。
    if observed and zeros.sum() / observed >= 0.10:
        return series.mask(zeros)
    return series


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, header=None, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"无法识别 CSV 编码: {path}")


def _is_wind_corner(value) -> bool:
    text = _norm_text(value).lower()
    return text in WIND_CORNER_LABELS or text.startswith("wind")


def _row_has_mapped_series(raw: pd.DataFrame, row_idx: int) -> bool:
    names = {_norm_text(c) for c in raw.iloc[row_idx].tolist()}
    return any(name in names for name in COLUMN_MAP)


def _find_header_cell(raw: pd.DataFrame) -> tuple[int, int]:
    """Locate the header row: 「指标名称」, or Windows Wind's A1 「Wind」."""
    scan_rows = min(len(raw), 40)
    for row_idx in range(scan_rows):
        for col_idx in range(raw.shape[1]):
            if _norm_text(raw.iat[row_idx, col_idx]) == "指标名称":
                return int(row_idx), int(col_idx)
    for row_idx in range(scan_rows):
        for col_idx in range(min(raw.shape[1], 3)):
            if _is_wind_corner(raw.iat[row_idx, col_idx]) and _row_has_mapped_series(raw, row_idx):
                return int(row_idx), int(col_idx)
    raise ValueError("找不到 Wind 表头（左上角应为「指标名称」或 Windows 导出的「Wind」）")


def _read_excel(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, header=None, sheet_name=None, engine="openpyxl")
    if not sheets:
        raise ValueError(f"空 Excel 文件: {path}")
    for frame in sheets.values():
        if frame.empty:
            continue
        try:
            _find_header_cell(frame)
        except ValueError:
            continue
        return frame
    return next(iter(sheets.values()))


def wind_date_to_datetime(value) -> pd.Timestamp | pd.NaT:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        ts = value.tz_localize(None) if value.tzinfo else value
        return ts.normalize()
    if isinstance(value, datetime):
        return pd.Timestamp(value).tz_localize(None).normalize()
    if isinstance(value, date):
        return pd.Timestamp(value)
    if isinstance(value, str):
        text = _norm_text(value)
        if not text or text in META_LABELS or _is_wind_corner(text) or text.startswith("数据来源"):
            return pd.NaT
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=False)
        if pd.isna(parsed):
            return pd.NaT
        return parsed.tz_localize(None).normalize() if getattr(parsed, "tzinfo", None) else parsed.normalize()
    try:
        number = float(value)
    except (TypeError, ValueError):
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return pd.NaT
        return parsed.tz_localize(None).normalize() if getattr(parsed, "tzinfo", None) else parsed.normalize()
    if 20000 <= number <= 80000:
        return (EXCEL_EPOCH + pd.to_timedelta(int(number), unit="D")).normalize()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return parsed.tz_localize(None).normalize() if getattr(parsed, "tzinfo", None) else parsed.normalize()


def load_wind_data(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else DEFAULT_DATA
    if not path.exists():
        raise FileNotFoundError(f"找不到 Wind 数据文件: {path}")

    suffix = path.suffix.lower()
    if suffix in EXCEL_SUFFIXES:
        raw = _read_excel(path)
    elif suffix == ".xls":
        raise ValueError("请用 Excel 另存为 data.xlsx（不要用旧版 .xls）")
    elif suffix == ".csv":
        raw = _read_csv(path)
    else:
        raise ValueError("支持的输入是根目录 data.xlsx 或 data.csv")

    if raw.empty:
        raise ValueError(f"空文件: {path}")

    header_idx, date_col_idx = _find_header_cell(raw)
    header = [_norm_text(c) for c in raw.iloc[header_idx].tolist()]
    data = raw.iloc[header_idx + 1 :].copy()
    data.columns = header
    date_col = header[date_col_idx]
    date_text = data[date_col].map(_norm_text)
    data = data.loc[
        ~(
            date_text.isin(META_LABELS)
            | date_text.map(_is_wind_corner)
            | date_text.str.startswith("数据来源")
            | date_text.isin({"", "nan", "None", "NaT"})
        )
    ].copy()
    data["date"] = data[date_col].map(wind_date_to_datetime)
    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    out = pd.DataFrame({"date": data["date"].to_numpy()})
    for wind_name, short_name in COLUMN_MAP.items():
        if wind_name in data.columns:
            out[short_name] = _wind_numeric(
                data[wind_name],
                allow_true_zero=short_name in ALLOW_TRUE_ZERO,
            ).to_numpy()
    return out.drop_duplicates("date", keep="last").set_index("date").sort_index()


if __name__ == "__main__":
    frame = load_wind_data()
    print(f"rows={len(frame)} cols={len(frame.columns)}")
    print(f"range={frame.index.min().date()} ~ {frame.index.max().date()}")
    print("columns:", ", ".join(frame.columns))
