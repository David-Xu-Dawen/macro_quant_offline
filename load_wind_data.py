#!/usr/bin/env python3
"""读取 Wind 导出的 Excel（Windows 纯 .xlsx）或 CSV。"""

from __future__ import annotations

import re
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
    "中债-国开债到期收益率:3年": "国开债_3Y",
    "中债国开债到期收益率:3年:日": "国开债_3Y",
    "中债中短期票据到期收益率(AA):3年": "中票AA_3Y",
    "中债-中短期票据到期收益率(AA):3年": "中票AA_3Y",
    "中债国债到期收益率:10年": "国债10Y",
    "中债-国债到期收益率:10年": "国债10Y",
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
CORE_EXPOSURE_ASSETS = {
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
}
# additional_asset 里这些是宏观输入，不是拿来画暴露的资产。
SKIP_AS_EXTRA_ASSET = CORE_EXPOSURE_ASSETS | {
    "pmi",
    "fai_yoy",
    "retail_yoy",
    "export_yoy",
    "import_yoy",
    "cpi_yoy",
    "ppi_yoy",
    "m2_yoy",
    "sf_yoy",
    "gpr",
    "国债10Y",
    "国开债_3Y",
    "中票AA_3Y",
    "国债净价",
    "国开财富_3_5",
    "企债财富_3_5",
    "猪肉",
    "螺纹钢",
    "CAD",
    "申万大盘市盈率",
    "申万小盘市盈率",
}


def additional_asset_path() -> Path | None:
    for name in ("additional_asset.xlsx", "additional_asset.xlsm", "additional_asset.csv"):
        path = ROOT / name
        if path.exists():
            return path
    return None


def pretty_asset_name(header: str) -> str:
    text = _normalize_header(header)
    for suffix in (
        ":收盘价(前复权)",
        ":收盘价(后复权)",
        ":收盘价(不复权)",
        ":收盘价",
        "(前复权)",
        "(后复权)",
        "(不复权)",
    ):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text or header


EXCEL_EPOCH = pd.Timestamp("1899-12-30")
EXCEL_SUFFIXES = {".xlsx", ".xlsm"}
FREQ_SUFFIX = re.compile(r"(:(日|周|月)|\[(日|周|月)\])$")
# 表头不完全一致时，用关键词兜底（全部命中且只匹配到一列才采用）。
KEYWORD_RULES: list[tuple[tuple[str, ...], str]] = [
    (("全球", "地缘政治风险"), "gpr"),
    (("LME", "铜"), "CAD"),
    (("南华沪铜",), "南华沪铜"),
    (("SHFE黄金",), "沪金"),
    (("沪金",), "沪金"),
    (("ICE布油",), "布伦特原油"),
    (("布伦特",), "布伦特原油"),
    (("SHFE螺纹钢",), "螺纹钢"),
    (("申万", "房地产"), "申万房地产"),
    (("国开债", "到期收益率", "3年"), "国开债_3Y"),
    (("国开行", "到期收益率", "3年"), "国开债_3Y"),
    (("中短期票据", "到期收益率", "3年"), "中票AA_3Y"),
    (("国债到期收益率", "10年"), "国债10Y"),
    (("国开行债券总财富", "3-5"), "国开财富_3_5"),
    (("企业债总财富", "3-5"), "企债财富_3_5"),
    (("国债总净价",), "国债净价"),
    (("国债总财富", "总值"), "中债国债"),
    (("企业债总财富", "总值"), "中债企业债"),
]


def expected_wind_name(short_name: str) -> str:
    for wind_name, mapped in COLUMN_MAP.items():
        if mapped == short_name:
            return wind_name
    return short_name


def _norm_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\u3000", " ").strip()


def _normalize_header(name: str) -> str:
    text = _norm_text(name).replace("：", ":").replace("（", "(").replace("）", ")")
    text = re.sub(r"\s+", "", text)
    return FREQ_SUFFIX.sub("", text)


def _exact_short_name(header: str) -> str | None:
    if header in COLUMN_MAP:
        return COLUMN_MAP[header]
    key = _normalize_header(header)
    for wind_name, short_name in COLUMN_MAP.items():
        if _normalize_header(wind_name) == key:
            return short_name
    return None


def _keyword_short_name(header: str, taken: set[str]) -> str | None:
    text = _normalize_header(header)
    hits: list[str] = []
    for keywords, short_name in KEYWORD_RULES:
        if short_name in taken:
            continue
        if all(k in text for k in keywords):
            hits.append(short_name)
    hits = list(dict.fromkeys(hits))
    if len(hits) == 1:
        return hits[0]
    return None


def map_wind_headers(headers: list[str]) -> tuple[dict[str, str], list[str]]:
    """返回 {原始表头: 短名}，以及未识别表头。"""
    mapping: dict[str, str] = {}
    taken: set[str] = set()
    unmatched: list[str] = []
    skip = META_LABELS | {"", "nan", "None", "date", "日期"}
    for header in headers:
        if not header or header in skip or _is_wind_corner(header):
            continue
        short = _exact_short_name(header)
        if short:
            mapping[header] = short
            taken.add(short)
    for header in headers:
        if not header or header in skip or header in mapping or _is_wind_corner(header):
            continue
        short = _keyword_short_name(header, taken)
        if short:
            mapping[header] = short
            taken.add(short)
        else:
            unmatched.append(header)
    return mapping, unmatched


def _wind_numeric(values, *, allow_true_zero: bool) -> pd.Series:
    """Wind 常把空格子导出成 0 或 --；空值必须回到 NaN，否则会当成真数。"""
    raw = pd.Series(values)
    text = raw.map(_norm_text).str.lower()
    raw = raw.mask(
        text.isin({"", "-", "--", "—", "n.a.", "na", "n/a", "null", "none", "无", "#n/a", "#na", "nan"})
    )
    series = pd.to_numeric(raw, errors="coerce")
    if not allow_true_zero:
        return series.mask(series <= 0)
    zeros = series.eq(0)
    if not zeros.any():
        return series
    observed = int(series.notna().sum())
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
    names = [_norm_text(c) for c in raw.iloc[row_idx].tolist()]
    mapping, _ = map_wind_headers(names)
    return bool(mapping)


def _drop_leading_empty_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Wind 表经常第一行是空的，真正表头从第二行「指标名称」开始。"""
    start = 0
    for i in range(len(raw)):
        values = raw.iloc[i].tolist()
        if any(_norm_text(v) for v in values):
            start = i
            break
    return raw.iloc[start:].reset_index(drop=True) if start else raw


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


def _prepared_wind_frame(path: Path) -> pd.DataFrame:
    """读 Wind 表到「date + 原始表头」宽表，尚未映射短名。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 Wind 数据文件: {path}")

    suffix = path.suffix.lower()
    if suffix in EXCEL_SUFFIXES:
        raw = _read_excel(path)
    elif suffix == ".xls":
        raise ValueError("请用 Excel 另存为 .xlsx（不要用旧版 .xls）")
    elif suffix == ".csv":
        raw = _read_csv(path)
    else:
        raise ValueError("支持的输入是 .xlsx 或 .csv")

    if raw.empty:
        raise ValueError(f"空文件: {path}")

    raw = _drop_leading_empty_rows(raw)
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
    if data.columns.duplicated().any():
        data = data.loc[:, ~data.columns.duplicated()].copy()
    return data


def load_wind_data(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else DEFAULT_DATA
    data = _prepared_wind_frame(path)
    mapping, unmatched = map_wind_headers([_norm_text(c) for c in data.columns])
    out = pd.DataFrame({"date": data["date"].to_numpy()})
    for wind_name, short_name in mapping.items():
        if wind_name not in data.columns:
            continue
        out[short_name] = _wind_numeric(
            data[wind_name],
            allow_true_zero=short_name in ALLOW_TRUE_ZERO,
        ).to_numpy()
    if unmatched:
        print("  未识别的 Wind 列: " + "、".join(unmatched))
    return out.drop_duplicates("date", keep="last").set_index("date").sort_index()


def load_additional_assets(path: str | Path | None = None) -> pd.DataFrame:
    """读取根目录 additional_asset.xlsx 里的额外资产价格。

    和现在这份表一样即可：第一行可以空着，第二行左边是「指标名称」，
    右边是各资产（Wind 常写成「中际旭创(300308):收盘价(后复权)」），
    第三行起是日期和价格。导出时用短名，例如 中际旭创(300308)。
    """
    path = Path(path) if path else additional_asset_path()
    if path is None:
        return pd.DataFrame()

    data = _prepared_wind_frame(path)
    mapping, unmatched = map_wind_headers([_norm_text(c) for c in data.columns])
    out = pd.DataFrame({"date": data["date"].to_numpy()})
    added: list[str] = []
    skipped: list[str] = []

    def _put(name: str, values) -> None:
        col = name
        i = 2
        while col in out.columns:
            col = f"{name}_{i}"
            i += 1
        out[col] = _wind_numeric(values, allow_true_zero=False).to_numpy()
        added.append(col)

    for wind_name, short_name in mapping.items():
        if wind_name not in data.columns:
            continue
        if short_name in SKIP_AS_EXTRA_ASSET:
            skipped.append(wind_name)
            continue
        _put(short_name, data[wind_name])

    skip_headers = META_LABELS | {"", "nan", "None", "date", "日期"}
    for header in unmatched:
        if header in skip_headers or _is_wind_corner(header):
            continue
        if header not in data.columns:
            continue
        name = pretty_asset_name(header)
        if name in SKIP_AS_EXTRA_ASSET:
            skipped.append(header)
            continue
        series = _wind_numeric(data[header], allow_true_zero=False)
        if int(series.notna().sum()) < 20:
            skipped.append(f"{header}(有效点过少)")
            continue
        _put(name, data[header])

    frame = out.drop_duplicates("date", keep="last").set_index("date").sort_index()
    extra_cols = [c for c in frame.columns]
    print(
        f"  additional_asset: {path.name} → 额外资产 {len(extra_cols)} 个"
        + (f"（{', '.join(extra_cols)}）" if extra_cols else "")
    )
    if skipped:
        print("  已跳过: " + "、".join(skipped))
    return frame


if __name__ == "__main__":
    frame = load_wind_data()
    print(f"rows={len(frame)} cols={len(frame.columns)}")
    print(f"range={frame.index.min().date()} ~ {frame.index.max().date()}")
    print("columns:", ", ".join(frame.columns))
