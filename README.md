# Macro Quant（纯离线版）

项目优先读取根目录的 `data1.xlsx`，其次 `data_new.xlsx`、`data.xlsx`，再次 `data.csv`。
不需要网络，也不需要后端服务。

## 使用方法

### 1. 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

### 2. 更新数据

用新的 Wind 导出文件覆盖根目录 `data1.xlsx`（或 `data_new.xlsx` / `data.xlsx`），然后执行：

```bash
python3 update_all_data.py
```

该命令会重新计算：

1. 低频和高频宏观因子；
2. 2021 年至今的月频、周频相关矩阵；
3. 最新资产因子暴露；
4. 周频波动分位、冲击警报和资产压力；
5. LightGBM + Black-Litterman 模型；
6. 稳健、均衡、进取三档模型结果和静态图片。

模型训练耗时较长。仅调试因子流程时可以运行：

```bash
python3 update_all_data.py --skip-model
```

### 3. 打开页面

```bash
python3 -m http.server 8765
```

浏览器打开：

<http://127.0.0.1:8765/macro_factor_corr_interactive.html>

页面完全读取本地 JSON/PNG，不需要后端服务。更新完成后按
`Command+Shift+R` 强制刷新。

## data.xlsx 要求

文件必须是 Wind 导出、或用 Excel 另存后的 `.xlsx`。放到项目根目录，优先命名为
`data1.xlsx`，也可以用 `data_new.xlsx` / `data.xlsx`。Windows 上直接用 Excel 打开再保存也可以识别。

第一列是日期，可以是 Excel 序列号，也可以是 `2021-01-04` 这种日期。
表头行左上角可以是「指标名称」，也可以是 Windows 终端导出常见的「Wind」。
后面可以保留频率、单位、指标 ID、来源。
如果 Wind 导出的是旧版 `.xls`，在 Excel 里「另存为」`.xlsx`。

Windows 上 Python 命令可能是 `python` 而不是 `python3`：

```bat
python -m pip install -r requirements.txt
python update_all_data.py
python -m http.server 8765
```

增长高频因子使用：

- `恒生指数`
- `期货收盘价(电子盘):LME3个月铜`
- `房地产(申万):收盘价(前复权)`

其中 Wind 的 LME 三个月铜对应旧数据源代码 `CAD`，不是加拿大元。项目固定
使用旧版权重和滞后，以保持增长暴露的历史可比性。

低频地缘因子使用 `全球:地缘政治风险指数` 或 `全球:地缘政治风险指数(参考十家报纸)` 的月末值。
高频地缘因子用沪金、布油的**绝对价格**线性拟合该指数水平：
`拟合值 ≈ 截距 + 系数×金价 + 系数×油价`。

## 主要文件

```text
data1.xlsx                            优先外部输入（含企业债总值 + 地缘指数）
data_new.xlsx / data.xlsx             备选 Excel 输入
update_all_data.py                    完整重算入口
update_from_xlsx.py                   因子与资产构建
load_wind_data.py                     Wind Excel/CSV 解析
plot_macro_factor_corr.py             月频矩阵
plot_macro_hf_corr.py                 周频矩阵与静态警报
factor exposure/
  compute_factor_exposure.py          Lasso 暴露
model prediction/
  run_all.py                          离线模型训练入口
macro_factor_corr_interactive.html    静态页面
```

## 静态输出

- `macro_factor_corr.json`
- `macro_hf_factor_corr.json`
- `factor exposure/factor_exposure_latest.json`
- `model_prediction_static.json`
- `model prediction/output/**/figures/`

低频、高频因子 CSV 还承担 Wind 起始日前历史拼接功能，不应手动删除。

## 常用检查

只查看将运行的步骤：

```bash
python3 update_all_data.py --dry-run
```

仅重算指定阶段：

```bash
python3 update_all_data.py --only data,corr,exposure
```

阶段名称为 `data`、`corr`、`exposure`、`model`。
