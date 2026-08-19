# Macro Quant

离线宏观因子与大类资产暴露系统。输入为万得（Wind）导出的本地 Excel，输出为因子时间序列、相关矩阵、资产暴露，以及可选的配置模型。全部计算在本机完成，不访问外部数据接口。

数据流：

```text
data/data1.xlsx
        │
        ▼
src/update_from_xlsx.py          低频 / 高频因子、资产收盘价
        │
        ├─► src/export_hf_factor_daily.py     日频因子总表
        ├─► src/plot_macro_factor_corr.py     月频相关
        ├─► src/plot_macro_hf_corr.py         周频相关与警报
        ├─► src/compute_factor_exposure.py    资产 × 因子暴露
        └─► src/model/                        LightGBM + Black-Litterman（可选）
        │
        ▼
output/  →  web/macro_factor_corr_interactive.html
```

一键入口为 `scripts/update_all_data.py`（根目录 `update_all_data.py` 为转发包装）。目录约定集中在 `src/paths.py`。

---

## 1. 目录与文件

### 1.1 根目录

| 文件 | 作用 | 如何得到 |
|------|------|----------|
| `README.md` | 本说明 | 仓库自带 |
| `requirements.txt` | Python 依赖 | 仓库自带 |
| `update_all_data.py` | 转发到 `scripts/update_all_data.py`，兼容旧命令 | 仓库自带 |
| `.gitignore` | 忽略缓存、锁文件、模型运行时中间表 | 仓库自带 |

### 1.2 `config/`

| 文件 | 作用 | 如何得到 |
|------|------|----------|
| `panel_config.json` | 面板参数：热力图起点与因子名单、暴露窗口 / Lasso / Bootstrap、资产×因子回归开关 `asset_factor_mask`、波动警报阈值 | 窗口、因子名单、0/1 开关手改。`asset_factor_mask` 的**行**在 data 阶段按 Excel 资产名单增删（新资产用默认开关，已有 0/1 保留）。改窗口或因子名单后需重跑对应阶段；滚动相关窗和警报阈值改完刷新网页即可 |

读取逻辑在 `src/panel_config.py`。缺字段时用该文件中的默认值。

### 1.3 `data/`

用户放入的原始表。程序不自动下载。

| 文件 | 作用 | 如何得到 |
|------|------|----------|
| `data1.xlsx` | 主输入。宏观指标 + 12 个默认大类资产的日频（或混频）序列 | Wind 导出后改名放入。识别顺序：`data1.xlsx` → `data_new.xlsx` → `data.xlsx` → `data.csv` |
| `additional_asset.xlsx` | 额外资产（行业、个股等），只进入暴露矩阵，不进入主因子合成 | 另做一张与 `data1` 同结构的 Wind 表。删列再跑更新，暴露图上对应行会消失 |

Excel 约定：第一行可空；从「指标名称」行起读；空值留空，不要填 0。详见第 6 节。

### 1.4 `scripts/`

| 文件 | 作用 | 如何得到 |
|------|------|----------|
| `update_all_data.py` | 按 `data → corr → exposure → model` 调用 `src/` 中各脚本 | 仓库自带。常用：`--skip-model`、`--only corr,exposure`、`--dry-run` |

### 1.5 `src/` — 核心代码

| 文件 | 作用 | 何时运行 |
|------|------|----------|
| `paths.py` | 仓库根目录、`config/`、`data/`、`output/` 下各子目录的唯一定义 | 被其它模块 import |
| `panel_config.py` | 读取 `config/panel_config.json`，解析因子名单与 `asset_factor_mask` | import |
| `load_wind_data.py` | 解析 Wind Excel/CSV：表头映射、假 0、额外资产 | `update_from_xlsx.py` 调用 |
| `update_from_xlsx.py` | 由 `data1.xlsx` 生成全部低频/高频因子，并写入资产收盘价面板 | 阶段 `data` |
| `export_hf_factor_daily.py` | 把七个高频因子拼成日频水平表与日变化表 | 阶段 `data` |
| `plot_macro_factor_corr.py` | 月频六因子相关矩阵（JSON / CSV / PNG） | 阶段 `corr` |
| `plot_macro_hf_corr.py` | 周频六因子相关矩阵，并附静态波动警报序列 | 阶段 `corr` |
| `compute_factor_exposure.py` | 周度对数收益对高频因子周变化做标准化 Lasso + 连续块 Bootstrap | 阶段 `exposure` |
| `export_static_model_prediction.py` | 把三档模型结果写成网页可读 JSON | 阶段 `model` |

### 1.6 `src/model/` — 配置模型（可选）

日常看板可 `--skip-model`。完整重训由 `src/model/run_all.py` 串联。

| 文件 | 作用 |
|------|------|
| `run_all.py` | 依次：本地 raw → 策略训练 → 作图 → 稳健/进取两档 |
| `prepare_local_raw_data.py` | 从 `output/exposure/combined_close.csv` 拆成模型用的单资产 CSV |
| `config.py` | 宇宙、标签 horizon、LightGBM 与 Black-Litterman 参数 |
| `macro_features.py` | 动量 / 波动 / 宏观截面特征与标签 |
| `model_lgbm.py` | LightGBM 排序 / 回归 |
| `black_litterman.py` | 观点与后验收益 |
| `portfolio_optimizers.py` | 组合优化 |
| `rolling_backtest.py` | Walk-forward 回测 |
| `run_macro_strategy.py` | 均衡档主流程 |
| `run_aggression_profiles.py` | 不重训，只换风险档位重跑配置 |
| `plot_results.py` | 净值、回撤、权重等图 |
| `model_summary.py` | 摘要结构，供静态 JSON 导出 |
| `tests/test_model_upgrade.py` | 离线单元测试 |

### 1.7 `output/factors/` — 因子时间序列

均由 `src/update_from_xlsx.py` 写入。新结果在接缝处与旧 CSV 对齐（水平因子可按衔接点缩放），**不要手工删这些文件**，否则历史接缝会断。

**增长 `output/factors/growth/`**

| 文件 | 内容 |
|------|------|
| `growth_factor.csv` | 低频：PMI 同比差、固投、社零、进出口加权的 `raw_growth_factor` |
| `hf_growth_factor_synthetic.csv` | 高频：铜 / 地产 / 恒生拟合后的净值、同比、周环比 |
| `growth_high_freq_daily.csv` | 高频日频中间表 |

**通胀 `output/factors/inflation/`**

| 文件 | 内容 |
|------|------|
| `inflation_factor.csv` | 低频：CPI、PPI 同比波动率倒数加权 |
| `hf_inflation_weekly.csv` | 高频周频净值、同比、周环比 |
| `hf_regression_results.json` | 猪肉 / 布油 / 螺纹钢权重与滞后期 |
| `commodities.csv` | 合成用到的商品序列 |

**利率 `output/factors/interest_rate/`**

| 文件 | 内容 |
|------|------|
| `rate_factor.csv` | 低频：十年国债收益率月末水平 |
| `hf_rate_factor_daily.csv` | 高频：国债净价取负后的水平与日变化 |
| `cn10y_yield_daily.csv` | 十年国债收益率日序列 |
| `cn_gov_bond_index_daily.csv` | 国债净价指数日序列 |

**信用 `output/factors/credit/`**

| 文件 | 内容 |
|------|------|
| `credit_factor.csv` | 低频：AA 中票 3Y − 国开 3Y 利差 |
| `hf_credit_factor_daily.csv` | 高频：企债与国开财富差去趋势后的指数 |

**汇率 `output/factors/exchange/`**

| 文件 | 内容 |
|------|------|
| `dxy_yahoo.csv` | 美元指数日收盘（列名沿用历史文件） |

**地缘 `output/factors/politics/`**

| 文件 | 内容 |
|------|------|
| `geo_factor.csv` | 低频：GPR 月末水平 |
| `hf_geo_factor_synthetic.csv` | 高频：沪金 + 布油拟合 GPR 的日水平 |
| `geo_fit_monthly.csv` | 月度拟合诊断 |
| `geo_high_freq_daily.csv` | 高频日频中间表 |
| `hf_regression_results.json` | 拟合系数、滞后、样本区间 |

**流动性 `output/factors/mobility/`**（只进暴露，不进六因子热力图）

| 文件 | 内容 |
|------|------|
| `mobility_factor.csv` | 低频：M2 同比 − 社融存量同比 |
| `hf_mobility_factor_synthetic.csv` | 高频：申万大小盘 PE 变化加权 |
| `mobility_high_freq_daily.csv` | 日频中间表 |
| `hf_regression_results.json` | 拟合元数据 |

### 1.8 `output/corr/` — 相关矩阵

| 文件 | 作用 | 生成脚本 |
|------|------|----------|
| `macro_factor_monthly.csv` | 月频六因子面板 | `plot_macro_factor_corr.py` |
| `macro_factor_corr.csv` / `.json` / `_heatmap.png` | 月频相关；JSON 供网页左侧热力图 | 同上 |
| `macro_hf_factor_weekly.csv` | 周频六因子面板 | `plot_macro_hf_corr.py` |
| `macro_hf_factor_corr.csv` / `.json` / `_heatmap.png` | 周频相关与警报序列；JSON 供网页右侧热力图 | 同上 |

网页读 JSON，不读 PNG。

### 1.9 `output/hf/` — 日频因子总表

由 `export_hf_factor_daily.py` 生成，**不进网页**，便于另存或核对。

| 文件 | 内容 |
|------|------|
| `hf_factor_daily.xlsx` | 工作表：`水平`、`日变化`（%）、`说明` |
| `hf_factor_daily.csv` | 水平 |
| `hf_factor_daily_change.csv` | 日变化 |

通胀在看板中按周计算；日频表用同一套猪肉 / 布油 / 螺纹钢权重做日收益，并对齐到最近周五的周频净值。

### 1.10 `output/exposure/` — 因子暴露

由 `compute_factor_exposure.py` 生成。网页暴露页读 JSON。

| 文件 | 内容 |
|------|------|
| `combined_close.csv` | 12 个默认资产 + `additional_asset` 的日收盘；模型 raw 也从此拆出 |
| `factor_exposure_latest.json` | 暴露系数、`factor_mask`（实际进回归的 0/1）、窗口元数据 |
| `factor_exposure_latest.csv` / `.png` | 同一矩阵的表与图 |
| `factor_exposure_weekly_panel.csv` | 回归用的周度资产收益与因子变化 |

### 1.11 `output/model/` — 配置模型产物

由 `src/model/` 与 `export_static_model_prediction.py` 生成。

| 路径 | 内容 |
|------|------|
| `model_prediction_static.json` | 网页「模型预测」页的静态摘要与图片 URL |
| `models/` | `macro_lgbm_bl.joblib`、特征重要性、元数据 |
| `output/figures/` | 均衡档图 |
| `output/aggression_conservative/figures/` | 稳健档图 |
| `output/aggression_aggressive/figures/` | 进取档图 |
| `data/raw/`、`data/panel/` | 运行时中间数据，`.gitignore` 忽略，可重建 |

### 1.12 `web/`

| 文件 | 作用 | 如何得到 |
|------|------|----------|
| `macro_factor_corr_interactive.html` | 本地看板：月/周相关、暴露、波动警报、三档配置 | 仓库自带。读 `config/` 与 `output/` 下 JSON，不写盘 |

用项目根目录启动 `python3 -m http.server 8765`，打开：

<http://127.0.0.1:8765/web/macro_factor_corr_interactive.html>

---

## 2. 因子构成

六个因子进入相关热力图；流动性只进入暴露。每个因子有低频（月末）与高频（日/周）两条线，刻画同一宏观维度的不同抽样频率，不是两套无关定义。

热力图用**水平或同比**（现在处在什么位置）；暴露用**周变化**（这一周冲击的方向与幅度）。因此同一因子在两张图上的数字不必一致。

### 2.1 增长

刻画国内经济活动加速或放缓。

**低频**（权重固定，缺项时在剩余项上重归一化，至少两项才取值）：

| Wind 指标 | 处理 | 权重 |
|-----------|------|------|
| 中国:制造业PMI | 月末值减 12 个月前 | 58% |
| 社会消费品零售总额同比 | 直接用同比 | 25% |
| 出口、进口同比平均 | 外贸 | 10% |
| 固定资产投资完成额累计同比 | 投资 | 7% |

1–2 月社零、固投空窗用相邻月填充。热力图使用 `raw_growth_factor`。

**高频**：用日频资产收益拟合上述月度因子——LME 三月铜（滞后 0，约 +77%）、申万房地产（约 1 个月，约 −15%）、恒生指数（约 3 个月，约 −8%）。对数收益加权累加成净值后再算同比。暴露使用周度日收益合计（`hf_mom_pct`）。

数值上升表示增长加快，或市场在交易加快。

### 2.2 通胀

**低频**：CPI、PPI 当月同比，按近一年波动率倒数加权；样本不足时等权。

**高频**：猪肉批发价（滞后 0，约 20%）、布伦特原油（约 1 个月，约 40%）、螺纹钢（约 3 个月，约 40%）。周五收盘算周涨跌，累加成净值并计算同比。暴露使用周环比 `hf_wow`。

数值上升表示通胀压力上升。

### 2.3 利率

**低频**：十年国债到期收益率月末水平（单位 %），不做同比。

**高频**：中债国债总净价指数取负。净价上涨对应利率因子下降。暴露使用净价的周对数变化（净价下跌 → 利率因子当周上升）。

数值上升表示利率水平更高，或当周利率在上行。

### 2.4 信用

利差走阔表示信用风险补偿上升，不是「信用越好」。

**低频**：`AA 中票 3 年收益率 − 国开债 3 年收益率`。

**高频**：企业债 3–5 年财富指数减国开 3–5 年财富指数，HP 滤波去慢趋势后取反、标准化。热力图用周末水平；暴露用指数周变化。

默认仅债券类资产将该因子纳入回归（见暴露一节）。

### 2.5 汇率

美元指数（DXY）收盘。低频取月末，高频取周五。没有单独的人民币因子；暴露表里的「美元兑人民币」是被解释资产。暴露使用美元指数周对数收益。

数值上升表示美元走强。

### 2.6 地缘

**低频**：Wind「全球:地缘政治风险指数」（或「……参考十家报纸」）月末值，即 GPR 水平，不是金价或油价。

**高频**：GPR 更新慢，用沪金、布油绝对价格对 GPR 做滞后网格线性拟合（金滞后 0–3 个月可选）。拟合 R² 通常不高（约 0.2 量级）：金油平滑，GPR 可跳跃，两条线不必同步见顶。沪金、原油默认仍可进入地缘暴露回归。

数值上升表示地缘风险定价更高。

### 2.7 流动性（仅暴露）

**低频**：`M2 同比 − 社会融资规模存量同比`。

**高频**：申万大盘 PE 变化为负权、小盘 PE 为正权，捕捉相对流动性与风格，不能直接等同于「央行放水」。中证 1000 与上证 50 的分化常与此有关。

---

## 3. 资产暴露

问题：在最近约 260 周（`config/panel_config.json` 可改）内，各资产周对数收益能被哪些宏观因子的周变化解释，以及方向、相对强度。

估计步骤：

1. Y：资产周对数收益；X：高频因子周变化。
2. 对 X、Y 做 `StandardScaler`，系数表示因子波动 1 个标准差时，资产收益变动的标准差倍数。
3. 在整段窗口上用 `LassoCV` 选惩罚强度，再乘 `alpha_scale`（默认 0.5，惩罚弱于纯 CV）。
4. 随机抽取 `bootstrap_samples`（默认 3000）段连续 `sample_length_weeks`（默认 104 周），每段拟合 Lasso，对每个因子取系数中位数。

网页格子为上述中位系数。灰色斜纹与 `—` 表示该因子未进入该资产回归（开关为 0），与 Lasso 把系数打成 0.000 不同。

默认资产：上证50、沪深300、中证500、中证1000、恒生指数、中债国债、中债企业债、中证转债、布伦特原油、沪金、标普500、美元兑人民币。额外资产来自 `data/additional_asset.xlsx`。

回归开关：

- `credit_only_for_bonds`：非债券默认不含信用因子。
- `asset_factor_mask`：资产 × 因子，`1` 进入回归，`0` 不进入。跑 `update_all_data` 的 data 阶段时，矩阵行与 `data1.xlsx` + `additional_asset.xlsx` 的资产名单对齐：Excel 多了就补默认行（非债券信用因子为 0），少了就删行。已有资产的 0/1 不会被覆盖。
- `asset_exclude_factors`：按资产剔除因子的简写；与 mask 冲突时以 mask 为准。

---

## 4. 运行

依赖：Python 3.10+（Windows 上命令多为 `python`，macOS 多为 `python3`）。

```bash
python3 -m pip install -r requirements.txt
```

将 Wind 表覆盖为 `data/data1.xlsx`。可选：`data/additional_asset.xlsx`。

```bash
python3 scripts/update_all_data.py              # 全流程，含模型
python3 scripts/update_all_data.py --skip-model # 只更新看板
python3 scripts/update_all_data.py --only exposure
python3 scripts/update_all_data.py --only corr
python3 scripts/update_all_data.py --dry-run
```

阶段：`data`（因子）→ `corr`（相关）→ `exposure`（暴露）→ `model`（配置）。模型失败时前三步结果仍在 `output/`。

```bash
python3 -m http.server 8765
```

浏览器打开 <http://127.0.0.1:8765/web/macro_factor_corr_interactive.html>。更新后强制刷新（Windows `Ctrl+F5`，Mac `Command+Shift+R`）。服务在项目根目录启动，以便 `/config/`、`/output/` 可被页面读取。

### 4.1 额外资产

`additional_asset.xlsx` 或 `.csv`，结构与 `data1` 相同。不要把额外资产写进 `data1.xlsx`。已在主表中的上证50、沪金等列若重复出现会被跳过。名称含「债」「转债」的按债券处理（默认纳入信用因子）。历史建议 3–5 年。改完后跑更新，暴露图和 `config/panel_config.json` 里的 `asset_factor_mask` 会一起增减行：

```bash
python3 scripts/update_all_data.py --skip-model
```

### 4.2 参数（`config/panel_config.json`）

| 目的 | 字段 |
|------|------|
| 热力图起点 | `heatmap.lf_start`、`heatmap.hf_start` |
| 相关矩阵因子 | `heatmap.include_factors` / `exclude_factors` |
| 暴露窗口与 Lasso | `rolling_window_weeks`、`sample_length_weeks`、`bootstrap_samples`、`alpha_scale` |
| 暴露因子列 | `exposure.include_factors` / `exclude_factors` |
| 按资产开关 | `exposure.asset_factor_mask`（1/0） |
| 信用是否仅债券 | `credit_only_for_bonds` |
| 滚动相关、警报 | `heatmap.rolling_corr_*`、`alerts.*`（改完刷新网页即可） |

只改暴露：`--only exposure`。只改热力图因子或起点：`--only corr`。

---

## 5. Wind 导出要求

- 左上可为「指标名称」或「Wind」。第一列为日期（`YYYY-MM-DD` 或 Excel 序列号）。频率 / 单位 / 指标ID / 来源行会跳过。
- 优先空值留空，0 也可以。价格、指数、PMI 出现 0 视为缺失；CPI 等同比允许真 0。
- 列名优先完全匹配；否则需一组关键词同时出现（例如国开 3 年收益率须同时含「国开债」「到期收益率」「3年」）。

**暴露用日收盘：** 上证50指数；沪深300指数；中证500指数；中证1000指数:收盘价；恒生指数；中证转债:收盘价(前复权)；标普500:收盘价(前复权)；中间价:美元兑人民币；美元指数；SHFE黄金:收盘价；ICE布油:收盘价；中债-国债总财富(总值)指数:收盘价(前复权)；中债-企业债总财富(总值)指数:收盘价(前复权)。

**因子合成：** 中国:制造业PMI；中国:固定资产投资完成额:累计同比；中国:社会消费品零售总额:当月同比(1-2月合并)；中国:出口金额:当月同比；中国:进口金额:当月同比；中国:CPI:当月同比；中国:PPI:当月同比；中国:M2:同比；中国:社会融资规模存量:同比；中债国债到期收益率:10年；中债国开债到期收益率:3年；中债中短期票据到期收益率(AA):3年；中债-国债总净价(总值)指数:收盘价(前复权)；中债-国开行债券总财富(3-5年)指数:收盘价(前复权)；中债-企业债总财富(3-5年)指数:收盘价(前复权)；中国:平均批发价:猪肉；SHFE螺纹钢:收盘价；期货收盘价(电子盘):LME3个月铜；房地产(申万):收盘价(前复权)；市盈率:申万大盘指数；市盈率:申万小盘指数；全球:地缘政治风险指数（或「……(参考十家报纸)」）。

---

## 6. 常见问题

**缺少某一列（如 `国开债_3Y`）**  
主表未包含对应 Wind 指标。按第 5 节补列后覆盖 `data/data1.xlsx` 再跑。

**只换了 Excel，网页没变**  
必须跑 `scripts/update_all_data.py`，并强制刷新浏览器。

**模型报错，热力图还有吗**  
有。模型在最后；`corr` 与 `exposure` 已写入 `output/`。

**个股暴露接近 0**  
七个因子服务大类资产。个股收益主要来自基本面与主题，宏观解释力低是预期现象。

**`python` 找不到**  
改用 `python3`，或反之。
