# 已确认Pro任务：Flash 4.1＋low独立重跑

本包自带559个META64物理任务、独立代码、Conda环境和旧结果标量对照。无需另下载原3000条数据，无需复制旧实验输出。**全部10种LLM初态及其父提案、审阅、构想/代码、FE反馈均在新目录重新生成，统一使用Flash 4.1 low。** 先运行50步，再继续到收敛或总2000步。

## 1. 这批包括什么

按实际返回的模型ID与成功API回执选择：任务至少有一次成功Pro响应才进入本包，不根据旧config中的默认模型猜测。

| 来源 | 本包cohort | 已确认Pro任务 | 正常新增调用 | 默认调用硬上限（含重试） |
|---|---|---:|---:|---:|
| 最早100任务 | `original100` | 100 | 1100 | 1320 |
| 本机200批次中使用过Pro的部分 | `local200` | 166 | 1826 | 2192 |
| 设备B300批次中使用过Pro的部分 | `deviceB300` | 293 | 3223 | 3868 |
| 合计，默认运行 | `all` | **559** | **6149** | **7379** |

559个不同任务、559个不同物理组，均属于原train1000。设备A和另一个新增100任务缺少完整可核实的逐请求模型回执，暂不计入；历史D1实验也不属于本包。选择快照见 `provenance/selection_audit.json`，调用清单见 `source_requests.csv`，逐任务/方法模型来源见 `source_method_lineage.csv`。快照后本机续跑不会自动改变这份任务列表。

选中任务的5590条历史LLM方法记录中，5472条为依赖链完整的纯Pro，3条为Pro/Flash混合，115条为其他模型或依赖不全。因此本包重跑5590条LLM结果，但比较纯Pro和Flash时只纳入历史来源为`pure_Pro`且双方结果有效的配对，不能把全部5590条都称为纯Pro对照。

模型：`deepseek-v4-1-flash-261017`；接口：`https://ark.cn-beijing.volces.com/api/v3`；`thinking=enabled`、`reasoning_effort=low`，最大输出32768 tokens。保持原有生成提示（含C50目标），不把此次实验改为新提示目标。

| 新算方法 | 说明 |
|---|---|
| A1 / A2 / A3 | 无FE直接矩阵／构想后代码／审阅A1 |
| B1 / B2 / B3 | Uniform FE辅助矩阵／构想后代码／审阅B1 |
| L1_PG / L1_H4 | 一次提案分别有界修改PG2和四孔 |
| L2_GEOM / L3_FE | 参数化杆件／读取新L1_PG自身FE后修改 |
| U / PG2 / H4_B035 | 重新计算的3种固定对照，无API |

共13种方法；本包不重复计算另外25种无LLM方法。数值协议为64×64、SIMP p=3、滤波rmin=2.5、OC move=0.2、统一物理体积校准。连续10步满足密度变化≤0.01、柔度逐步相对变化≤1e-4、窗口相对极差≤5e-4，且体积误差≤1e-8、FE残差<1e-6才停止；最早可认证60步，总上限2000。到上限单列，不当作收敛。

## 2. 获取最新代码与数据

已克隆仓库，在仓库根目录执行：

```bash
git pull --ff-only origin main
```

如果此前用了稀疏检出，再执行：

```bash
git sparse-checkout add META64_Pro_to_Flash_low_v1
```

没有克隆过：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/hsksfje/topo.git topo_flash
cd topo_flash
git sparse-checkout set META64_Pro_to_Flash_low_v1
```

然后进入包目录：

```bash
cd META64_Pro_to_Flash_low_v1
```

物理输入全部在 `physical_inputs.zip`，约9.1 MB，运行时按清单解压和SHA256核验。无需手动解压，路径按当前设备重新建立，不依赖原服务器绝对路径。

## 3. Conda环境及API配置

```bash
conda env create -f environment.yml
conda activate meta64-flash-low
```

环境包含Python 3.11、NumPy 1.26.4、SciPy 1.12.0、Matplotlib 3.8.4、OpenAI兼容SDK 3.6.0；无需GPU。原`meta64-fresh`环境如果这些版本相同也可直接使用。

Windows PowerShell：

```powershell
Copy-Item .env.example .env.local
notepad .env.local
```

Linux：`cp .env.example .env.local`，再编辑。仅填写本地 `MODEL_API_KEY` 和授权接口 `MODEL_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3`。运行时使用新配置指定的Flash模型，不采用环境文件中的旧模型名。密钥不上传、不放进回传压缩包。

## 4. 预检、正式运行与实时查看

预检不调用API：

```bash
python run.py --stage verify
python run.py --stage preflight
```

正式运行全部559任务（这个命令会开启付费调用）：

```bash
python -u run.py --execute-api
```

默认数值进程数为当前可用逻辑CPU数减1，BLAS单线程，API默认4并发。i7-10700若系统识别16个逻辑CPU，则自动使用15个数值进程，预留1个CPU；无需手工填写核心数量。需要限制资源可加`--workers 7 --api-workers 4`。

**同一输出目录只能启动一次。** 各阶段会复用本次Flash运行已完成的结果；不复用旧Pro初态。瞬时连接错误、408/409/429/5xx等待后重试，单请求最多8次重试，退避最长900秒，所有尝试均计入上限；永久错误、未知响应、额度用尽会暂停并保留证据，不无限扣费。达到上限不自动加额度，检查后另建授权版本。

实时查看（另开一个已激活环境的终端，在包目录运行）：

```bash
python monitor.py
```

每10秒刷新：任务数、API次数、50步记录数、收敛结算数、错误及更新时间。报告写在 `outputs/flash_low_all_run01/reports/REPORT_zh.md`；50步逐条数据为 `reports/OC50_results.csv`，收敛逐条数据为 `convergence/reports/results.csv`；同任务历史比较为 `reports/Flash_vs_historical_paired.csv`。

预算参考：按本机335次Flash-low调用的平均token量外推，本批正常调用约空闲236元／高峰472元，留余量约300～600元；这只是估算，不是人民币硬上限。调用硬上限7379次，含约20%余量；代码不会在本机替你调用本批API。

## 5. 分批、暂停和恢复

只先运行最早100任务：

```bash
python -u run.py --cohort original100 --execute-api
python monitor.py --cohort original100
```

另外两批分别用 `--cohort local200`（166任务）和`--cohort deviceB300`（293任务）。各自输出目录与API上限独立。**选择all后不要再运行这三批，否则重复付费；选择分批时三批彼此不重叠。** 只想先看50步，可加 `--stop-after50`，随后同一命令去掉这个参数会继续到收敛。

暂停：`python monitor.py --pause`。等旧进程退出后恢复：

```bash
python -u run.py --execute-api --resume
```

分批运行时暂停/恢复命令都加对应`--cohort`。不要删除request/receipt、检查点或修改冻结配置；未知API结果需先检查，不能靠删除回执强制重发。Windows临时文件占用会短暂重试写入，仍失败则保留错误供检查。

## 6. 结果回传到原设备

全部结算后自动生成：

```text
exports/flash_low_all_run01_results.zip
exports/flash_low_all_run01_results.zip.sha256
```

将这两个文件一起复制到原设备：

```text
InitSkill-Lab/outputs/exports/flash_low_rerun/
```

分批文件名分别为 `flash_low_original100_run01_results.zip`、`flash_low_local200_run01_results.zip`、`flash_low_deviceB300_run01_results.zip`。每个包都有逐文件SHA256返回清单，方便解压校验并与历史Pro同任务匹配。若自动打包未完成，使用原分组参数执行 `python run.py --stage export`；没有完成标记时不会把部分结果伪装为完整回传。

## 数据来源与改动

原数据：Igor Kuszczak、Miguel Bessa、Gawel Kus、Federico Bosi，*Meta-neural Topology Optimization: Knowledge Infusion with Meta-learning*。[Zenodo 21893871](https://zenodo.org/records/21893871)，[DOI](https://doi.org/10.5281/zenodo.21893871)，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本包仅选择已确认Pro的任务，并把原条件映射为求解器节点自由度和物理输入；不含作者优化标签。选择和坐标映射见任务JSON、输入SHA256及provenance。方法代码来自本项目已冻结的META64实验；新增独立Flash-low调度、任务来源审计和Windows临时文件占用重试，不表示原作者认可本项目。
