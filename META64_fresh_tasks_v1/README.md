# 新100任务：从原始工况运行28种固定初始化＋10种LLM，再继续到收敛

这是一套**从头生成初始化**的运行入口，独立于此前的`META64_remaining50_convergence_v1`续跑包。本次默认任务从`train1000`排除此前100条任务和等价物理组后抽取，体积分数十档各10条，共100条；抽样种子`2026112009`，不使用任何优化成绩挑选。

完整数据另放在仓库同级目录`META64_datasets_v1/`。包括训练3000、验证300、分布内/分布外/跨分辨率测试各1000条及其小子集，源数组只存一次。**本轮只解压训练1000对应的分片即可，不需要先下载全部3.53 GB原始数组。**

## 1. 下载新代码和这次需要的数据

已经完整克隆过仓库时，在仓库根目录执行：

```bash
git pull
cd META64_fresh_tasks_v1
```

首次下载、希望避免把所有测试数据一并下载时：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/hsksfje/topo.git topo_new100
cd topo_new100
git sparse-checkout set META64_fresh_tasks_v1 META64_datasets_v1/metadata
cd META64_fresh_tasks_v1
```

若不熟悉稀疏克隆，也可以正常`git clone`或从GitHub下载ZIP，但会同时下载所有数据分片，耗时和下载量更大。

## 2. Conda环境

在Anaconda Prompt或配置好Conda的终端中，进入本目录：

```bash
conda env create -f environment.yml
conda activate meta64-fresh
```

依赖固定为Python3.11、NumPy1.26.4、SciPy1.12.0、Matplotlib3.8.4，及通过pip安装的OpenAI兼容SDK3.6.0。SDK用于火山方舟兼容接口，不代表使用OpenAI模型。无GPU要求。

```bash
python -c "import sys,numpy,scipy,matplotlib,openai; print(sys.version); print(numpy.__version__,scipy.__version__,matplotlib.__version__,openai.__version__)"
```

## 3. 解压并校验本次任务数据

```bash
python ../META64_datasets_v1/fetch_data.py --subset train100_new100
python ../META64_datasets_v1/fetch_data.py --subset train100_new100 --verify-only
python code/run.py --stage verify
```

本地已有所需ZIP时直接解压，无需重复下载；若使用稀疏克隆，程序从本仓库下载缺少的ZIP并核验SHA256。解压位置固定为`../META64_datasets_v1/extracted/`，仍保留原来的`raw/train/`相对路径。

新100任务清单：`../META64_datasets_v1/metadata/manifests/train100_new100.jsonl`。抽样及排除审计：`../META64_datasets_v1/metadata/audits/train100_new100_selection.json`。**不要把旧`train100.jsonl`当成本次新任务清单。**

## 4. 本地配置API，再运行全部38种

Windows PowerShell：

```powershell
Copy-Item .env.example .env.local
notepad .env.local
```

Linux：

```bash
cp .env.example .env.local
```

在`.env.local`中填写你自己的密钥，保留如下字段格式：

```dotenv
MODEL_API_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
MODEL_API_KEY="填写你的本地密钥"
```

密钥文件留在本机，已被`.gitignore`排除，不进入结果导出。仓库只有不含真实密钥的`.env.example`。

先做无API预检，再启动全流程：

```bash
python code/run.py --stage preflight
python -u code/run.py --stage run --execute-api --env-file .env.local
```

默认阶段依次为：

1. 数据转换、哈希及物理预检；
2. 28种无LLM方法全部新跑OC50；
3. 10种LLM方法生成同任务新初态并新跑OC50；
4. 全部38种方法从各自第50步继续，满足统一判据或达到总2000步；
5. 自动输出报告和回传清单。

固定初始化阶段结束前，LLM完成数为0属于阶段安排。整个试验计划3800条方法记录；无效提案和数值失败保留状态，不以Uniform回填。

**本次用户已授权这100条新任务最多1100次新增付费调用**，默认模型/提供方与前一批相同，配置见`protocol.json`，授权清单见`authorizations/new100.json`。发送工况、同任务初态和新算初态FE摘要；不发送源参考柔度或保存的应变能标签。原始参考标签只是数据包的一部分，不进入LLM输入。

请求先记账再发送，OpenAI SDK重试次数为0，未知网络结果不自动重发。遇到API错误时程序安全暂停，查看`requests/<请求ID>/attempt.json`、`receipt.json`和`error.json`；不要删除回执文件以强行重发。额度上限是调用次数，不是费用保证；实际费用以服务商账单为准，`budget_status.json`保存调用及token用量。

只想先运行28种固定初始化的OC50时：

```bash
python -u code/run.py --stage fixed
```

之后运行上面的全流程命令，会复用这批已算好的固定结果，再接LLM和收敛阶段。没有`--execute-api`不会产生网络模型调用，也不会假装已经完成全38方法。

## 5. CPU：自动留一个逻辑处理器

默认数值进程数为**可用逻辑CPU数减1**，每个数值库只使用1线程。若i7设备识别到16个可用逻辑CPU，就运行15个数值进程；Windows/Linux同时设置CPU亲和性，排除一个逻辑CPU。API阶段默认最多6个并发任务，避免将15路数值并行等同于15路外部API请求；可用`--api-workers`调整，但不能超过数值进程数。

例如明确使用15个数值进程和6个API并发：

```bash
python -u code/run.py --stage run --execute-api --env-file .env.local --workers 15 --api-workers 6
```

若另一台设备还在跑原来的剩余50任务，建议等该批结束后启动本批，避免两个队列都按N−1占用CPU。若内存紧张，可安全暂停后减少`--workers`。不要在同一输出目录重复启动程序。Windows建议将仓库放在较短路径，例如`C:\topo`。

## 6. 后台运行与实时查看

输出固定为**本运行包下`outputs/train100_new100_run01/`**。不要与之前的`remaining50_run01`混用。

Windows PowerShell（已激活环境、已填写`.env.local`）：

```powershell
New-Item -ItemType Directory -Force outputs/train100_new100_run01 | Out-Null
Start-Process -FilePath (Get-Command python).Source -ArgumentList '-u','code/run.py','--stage','run','--execute-api','--env-file','.env.local' -WorkingDirectory (Get-Location).Path -RedirectStandardOutput 'outputs/train100_new100_run01/run.log' -RedirectStandardError 'outputs/train100_new100_run01/stderr.log'
```

Linux：

```bash
mkdir -p outputs/train100_new100_run01
nohup python -u code/run.py --stage run --execute-api --env-file .env.local > outputs/train100_new100_run01/run.log 2>&1 < /dev/null &
```

Windows PowerShell实时进度：

```powershell
while ($true) { Clear-Host; Get-Content outputs/train100_new100_run01/progress.json; Start-Sleep -Seconds 10 }
```

Linux：

```bash
watch -n 10 'cat outputs/train100_new100_run01/progress.json'
```

主要输出：

| 路径（均相对输出目录） | 内容 |
|---|---|
| `progress.json` | 当前阶段、固定/LLM OC50完成数、API调用数、收敛进度 |
| `reports/OC50_results.csv` | 全方法50步结果及相对Uniform改善 |
| `convergence/reports/REPORT_zh.md` | 收敛结果主报告 |
| `convergence/reports/SPEED_QUALITY_zh.md` | 速度与最终柔度，含更快且更好、效果接近、更慢但改善>1%/>5%等列 |
| `convergence/reports/results.csv` | 逐任务停止步数、柔度和共同质量目标步数 |
| `corpus/`、`extra/`、`tasks/` | 固定17、补充11、LLM10的初态与OC50记录 |
| `convergence/tasks/` | 第50步以后的检查点、完整历史和最终密度 |
| `requests/`、`budget_status.json` | API请求回执、调用次数及token账本 |
| `environment.json`、`process.json` | 实际环境与CPU配置 |

## 7. 暂停与恢复

在`outputs/train100_new100_run01/`下创建`PAUSE_REQUESTED`文件请求暂停。程序保存当前检查点；在途API正常返回后保存回执，不再提交新请求。

Windows PowerShell：

```powershell
New-Item -ItemType File -Force outputs/train100_new100_run01/PAUSE_REQUESTED
```

Linux：

```bash
touch outputs/train100_new100_run01/PAUSE_REQUESTED
```

等待原进程退出后，移除根目录及收敛子目录的暂停标记，再用原命令恢复。不要在原进程仍运行时删除标记或启动第二个进程。

Windows PowerShell：

```powershell
Remove-Item outputs/train100_new100_run01/PAUSE_REQUESTED -ErrorAction SilentlyContinue
Remove-Item outputs/train100_new100_run01/convergence/PAUSE_REQUESTED -ErrorAction SilentlyContinue
python -u code/run.py --stage run --execute-api --env-file .env.local
```

Linux：

```bash
rm -f outputs/train100_new100_run01/PAUSE_REQUESTED outputs/train100_new100_run01/convergence/PAUSE_REQUESTED
python -u code/run.py --stage run --execute-api --env-file .env.local
```

## 8. 数值与实验边界

与此前100任务保持相同的64×64网格、p=3、rmin=2.5、move=0.2及体积可行侧OC。LLM仍按原提示生成针对C50的初态，后续再额外评价收敛效果；没有悄悄改成针对收敛终点的新提示。

从第50步后，连续10步同时满足max|Δx|≤0.01、单步|ΔC|/C≤1e-4、窗口柔度相对极差≤5e-4，体积误差≤1e-8且FE残差<1e-6才认证收敛。总上限2000步包含原50步；达到上限标记`max_steps`，不当作收敛。失败不回填。COARSE30/80、OC5前置优化费用单列。

PLSM/MMC、应力追踪/脊线方法是本地初始化适配，不是原论文全算法复现。此次从训练池抽样，属于新增训练实验；验证和测试数据已提供，但保持独立，不应在反复调Skill时当作训练反馈。

代码以原冻结实现为基础，仅抽取所需函数并增加迁移、Windows文件锁/子进程资源限制和资源管理。Windows提供原生兼容代码，但尚未在你的设备实测，请先运行预检。Linux的验证摘要见`validation/`：100条Uniform物理核验、一个新任务的28种固定方法OC50、10种LLM协议的11次模拟响应、Uniform从50步续跑到291步。模拟响应只验证流程，实际外部API调用为0，不是这批新任务的真实LLM效果。

## 9. 结果带回本机

完成后输出目录会自动生成`RETURN_READY.json`和`return_manifest.json`。复制整个`outputs/train100_new100_run01/`回来，放到：

```text
/public/home/cairn/topo/InitSkill-Lab/outputs/meta64_train100_new100_remote_v1/run01/
```

这个`run01/`下直接放`config.json`、`convergence/`、`corpus/`、`extra/`、`tasks/`、`requests/`等，不再多套一层目录。也可生成便于传输的ZIP：

```bash
python code/export_results.py --zip exports/META64_new100_results.zip
```

导出只封存完整结果并计算哈希，不调用API、不重新求解。不把根目录`.env.local`放进回传包。

## 10. 换其他数据清单

同一个代码可以对其他**64×64**清单运行固定初始化，例如指定`--manifest ../META64_datasets_v1/metadata/manifests/train3000.jsonl`并使用独立`--output`。本次`--execute-api`授权只覆盖冻结的新100任务；不能把1100次上限自动扩成对3000条都调用LLM。

`test_cr1000`/`test_cr300`为256×256，本入口会明确拒绝，不缩放或冒充64×64数据。它们的原始数组已完整上传，适用于单独设计的跨分辨率实验。
