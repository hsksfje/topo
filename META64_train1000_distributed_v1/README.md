# 跑完train1000：剩余800任务按原顺序分成200＋300＋300

**不随机抽取，不重复运行前两批200任务。** 从原`train1000.jsonl`顺序删除最早100条和另一台设备当前运行的新增100条，剩余800条原顺序划分如下：

| 分工 | profile | 任务数 | 剩余清单内位置（从1开始） | 首／末源任务ID | API上限 |
|---|---|---:|---|---|---:|
| 本机 | `local200` | 200 | 1—200 | problem_000088 → problem_007709 | 2200 |
| 另一台设备A | `deviceA300` | 300 | 201—500 | problem_007726 → problem_018425 | 3300 |
| 另一台设备B | `deviceB300` | 300 | 501—800 | problem_018439 → problem_029991 | 3300 |

源任务ID不连续，因为原train1000本身是子集且删除了前两批。每批必须使用对应的JSONL，不要自行按编号区间取样。**三批加前两批合计1000个不同任务和1000个物理组，互相不重叠。** 完整覆盖审计见`provenance/train1000_partition.json`，三批清单见`manifests/`。

三批均采用相同38种方法：28无LLM＋10LLM，新算OC50，再继续到相同数值收敛判据或总2000步。各批API上限是11×任务数，总剩余上限8800次；已按用户“跑完1000并按200/300/300分设备”的指令记录在`authorizations/`。只发同任务工况、初态和新算FE摘要，不发源参考柔度或保存的能量标签。

## 1. 每台设备获取代码和数据

首次下载可用稀疏克隆，避免一次下载全部测试数据：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/hsksfje/topo.git topo_train1000
cd topo_train1000
git sparse-checkout set META64_train1000_distributed_v1 META64_datasets_v1/metadata
cd META64_train1000_distributed_v1
```

已有仓库则`git pull`。如果使用了稀疏检出，还要在仓库根目录执行：

```bash
git sparse-checkout add META64_train1000_distributed_v1 META64_datasets_v1/metadata
```

已有`meta64-fresh`环境，直接激活；没有则创建：

```bash
conda env create -f environment.yml
conda activate meta64-fresh
```

环境：Python3.11、NumPy1.26.4、SciPy1.12.0、Matplotlib3.8.4、OpenAI兼容SDK3.6.0（用于火山方舟，不代表换用OpenAI模型）。无需GPU。

准备整个train1000：

```bash
python ../META64_datasets_v1/fetch_data.py --subset train1000
python code/run.py --stage verify
```

三个训练分片约32.4 MB，解压约169 MB。之前准备过新增100的数据时，整个train1000已经解压；脚本会校验并复用，不重复下载。完整6300任务仍在数据目录，不会在本次额外运行。

## 2. 每台设备本地配置API密钥

Windows：

```powershell
copy .env.example .env.local
notepad .env.local
```

Linux用`cp .env.example .env.local`，再编辑。填写MODEL_API_KEY，保留授权的HTTPS接口MODEL_API_BASE_URL。密钥不进Git，不复制到实验结果。

**设备A只用A启动脚本，设备B只用B启动脚本。程序不会默认替你选择批次，避免两台误跑同一批。**

## 3. 设备A：运行中间300个

```bash
python run_deviceA300.py --stage preflight
python -u run_deviceA300.py
```

等价的完整命令：

```bash
python -u code/run.py --profile deviceA300 --stage run --execute-api --env-file .env.local
```

输出：`outputs/deviceA300_run01/`。

## 4. 设备B：运行最后300个

```bash
python run_deviceB300.py --stage preflight
python -u run_deviceB300.py
```

等价命令：

```bash
python -u code/run.py --profile deviceB300 --stage run --execute-api --env-file .env.local
```

输出：`outputs/deviceB300_run01/`。

## 5. 本机：运行前200个

本机后台队列使用原数据和现有本地凭据，运行位置在原项目：

```bash
cd /public/home/cairn/topo/InitSkill-Lab
/public/home/cairn/miniconda3/envs/topo-py3.11/bin/python -u portable/META64_train1000_distributed_v1/code/run.py --profile local200 --stage run --execute-api --env-file /public/home/cairn/topo/.env.zj --data-root /public/home/cairn/topo/InitSkill-Lab/data/META64_stratified_3000_v1 --output /public/home/cairn/topo/InitSkill-Lab/outputs/meta64_train1000_local200_v1/run01 --workers 5 --api-workers 2
```

本机启动器最多使用7个逻辑CPU；当前实际可用6个，保留1个，因此实际运行5个数值进程，以输出process.json为准。其他设备默认数值进程为可用逻辑CPU数减1，例如16个可用逻辑处理器用15个数值进程，各数值库1线程。

三设备同时运行时默认每台API并发2个，总计6个；这不限制数值阶段的CPU并行。API并发可通过`--api-workers`指定，数值并行可用`--workers`减少。若某设备还在运行前一批任务，先等该批结束或分配较少进程，避免两个队列各自占满CPU。

执行顺序为28种固定OC50→10种LLM OC50→全部续跑至收敛。前一阶段没结束时后一阶段计数为0是正常安排。固定初态失败、LLM无效和2000步仍未收敛均保留，不用Uniform回填。

## 6. 后台运行与实时查看

设备A的Windows PowerShell示例（激活环境后）：

```powershell
New-Item -ItemType Directory -Force outputs/deviceA300_run01 | Out-Null
Start-Process -FilePath (Get-Command python).Source -ArgumentList '-u','run_deviceA300.py' -WorkingDirectory (Get-Location).Path -RedirectStandardOutput 'outputs/deviceA300_run01/run.log' -RedirectStandardError 'outputs/deviceA300_run01/stderr.log'
```

设备B将其中所有`deviceA300`替换为`deviceB300`。

Windows PowerShell实时查看设备A：

```powershell
while ($true) { Clear-Host; Get-Content outputs/deviceA300_run01/progress.json; Start-Sleep -Seconds 10 }
```

Linux设备A：

```bash
mkdir -p outputs/deviceA300_run01
nohup python -u run_deviceA300.py > outputs/deviceA300_run01/run.log 2>&1 < /dev/null &
watch -n 10 'cat outputs/deviceA300_run01/progress.json'
```

本机进度：

```bash
watch -n 10 'cat /public/home/cairn/topo/InitSkill-Lab/outputs/meta64_train1000_local200_v1/run01/progress.json'
```

各批主输出下：`reports/OC50_results.csv`为50步结果；`convergence/reports/SPEED_QUALITY_zh.md`为速度和最终效果表，`convergence/reports/results.csv`为逐任务记录。API账本`budget_status.json`和`requests/`，环境及资源信息`environment.json`和`process.json`。

## 7. 暂停和恢复

在对应输出目录创建`PAUSE_REQUESTED`；等进程退出后删除根目录及`convergence/PAUSE_REQUESTED`两个标记，再运行同一启动命令。已完成结果、检查点和API回执会复用，不重发结果未知的请求。不要同时启动同一批次的两个进程。

例如设备A的Windows PowerShell：

```powershell
New-Item -ItemType File -Force outputs/deviceA300_run01/PAUSE_REQUESTED
# 等原进程退出，再执行下面几行
Remove-Item outputs/deviceA300_run01/PAUSE_REQUESTED -ErrorAction SilentlyContinue
Remove-Item outputs/deviceA300_run01/convergence/PAUSE_REQUESTED -ErrorAction SilentlyContinue
python -u run_deviceA300.py
```

出现API错误时检查`error.json`与请求回执，不删除回执强行重试。次数上限不是费用保证，真实费用以服务商账单为准。

## 8. 回传结果：A和B分开复制

各批完成后会生成`RETURN_READY.json`和逐文件SHA256返回清单。复制整个输出目录，不能只复制报告。

| 来源 | 当前设备输出 | 回到本机的位置 |
|---|---|---|
| 设备A | `outputs/deviceA300_run01/` | `/public/home/cairn/topo/InitSkill-Lab/outputs/meta64_train1000_deviceA300_remote_v1/run01/` |
| 设备B | `outputs/deviceB300_run01/` | `/public/home/cairn/topo/InitSkill-Lab/outputs/meta64_train1000_deviceB300_remote_v1/run01/` |

也可各自打ZIP：

```bash
python code/export_results.py --output outputs/deviceA300_run01 --zip exports/META64_deviceA300_results.zip
python code/export_results.py --output outputs/deviceB300_run01 --zip exports/META64_deviceB300_results.zip
```

到上限不等于收敛。三批均沿用相同SIMP/OC、物理体积校准、连续10步密度/柔度稳定判据、总2000步上限及原C50导向LLM提示。候选数量、初始化额外FE、粗网格前置优化成本按原协议记录。跨设备不要直接拿墙钟时间排序。

最早100任务中另一设备的50条已由用户确认跑完，结果待回传；另一个新增100任务仍由原设备负责。本包只接剩余800，五个清单合计正好1000，不会重跑前两批。
