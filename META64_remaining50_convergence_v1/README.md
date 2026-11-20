# META64 剩余50任务：从已有OC50初态继续到收敛

本文件夹可独立复制到另一台设备运行，不需要原`InitSkill-Lab`项目、不需要API密钥、不需要重新调用LLM，也不需要GPU。建议使用64位Windows（Anaconda Prompt）或Linux。

## 1. 这50个任务是什么

- 原始META64 Train100有100个任务，每个体积分数档10个。
- 本机正在跑的50个任务：原冻结清单中每档前5个。
- **本运行包是另外50个任务，十档各5个，与本机队列没有任务重叠。** ID和分区核验见`provenance/task_partition.json`。
- 包含38种初始化：原17种固定方法（已包含Uniform）、10种LLM方法、11种新增无LLM方法，共1900条方法记录。已有生成无效或初始FE失败的记录保留为`source_unavailable`，不回填Uniform、不重新生成。
- 1900条记录中，**1891条具有有效OC50检查点，9条为先前无效提案**；这9条会直接结算为源初态不可用，不会发起新的付费调用。
- 有效记录已提供设计密度`state_0050.npz`和0—50步历史。这里从**第50步继续**；没有重新生成初态，也不会重跑前50步。

## 2. Conda环境：需要哪些包

已在Linux打包机器上以独立迁移目录验证以下版本；Windows已提供兼容代码和命令，尚未在你的设备上实际执行，请先运行预检：

| 包 | 版本 | 用途 |
|---|---|---|
| Python | 3.11 | 主程序及多进程 |
| NumPy | 1.26.4 | 数组、密度场 |
| SciPy | 1.12.0 | 稀疏有限元求解、滤波 |
| Matplotlib | 3.8.4 | 自动报告图表 |
| pip | Conda自动选择 | 可选安装工具，运行本身不依赖它 |

`environment.yml`固定了上述主要版本。打开**Anaconda Prompt**（Windows）或已经安装Conda的终端，先进入本文件夹根目录，再执行：

```bash
conda env create -f environment.yml
conda activate meta64-convergence
python -c "import sys,numpy,scipy,matplotlib; print(sys.version); print(numpy.__version__,scipy.__version__,matplotlib.__version__)"
```

如果不想用Conda，也可在已有Python 3.11虚拟环境中运行`python -m pip install -r requirements.txt`。两种安装方式任选其一，不必重复安装。安装环境需要联网，环境搭好后实验可离线运行。

## 3. 下载或搬到另一台电脑

可以直接复制整个`META64_remaining50_convergence_v1`文件夹；**必须包含`data/`，只复制代码不能续跑。** 所有运行路径都是包内相对路径，复制后不需要修改原机器路径。包内`.gitattributes`禁止Git自动转换换行，保证Windows/Linux校验同一SHA256；请保留此文件。

也可以从用户指定的GitHub仓库下载。仓库中的目录是`META64_remaining50_convergence_v1/`：

```bash
git clone https://github.com/hsksfje/topo.git
cd topo/META64_remaining50_convergence_v1
```

如果没有Git，可使用GitHub的`Code → Download ZIP`，解压后进入该目录。不要把嵌套的外层仓库目录误认为运行包根目录；根目录应能看到`environment.yml`、`MANIFEST.json`和`code/`。

## 4. 先校验，再运行

激活上面的环境，在包根目录执行：

```bash
python code/run.py --stage verify
python code/run.py --stage preflight
python -u code/run.py --stage run
```

- `verify`：验证包内SHA256，检查代码和数据是否完整；不做FE、不调用API。
- `preflight`：重算三类方法的C50，并核对Uniform从50步再跑5步与从头跑55步一致；只做本地有限元计算。结果见`outputs/remaining50_run01/preflight.json`。
- `run`：自动执行必要预检、继续剩余轨迹，并自动更新报告。已经做过的预检不会重复运行。

### 你的i7设备：留一个逻辑CPU，其余并行

**默认自动检测当前进程可用的逻辑CPU数，并使用N−1个计算进程；NumPy/OpenBLAS/MKL/OMP各设置为1线程。** 例如系统识别到16个可用逻辑处理器，就启动15个计算进程；以启动日志打印的实际检测值为准。Windows/Linux同时会从本进程及子进程的CPU亲和性中排除一个逻辑CPU。这里预留的是一个逻辑处理器，并不保证独占保留一个物理核心，也不限制其他软件的CPU使用。

默认运行不需要指定CPU数：

```bash
python -u code/run.py --stage run
```

需要降低占用时，可手动指定进程数，例如：

```bash
python -u code/run.py --stage run --workers 8
```

手动指定也不能超过检测到的N−1。只有一个可用CPU时会用一个进程。多进程会增加内存需求；出现内存压力时可安全暂停，再用较少`--workers`恢复。无需GPU或CUDA环境。

### 可选后台运行

Windows PowerShell（先激活Conda环境）：

```powershell
New-Item -ItemType Directory -Force outputs/remaining50_run01 | Out-Null
Start-Process -FilePath (Get-Command python).Source -ArgumentList '-u','code/run.py','--stage','run' -WorkingDirectory (Get-Location).Path -RedirectStandardOutput 'outputs/remaining50_run01/run.log' -RedirectStandardError 'outputs/remaining50_run01/stderr.log'
```

Linux：

```bash
mkdir -p outputs/remaining50_run01
nohup python -u code/run.py --stage run > outputs/remaining50_run01/run.log 2>&1 < /dev/null &
```

同一输出目录不要重复启动；程序有独占运行锁。后台运行后可关闭该终端；电脑休眠、关机仍会中断计算，可按后面的方式恢复。

## 5. 实时查看进度和结果

Windows PowerShell：

```powershell
while ($true) { Clear-Host; Get-Content outputs/remaining50_run01/progress.json; Start-Sleep -Seconds 10 }
```

Linux：

```bash
watch -n 10 'cat outputs/remaining50_run01/progress.json'
```

主要输出：

| 路径 | 内容 |
|---|---|
| `outputs/remaining50_run01/progress.json` | 1900条记录的结算数、收敛数、上限未收敛数及当前步数 |
| `outputs/remaining50_run01/reports/REPORT_zh.md` | 数值计算主报告 |
| `outputs/remaining50_run01/reports/SPEED_QUALITY_zh.md` | 速度与最终质量对照，保留所有新增分类列 |
| `outputs/remaining50_run01/reports/results.csv` | 每任务每方法的C50、终点柔度、停止步数及共同质量目标步数 |
| `outputs/remaining50_run01/reports/speed_quality_analysis.csv` | “更快且改善>1%”“更快且差异≤1%”“更快但变差>1%”“更慢但改善>1%”“更慢但改善>5%”统计 |
| `outputs/remaining50_run01/tasks/<任务>/<方法>/history.json` | 原0—50步加上本轮续跑的全部历史 |
| `outputs/remaining50_run01/tasks/<任务>/<方法>/state_final.npz` | 终态设计密度、物理密度及位移 |
| `outputs/remaining50_run01/process.json` | 实际进程数、可用CPU与预留CPU |

报告由主程序自动刷新，无需另开分析进程；只有完成的结果才进入汇总。要在计算结束后单独重建报告：

```bash
python code/run.py --stage report
python code/analyze.py
```

## 6. 暂停、中断恢复

**安全暂停**：在输出目录新建`PAUSE_REQUESTED`文件。程序在当前一次FE结束后保存检查点并停止。

Windows PowerShell：

```powershell
New-Item -ItemType File -Force outputs/remaining50_run01/PAUSE_REQUESTED
```

Linux：

```bash
touch outputs/remaining50_run01/PAUSE_REQUESTED
```

等程序退出后，删除这个标记，再运行原命令：

Windows PowerShell：

```powershell
Remove-Item outputs/remaining50_run01/PAUSE_REQUESTED
python -u code/run.py --stage run
```

Linux：

```bash
rm outputs/remaining50_run01/PAUSE_REQUESTED
python -u code/run.py --stage run
```

已结算记录不会重跑；未结束轨迹从最近检查点恢复。意外关机可能需要重算最近不足10步。FE计数保留已尝试次数，恢复时重新评价检查点也计入成本。不要修改包内代码/数据/配置后混续旧输出；SHA256会拒绝身份不一致的续跑。

如果要把已开始的计算再搬家，连同`outputs/`一起复制，在新路径运行同一命令即可；包内身份校验不绑定原机器绝对路径。

## 7. 数值协议及结果解读

与本机50任务保持相同：二维64×64、SIMP p=3、rmin=2.5、move=0.2、体积可行侧OC更新。连续10次更新需同时满足：

- 最大设计密度变化≤0.01；
- 单步柔度相对变化≤1e-4；
- 最近10次更新及前一状态的柔度相对极差≤5e-4；
- 物理体积误差≤1e-8、FE残差<1e-6。

**总细网格OC上限2000步，包含之前的50步，最多再更新1950步。** 到上限仍未满足判据标记`max_steps`，不能称为已收敛；历史0—50步未保存逐步密度变化，因此最早可认证停止点为60步。

“更慢/更快”按同任务停止步数严格比较；质量按双方各自收敛柔度比较。“更慢但改善>1%”包含“>5%”的任务；“效果接近”按柔度差异不超过±1%，不是统计等效或图像相同。

COARSE30/80已有31/81次粗FE前置计算，OC5已有前置5次OC；它们不能作为免费初态。这里保存的新增FE/时间是续跑成本，不等于完整端到端成本。不同CPU、BLAS库可能有末位数值及收敛步数差异；因此先通过预检，比较时保留环境信息。结果为原训练100任务上的探索，不是独立盲测。

## 8. 运行代码分别在哪里

| 文件 | 用途 |
|---|---|
| `code/run.py` | 迁移校验、预检、并行续跑、检查点、暂停恢复、主报告 |
| `code/physics.py` | 原冻结的网格、密度滤波、Q4线弹性FE及灵敏度实现，逐字复制 |
| `code/solver.py` | 从原冻结模块逐字提取的可行侧OC类，算法不改 |
| `code/storage.py` | 原冻结的原子JSON/NPZ写入与SHA256工具 |
| `code/analyze.py` | 速度—质量、共同目标、Pareto及图表分析，不新增FE/API |
| `data/config.json` | 另外50任务的冻结清单、38方法及求解协议 |
| `data/inputs/` | 自包含物理输入，已解除外部符号链接 |
| `data/trajectories/` | 原第50步检查点、历史和原结果 |
| `provenance/` | 任务互斥核验、原代码身份、源文件哈希及打包说明 |
| `validation/` | 打包机器上的预检及单轨迹续跑验证记录 |
| `MANIFEST.json`、`MANIFEST.sha256` | 包内文件与清单自身的完整性校验 |

原项目对应入口是`InitSkill-Lab/scripts/run_meta64_convergence50.py`与`analyze_meta64_convergence50.py`；本包已去掉原机绝对路径和等待本机试验的依赖，并将默认资源策略改为预留一个逻辑CPU。数学求解器和收敛判据保持一致。

## 9. 跑完后把结果带回第一台设备

**另一台设备的输出固定为本包下`outputs/remaining50_run01/`。** 不指定`--output`即可。程序完成后会自动生成：

- `outputs/remaining50_run01/return_manifest.json`：返回文件逐个SHA256；
- `outputs/remaining50_run01/RETURN_READY.json`：表示输出已完整封存，可以复制；
- `outputs/remaining50_run01/environment.json`：实际Python、NumPy、SciPy及Matplotlib版本和操作系统。

看到`RETURN_READY.json`后，**复制整个`remaining50_run01`目录**，保留原目录层次，包括`tasks/`中的历史、终态及检查点。不要只复制Markdown/CSV。保留失败和`max_steps`记录，避免回传时只挑好结果。

也可以打成一个ZIP方便传输（不会重新计算）：

```bash
python code/export_results.py --zip exports/META64_remaining50_results.zip
```

生成`exports/META64_remaining50_results.zip`及其`.sha256`校验文件；解压得到`remaining50_run01/`。若程序已结束但因中断未生成返回清单，可运行`python code/export_results.py`补生成。程序未完成时此命令会拒绝封存，避免把部分输出误认为完整50任务。

**带回当前第一台设备后，目录放到：**

```text
/public/home/cairn/topo/InitSkill-Lab/outputs/meta64_convergence_remaining50_remote_v1/run01/
```

该`run01/`下应该直接看到`config.json`、`RETURN_READY.json`、`reports/`和`tasks/`，不要再嵌套一层`remaining50_run01/`。这个接收目录与当前本机50任务目录分开，不覆盖本机结果。

等本机50任务也完成后，在第一台设备执行：

```bash
cd /public/home/cairn/topo/InitSkill-Lab
/public/home/cairn/miniconda3/envs/topo-py3.11/bin/python portable/META64_remaining50_convergence_v1/code/merge_results.py --local outputs/meta64_convergence50_v1/run01 --remote outputs/meta64_convergence_remaining50_remote_v1/run01 --output outputs/meta64_convergence100_v1/run01
```

合并脚本会核验返回文件哈希、双方任务互斥且总数100、38种方法及数值协议一致，再合并3800条方法记录；生成：

```text
outputs/meta64_convergence100_v1/run01/reports/SPEED_QUALITY_zh.md
outputs/meta64_convergence100_v1/run01/reports/results.csv
outputs/meta64_convergence100_v1/run01/reports/speed_quality_analysis.csv
outputs/meta64_convergence100_v1/run01/merge_audit.json
```

合并目录是分析汇总，原始轨迹仍分别保存在两个50任务目录。它不会重新做FE或调用LLM，也不会把上限未收敛记录算作收敛。若只想提前检查部分结果，可显式加`--allow-partial`，此时报告和审计会标为部分合并。
