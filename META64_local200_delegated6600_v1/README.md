# 本机200任务：委托6600条OC50记录续跑至收敛

这是可独立搬到另一台设备的离线续跑包。**不需要API密钥、原项目或GPU，不重新生成初始化，不重跑前50步。**

## 本次分工

| 运行位置 | 范围 | 记录数 |
|---|---|---:|
| 本包，另一台设备 | 本机200任务的全部28种无LLM初始化（包含Uniform） | 5600 |
| 本包，另一台设备 | 同批任务中，冻结时前1000条有效完成OC50的LLM初始化 | 1000 |
| 原服务器 | 同批200任务中其余LLM初始化，完成OC50后续跑 | 1000 |

本包共6600条任务—方法记录，不是6600个独立工况。LLM按原冻结任务清单、方法顺序选择已有效完成50步的前1000条，涉及103个任务；不按性能挑选。固定5600条中有8条原始失败记录，原样保留，续跑时标为source_unavailable；其余6592条具有有效50步检查点。无效LLM不占本包1000条有效名额，留在服务器互补清单中处理。

完整分工见`provenance/delegation.json`：远端6600与本机1000无交集，并集覆盖本批200×38=7600条。不同方法的样本范围不同；固定方法覆盖200任务，LLM是已完成的子集，不能直接把不同样本范围的均值当作公平排名。正式比较应在共同配对任务上进行。

## 1. 更新GitHub代码与数据

已有topo仓库，在仓库根目录：

```bash
git pull --ff-only
```

如果之前使用了稀疏克隆，再执行：

```bash
git sparse-checkout add META64_local200_delegated6600_v1
```

进入运行包：

```bash
cd META64_local200_delegated6600_v1
```

没有仓库则首次下载：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/hsksfje/topo.git topo_convergence6600
cd topo_convergence6600
git sparse-checkout set META64_local200_delegated6600_v1
cd META64_local200_delegated6600_v1
```

GitHub提供`data_shards/*.zip`分片，每片小于GitHub单文件限制，逐片SHA256写在`DATA_SHARDS.json`。这些分片只含本次输入和已有50步检查点，不含密钥或API请求原文。准备数据：

```bash
python prepare_data.py
```

已有正确数据会校验并复用。若没有通过Git下载分片，脚本会按目录中的固定GitHub地址补下载。没有网络时，复制整个包（包含data_shards），同一命令可离线解压。

## 2. Conda环境

已有`meta64-convergence`环境可直接激活；已有`meta64-fresh`且以下核心版本一致也可使用，无需再装一份。首次创建：

```bash
conda env create -f environment.yml
conda activate meta64-convergence
```

需要Python3.11、NumPy1.26.4、SciPy1.12.0、Matplotlib3.8.4；详细版本在`environment.yml`和`requirements.txt`。无需OpenAI SDK、CUDA或API密钥。

如果系统还没有可运行的Python，先创建并激活Conda环境，再执行上面的`python prepare_data.py`。

## 3. 校验、预检和运行

在运行包根目录、已激活环境的终端执行：

```bash
python code/run.py --stage verify
python code/run.py --stage preflight
python -u code/run.py --stage run
```

`verify`核对完整代码/数据SHA256；`preflight`分别复算Uniform、LLM和粗网格初始化的C50，再续跑5步，并验证Uniform从50续跑5步与从头运行55步一致。没有任何付费调用。

默认检测可用逻辑CPU，保留1个，其余单线程多进程并行；例如16个可用逻辑CPU使用15个进程。需要降低CPU/内存占用可指定：

```bash
python -u code/run.py --stage run --workers 8
```

不要同时向同一输出目录启动多个队列。若设备还在跑其他批次，等其结束或主动降低本包workers，避免两个队列各自占满CPU。

## 4. 实时查看

输出固定在`outputs/delegated6600_run01/`。

Windows PowerShell，另开终端进入运行包目录：

```powershell
while ($true) { Clear-Host; Get-Content outputs/delegated6600_run01/progress.json; Start-Sleep -Seconds 10 }
```

Linux：

```bash
watch -n 10 'cat outputs/delegated6600_run01/progress.json'
```

输出中的`planned`应为6600。优先安排200条Uniform参考，再安排其余选定记录。进度文件的`settled`包括收敛、达到上限和源初态不可用；不能把全部结算等同于全部收敛。

| 文件 | 内容 |
|---|---|
| `outputs/delegated6600_run01/progress.json` | 6600条结算进度、状态和活动轨迹 |
| `outputs/delegated6600_run01/reports/REPORT_zh.md` | 数值收敛主报告 |
| `outputs/delegated6600_run01/reports/SPEED_QUALITY_zh.md` | 收敛速度与最终质量，包括更快且接近、停得更慢但改善>1%/>5% |
| `outputs/delegated6600_run01/reports/results.csv` | 逐任务/方法的停止步数、C50、最终柔度和相对Uniform指标 |
| `outputs/delegated6600_run01/tasks/` | 完整续跑历史、检查点和终态数组 |
| `outputs/delegated6600_run01/process.json` | 实际CPU与进程数 |

## 5. 暂停与恢复

在输出目录新建`PAUSE_REQUESTED`文件。程序在当前FE后保存检查点并退出。等进程退出后删除标记，再运行原命令；已结算记录不重算，未结束记录从最近检查点继续。

PowerShell：

```powershell
New-Item -ItemType File -Force outputs/delegated6600_run01/PAUSE_REQUESTED
# 等原进程退出后执行
Remove-Item outputs/delegated6600_run01/PAUSE_REQUESTED
python -u code/run.py --stage run
```

Linux分别使用`touch`和`rm`处理相同标记。不要删除原始结果或修改冻结的data/config.json。

## 6. 结束后打包回传

完成后自动生成`RETURN_READY.json`和`return_manifest.json`。使用下面的命令打包整个输出，避免逐个复制小文件漏传：

```bash
python code/export_results.py --zip exports/META64_delegated6600_results.zip
```

同时生成ZIP的`.sha256`。将ZIP和校验文件一起带回原服务器：

```text
/public/home/cairn/topo/InitSkill-Lab/outputs/meta64_local200_delegated6600_remote_v1/exports/
```

之后由原服务器校验解压，最终原始输出接收位置为：

```text
/public/home/cairn/topo/InitSkill-Lab/outputs/meta64_local200_delegated6600_remote_v1/run01/
```

不要只传Markdown/CSV，也不要混到最早100任务或另一设备300任务的目录中。回传后与本机剩余1000条LLM合并，才能汇总本批200任务全部7600条；远端结果不完整时不会宣称整批完成。

## 7. 数值协议与代码位置

- `code/physics.py`、`code/solver.py`保持此前冻结的FE和可行侧SIMP/OC，不改变材料、滤波、约束或优化器。
- `code/run.py`只调度冻结清单中的6600条，从`state_0050.npz`继续，不重新调用LLM。
- `data/inputs/`含200个任务的物理输入；`data/trajectories/`含对应原0—50步历史、结果和有效检查点。
- `code/analyze.py`生成速度—质量报告；`code/export_results.py`完成后封存回传。
- `MANIFEST.json`、`MANIFEST.sha256`提供代码和数据完整性；`provenance/delegation.json`提供分工及原文件哈希。

仍为总细网格OC最多2000步（包含前50步），连续10次更新满足密度变化≤0.01、单步柔度相对变化≤1e-4、窗口柔度相对极差≤5e-4、物理体积误差≤1e-8、FE残差<1e-6。达到2000步仍不满足判据标为max_steps，不称为收敛。最早可认证停止点为60步。COARSE/OC5的前置成本不能当作免费；跨设备不直接比较墙钟时间。

## 8. 回传后在原服务器合并全部7600条

远端6600条和本机剩余1000条都完成、且ZIP已经校验解压后，在原服务器项目目录执行：

```bash
cd /public/home/cairn/topo/InitSkill-Lab
/public/home/cairn/miniconda3/envs/topo-py3.11/bin/python portable/META64_local200_delegated6600_v1/code/merge_with_local1000.py --local outputs/meta64_train1000_local200_v1/run01/convergence --remote outputs/meta64_local200_delegated6600_remote_v1/run01 --output outputs/meta64_train1000_local200_merged_v1/run01
```

该命令核验远端返回哈希、双方6600/1000清单、协议和任务方法互斥，再生成200任务完整7600条报告：`outputs/meta64_train1000_local200_merged_v1/run01/reports/SPEED_QUALITY_zh.md`。只分析，不重新计算。
