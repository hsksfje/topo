# META64 Train1000：保留23种初始化的缺口补跑包

这是**补齐已有结果**的独立运行包，不重跑整个1000×23网格。代码、这批所需的272个工况物理输入、可恢复初态、Uniform比较参考和缺口计划均已随包提供。无需下载此前全部3000条数据，也无需复制本机其他目录。

## 1. 本批运行内容

|内容|组合数|处理方式|
|---|---:|---|
|有历史50步状态|109|核对保存状态的柔度，再从第50步续跑|
|有历史初态|33|保留初态，从第0步重新优化|
|固定方法缺口|495|使用原生成器重新生成初态并优化|
|缺少可恢复初态的LLM组合|49|默认标记blocked；可单独用Flash4.1 low生成新提案|
|合计|686|272个不同训练任务中的指定缺口|

前637个组合不需要API。49个LLM缺口为A1 13个、A3 19个、B1 17个；A3有15个额外A1父提案需要重新生成，因此顺利时共64次新调用，不计重试。生成入口默认最多200次请求，重试也计入上限。新提案单独标记`new_Flash_low`，不能称为历史Pro初态恢复。

固定方法共需补523个；LLM共需补163个，其中114个已有可恢复历史初态。U、HC1_B020、RAND_IID_S03已各有1000条有效轨迹，本包不重复计算它们。逐方法数量见`coverage_by_method.csv`，逐任务计划见`plan.json`。

## 2. 下载更新和环境

已clone过仓库，在仓库根目录执行：

```bash
git pull --ff-only
cd META64_retained23_backfill_v1
```

首次下载：

```bash
git clone https://github.com/hsksfje/topo.git
cd topo/META64_retained23_backfill_v1
```

建议新建环境，Windows Anaconda Prompt与Linux终端均可：

```bash
conda env create -f environment.yml
conda activate meta64-backfill
python verify.py
python run.py --stage prepare
```

已有等价环境可直接激活。依赖为Python 3.11、NumPy 1.26.4、SciPy 1.12.0、Matplotlib 3.8.4；只有可选付费生成需要OpenAI Python SDK 3.6.0。环境名不同不影响运行。不使用GPU，无需CUDA/PyTorch。

`payload.zip`约5.4 MB，首次运行自动解压并逐文件校验到`data/`。输入不含源最终拓扑、参考柔度或保存的源应变能标签；Uniform参考只用于本机结果比较，不进入LLM提示。

## 3. 多CPU运行：保留原2000步上限

```bash
python -u run.py
```

默认按可用逻辑CPU数量减1启动进程；Linux同时考虑CPU亲和性及标准cgroup v2 CPU配额，Windows读取进程可用CPU掩码。**16逻辑处理器的i7-10700通常启动15个进程**，实际数量显示在`progress.json`。每个进程的BLAS/OpenMP均固定1线程，避免多层线程过量。按单个“任务×方法”调度，某个方法慢不会占住整个任务的其他方法。

可以手动减少并发，不能超过自动检测的留1核上限：

```bash
python -u run.py --workers 15
```

默认与历史数据一致：总细网格OC上限2000，达到收敛立即停止，同时输出50步与500步指标。500步仍未真正收敛的轨迹，不能计入500步严格双赢。

如果本次**只跑到500步**，使用单独的输出目录：

```bash
python -u run.py --max-step 500
```

两者选一个即可。不要同时在两个终端启动同一输出。不同上限属于不同运行配置，不能对同一输出目录改上限。需要从500继续至2000时，应先回传或另行做续跑适配；默认2000运行已能给出500步结果，避免重复计算。

代码沿用原SIMP/OC、`penal=3`、`rmin=2.5`、`move=0.2`、64×64网格。连续10次更新满足最大密度变化≤0.01、逐步柔度相对变化≤1e-4、11个柔度状态相对极差≤5e-4，体积误差≤1e-8且FE残差<1e-6。为匹配历史从50步续跑的认证口径，最早认证收敛步数为60。粗网格初始化额外FE单独记录。

## 4. 49个LLM缺失初态：可选Flash4.1 low新生成

仅运行`run.py`会完成可运行部分，并将这49个缺口标记`blocked_missing_LLM`，不暗中使用Uniform替代，不产生API费用。退出码2表示仍有失败或缺初态，详情见报告。

如要补齐这49个，在本设备创建**仓库以外**的凭据文件，例如`C:/private/ark.env`或`/home/yourname/private/ark.env`，内容为：

```dotenv
MODEL_API_KEY=你在本设备配置的密钥
MODEL_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
```

不要把凭据提交GitHub或放入结果回传包。已有相同格式凭据文件可直接使用。

Linux示例：

```bash
python -u generate_missing.py --execute-api --env-file /home/yourname/private/ark.env --api-workers 4
python -u run.py --retry-failed
```

Windows Anaconda Prompt示例：

```bat
python -u generate_missing.py --execute-api --env-file C:/private/ark.env --api-workers 4
python -u run.py --retry-failed
```

若选择500步，最后一条需改成`python -u run.py --max-step 500 --retry-failed`。也可以先完成生成，再首次启动数值计算。生成过程和数值进程入口分开；API并发默认4，数值并发仍按CPU数量自动决定。

也提供一条命令：先生成缺失提案，再启动全部可运行的CPU任务。例如Windows：

```bat
python -u start_all.py --execute-api --env-file C:/private/ark.env
```

Linux将凭据路径换成本机路径即可。`start_all.py`同样支持`--max-step 500`、`--workers 15`和`--resume`。若部分API仍失败，也会继续计算其他已具备初态的任务，最后返回未解决状态。

模型沿用已配置的`deepseek-v4-1-flash-261017`，明确发送`reasoning_effort=low`和`thinking.type=enabled`。429、408、409、5xx及连接/超时错误退避等待后自动重试，间隔60秒起、逐渐增至900秒，并尊重更长的Retry-After；每个请求最多8次重试，全批最多200次尝试。永久参数/鉴权错误停止对应任务；不明结算状态不会盲目重复收费。修复后可重跑同一命令；保留请求回执。不要手改已冻结模型或删除回执。

A3的父A1由同一轮Flash重新生成，只作为依赖，不擅自覆盖主表已有效的A1。B1仅使用本任务新计算的Uniform初态FE摘要；不会读取Uniform最终柔度或参考标签。

## 5. 实时查看、暂停和恢复

另开一个终端并激活环境，进入本包目录：

```bash
python monitor.py
```

只看一次：

```bash
python monitor.py --once
```

500步运行：

```bash
python monitor.py --output outputs/backfill_500_run01
```

监控每10秒刷新，显示结算数、收敛/到上限/失败/缺初态数量、进程数和当前运行步数。

请求安全暂停：

```bash
python -c "from pathlib import Path; Path('outputs/backfill_2000_run01/PAUSE_REQUESTED').touch()"
```

当前工况保存检查点后退出，控制器停止添加新任务。重新启动：

```bash
python -u run.py --resume --retry-failed
```

500步运行请将暂停路径改为`backfill_500_run01`，恢复命令同时加`--max-step 500`。正常运行每10步保存检查点；意外强制退出最多重做最近未保存的约10步。成功结果不会重跑；失败结果默认保留，`--retry-failed`会归档旧错误并尝试从检查点恢复。已恢复初态的blocked组合会自动重试。

针对此前Windows文件替换失败，JSON/NPZ写入对PermissionError已加入有限退避重试；仍失败会明确记录，不会把缺失文件当成成功。避免将实时输出放在云盘同步目录，运行时不要用Excel等程序占用结果文件。

## 6. 输出和回传

默认输出：

```text
outputs/backfill_2000_run01/
  config.json
  progress.json
  tasks/<任务ID>/<方法>/
    history.json
    checkpoint.npz
    state_0050.npz         # 从0跑的任务新生成
    state_0500.npz         # 到达500时保存
    state_final.npz
    result.json
  reports/results.csv
  reports/REPORT_zh.md
generated/                # 可选新Flash提案、API回执与用量
```

从历史50步状态续跑的任务，其原50步状态在本包`data/seeds/`中，运行输出保留原0—50历史前缀和新增后续轨迹。报告只统计这批补跑样本，不代表全1000任务成绩。最终合并会优先填原来的缺口，不按柔度挑选重复运行中的最好值。

停止计算后打包回传：

```bash
python run.py --stage export
```

500步运行：

```bash
python run.py --stage export --max-step 500
```

将`exports/`中新生成的**`.zip`和`.zip.sha256`两个文件一起复制**到原设备：

```text
InitSkill-Research/incoming/META64_retained23_backfill_v1/
```

压缩包包含本次新轨迹、检查点、报告、配置、可选Flash请求回执、逐文件SHA256清单；不打包凭据。无需复制整个环境或原数据集。回传后由原设备核验、按任务ID/方法与模型来源合并。

## 7. 文件索引与验证

|文件|用途|
|---|---|
|`run.py`|任务×方法多进程、断点续跑、50步/500步/最终结果|
|`start_all.py`|一条命令串联可选付费生成与多进程数值计算|
|`generate_missing.py`|可选Flash-low付费生成，仅缺少历史初态的49个组合|
|`monitor.py`|跨平台实时进度|
|`export_backfill.py`|轨迹与回执归档、校验清单|
|`code/`|原SIMP/OC、固定生成器、LLM解码器及重试客户端|
|`plan.json`|686个缺口、输入、初态来源及Uniform参考|
|`payload.zip`、`PAYLOAD_HASHES.json`|物理输入与历史初态、逐文件校验|
|`BUILD_AUDIT.json`|构包覆盖和缺失初态清单|
|`VALIDATION.json`|本次离线与数值检查记录；不是完整补跑成绩|

`--max-step 52`仅用于工程验证，不能用作正式收敛结果。完整补跑尚未在本机启动；本机仅验证打包与少量数值接口。包内不含API密钥。
