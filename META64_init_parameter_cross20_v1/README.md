# 原Train20：PG1／L1_PG初始化 × 启动参数交叉实验

**直接运行 `python -u run.py`，只新增120条轨迹；不调用LLM，无需API密钥。** 本包自带20个物理输入、40份实际初态、120条历史对照，无需下载完整数据集或其他运行包。

## 实验回答的问题

同一个冻结初态在不同启动参数下，是否仍比**相同参数下的Uniform**得到更低柔度、更少OC步数？例如PG1+P39只与U+P39配对，不与U+P34混比。平均收益是否保持、退化是否增多、是否有初始化×策略交互，都由同任务配对报告回答。

这是一轮已见任务上的开发诊断，不预设PG1/L1_PG必然有效，不声称任意参数或每个任务都胜出。PG1是无LLM规则，L1_PG是历史LLM对PG2的有界修改，不能称作PG1的修改或新训练的收敛预测器。

## 任务与12格设计

精确复用原 `train100.jsonl` 的任务顺序，体积分数十档**各取前2条**，共20个不同物理组。没有按成绩挑选。`data/tasks.json`与`provenance/data_identity.json`保留任务、物理输入和源初态身份。

|初始场|P34默认：3/2.5/0.2|P39：3/3/0.2|P38：3/3/0.15|P20：2/3/0.3|
|---|---|---|---|---|
|Uniform U|历史参考|历史参考|历史参考|历史参考|
|PG1|历史参考|**新增20条**|**新增20条**|**新增20条**|
|L1_PG|历史参考|**新增20条**|**新增20条**|**新增20条**|

参数括号顺序为p/rmin/move。6个新增组合×20任务＝120条；历史参考为6格×20任务＝120条；完整分析矩阵12格×20＝240条。仅新增120条，不重复求解Uniform或默认策略，不重新生成任何LLM提案。

历史Uniform四策略来自已完成的参数60组实验；PG1/L1_PG默认策略来自原100任务初始化收敛结果。所有120条参考均已收敛。原参数返回包记录的源码包哈希与本地发布包不同；此前已逐项核对协议、输入和默认数值一致性，但不声称原设备逐文件源码完全一致，该边界保留在来源记录中。新包使用本地已验证、冻结的原SIMP/可行侧OC内核。

## 初态与滤波体积处理

`data/starts/<task>.npz`保存PG1、L1_PG实际第0步的设计变量x，逐元素复制，不用配方重生成。两种初态在所有策略中共享相同的原始场SHA256。

改变rmin会改变物理密度映射，非均匀初态的物理体积可能变化，因此流程为：

1. 读取并核对原始x，保存 `state_source.npz`。
2. 在启动滤波器下检查体积；超出1e-8时，按原参数实验的规则对设计密度做**整体加法平移＋上下界截断**，以1e-11为修正目标。已在容差内则原样保留。保存原始/修正后哈希、体积和最大修正量，不作OC更新、不调用LLM，该校准本身不增加FE。
3. 评价实际启动初态FE，前50次OC使用指定p/rmin/move。
4. 第50步后恢复共同 `p=3、rmin=2.5、move=0.2`，重建滤波；必要时再次校准体积，并重新FE评价，不能继续使用旧梯度。
5. 从共同阶段累计稳定窗口，收敛或总计2000次OC更新。启动50步包含在2000步内。

这比较的是“固定原始初态＋必要的滤波体积修正＋同一启动策略”。修正后的实际初态可能与原x有小幅差异，记录中会明确展示；不能把修正影响隐去。原始场从不覆盖，原实验目录不修改。

共同判据与原实验一致：连续10步最大设计密度变化≤0.01、逐步相对柔度变化≤1e-4、窗口及前一状态柔度相对极差≤5e-4、物理体积误差≤1e-8、FE残差<1e-6。最早第60步认证。2000步仍不满足判据保留为 `max_steps`，不当作收敛解，不回填Uniform。

## 下载与环境

从GitHub只下载本目录，避免拉取旧实验的大数据：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/hsksfje/topo.git topo_cross20
cd topo_cross20
git sparse-checkout set META64_init_parameter_cross20_v1
cd META64_init_parameter_cross20_v1
```

已有普通仓库可 `git pull` 后进入该目录；已有稀疏克隆可 `git sparse-checkout add META64_init_parameter_cross20_v1`。

```bash
conda env create -f environment.yml
conda activate meta64-cross20
```

若已有 `meta64-parameter60`、`meta64-convergence` 或 `meta64-fresh`，且Python3.11、NumPy1.26.4、SciPy1.12.0一致，可直接使用。也可在Python3.11虚拟环境中 `python -m pip install -r requirements.txt`。不需要PyTorch、OpenAI SDK、Matplotlib或.env配置。

## 核验与正式运行

```bash
python run.py --stage verify
python run.py --stage preflight
python -u run.py
```

`run`会自动执行尚未完成的预检。预检核对包内SHA256、20任务×3初态的默认C0、一个任务三种初态的默认OC50，以及全部120个启动体积修正，不重新请求LLM。预检FE单独记账，不混入正式120条轨迹。

默认可用逻辑核减一的多进程、BLAS/OMP/MKL各单线程；Linux还扣除一个CPU亲和性核。可减小并行度：

```bash
python -u run.py --workers 4
```

默认输出 `outputs/train20_init_parameter_cross_run01/`。各轨迹如 `tasks/<task>/PG1__P39/`，保留原始场、启动修正、50步切换前后、检查点、历史、最终场和实际FE尝试；重复启动同一输出会被跨进程锁阻止。

先做独立烟雾检查也可以（1个真实任务×6新组合，各最多70步）：

```bash
python -u run.py --stage smoke --workers 1
```

输出 `outputs/smoke_cross_validation/`。此模式标记 `validation_only=true`，70步上限不是正式收敛实验。它不导入正式120条会话。

## 查看、暂停、恢复

```bash
python run.py --stage status
python run.py --stage pause
```

在途任务在下一次OC边界安全保存后退出。待原进程退出再恢复：

```bash
python -u run.py --stage resume
```

已结算的收敛、上限或失败记录不重跑。恢复沿用同一包、协议、任务、原始初态及检查点；可以改变worker数量。每10步及两个体积修正边界原子保存，正常暂停立即保存；硬中断可能重算最后未保存区间，已记录FE尝试仍保留。

显式输出目录通过 `--output`指定；查看/暂停/恢复时使用同一目录。不要修改包内protocol或源码后续跑旧会话，改协议要开新版本。Windows/macOS锁和多进程已做兼容实现，发布前实际数值验证环境为Linux。

## 报告怎样读

`reports/REPORT_zh.md`自动给出12格表。核心指标：

```text
I(init,strategy) = 100 * (1 - C_init,strategy / C_U,strategy)
S(init,strategy) = 100 * (1 - N_init,strategy / N_U,strategy)
```

正数表示相同策略下初始化更好/更快。逐任务配对后再汇总，不比较不同工况的绝对柔度均值。报告包含：

- 平均/中位收益、最差收益、改善与退化>1%的任务数。
- 更快且改善>1%、更快且接近±1%、更快但变差>1%，以及严格双赢（质量改善>1%且步数减少≥10%）。
- 初始化收益随策略的变化：同一组共同收敛任务上，`I(init,strategy)-I(init,P34)`，单位为百分点。它不是两组柔度的直接相对改善。
- 失败、未收敛、有效配对数和实际新增FE。均值只用双方收敛配对，失败不填Uniform。

`paired_task_results.csv`保存240格的任务明细（120新增＋120参考）；`same_strategy_summary.csv`保存12格汇总；`initialization_strategy_interaction.csv`保存6项初始化收益变化。处理中未完成的新增格标记pending。

如果四种策略下平均收益都为正，只能说在**这四种策略和这20条开发任务**上平均收益一致，不能说每个任务、任意参数都胜出。看到结果后选定组合，需要在独立任务确认。L1_PG目前也未被预设为“更好”的初态。

历史参考此次新增计算成本为0，但其原始生成与求解成本不为0；LLM生成、初始化前置计算及跨设备墙钟时间不混为主轨迹加速。新实验两次体积修正和FE重算的成本均保留。

## 回传结果

全部120条结算后自动生成 `COMPLETION.json`、`return_manifest.json`、`RETURN_READY.json`。完成仅表示全部结算，不表示全部收敛。封存后不再改写报告。

```bash
python run.py --stage verify-return
python run.py --stage export
```

生成 `outputs/train20_init_parameter_cross_run01_return.zip` 及 `.sha256`。包含实际执行源码快照、冻结输出配置、120条历史参考标量、120条新增轨迹及完整报告，不依赖原机器路径。源码快照按原包清单再次校验，便于回传后核对真实代码。请原样回传到原设备：

```text
InitSkill-Lab/outputs/meta64_init_parameter_cross20_remote_v1/
```

也可复制整个输出目录。独立校验：

```bash
python code/export_results.py --output /path/to/train20_init_parameter_cross_run01 --verify-only
```

## 复现与来源

`code/physics.py`、`code/solver.py`与原参数实验物理/OC内核逐字一致；新增内容只涉及冻结初态读取、启动体积校准、交叉调度和同策略配对报告。`storage.py`显式UTF-8写JSON，以兼容Windows。来源与参考哈希见 `provenance/`。

物理输入来自Meta-neural Topology Optimization，作者Igor Kuszczak、Miguel Bessa、Gawel Kus、Federico Bosi；[Zenodo21893871](https://zenodo.org/records/21893871)，[DOI10.5281/zenodo.21893871](https://doi.org/10.5281/zenodo.21893871)，原数据[CC BY4.0](https://creativecommons.org/licenses/by/4.0/)。本包使用既有坐标转换和分层选择，并增加本项目生成的PG1/L1_PG初态，不代表原作者认可。

```bash
python -m unittest discover -s tests -v
```

发布前验证见 `validation/VALIDATION_zh.md`。本机只进行工程和代表轨迹验证，不自动启动正式120条队列。
