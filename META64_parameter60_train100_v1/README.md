# 原Train100：Uniform启动参数60组合收敛实验

本包对应**已经完成38种初始化收敛比较的最早100个任务**，即 `META64_stratified_3000_v1/manifests/train100.jsonl`；不是新增100、local200或其他训练任务。100个不同物理组，体积分数十档各10条。包内自带稀疏载荷、支承、材料预算及原38×100=3800条标量结果，无需完整数据下载，无API调用或密钥。

## 实验定义

每个任务运行完整笛卡尔积，共60组、全实验6000条轨迹：

|参数|启动值|
|---|---|
|SIMP p|2、3、4|
|滤波rmin，单位为64×64网格单元边长|1.5、2、2.5、3|
|OC move|0.05、0.10、0.15、0.20、0.30|

编号按p→rmin→move嵌套顺序递增，P01～P60。**P34=(3,2.5,0.2)为原默认Uniform基线。** `protocol.json`保存全部精确配置。

1. 从Uniform `x=volume_fraction`开始。
2. 前50次OC使用该组固定启动三元组，不做在线LLM或线性p延续。
3. 第50步结束，统一切换到原默认 `p=3、rmin=2.5、move=0.2`。
4. 重建共同滤波器，检查体积；必要时做带截断的设计密度整体加法平移，以满足共同滤波下的物理体积。若已在1e-8容差内，原样保留，不强行修改。修正以1e-11为目标，记录幅度。
5. 参数或设计密度改变时重新FE求解。切换前后状态分别保存；共同阶段不沿用旧模型梯度。
6. 在共同模型下运行至原稳定判据或总计2000次OC更新；前50步计入2000步。

这是**固定启动参数＋共同收敛阶段**，不是全程不同常数的优化问题对比，也不是完整论文在线四参数控制复现。原初始化路线从不同初始场使用默认求解器；本路线从Uniform使用50步不同参数启动，再用同一默认求解器收敛。两条路线的前置计算均应计入成本。

共同收敛判据：切换后连续10次更新满足最大设计密度变化≤0.01、逐步相对柔度变化≤1e-4、窗口包含前一状态的柔度相对极差≤5e-4、物理体积误差≤1e-8、FE残差<1e-6。最早第60步认证；2000步上限不等于已收敛。材料及数值下限不搜索：xmin=1e-6、Emin=1e-9、Emax=1、ν=0.3，OC采用原冻结可行侧二分实现、容差1e-8。

## 在另一台机器下载

从GitHub仅取本实验目录，避免下载其他实验的大数据分片：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/hsksfje/topo.git topo_parameter60
cd topo_parameter60
git sparse-checkout set META64_parameter60_train100_v1
cd META64_parameter60_train100_v1
```

已有该仓库的普通完整克隆可先 `git pull`，再进入本目录。ZIP下载仓库也可以；不要改动包内源码、数据或protocol，否则身份校验会拒绝继续旧实验。

## 环境

推荐匹配原数值版本：

```bash
conda env create -f environment.yml
conda activate meta64-parameter60
```

已有 `meta64-convergence` / `meta64-fresh` 环境，若Python3.11、NumPy1.26.4、SciPy1.12.0一致，可以直接使用。

没有Conda时，在Python3.11虚拟环境安装：

```bash
python -m pip install -r requirements.txt
```

支持Linux、macOS、Windows。只依赖NumPy/SciPy及Python标准库；不需要Matplotlib、PyTorch、OpenAI SDK或任何.env文件。

## 核验与完整运行

```bash
python run.py --stage verify
python run.py --stage preflight
python -u run.py
```

预检会核对包内每个文件SHA256、100个Uniform初态FE与原C0，并检查一个任务50次默认OC后与原C50一致。预检成本单列，不混入参数轨迹成本。`run`在没有预检记录时也会自动执行预检。

默认预留一个可用逻辑CPU，其余采用多进程，每个数值库强制单线程。Linux同时将运行进程限制在扣除一个CPU后的亲和性集合；其他平台按worker数量预留计算容量。需要减少负载可指定：

```bash
python -u run.py --workers 4
```

不要在同一台机器上启动多个大CPU队列造成超额占用。具体运行时间依赖机器及收敛轨迹，不以OC50耗时估算全程。

默认输出：

```text
outputs/train100_parameter60_run01/
  config.json / config_identity.json       冻结任务、参数、代码身份
  preflight.json / environment.json        输入与环境核验
  process.json / progress.json             进程和6000条结算进度
  tasks/<task_id>/P01...P60/
    checkpoint.npz / cost.json             数值检查点、实际FE尝试
    state_0000.npz                         Uniform初态
    state_0050_launch.npz                  启动阶段结束、切换前
    state_0050_common.npz / transition.json 共同模型切换、体积修正
    history.json / state_final.npz         逐步标量和最终设计
    result.json                            收敛、达到上限或失败
  reports/                                自动配对统计
  COMPLETION.json                         全部6000条已结算
  return_manifest.json / RETURN_READY.json 返回文件哈希及完整性标记
```

完成表示所有结果已经结算，不表示全部收敛。数值失败和2000步未收敛保留原记录，不回填默认解、不静默重试。若要更改协议或重试失败，应建立新输出版本；不要删除原失败结果后混写旧会话。

## 可选的小规模试运行

先验证一个真实64×64任务的4种代表配置，各最多70步，输出与正式实验完全隔离：

```bash
python -u run.py --stage smoke --workers 1
```

其 `max_steps` 通常是预期的验证终点，不作为2000步实验成绩。可查看 `outputs/smoke_validation/`。

也可先跑体积分数十档各2条、全部60组，共1200条完整收敛轨迹：

```bash
python -u run.py --task-count 20 --output outputs/train20_parameter60_run01
```

该子集按原manifest顺序、每档取前2条，不按成绩筛选。它与正式100输出隔离，当前入口不自动将其结果导入正式会话。要完整跑100任务，直接使用上面的正式命令。

## 查看、暂停、恢复

```bash
python run.py --stage status
python run.py --stage pause
```

暂停请求写入根输出目录的 `PAUSE_REQUESTED`。在途worker在下一次OC边界保存；等待原进程退出后恢复：

```bash
python -u run.py --stage resume
```

恢复保留原选择、协议和检查点，不重复已结算轨迹，可通过 `--workers` 改并行度。每10步及切换边界原子保存检查点，正常暂停会即时保存；中断重启直接恢复设计、当前FE梯度和状态，无需恢复FE。硬崩溃可能重算最后一个未保存区间，所有已记录FE尝试保留；墙钟时间为已保存区间累计，不声称完整硬崩溃耗时。

子集或自定义输出的查看/暂停/恢复需加相同 `--output`，例如：

```bash
python run.py --stage status --output outputs/train20_parameter60_run01
python run.py --stage pause --output outputs/train20_parameter60_run01
python -u run.py --stage resume --output outputs/train20_parameter60_run01
```

POSIX/Windows跨进程输出锁防止重复写同一会话。运行时可直接只读查看 `progress.json` 和报告；需要手动重建报告时，先暂停并等进程退出，再用 `--stage report`。已封存结果不再改写报告，避免破坏返回哈希。

## 怎样比较参数与初始化

- `reports/results.csv`：每任务每参数的Cend、总OC步数、实际FE、切换体积修正、失败信息。
- `reports/parameter_summary.csv`：每个参数配置相对历史Uniform的均值、中位数、明显退化、严格双赢及配对数量。
- `reports/initialization_summary.csv`：已有38种初始化，在同一批任务上的参考汇总。
- `reports/parameter_vs_initialization.csv`：60×38个配对比较，各自附双方已收敛样本数。
- `reports/REPORT_zh.md`：中文进度与汇总。

最终柔度改善：`100*(1-C_parameter/C_reference)`，正数更好。OC步数节省：`100*(1-N_parameter/N_reference)`，包含启动50步。严格双赢：柔度改善>1%且OC步数减少≥10%，分母为全部冻结任务；失败、未收敛及未完成均不计成功。有效配对数量不同时，不把均值直接当作全任务排名。

`C50_launch`来自不同p/rmin的启动模型，不跨组排名。`C50_common`是在切换及必要校准后共同模型的评价。到共同质量的首次步数只检查共同阶段、连续5个状态达标，不能推断切换前是否已经达标。收敛步数收益不等于端到端时间收益；本包记录新实验实际FE，历史初始化的候选搜索、前置粗优化和API成本不从旧标量表中猜测重建。

这100条是已有开发数据。每个任务在60组中事后选最好只作为搜索上界，不能宣称为一个已部署选参器或未见测试成绩。若依据结果选择配置，后续应在独立任务确认。

## 回传结果

全组结算后会自动生成 `RETURN_READY.json`。原样复制整个 `outputs/train100_parameter60_run01/` 即可；也可生成ZIP：

```bash
python run.py --stage verify-return
python run.py --stage export
```

生成 `outputs/train100_parameter60_run01_return.zip` 及同名 `.sha256`。运行未完成时拒绝封存。ZIP包含全部已封存输入、轨迹、状态、失败记录和报告。解压后校验：

```bash
python code/export_results.py --output /path/to/train100_parameter60_run01 --verify-only
```

建议回传到原机器：`InitSkill-Lab/outputs/meta64_parameter60_train100_remote_v1/run01/`，保留完整目录结构。不要只传汇总均值。

## 来源与修改说明

物理输入来自Meta-neural Topology Optimization，作者Igor Kuszczak、Miguel Bessa、Gawel Kus、Federico Bosi；[Zenodo21893871](https://zenodo.org/records/21893871)，DOI[10.5281/zenodo.21893871](https://doi.org/10.5281/zenodo.21893871)，原数据许可[CC BY4.0](https://creativecommons.org/licenses/by/4.0/)。本包对原始输入进行了既有坐标转换，并以稀疏JSON打包载荷与约束；源任务、原输入资产哈希、100任务合并审计见 `provenance/input_identity.json`。参考标签没有用于选择新参数配置。

`code/physics.py`逐字复制原Train100冻结物理内核；`code/solver.py`提取原可行侧OC类。新增部分为两阶段运行、滤波切换体积修正、独立检查点、统计与回传封存。源身份见 `provenance/solver_identity.json`。

本地回归测试：

```bash
python -m unittest discover -s tests -v
```

验证结果与限制见 `validation/VALIDATION_zh.md`。包内不包含任何API密钥、账户凭证或付费调用入口。
