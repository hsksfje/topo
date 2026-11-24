# META64 train1000：尚未运行的文献类别初始化

本包自带原 `train1000` 的1000条物理输入，**不需要另外下载数据、其他运行包或配置API**。默认先全部跑50步，再从各自第50步继续到收敛。直接执行 `python -u run.py` 即可。

## 本次跑哪些方法

上一张有溯源方法表中，尚未运行的是中心孔和独立随机密度三次重复。本包只新增这4个配置，并独立重算Uniform，确保1000任务都有同协议配对对照。

|名称|状态／数量|配方与参考|
|---|---|---|
|U|已有方法，本包重算1000条对照|所有单元x=VF；[Sigmund2001](https://doi.org/10.1007/s001580050176)|
|HC1_B020|新配置，1000条|中心(0.5W,0.5H)、半径0.25H；截断距离场→sigmoid体积校准，再20%几何密度+80%Uniform|
|RAND_IID_S01|新重复，1000条|独立随机密度，第1个冻结种子；不做空间平滑|
|RAND_IID_S02|新重复，1000条|同上，第2个冻结种子|
|RAND_IID_S03|新重复，1000条|同上，第3个冻结种子|

共5000条轨迹，其中4000条新增配置、1000条Uniform对照。**50步与收敛不是两次重新初始化的实验**，而是同一轨迹的中间点和终点。已运行的PLSM15、MMC_CROSS、应力轨迹、H4与粗网格方法不在本队列重跑。

中心孔与随机密度参考：Alvarez、Zambrano、Silva (2019)，*Influence of Density-Based Topology Optimization Parameters on the Design of Periodic Cellular Materials*，[论文页](https://doi.org/10.3390/ma12223736)，[开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6888502/)。原研究比较Center Hole、Fully Random等初始猜测，对象是周期胞元；本包采用其**初始化类别**，孔径、混合强度、随机分布、体积映射和梁任务为本地适配，不能写成原论文逐项复现。

HC1距离场为 `clip((distance_to_center-0.25H)/(0.04H),-3,3)`，共享校准使滤波后物理体积达到目标，再线性混合。每个随机重复先按 `default_rng(seed).uniform(0.01,0.99,(ny,nx))` 取值，再logit映射和体积校准；种子为 `int(sha256((task_id+'|RAND_IID|'+str(replicate)).encode()).hexdigest()[:16],16)`。矩阵第0行是顶部，F序展开。随机三次不按结果挑选，报告按同任务三个重复取平均。

## 更新或下载GitHub代码

在已有仓库根目录执行：

```bash
git pull --ff-only origin main
git sparse-checkout add META64_literature_pending_train1000_v1
cd META64_literature_pending_train1000_v1
```

`git sparse-checkout add`仅用于此前采用稀疏下载的仓库；普通克隆跳过这一行。若有本地源码改动，先保留自己的修改再更新，不要强制覆盖。计算输出被本目录.gitignore排除。

也可以全新只下载此包，避免其他实验的大文件：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/hsksfje/topo.git topo_literature
cd topo_literature
git sparse-checkout set META64_literature_pending_train1000_v1
cd META64_literature_pending_train1000_v1
```

## Conda环境

```bash
conda env create -f environment.yml
conda activate meta64-lit-init
```

依赖只有Python3.11、NumPy1.26.4、SciPy1.12.0。不需要GPU、PyTorch、OpenAI SDK、Matplotlib或API密钥。若现有环境版本一致，可以直接使用；也可用Python3.11虚拟环境执行 `python -m pip install -r requirements.txt`。

## 核验、试运行与正式运行

```bash
python run.py --stage verify
python -m unittest discover -s tests -v
python -u run.py --stage smoke --workers 1
python -u run.py
```

`smoke`是第1个真实任务×5方法、总计70步上限的工程验证，输出到 `outputs/smoke_validation/`，标记validation_only，不进入正式1000条结果。重复执行会跳过已结算的烟雾轨迹。

正式输出固定为：

```text
outputs/train1000_literature_pending_run01/
```

默认使用**可用逻辑CPU数量减1**个并行进程，每个进程的BLAS/OMP/MKL设为单线程。Linux同时从进程CPU亲和性中扣除一个逻辑CPU；Windows限制worker数量。若系统识别16个可用逻辑CPU，则默认15并发；识别8个则默认7并发。仅有1个可用CPU时使用1个进程。

可主动减少并发，例如：

```bash
python -u run.py --workers 6
```

不能在同一输出目录同时启动两个计算进程，程序使用跨平台操作系统文件锁阻止重复写入。

只先跑50步，之后再续跑收敛：

```bash
python -u run.py --stage 50
python -u run.py --stage converge
```

`converge`要求本批所有未失败轨迹已有50步结果；`all`和`resume`会先自动补齐50步，再进入收敛阶段。已有最终结果不会重算。

## 收敛判据与50步协议

沿用既有可行侧OC、Q4平面应力SIMP：p=3、rmin=2.5单元、move=0.2、Emin=1e-9、Emax=1、nu=0.3、xmin=1e-6。设计密度x先经密度滤波得到物理密度rho；初始体积校准和后续求解器均不改变载荷、支承及目标体积。

每条必须先保存50次OC更新后的结果；从第60步起，连续10步同时满足下列条件才认证收敛：

- 最大设计密度变化≤0.01。
- 每步相对柔度变化≤1e-4。
- 10步窗口及其前一状态的柔度相对极差≤5e-4。
- 滤波后物理体积误差≤1e-8，FE相对残差<1e-6。

**总上限2000次OC更新，包含前50步。** 到上限仍未满足标记`max_steps`，不当作收敛；数值或执行错误记录为`failed`，不以Uniform替代。这套密度SIMP协议与原周期胞元论文优化流程不同。

## 实时查看

在另一个终端、同一包目录执行：

```bash
python monitor.py --watch
```

每10秒显示50步完成数、已收敛、到上限、失败和近期轨迹步数，Ctrl+C仅退出监视器。也可单次查看：

```bash
python run.py --stage status
```

主进程每30秒更新 `progress.json`、CSV和Markdown报告；worker每10步更新自己的检查点。近期更新时间不是PID存活证明；后台运行请同时观察主终端是否仍有输出。Windows运行时请关闭自动睡眠。

## 暂停、恢复与异常

```bash
python run.py --stage pause
```

worker会在下一次OC边界保存退出。等原进程退出后：

```bash
python -u run.py --stage resume
```

主计算终端Ctrl+C也会请求安全暂停；再次强杀可能重算最近10步内未保存的区间。每次FE尝试在求解前记账，恢复重算单独记录。恢复可以改变worker数量，但不能改变代码、任务范围或收敛协议；改设置请使用新包／新输出目录。

失败默认保留且跳过，避免无限重复。修复设备磁盘、权限等问题后可显式重试：

```bash
python -u run.py --stage resume --retry-failed
```

原失败文件移到各轨迹的`failure_archive/`，检查点和累计FE成本保留。程序不访问API，因此不存在API重试或付费。

## 文件、报告与分析

|路径（相对正式输出）|内容|
|---|---|
|`config.json`、`config_identity.json`|冻结协议、任务范围、源码包身份|
|`inputs/<task>/`|实际载荷、支承、设计域和任务信息|
|`source_snapshot/`|本次实际执行的代码、协议及输入包快照|
|`tasks/<task>/<method>/initializer.json`|初态参数、种子与文献适配说明|
|`state_0000.npz`、`state_0050.npz`、`state_final.npz`|位于上述各轨迹目录，保存初态、50步和终态x/rho；终态另含位移u|
|`history.json`、`checkpoint.npz`、`cost.json`|逐步柔度、体积、残差、收敛变化、恢复检查点与FE成本|
|`result_50.json`、`result.json`、`failure.json`|50步、最终结算及失败信息；按状态出现|
|`reports/REPORT_zh.md`|自动汇总分析表|
|`reports/results.csv`|全部5000任务×方法格的明细，含未完成状态|
|`reports/method_summary.csv`|各配置统计、共同收敛范围、双赢等指标|
|`reports/random_replicates_task_mean.csv`|随机方法任务内3重复平均，不取最好一次|

所有质量收益都与**同任务Uniform**配对，`I=100*(1-Cmethod/CUniform)`。最终统计只计双方均收敛；上限、失败、缺失单列，不能用部分幸存样本声称完整1000任务优势。I50和最终均值可能覆盖不同任务，不能直接相减解释优化收益。

主表包括配对数、平均I50、平均最终改善、中位停止步、更快且改善>1%、更快且接近±1%、更快但变差>1%、更慢但改善>1%及>5%。停止步数是累计OC次数，不是直接的墙钟加速；不同设备时间也不直接混比。

## 回传这台设备

运行结束后，或安全暂停后希望回传阶段性结果：

```bash
python run.py --stage export
python run.py --stage verify-return
```

产生：

```text
outputs/train1000_literature_pending_run01_return.zip
outputs/train1000_literature_pending_run01_return.zip.sha256
```

ZIP包含完整输出目录、输入、执行源码、报告、逐文件哈希与结算状态；`all_settled`和`all_converged`分开，部分回传不会标成完整收敛。继续运行会使旧封存失效，之后请重新export，不要把两次ZIP混拼。

请将ZIP与.sha256原样复制到原设备：

```text
InitSkill-Lab/outputs/meta64_literature_pending_train1000_remote_v1/exports/
```

整个输出目录复制回去也可，但ZIP＋哈希更方便判断完整性。解压后的相对路径可迁移，不依赖另一台设备盘符。

## 多设备可选分段

默认就是全部1000条，不再抽样。如需两台设备分担，可按**原清单顺序**分为前500和后500，分别在各自设备执行：

```bash
# 设备A
python -u run.py --start 0 --count 500 --output outputs/train1000_partA_0001_0500
# 设备B
python -u run.py --start 500 --count 500 --output outputs/train1000_partB_0501_1000
```

分段恢复时必须使用相同`--start/--count/--output`；查看、暂停、导出加同一`--output`即可。监视器也支持`--output`。不要同时再运行默认全1000队列造成重复。

## 数据与源码来源

精确沿用原`train1000.jsonl`顺序，1000个不同物理组、体积分数十档各100个；不重新抽样。`provenance/train1000.jsonl`保留原清单，`data_identity.json`保存输入SHA256。打包前核对原6000个NPY文件，仅发布转换后的载荷、支承、体积分数和几何；不发布或读取源参考柔度、已优化应变能标签用于初始化。

源数据：Igor Kuszczak、Miguel Bessa、Gawel Kus、Federico Bosi，*Meta-neural Topology Optimization: Knowledge Infusion with Meta-learning*；[Zenodo21893871](https://zenodo.org/records/21893871)，[DOI10.5281/zenodo.21893871](https://doi.org/10.5281/zenodo.21893871)，原数据[CC BY4.0](https://creativecommons.org/licenses/by/4.0/)。本包进行既有分层子集选择、坐标转换与初始化实验，不代表原作者认可。

`code/physics.py`、`solver.py`、`storage.py`逐字复用既有交叉实验包，哈希见`provenance/solver_identity.json`。新代码只涉及本次两类初始化、批处理、恢复与报告。测试与发布前数值验证见`validation/VALIDATION_zh.md`；实际验证平台是Linux，Windows路径、spawn和文件锁已按跨平台实现，但尚未经Windows实机验证。
