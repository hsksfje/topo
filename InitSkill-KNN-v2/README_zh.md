# 独立KNN修正版 v2

这是单独的Python CPU实验包，比较Uniform、旧1NN规则和修正版KNN。无需Torch、CUDA、JAX或原Benchmark环境，不读取原目录的结果，不会重训其他神经网络。

本版是在查看v1测试结果及失败日志之后设计的**探索性修复实验**。不能将复用同一测试集后的成绩称为一次性盲测确认，也不保证修正版一定提高最终柔度。

## 下载与运行（Windows11 / Python3.11）

如果旧Benchmark还在运行，先在其原目录执行 `python benchmark.py stop`，避免两个包争用同一CPU/内存预算。旧结果会保留。

```powershell
cd D:\2026-11-20--topo-test
git clone --depth 1 --filter=blob:none --sparse https://github.com/hsksfje/topo.git topo_knn_v2
cd topo_knn_v2
git sparse-checkout set InitSkill-KNN-v2
cd InitSkill-KNN-v2
conda activate meta64-fresh
python setup.py
python run.py verify
python run.py start --cpus 28 --ram-gb 64
```

逐行执行。setup会在本目录创建独立venv，仅安装NumPy1.26.4、SciPy1.12.0、psutil7.0.0；Decimal属于Python标准库。不要给旧 `benchmark.py` 传本包的命令。本包入口是 `run.py`。

## 三种对照与修复

| 名称 | 初始化 | 公共求解器 |
| --- | --- | --- |
| uniform | 均匀目标体积分数 | float64 SuperLU；必要时Decimal40精化 |
| knn_legacy | 原1157维栅格特征逐坐标标准化、欧氏1NN、原体积校准；没有软化 | 同一修复后的公共求解器 |
| knn_v2 | 空间平滑、载荷整体反向对称距离、分块归一化、3NN加权平均；校准后10%均匀混合 | 同一修复后的公共求解器 |

旧1NN保留用于判断改善究竟来自数值修复，还是新检索/初态。其新结果不是旧Windows数值路径的原始成绩，必须区分版本。

### 数值修复

原先的np.longdouble在标准Windows上不能保证比float64更高精度。新路径用Python Decimal40有效十进制位重新构造Q4积分/刚度、系数乘积和装配求和，在高精度中计算残差和位移修正。原float64 LU只作预条件器，能正常通过的float64路径保持原样。精化最后检查能量与外力功一致性。

没有放宽残差阈值1e-6，没有改变Emin=1e-9、penal=3、nu=.3、move=.2或体积条件。Decimal修正是一次FE求解内部的修正，不算额外OC步；实际修正次数、残差、精度路径会保留。后备路径只在float64不通过时启用，病态场中可能明显增加计算时间。

### 检索与初态修复

4个17×17工况栅格做sigma=1.25的高斯平滑。载荷两通道整体L2归一化，两个约束方向各自L2归一化，避免逐坐标标准化放大稀少位置。三个距离块为：

- 载荷：`min(||f-f_i||², ||f+f_i||²)`，只允许整体反向，保留多个载荷之间的相对方向。
- 支承：两个方向的平滑约束通道平方距离。
- 体积分数：`((VF-VF_i)/span)²`，span仅由Train1000的95%和5%分位数之差确定。

三块权重固定为1、2、2。取3个最近邻，以距离平方根的倒数归一化加权；出现零距离时直接取首个零距离设计。64→256使用半像素中心对齐的双线性插值和边缘复制，保持原x-major顺序。先以原校准达到目标体积，再 `x=0.9*x_calibrated+0.1*VF`。线性过滤下该混合保持目标体积，也给原近空区域留出恢复材料的起点。

参数是公开固定的工程变体，不是已经通过验证集搜索得出的最优参数；本版不进行测试集参数搜索。Valid100用于对这一固定版本做完整验证审计。不能声称这些参数一定优于其他选择。

## 会全量跑到哪里

- Train1000：建立训练库；直接使用随包已审计的历史Uniform标签，997收敛、3到原2000步预算上限，不是全局最优。生成成本快照在data/historical_teacher_costs.json。
- Valid100：三种方法各跑100个任务，共300条，用于验证审计，单独保存，不混进测试排名。
- ID/OOD/CR：每类固定300个任务，三种方法各跑，共2700条正式测试。
- 总计3000条优化运行。每条最少200步、最多500步，保存C50、C100、C200及终态。500步仍不满足条件记为未收敛；数值失败单独记录。
- 完成后生成results/reports/REPORT_zh.md的两张目标表、summary.json、paired_tasks.csv和覆盖率。完成标记为results/campaign/COMPLETE.json，表示任务全部结束（可能含明确失败），不等于全部收敛。

达标目标为同任务Uniform终态柔度×1.01。未达标/执行失败按500步封顶统计，Uniform参考失败时单独处理。柔度比只在配对成功结果上计算，必须结合失败率解释；同时给出均值与中位数，避免长尾被掩盖。

三个测试集、Train/Valid清单、数据ZIP与原Benchmark一致。包中不包含测试最终拓扑或测试柔度答案。8个旧失败初态仅作为tests/fixtures的数值回归输入，训练/检索不读取该文件；其来源和测试期间调试事实均公开记录。

## 并行与内存

默认28CPU、64GiB计划预算。新包没有教师训练流水线，因此采用独立的小/大网格进程池；28CPU/64GiB且实际可用物理内存不少于64GiB时，是小网格12＋大网格16，合计不超过28个优化进程。每个进程的BLAS/数值线程固定为1；一个任务进程按顺序运行3种初始化。

程序会读取实际物理内存，并以声明值和实际值中较小者规划并发。为系统预留6–16GiB，大任务按2GiB估算，小任务按0.35GiB估算，实际阶段峰值可能不同。若设备实际上为24GiB，即使传64也不会按64规划。Linux容器应自行填写容器实际限制，宿主机物理内存不等于cgroup限额。

验证阶段只有64网格任务，所以不会一直用满所有CPU；测试阶段大小网格独立推进。此并行分配仅适用于本包，和原Benchmark的2＋24分配不同。

## 查看、停止与恢复

另开一个终端，在同一目录执行：

```powershell
python run.py status
python run.py stop
python run.py start --cpus 28 --ram-gb 64
```

stop只停止本目录记录的进程树，检查PID创建时间和根目录。已经完成的结果直接保留；中断的OC从两个一致性检查点槽恢复。数值失败会保留failure.json并跳过，不会悄悄重试或覆盖；需要修复时应明确建立新版本/重试审计。完整耗时可能不包含中断前未提交的部分，FE逐次尝试记录保留。

## 返回结果

全部完成后：

```powershell
python run.py export
```

将打印出的 `returns/时间戳/` 整个文件夹返回，包括RETURN_MANIFEST.json及全部results.zip.partXXX分卷。每卷100MiB并带SHA256。提前反馈错误请先停止运行，再用 `python run.py export --allow-partial`，报告会标记为中间结果。分卷是分析材料，不是用于在另一设备上恢复运行的完整工作目录。

## 验证范围

`verify`校验9200份数据资产，并检查：数据隔离、整体载荷反向不变性、半像素跨分辨率插值、软化后的体积、Decimal与健康float64结果/梯度/能量一致性、8个实际旧Windows失败初态、精确OC续跑、Train1000完整建库、3种方法训练任务上的轨迹以及spawn进程路径。

这些预检不等于3000条全量实验已经跑完，也不证明修正版会改善最终柔度。源码和数据协议会冻结，修改算法后应使用新结果目录。本包不改动原InitSkill-Benchmark文件。
