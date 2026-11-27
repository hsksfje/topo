# 其余初始化方法：独立ID/OOD运行包

本包用于另一台设备全量运行Ulu、Wang、CNN、Meta、Early，并以Uniform为参照。默认只训练和测试64×64工况；跨分辨率测试、256×256教师及CNN＋cGAN流程已准备，默认关闭。KNN使用已经独立上传的`InitSkill-KNN-v2`，本包不重复运行。后续合并两包结果时，要核验相同测试清单、公共求解器版本和Uniform配对参照。

## 1. 核对结果：哪些方法已有代码

最初清单的7项均已存在实现，没有发现整项缺失。此次补齐的是独立运行流程、默认关闭跨分辨率的控制、统一数值修复及运行诊断。详细逐项核对见[METHODS_STATUS_zh.md](METHODS_STATUS_zh.md)。

| 对照 | 本包状态 | 默认执行 |
| --- | --- | --- |
| Uniform | 公共OC参照，使用新Decimal数值后备路径 | 是 |
| Ulu PCA＋NN | 独立目录、Python SCG、训练/验证/测试 | 是 |
| Wang PCA＋前馈网络 | 独立目录、Python LM、训练/验证/测试 | 是 |
| Yu CNN编码器—解码器 | 论文网络图重建、训练/验证/测试，增加预测塌缩诊断 | 是 |
| Meta-neural TO | 作者代码适配，6000外循环、每批5、内循环10步 | 是 |
| Early Trajectories | 独立从头训练，6000外循环、每批5、内循环5步 | 是 |
| Yu CNN＋cGAN | 两阶段预测、1100份高分辨率教师、训练和测试代码保留 | 延后，依赖跨分辨率教师 |
| KNN | 使用原独立修正版，不在这里重复 | 否 |

“已有实现”与“在本数据上完成全部性能复现”分别记录。本文不把预检通过当成全量训练/测试完成，也不把论文重建或平台适配表述为作者原始流程完全一致。

## 2. Windows下载、安装和运行

在准备运行的设备上执行。建议单独克隆本目录，避免受旧仓库的未提交修改影响。

```powershell
cd D:\2026-11-20--topo-test
git clone --depth 1 --filter=blob:none --sparse https://github.com/hsksfje/topo.git topo_id_ood
cd topo_id_ood
git sparse-checkout set InitSkill-Benchmark-ID-OOD
cd InitSkill-Benchmark-ID-OOD
conda activate meta64-fresh
python scripts/setup_env.py --profile cpu --cpus 28 --ram-gb 64
python benchmark.py plan
python benchmark.py verify
python benchmark.py start
```

逐行执行，verify通过后再start。主环境和JAX环境分别隔离创建，安装阶段有uv时优先使用uv；没有uv时使用独立venv内的pip。主环境NumPy1.26.4、SciPy1.12.0、Torch2.5.1 CPU；JAX环境单独使用已固定的NumPy2/JAX/CVXOPT依赖。禁止`--use-current-main`，避免把原Conda中的OpenMP运行库混入。

`plan`不需要先安装模型环境，会打印：

```text
methods: uniform, ulu, wang, cnn, meta, early
actors: train, test64
test_subsets: test_id300, test_ood300
planned_test_runs: 3600
high_resolution_teachers: 0
native_export_tasks_per_method: 600
```

默认启动不会生成高分辨率教师，不会训练cGAN，不会运行CR300，也不会为Meta/Early导出CR密度场。随包仍保存CR清单与物理输入以备将来使用；文件解压/哈希校验与执行跨分辨率优化是不同操作。

## 3. 全量流程和停止条件

训练1000、验证100来自原固定清单；ID300/OOD300也与前面的包一致。训练PCA/CNN使用已审计的历史优化标签，Meta/Early使用工况、参考应变能和任务内物理优化损失。验证100用于各方法自身的验证与模型选择；测试不参与选择。

训练按Ulu→Wang→CNN→Meta→Early依次进行，同一时刻只启动一个训练阶段。测试流水线可在某方法准备好后使用它进行OC优化。训练流程和OC测试可以同时工作；全程不涉及LLM调用或本地LLM推理。

- Ulu：80维PCA、80宽隐藏层、SCG，最多1000轮，验证连续6轮无改善时停止。
- Wang：200维PCA、两个200宽隐藏层、矩阵自由LM，最多1000轮，验证连续6轮无改善时停止。
- CNN：批量16、最多500轮、验证连续30轮无改善时停止。
- Meta：6000外循环、每批5任务、内循环10步；每200外循环完整验证100。
- Early：独立模型、6000外循环、每批5任务、内循环5步；不读取训练好的Meta权重。
- 每个测试任务的公共OC至少200步、最多500步，保存C50、C100、C200及终态。500步未满足判据记为未收敛，数值失败单列。

默认6种对照×2组×300任务，共3600条测试运行。输出在`results_id_ood/`。`campaign/COMPLETE.json`表示本范围完整性检查通过；不等于所有任务都收敛或没有明确记录的失败。CPU上Meta/Early6000外循环及全量OC可能耗时较长，预算不会自动缩减。

## 4. 并行和数值路径

默认28CPU、64GiB规划。ID/OOD阶段最多26个单线程OC任务，约2核留给训练和调度；实际内存按声明值与检测物理内存的较小者计算，可降低并发。每个数值工作进程的BLAS线程为1。Linux容器应填写实际cgroup内存预算，宿主机物理内存不是容器限额。

沿用独立KNN包已验证的Decimal40后备路径：float64求解若未通过残差、正柔度、能量符号或能量/外力功一致性检查，进入真正跨平台的高精度装配与残差精化。所有方法的公共OC统一使用该版本；Emin、材料参数和1e-6残差阈值保持不变。原生Meta训练仍使用作者CVXOPT/JAX计算，不能把它与公共OC后备路径混为一体。

默认有两层范围保护：执行计划不创建大网格任务；公共求解器和原生采样也拒绝大于64×64的输入。Meta导出器只允许当前启用的测试子集。

## 5. 单方法训练入口和目录

在包根目录执行以下命令，仅完成相应方法的训练准备；完整测试仍由默认start统一调度。

```powershell
python benchmark.py ulu
python benchmark.py wang
python benchmark.py cnn
python benchmark.py meta
python benchmark.py early
```

方法实现分别在`src/initskill/methods/benchmark_init_v1/`下的`ulu_pca/`、`wang_pca/`、`cnn_encoder_decoder/`、`meta_neural/`、`early_trajectories/`、`cnn_cgan/`。各目录README说明论文、实现和运行步骤。方法流程与适配边界另见`docs/METHODS.md`。

PCA/CNN增加验证集训练健康报告`training_health.json`：检查相对平均拓扑系数基线的误差、CNN预测方差及近全空/全实体输出。它们是诊断，不改变原损失、早停或模型选择，诊断不会触发额外的Uniform回退。常量预测经过统一体积校准可能成为均匀场；原始和校准数组都会保存，这一现象需连同训练诊断解释。

## 6. 查看、停止、恢复与报告

另开终端，在本目录执行：

```powershell
python benchmark.py status
python benchmark.py stop
python benchmark.py start
python benchmark.py analyze --allow-partial
```

停止只针对本包、当前范围记录的PID/创建时间/目录匹配的进程树。完整的Meta/Early模型、Adam与采样器状态支持一致性续跑；OC支持一致性检查点恢复。未完成的PCA/CNN训练阶段按固定种子重新训练；已冻结且核验通过的模型复用，不重新训练。中断前尚未提交的训练/OC耗时可能缺失，FE尝试账本保留。

失败记录不会自动删除。训练流水线异常会停止本范围的其他进程并保存日志，修复后重启；模型或数值协议发生变化时需要新版本结果目录。

默认生成两个目标表：

1. 达到同任务Uniform终态柔度×1.01所需步数，未达到和失败按500步封顶；参考失败单列。
2. 同预算C50/C100/C200和终态柔度，逐任务相对Uniform配对；成功子集的柔度统计必须连同失败率解释。

报告在`results_id_ood/reports/FINAL_REPORT_zh.md`；未完整覆盖时只生成中间报告。Meta/Early严格检查实际6000更新及30次完整验证，未用工程模型冒充正式模型。

## 7. 未来的跨分辨率代码

只查看未来计划可以运行：

```powershell
python benchmark.py plan --enable-cross-resolution
```

这条命令只打印计划，不做训练或FE。未来在明确启用时，给start添加同名开关；才会运行1100份256网格教师、cGAN和CR300，并将结果写入独立`results_cross/`，其范围为7种方法×3组×300=6300条测试。该范围采用独立模型/协议，不能直接把旧ID/OOD结果混入，也不会改写当前`results_id_ood/`。本次交付与验证没有启动该全量跨分辨率流程。

## 8. 导出回传

完成后执行：

```powershell
python benchmark.py export
```

提前返回错误时先stop，再执行`python benchmark.py export --allow-partial`。将打印出的`returns/时间戳/`整个目录（含RETURN_MANIFEST.json和全部分卷）返回。每卷带SHA校验。导出时会重新运行对应范围的分析检查。

## 9. 复现边界

Ulu/Wang均有Python优化器数值对照，未提供MATLAB二进制逐位交叉训练证明；Yu按论文图重建并记录未披露细节；Meta/Early比较“零步神经适应密度→公共OC”，不同于原论文继续在神经参数空间优化。历史测试曾用于数值调试，因此这些修复后的结果应按探索性平台比较解释。

`verify`包括固定数据隔离、SCG/LM、训练任务FEM/CNN反向传播、原生模型/Adam/采样器续跑、8个已知失败64网格初态的数值回归，以及默认流程没有CR/cGAN任务的检查。默认预检不执行256网格FE或cGAN操作。
