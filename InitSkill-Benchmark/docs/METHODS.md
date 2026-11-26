# 方法与复现差异

当前范围固定为Train1000、Valid100及ID/OOD/CR各Test300。所有模型使用相同清单；测试集不参与模型选择。不提供训练集效果排名。完整调用流程由 `benchmark.py` 管理，各方法源目录独立。

| 方法 | 源目录 | 当前实现与需要说明的边界 |
| --- | --- | --- |
| Uniform | common.py / campaign.py | 统一体积分数初态与公共OC参考，不训练 |
| 1NN | knn/ | Train1000工况特征标准化后取最近1个任务并迁移密度；这是检索基线，不称Ulu论文原始KNN实现 |
| Ulu PCA＋NN | ulu_pca/ | 训练拓扑PCA，80维系数、80宽隐藏层、SCG；可变支承栅格输入、tanh/线性层和未披露默认值均属于平台适配或实现假设 |
| Wang PCA＋NN | wang_pca/ | PCA＋前馈网络、LM；200维/宽度用于1000任务的外推，矩阵自由正规方程与稠密/AD数值对照，不声称作者MATLAB训练逐位一致 |
| Yu CNN | cnn_encoder_decoder/ | 按论文网络图重建编码器/解码器，使用载荷、支承与体积分数；未获得完整Yu作者TO训练仓库，不能把通用Keras-GAN仓库当作该论文官方代码 |
| Meta | meta_neural/ | 保留固定commit的作者ResidualSiren、Reptile、CVXOPT FEM、损失与内层/外层Adam；清单迭代器替代DALI；6000×B5、k10，Valid100每200步完整验证 |
| Early | early_trajectories/ | 独立6000×B5、k5短轨迹消融，不从Meta权重初始化；并非另一篇独立算法论文 |
| CNN＋cGAN | cnn_cgan/ | Yu两阶段重建；同任务64→256教师，原论文32→128的比例适配；全Train1000＋Valid100教师，20epochs、batch16；64测试时面积平均回目标网格 |

SCG/LM验证包含正定二次型、线性最小二乘、非线性AD对稠密步长对照；没有MATLAB/Octave二进制交叉训练证明。PCA网络max_fail=6、CNN早停等未明确作者细节保留于代码/config与历史审计，不用测试柔度事后调参。

Meta/Early原生过滤半径为1/32，公共OC为2.5/64物理尺度。导出的物理rho以明确的rho-as-x方式进入公共体积校准和滤波，热身步数0，参考场FE与导出费用单独记账。比较的是初始化交给相同OC后的效果，不推断原论文在线神经Adam的性能排名。

cGAN的Adam lr=2e-4、beta1=.5取自被引用的通用Keras-GAN来源，论文TO作者未完整披露；生成器用非饱和BCE＋.001MAE，判别器以低分辨率教师为条件。论文措辞与图之间的歧义没有被隐藏为“完全官方实现”。

公共物理：Q4平面应力、p=3、Emax=1、Emin=1e-9、nu=.3、xmin=1e-6、move=.2；64网格rmin2.5、256网格rmin10。保存原始载荷，固定DOF的载荷只影响反力。所有方法同一可行侧OC更新、体积校准及验收阈值。

OC最多500步、至少200步，完整保存C50/C100/C200。连续10步满足max_dx≤.01、单步相对柔度变化≤1e-4、11状态柔度跨度≤5e-4、VF误差≤1e-8与残差<1e-6才称收敛。到上限保留为budget-limited。失败保留，不编造柔度；未达到统一目标以500步计入封顶统计。

Train1000标签来自已审计的Uniform历史库（原历史生成上限2000步），997收敛、3到原预算上限；它们是局部优化标签，不是全局最优证明。Valid100使用对应的独立教师标签。共享库构造成本在随包快照中保存，不能因标签已经存在而称为零训练准备成本。高分辨率1100份教师在新设备实际求解，不能用插值替代。

本数据和实现曾经历历史测试期间数值调试；固定抽样没有依据柔度表现选择，但这不是一次性完全盲测。历史审计保留在 `HISTORICAL_REPRODUCTION_AUDIT.md`，早期Valid300/Test1000结果不混入本独立运行版。

论文和作者来源清单位于 `references/init_benchmark_v1/SOURCES.json`，含原始URL、PDF SHA与代码commit；第三方许可证随源码保存。完整方法适用性分析见 `METHOD_REVIEW_zh.md`（历史初始方案，当前规模以本目录README为准）。
