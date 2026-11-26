# Wang PCA + Feedforward NN

Python实现：`core.py / optimizer.py`。

Train1000 PCA200、两隐藏层200、Python LM；Valid300 早停。

使用 AD 的 JVP/VJP 与 CG 求 LM 正规方程；提供显式 Jacobian 数值核对。1000样本配置按论文比例延伸。

所有方法接续相同的可行侧OC求解器；完整评测ID/OOD/CR各1000项，最多2000步，至少保留C50/C100/C200。两项目标与账本统一由`../campaign.py`汇总。

[论文与源码审计](../../../../../references/init_benchmark_v1/REPRODUCTION_AUDIT.md)
