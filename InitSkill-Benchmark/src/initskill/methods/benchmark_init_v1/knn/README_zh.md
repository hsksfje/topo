# 1NN 密度检索

Python实现：`core.py`。

仅 Train1000 的完整工况特征和 Uniform 标签构建库；无监督网络训练。

自定义历史迁移基线，不冒称 Ulu 原算法。

所有方法接续相同的可行侧OC求解器；完整评测ID/OOD/CR各1000项，最多2000步，至少保留C50/C100/C200。两项目标与账本统一由`../campaign.py`汇总。

[论文与源码审计](../../../../../references/init_benchmark_v1/REPRODUCTION_AUDIT.md)
