# Meta短轨迹消融

Python实现：`core.py / run.py`。

独立训练checkpoint；同一Reptile，内层5步，相对于主方法10步。

这是同一论文机制消融，不作为独立发表算法；等外层预算，FE成本不同并单列。

所有方法接续相同的可行侧OC求解器；完整评测ID/OOD/CR各300项，最多500步，至少保留C50/C100/C200。两项目标与账本统一由`../campaign.py`汇总。

[论文与源码审计](../../../../../docs/METHODS.md)
