# Meta-neural TO

Python实现：`core.py`。

直接调用固定提交的作者JAX模型、滤波、FEM、Reptile与损失；只替换manifest迭代器。

从随机参数开始；1/32原生滤波与平台2.5/64不同。零步密度以rho-as-x明确交接，不隐藏二次滤波。

所有方法接续相同的可行侧OC求解器；完整评测ID/OOD/CR各300项，最多500步，至少保留C50/C100/C200。两项目标与账本统一由`../campaign.py`汇总。

[论文与源码审计](../../../../../docs/METHODS.md)
