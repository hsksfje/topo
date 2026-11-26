# Ulu PCA + NN

Python实现：`core.py / optimizer.py`。

Train1000 PCA80、单隐藏层80、Python SCG；Valid300 早停。

变化支承栅格输入、tanh 与标准化是显式选择；无作者完整代码，不能保证 MATLAB 位级一致。

所有方法接续相同的可行侧OC求解器；完整评测ID/OOD/CR各1000项，最多2000步，至少保留C50/C100/C200。两项目标与账本统一由`../campaign.py`汇总。

[论文与源码审计](../../../../../references/init_benchmark_v1/REPRODUCTION_AUDIT.md)
