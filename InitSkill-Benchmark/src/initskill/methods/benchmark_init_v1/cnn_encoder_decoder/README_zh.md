# Yu CNN Encoder–Decoder

Python实现：`core.py`。

完整四通道节点输入、VF瓶颈、论文图7通道、MAE 与 Adam；Train1000/Valid300。

图7有7个编码卷积和8个解码卷积，正文称9个解码卷积；本实现跟随图示，保留最终64→2→1。

所有方法接续相同的可行侧OC求解器；完整评测ID/OOD/CR各1000项，最多2000步，至少保留C50/C100/C200。两项目标与账本统一由`../campaign.py`汇总。

[论文与源码审计](../../../../../references/init_benchmark_v1/REPRODUCTION_AUDIT.md)
