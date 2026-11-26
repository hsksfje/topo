# Yu CNN + cGAN

Python实现：`core.py / pairs.py`。

八残差块、两次2倍上采样、条件判别器、对抗与MAE损失；生成同一Train/Valid的64/256配对教师。

图9优先于所引用通用Keras-GAN，不能原样采用VGG感知损失。CR条件粗化与ID/OOD输出降采样均记录。

所有方法接续相同的可行侧OC求解器；完整评测ID/OOD/CR各1000项，最多2000步，至少保留C50/C100/C200。两项目标与账本统一由`../campaign.py`汇总。

[论文与源码审计](../../../../../references/init_benchmark_v1/REPRODUCTION_AUDIT.md)
