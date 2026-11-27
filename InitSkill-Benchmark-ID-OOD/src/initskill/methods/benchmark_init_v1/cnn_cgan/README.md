# Yu CNN＋cGAN（已准备，当前延后）

代码包含CNN粗结构预测、同任务64/256教师配对、条件生成器/判别器训练及目标网格适配。该方法依赖高分辨率教师，因此当前默认不生成教师、不训练、不测试。

根目录`python benchmark.py plan --enable-cross-resolution`只打印未来计划，不执行训练或FE；实际启用需要明确的同名开关，使用独立results_cross目录。当前请按默认ID/OOD说明运行其余方法。
