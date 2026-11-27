# 原始清单与代码逐项核对

## 核对结论

最初7项均有实现。此处没有新增虚构的方法名称；此次开发修复的是跨分辨率默认开关、独立打包、数值一致性、流程完整性及诊断。

| 原始项目 | 已有代码位置 | 原文献／性质 | 本次动作 |
| --- | --- | --- | --- |
| KNN Initialization | `knn/core.py`；另有独立`InitSkill-KNN-v2` | 本研究历史检索基线 | 本包不重复；继续使用独立KNN修正版 |
| Ulu PCA＋NN | `ulu_pca/core.py`和`optimizer.py` | Ulu、Zhang与Kara，期刊版，DOI10.1080/21681163.2015.1030775；拓扑预测及传统优化初态迁移 | 纳入默认ID/OOD；增加验证基线诊断 |
| Wang PCA＋前馈NN | `wang_pca/core.py`和`optimizer.py` | Wang等2023，DOI10.3390/pr11030802；PCA＋前馈网络实时结构预测 | 纳入默认ID/OOD；增加验证基线诊断 |
| CNN编码器—解码器 | `cnn_encoder_decoder/core.py` | Yu等2019，DOI10.1007/s00158-018-2101-5；两阶段生成方法的低分辨率预测部分 | 纳入默认ID/OOD；保存预测塌缩与方差诊断 |
| CNN＋cGAN | `cnn_cgan/core.py`和`pairs.py` | 同一篇Yu论文的高分辨率细化方案 | 保留代码；其教师生成、训练、测试默认全部延后 |
| Meta-neural TO | `meta_neural/core.py`、`export.py`和`adapter.py` | Kuszczak等2026，DOI10.1016/j.engappai.2026.115763；从部分轨迹学习初始化 | 纳入默认ID/OOD；原生导出默认由900例改为600例，拒绝暗中导出CR |
| Early Trajectories | `early_trajectories/core.py` | 本平台基于Meta框架的独立k=5短轨迹消融 | 纳入默认ID/OOD，独立训练并保留完整预算 |

统一的Uniform参考包含在默认范围内。上述路径均相对`src/initskill/methods/benchmark_init_v1/`。

## 旧入口为何不适合“先不跑跨分辨率”

旧包默认启动`pairs`、`testcr`和`cgan`；原生Meta导出器也固定遍历ID、OOD、CR。只是不打开CR报告并不能阻止这些计算。本包把范围写进执行计划和冻结协议，并在任务准备/求解器及原生采样入口加入大网格限制。

## 数值与训练核验

- 公共OC合并已验证的Decimal40修复，并保留原物理参数与阈值。
- 失败包装保留原异常阶段、秒数、FE等字段，不再覆盖掉内层信息。
- PCA/CNN的训练健康结果单独保存，并不因验证表现不好而自动切换算法或初态。
- 默认训练流水线串行；测试可与训练并行。测试失败保留在300例分母中，未收敛与数值失败分列。
- 没有声称通过代码审计就完成全部6000元更新或3600条测试。本机只进行工程预检，正式全量运行留给外部设备。
