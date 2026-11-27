# Wang PCA＋前馈网络

本方法已有独立实现；本包默认训练1000、验证100，在ID300和OOD300上各测试300例。

在包根目录执行`python benchmark.py wang`可单独完成本方法的训练准备；`python benchmark.py start`会统一运行全部默认方法及Uniform。

方法原理、配置、原文献及平台适配边界见根目录METHODS_STATUS_zh.md与docs/METHODS.md。

跨分辨率默认关闭。Meta/Early的默认导出仅包含ID/OOD共600例，公共OC最多500步。完整预算、失败和健康诊断会保留；安装或预检通过不等于全量复现完成。
