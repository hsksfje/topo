# Wang PCA + LM前馈网络

Train1000/Valid300，tansig用tanh实现。独立Python LM以矩阵自由AD和CG解法避免显式巨型Jacobian；稠密方程数学验证已通过，没有MATLAB引擎逐位对照。

本目录仅包含此方法；共享物理求解、数据清单、约束校准及完整优化停止规则位于上级目录。

完整工程入口：在 `InitSkill-Research` 根目录执行 `python run_gpu_initialization_benchmark.py start`。当前正式输出为 `research_results/benchmark_init_gpu_v2`，训练模型与已审计标签复用V1目录。运行中的正式实验请使用这一可续跑入口，避免另建重复训练。

单方法复现命令模板（新输出目录先建立同一协议、数据与标签；设置 `PYTHONPATH=src`、`INITSKILL_FE_SOLVER=cudss`；GPU可见）：

```bash
python -m initskill.methods.benchmark_init_v1.run train-pca --method wang --epochs 1000 --out "$OUT"
python -m initskill.methods.benchmark_init_v1.campaign --stage evaluate --method wang --out "$OUT" --workers 4
```

`campaign --stage evaluate`执行ID/OOD/CR各1000例、最多2000步及共同停止准则；它与旧的固定50步诊断入口不同。Meta训练/导出用独立JAX环境（训练注册cuda,cpu，导出CPU），公共OC评测用CUDA环境；其余依赖、论文下载及实现差异详见根目录 `references/init_benchmark_v1/REPRODUCTION_AUDIT.md`。该模板说明实现入口，不表示当前各阶段均已完成。
