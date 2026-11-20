# META64 拓扑优化初始化收敛实验

独立运行包：[`META64_remaining50_convergence_v1`](META64_remaining50_convergence_v1/README.md)。

包含另外50个META64任务的38种初始化OC50检查点、冻结SIMP/OC求解器、Conda环境和运行/回传说明。无需API密钥，默认预留一个逻辑CPU，其余并行。

```bash
git clone https://github.com/hsksfje/topo.git
cd topo/META64_remaining50_convergence_v1
conda env create -f environment.yml
conda activate meta64-convergence
python code/run.py --stage verify
python code/run.py --stage preflight
python -u code/run.py --stage run
```

完整环境、Windows后台运行、暂停恢复、输出回传和100任务合并见运行包README。
