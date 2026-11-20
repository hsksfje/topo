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

## 完整数据与新100任务

- [META64完整分层数据包：训练3000、验证300、三类测试各1000](META64_datasets_v1/README.md)
- [新100任务：Conda环境、原始数据到38种初始化及收敛](META64_fresh_tasks_v1/README.md)

新100从train1000排除原100条及等价物理组后抽取，体积分数十档各10条。原始3.53 GB数据分片压缩约660 MB；只运行本次100条需准备约32.4 MB的训练分片。
