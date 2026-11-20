# META64 数据与拓扑优化初始化实验

## 最新：跑完train1000，剩余按200＋300＋300分设备

[三设备分配清单、独立启动脚本与README](META64_train1000_distributed_v1/README.md)。排除最早100和另一设备正在跑的新增100，剩余800按原train1000清单顺序分为本机200、设备A300、设备B300；不抽样、不重叠，总计覆盖1000。

设备A：`python -u run_deviceA300.py`；设备B：`python -u run_deviceB300.py`。先按该目录README激活环境、准备train1000数据并配置本地API密钥。三批API上限2200/3300/3300，已获得用户明确的外部数据发送及付费调用授权。


## 本次：新100任务，从头运行全部38种方法

[完整运行说明与Conda环境](META64_fresh_tasks_v1/README.md)

从train1000排除此前100条及等价物理组后，按体积分数十档各抽10条。28种固定初始化＋10种LLM，先OC50，再继续到收敛（总上限2000步）。默认预留一个逻辑CPU，其余数值进程并行；LLM阶段默认最多6并发，最多1100次新增调用，API密钥在本地配置。

快速下载代码与元数据，不一次下载所有测试分片：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/hsksfje/topo.git topo_new100
cd topo_new100
git sparse-checkout set META64_fresh_tasks_v1 META64_datasets_v1/metadata
cd META64_fresh_tasks_v1
conda env create -f environment.yml
conda activate meta64-fresh
python ../META64_datasets_v1/fetch_data.py --subset train100_new100
```

按运行说明复制`.env.example`为`.env.local`并填入本地密钥，然后：

```bash
python code/run.py --stage preflight
python -u code/run.py --stage run --execute-api --env-file .env.local
```

输出固定为`META64_fresh_tasks_v1/outputs/train100_new100_run01/`，包含自动报告、轨迹、API回执及完成后的返回清单。

## 全部原始子集数据

[数据说明、来源署名、分片及校验](META64_datasets_v1/README.md)

训练3000、验证300、分布内/分布外/跨分辨率测试各1000，共6300个独立源任务、37800个原数组，原始3.53 GB、51个压缩分片约660 MB。train1000、train100、验证100和各测试300是嵌套子集，不重复存数组。只准备本次新100任务需约32.4 MB训练分片。

数据来源为Meta-neural Topology Optimization，原记录CC BY4.0，作者和DOI见数据说明。跨分辨率测试为256×256，当前新实验入口仅支持64×64，不自动缩放。

## 此前：原100任务中另外50条的OC50续跑包

[旧50任务独立运行包](META64_remaining50_convergence_v1/README.md)。该目录仍保留原任务、检查点、运行及回传说明，与本次新100任务不同。
