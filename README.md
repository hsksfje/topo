# META64 数据与拓扑优化初始化实验

## 原Train20交叉实验：初始化在不同启动参数下是否仍有效

[独立运行包与完整说明](META64_init_parameter_cross20_v1/README.md)。原100任务按体积分数十档各取前2条；PG1、L1_PG的实际初态分别与P39/P38/P20启动策略交叉，只新增120条轨迹。Uniform四策略及两种非Uniform初态的默认策略共120条历史对照随包提供，完整矩阵为20×3×4=240格。

不重新生成LLM初态，无API调用，无需下载完整数据集。启动及第50步滤波切换时必要的体积修正均保留；统一跑至收敛或总2000步。报告按相同策略的Uniform逐任务配对，含初始化收益与策略的交互。返回包附实际执行源码快照、初态身份及结果哈希。

```bash
cd META64_init_parameter_cross20_v1
conda env create -f environment.yml
conda activate meta64-cross20
python run.py --stage verify
python run.py --stage preflight
python -u run.py
```

已有Python3.11/NumPy1.26.4/SciPy1.12.0的meta64-parameter60环境可直接使用。输出为outputs/train20_init_parameter_cross_run01/；运行完用`python run.py --stage export`生成回传ZIP。完整下载、暂停恢复和回传路径见包内README。

## 原Train100新增：Uniform启动参数60组合收敛实验

[独立运行包、环境和完整说明](META64_parameter60_train100_v1/README.md)。使用已完成38种初始化收敛比较的最早100个任务，不是新增100或local200。Uniform起点，p={2,3,4}、rmin={1.5,2,2.5,3}、move={0.05,0.10,0.15,0.20,0.30}全组合，共6000条轨迹；前50步使用该组参数，再回到p3/rmin2.5/move0.2至共同收敛或总2000步。

包内自带物理输入和原3800条标量对照，无需下载完整数据集或API密钥。默认CPU进程数为可用逻辑核减一、BLAS单线程，支持暂停恢复和结果回传哈希。进入目录后：

```bash
conda env create -f environment.yml
conda activate meta64-parameter60
python run.py --stage verify
python run.py --stage preflight
python -u run.py
```

本机仅完成预检、代表轨迹与恢复验证，没有启动正式6000条队列。完整运行与回传说明见包内README。

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


## 本机200任务：离线委托6600条OC50记录续跑（2026-11-22）

运行包与完整使用指南：[META64_local200_delegated6600_v1](META64_local200_delegated6600_v1/README.md)。包含200任务全部28种固定初始化5600条，以及冻结时前1000条有效LLM OC50轨迹；原服务器只继续其余1000条LLM。分工清单互斥且覆盖本批200×38=7600条。

数据分成10个ZIP分片（约553MB十进制，约528MiB），更新仓库后进入该目录，依次执行：

```bash
conda activate meta64-convergence
python prepare_data.py
python code/run.py --stage verify
python code/run.py --stage preflight
python -u code/run.py --stage run
```

已有meta64-fresh且核心数值包版本一致也可使用。不调用LLM，无需API密钥；默认保留一个逻辑CPU，其余并行。输出固定outputs/delegated6600_run01/，运行结束后用code/export_results.py打ZIP回传，详细路径见运行包README。
