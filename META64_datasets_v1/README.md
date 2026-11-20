# META64：完整分层子集数据包

本目录发布此前下载、分层抽取的Meta-neural TO数据子集。原数组按任务保存，未修改数值、分辨率或原始NPY字节；重复引用的数组只打包一次。所有源数组已按原清单SHA256核验。

| 清单 | 工况数 | 分辨率 | 关系 |
|---|---:|---|---|
| `train3000` | 3000 | 64×64 | 主训练池 |
| `train1000` | 1000 | 64×64 | train3000子集 |
| `train100` | 100 | 64×64 | 此前已做初始化比较的100条 |
| **`train100_new100`** | **100** | **64×64** | 从train1000排除前100条及等价物理组后新抽；十档体积分数各10条 |
| `valid300` / `valid100` | 300 / 100 | 64×64 | 验证池及其子集 |
| `test_id1000` / `test_id300` | 1000 / 300 | 64×64 | 分布内测试及子集 |
| `test_ood1000` / `test_ood300` | 1000 / 300 | 64×64 | 分布外测试及子集 |
| `test_cr1000` / `test_cr300` | 1000 / 300 | **256×256** | 跨分辨率测试及子集 |

不重复计算这些嵌套清单：总共**6300个独立源任务、37800个NPY文件、原始大小3,533,076,000字节**。压缩成51个ZIP分片，共659,685,542字节（约660 MB）。每个分片小于90 MiB，逐分片和逐数组均有SHA256。源任务唯一性按source_key计；物理等价规则和原划分审计见`metadata/`。

## 只准备本次新100任务

只需Python标准库；在本目录执行：

```bash
python fetch_data.py --subset train100_new100
python fetch_data.py --subset train100_new100 --verify-only
```

会使用或下载`train1000`对应的3个分片，压缩大小约32.4 MB，解压得到整个train1000的数组（约169 MB），然后核验本次100任务所需的600个数组。源数据不再重复下载，不会调用任何模型接口。

准备其他子集：

```bash
python fetch_data.py --subset train3000
python fetch_data.py --subset valid300
python fetch_data.py --subset test_id1000
python fetch_data.py --subset test_ood1000
python fetch_data.py --subset test_cr1000
```

一次准备全部数据：

```bash
python fetch_data.py --all
```

默认解压位置`extracted/`，内部保持`raw/train/`、`raw/validation/`、`raw/test_id/`、`raw/test_ood/`、`raw/test_cr/`。可用`--root`指定其他解压目录。已经完整克隆仓库时直接使用本地ZIP；若只下载元数据，程序会从本GitHub仓库下载所需分片，核验后解压。

`catalog.json`记录子集→分片→数组的映射。`metadata/manifests/*.jsonl`保留每条工况的完整6数组引用，包括forces、normals、target_vf、id、reference_compliance、strain_energy。参考标量和保存的应变能数组供离线研究使用；本次LLM运行程序不把它们发送给模型。

`test_cr`的原始网格是256×256，不是64×64；仓库中的`META64_fresh_tasks_v1`入口只运行64×64，读取跨分辨率清单时会拒绝执行，不自动缩放。

## 新100任务的可运行代码

见同级目录[`META64_fresh_tasks_v1/README.md`](../META64_fresh_tasks_v1/README.md)，包括Conda环境、API本地配置、全部38种方法、OC50及到收敛的流程。不要用此前的“剩余50任务续跑包”来直接生成新初态。

## 来源、署名与许可

原数据：**Meta-neural Topology Optimization: Knowledge Infusion with Meta-learning**，作者Igor Kuszczak、Miguel Bessa、Gawel Kus、Federico Bosi。发布记录：[Zenodo 21893871](https://zenodo.org/records/21893871)，DOI：[10.5281/zenodo.21893871](https://doi.org/10.5281/zenodo.21893871)。源记录声明**CC BY 4.0**，许可见[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)。使用和再分发时请保留作者及来源署名。

本目录的改动仅为分层选择、清单组织、压缩分片和新100任务抽样；原数组内容不变，不代表原作者认可本项目。原始元数据、下载来源、选择种子、等价分组和哈希审计保存在`metadata/source_metadata/`、`metadata/audits/`及`metadata/manifest.json`。
