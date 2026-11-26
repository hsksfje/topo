# InitSkill 初始化 Benchmark（独立运行版）

本目录可单独下载运行，不依赖原服务器的目录、虚拟环境、API密钥或旧测试结果。默认适配 **Windows11 AMD64 / Python3.11**，与你的 `meta64-fresh` 环境一致。代码、固定抽样、必要数据和结果回传工具均在此目录内。

| 数据用途 | 数量 | 用法 |
| --- | ---: | --- |
| Train1000 | 1000 | 训练模型、建立KNN库；不比较训练集效果 |
| Valid100 | 100 | 从原Valid300均匀无放回抽取，只做验证和模型选择 |
| ID测试 | 300 | 从原ID1000均匀无放回抽取，比较各方法 |
| OOD测试 | 300 | 从原OOD1000均匀无放回抽取，比较各方法 |
| CR测试 | 300 | 从原CR1000均匀无放回抽取，比较各方法 |

固定种子42，四个独立子随机流；抽样清单、原始索引和SHA见 [SAMPLING.json](data/META64_uniform_1000_100_300_v3/SAMPLING.json)。各方法使用完全相同的测试任务。数据包约16MB，解压后约0.88GB；不含测试参考柔度、测试优化拓扑或测试应变能答案。

## Windows PowerShell：安装、检查、运行

建议在较短路径，例如 `C:\work` 下克隆。使用稀疏克隆只下载这个目录：

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/hsksfje/topo.git
cd topo
git sparse-checkout set InitSkill-Benchmark
cd InitSkill-Benchmark
conda activate meta64-fresh
$env:PYTHONUTF8 = "1"

# 保留当前 NumPy1.26.4 / SciPy1.12.0 / Matplotlib3.8.4，安装Torch等依赖。
# JAX/CVXOPT装在独立.runtime环境，不升级主环境的NumPy。
python scripts/setup_env.py --profile cpu --use-current-main

# 解压、核对9200个文件及抽样；验证SCG/LM、FE梯度、CNN/GAN与断点续跑。
python benchmark.py verify
python -m unittest discover -s tests -v

# 正式运行全部8种方法 × 3类测试 × 每类300例。
python benchmark.py start
```

默认Windows模式：公共优化使用SciPy/SuperLU；PyTorch与JAX使用CPU。无需安装CUDA，也不需要openai SDK。JAX的WindowsCPU支持在官方文档中标为实验性，安装脚本固定版本并实测导入CVXOPT CHOLMOD；以 `verify` 的结果为准。版本与官方安装来源见 [环境说明](docs/ENVIRONMENT.md)。

如果希望完全不改现有Conda环境，省略 `--use-current-main`，脚本会另外建立主环境。Linux/WSL2＋NVIDIA可选择 `--profile cuda`，启用相同Q4/OC方程的cuDSS求解路径；不同后端的独立实验不要混入同一个 `results` 目录。

**OC上限按用户最新要求统一为500步，达到上限仍未满足条件即记为未收敛。** 全部任务的OC上限500步，保留C50/C100/C200；Meta/Early各6000次外层训练，cGAN需要1000＋100份同任务高分辨率教师，新生成教师也以500步为上限。已有训练/验证标签保留其历史来源，不重新定义为500步优化结果。`verify` 使用训练任务做工程检查，不把少量检查结果冒充正式Benchmark。

## 进度、停止、续跑

```powershell
python benchmark.py status
python benchmark.py analyze
python benchmark.py stop
python benchmark.py start
```

停止会保留最近完整检查点、模型、Adam状态、任务采样器和FE尝试账本。重复 `start` 跳过已完成任务。OC与Meta/Early支持已验证的精确续跑；PCA/CNN/cGAN的未完成训练阶段会按固定种子重新训练，不能把这些阶段称为完整优化器续跑。进程级失败写入 `results/campaign/ACTOR_FAILURE_*.json`；不要删除失败记录来制造更好的成绩。

主要文件：

- `results/campaign/logs/`：按阶段保留的日志。
- `results/models/<方法>/`：各方法独立模型、配置和训练历史。
- `results/early_trajectories/`：Early独立元训练，未从Meta模型热启动。
- `results/full_evaluation/<测试集>/<方法>/<任务>/`：初始场、逐步轨迹、终态或失败记录。
- `results/reports/INTERIM_REPORT_zh.md`：运行中两张对比表，仅统计测试集。
- `results/reports/FINAL_REPORT_zh.md` 和 `FINAL_REPORT_READY.json`：全部覆盖、模型、物理轨迹和账本核验通过后生成。

第一张表比较达到同任务Uniform终态柔度×1.01所需步数；未达到或失败按500步计入封顶均值，并单列失败。第二张表比较同预算和终态柔度；到500步上限不称充分收敛或全局最优。

## 跑完后回传

```powershell
python scripts/export_results.py
```

把打印出的整个 `returns\时间戳\` 文件夹回传，包含 `RETURN_MANIFEST.json` 和全部zip/分卷。默认每卷100MB，带完整SHA校验。导出保留模型、初始场、逐步柔度、终态、失败和FE账本，省略可重建的大型输入缓存及中间检查点。

若运行报错或被中断，先停止，再回传诊断：

```powershell
python benchmark.py stop
python scripts/export_results.py --allow-partial
```

另一设备在同版本的干净目录中导入：

```powershell
python scripts/import_return.py "回传文件夹"
python benchmark.py analyze
```

脚本拒绝覆盖不相同的已有产物。回传的Windows运行环境只作为证据保存，不会覆盖分析设备的Python路径。

## 方法、文献与验证边界

八种方法为Uniform、1NN、Ulu PCA＋SCG、Wang PCA＋LM、Yu CNN、Meta、Early和CNN＋cGAN，各自独立目录见 [方法与实现差异](docs/METHODS.md)。作者源码和许可证位于 `references/init_benchmark_v1/upstream/`；论文URL及原始SHA位于 `SOURCES.json`。需要PDF可运行 `python scripts/fetch_references.py`，正式计算不依赖PDF下载。

SCG/LM已用Python做数学/数值对照，未取得MATLAB二进制训练器的逐位一致性证明；不能把实现假设表述为作者明确参数。Meta/Early比较的是预测密度交给公共OC后的初始化效果，不是原论文神经参数空间在线Adam的速度排名。Windows的`longdouble`精度可能不同，预检记录实际位数，残差验收阈值不因此放宽。历史局限和全部假设见方法说明及历史审计。

[本机数值检查证据](docs/LOCAL_VERIFICATION.json)来自LinuxCPU，不冒称Windows完整实验成绩。原生Windows检查由仓库的跨平台CI和你本机的 `verify` 命令执行；最终研究结果必须来自实际完整运行。
