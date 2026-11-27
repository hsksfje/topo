# 已有Pro结果、尚无有效Flash结果：三种LLM方法独立重跑

本包针对META64 Train1000中**确认逐方法使用过Pro、但尚无有效Flash独立结果**的部分。重新生成Flash初态，不能复用Pro初态，也不把模型对比重跑当成修复原Pro实验。代码和本批358个任务的物理输入均已包含，无需下载完整数据集。

|方法|本轮任务数|输入与输出|
|---|---:|---|
|A1|358|工况→直接10×20粗密度矩阵，无FE提示|
|A3|354|同一轮Flash生成的A1矩阵→独立审阅与修改|
|B1|353|工况＋同任务新算Uniform初态FE摘要→粗密度矩阵|
|合计|1065个组合、358个工况|local200的496个组合＋deviceB300的569个组合|

A3任务均已包含在本轮A1中，无额外父提案调用；正常需1065次付费调用，默认硬上限1500次尝试（含重试）。使用`deepseek-v4-1-flash-261017`，`thinking.type=enabled`、`reasoning_effort=low`。已有有效Flash重跑的组合不进入本计划。它与`META64_retained23_backfill_v1`的686个组合**没有交集**。

## 更新与环境

已下载仓库，在仓库根目录：

```bash
git pull --ff-only
cd META64_Pro_remaining_to_Flash_low_v1
```

若已为686缺口补跑包建好`meta64-backfill`环境，直接激活：

```bash
conda activate meta64-backfill
python verify.py
python run.py --stage prepare
```

没有该环境则先运行：

```bash
conda env create -f environment.yml
conda activate meta64-backfill
```

依赖Python 3.11、NumPy 1.26.4、SciPy 1.12.0、Matplotlib 3.8.4、OpenAI SDK 3.6.0。不需要GPU/CUDA。本包会自动校验并解压`payload.zip`到`data/`，包含物理输入，不包含Pro初态、Pro响应、源最优拓扑或源应变能标签。比较用标量参考不进入LLM提示。

## 一条命令运行全部

凭据文件放在仓库之外，格式：

```dotenv
MODEL_API_KEY=在另一台设备自行配置的密钥
MODEL_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
```

Windows示例：

```bat
python -u start_all.py --execute-api --env-file C:/private/ark.env
```

Linux示例：

```bash
python -u start_all.py --execute-api --env-file /home/yourname/private/ark.env
```

先以4并发完成Flash提案，再自动启动CPU并行优化。默认进程数=可用逻辑CPU减1，每进程的BLAS/OpenMP为1线程；i7-10700的16个可用逻辑处理器通常用15个进程。Linux考虑亲和性及标准cgroup v2配额，Windows按进程可用CPU掩码计算。可以追加`--workers 15`减少并发。请不要同时在同一设备用两个包各开满全部CPU；按顺序运行或手动分配进程数。

默认沿用历史总上限2000步，提前收敛即停，同时保存第50步和500步统计。只跑500步：

```bat
python -u start_all.py --execute-api --env-file C:/private/ark.env --max-step 500
```

模型生成和数值计算也可分开：

```bat
python -u generate_missing.py --execute-api --env-file C:/private/ark.env --api-workers 4
python -u run.py --retry-failed
```

数值入口`run.py`不会发起API，没有新初态的组合会明确标记blocked，不会替换成Uniform。若API部分存在失败，一键入口仍继续计算其他已具备初态的任务，最终返回未解决状态。

API对408/409/429、5xx、网络连接与超时错误退避重试，60秒起逐步增至900秒，并尊重更长的Retry-After；每请求最多8次重试，全批最多1500次尝试，重试计入预算。请求回执持久化，已收到的响应不重复调用；不明结算状态需要核查，不盲目再扣费。永久鉴权或参数错误需修复后再继续。每10秒最多新发一个请求的全局节流仍保留，4个请求可同时等待响应，不能按CPU并发数估计API速度。凭据不上传GitHub，不进入结果回传包。

## 监控、恢复

另开终端，进入包目录并激活环境：

```bash
python monitor.py
```

500步运行：

```bash
python monitor.py --output outputs/flash_remaining_500_run01
```

监控显示API尝试/成功回执数量、数值结算数、收敛/上限/失败/缺初态计数和当前步数。

数值运行安全暂停：

```bash
python -c "from pathlib import Path; Path('outputs/flash_remaining_2000_run01/PAUSE_REQUESTED').touch()"
```

恢复数值计算：

```bash
python -u run.py --resume --retry-failed
```

500步运行需相应修改暂停路径并给恢复命令加`--max-step 500`。API阶段使用`generated/PAUSE_REQUESTED`暂停；恢复前删除该标记，再重跑原生成命令。不要删除请求回执。每10步保存数值检查点，已有成功结果不重跑。

## 输出、比较与回传

默认输出：

```text
generated/                         # Flash提案、请求回执、模型与用量
outputs/flash_remaining_2000_run01/
  config.json
  progress.json
  tasks/<任务ID>/<方法>/history.json
  tasks/<任务ID>/<方法>/checkpoint.npz
  tasks/<任务ID>/<方法>/state_0050.npz
  tasks/<任务ID>/<方法>/state_0500.npz
  tasks/<任务ID>/<方法>/state_final.npz
  tasks/<任务ID>/<方法>/result.json
  reports/results.csv
  reports/REPORT_zh.md
```

500步选项使用`outputs/flash_remaining_500_run01/`。500与2000步使用不同输出，不能改同一个输出的冻结上限。运行保留原SIMP/OC、惩罚3、滤波半径2.5、move=0.2、64×64网格；连续10次密度变化≤0.01、柔度相对变化≤1e-4、11个柔度状态相对极差≤5e-4、体积误差≤1e-8、FE残差<1e-6，最早60步认证。达到上限不称为收敛。

报告分别列出对Uniform、对同任务纯Pro的500步表现，并保留I50。只有双方500步内真正收敛，才计“更快且改善>1%”或“更快且柔度差异绝对值<1%”。运行中是部分配对样本，不代表全1000任务效果。Pro参考C50已与主归档逐条核对一致；Pro输入和结果不提供给Flash。

完成或暂停后导出：

```bash
python run.py --stage export
```

500步则：

```bash
python run.py --stage export --max-step 500
```

将`exports/`中新生成的ZIP和同名SHA256一起回传到原设备：

```text
InitSkill-Research/incoming/META64_Pro_remaining_to_Flash_low_v1/
```

归档包含新轨迹、提案、API回执、配置、报告和逐文件哈希；不含密钥。原设备按任务ID/方法合并为独立Flash实验，不覆盖Pro结果。

## 其余历史方法

`coverage_by_method.csv`列出历史全部10种方法：总计仍有3567个组合缺少有效Flash结果，其中74个曾尝试Flash但失败/无有效轨迹。本包只运行最终保留的三种方法，不自动扩跑A2、B2、B3、L1_PG、L1_H4、L2_GEOM、L3_FE。

纯Pro追溯范围是559个任务；其他任务若没有完整逐请求来源，不能直接断言使用Pro。尚未回传的其他设备结果无法在此确认。若另有Flash回传，请先合并去重再执行，避免重复调用。

`VALIDATION.json`注明数值内核继承验证与本包离线检查范围。本机未启动这1065次付费生成。
