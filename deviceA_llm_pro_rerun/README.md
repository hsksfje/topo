# 设备A LLM 任务:用 flash 重跑 pro 那批

之前设备A的 LLM 阶段有 **166 个任务是用 `deepseek-v4-pro-ga-260813`(pro)跑的**,
其余 134 个任务已经是 `deepseek-v4-1-flash-261017`(flash)。
为了让 LLM 结果统一成 flash,这里提供"只重跑 pro 那 166 个任务"的清单和跳过脚本。

## 本目录文件
- `pro_tasks.json` —— 用 pro 跑过的 166 个 task_id(要重跑);
- `skip_flash_tasks.json` —— 已经是 flash 的 134 个 task_id(跳过);
- `setup_skip.py` —— 在输出目录里创建跳过标记(跳过固定阶段 + 跳过 flash 任务);
- 本 README。

## 前置条件
- 已 clone 仓库并 `git pull` 到最新;
- conda 环境 `meta64-fresh`(Python 3.11 + numpy 1.26.4 + scipy 1.12.0 + matplotlib 3.8.4 + openai 3.6.0);
- 已配置 `META64_train1000_distributed_v1/.env.local`(MODEL_API_KEY 填 flash 可用的 key,接口保持 `/api/v3`)。

## 步骤(在 topo 仓库根目录)

```bash
# 1. 拉代码
git pull --ff-only

# 2. 准备数据(会复用已下载的)
conda activate meta64-fresh
cd META64_train1000_distributed_v1
python ../META64_datasets_v1/fetch_data.py --subset train1000

# 3. 先做 prepare(生成输出目录和配置)
python run_deviceA300.py --stage prepare

# 4. 创建跳过标记(跳过固定阶段 + 跳过已经是 flash 的 134 个任务)
python ../deviceA_llm_pro_rerun/setup_skip.py outputs/deviceA300_run01

# 5. 正式跑(固定阶段被跳过,只重跑 166 个 pro 任务,用 flash + low 思考)
python -u run_deviceA300.py --api-workers 6
```

## 结果
- 输出目录 `outputs/deviceA300_run01/` 里,这 166 个任务的 LLM 结果会被 flash 重新生成;
- 收敛阶段(continue_all)会自动跳过"没有 OC50 结果"的任务,只对重跑的这 166 个任务做收敛;
- 跑完后把 `outputs/deviceA300_run01/` 回传给设备A,用于替换掉原来 pro 的那批结果。

## 注意
- 混合任务(4 个同时有 pro 和 flash 的)也会被整体重跑一遍(flash),结果是统一的;
- `--api-workers` 可按机器核数调整,建议 4~6;过大会触发限流(代码里已有自动重试兜底)。
