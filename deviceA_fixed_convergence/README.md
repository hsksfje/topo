# 设备A 固定初始化收敛子集(协助跑包)

本目录用于把设备A剩余的一部分**固定(非LLM)初始化收敛任务**交给另一台设备帮忙跑。
数据包是 `deviceA_fixed_convergence.zip.001` ~ `deviceA_fixed_convergence.zip.006` 共 6 个分片。

## 前置条件
- 已 clone 仓库并 `git pull` 到最新;
- conda 环境 `meta64-fresh`(或等价:Python 3.11 + numpy 1.26.4 + scipy 1.12.0 + matplotlib 3.8.4;**不需要 openai**)。

## 步骤

### 1. 拉代码(在 topo 仓库根目录)
```bash
git pull --ff-only
```

### 2. 合并分片成一个 zip
在 `topo/deviceA_fixed_convergence/` 目录下:

Windows(PowerShell,用 Python 合并最稳):
```powershell
python -c "import pathlib; p=pathlib.Path('.'); data=b''.join(f.read_bytes() for f in sorted(p.glob('deviceA_fixed_convergence.zip.*'))); pathlib.Path('full.zip').write_bytes(data)"
```

Linux/macOS:
```bash
cat deviceA_fixed_convergence.zip.* > full.zip
```

### 3. 解压
```bash
# 例如解压到 ../deviceA_fixed_data
python -c "import zipfile,pathlib; zipfile.ZipFile('full.zip').extractall('../deviceA_fixed_data')"
```
解压后目录结构应为:
```
deviceA_fixed_data/
  config.json
  corpus/<task_id>/<method>/state_0050.npz + history.json
  extra/<task_id>/<method>/state_0050.npz + history.json
  inputs/<task_id>/...
```

### 4. 跑收敛
```bash
conda activate meta64-fresh    # 或你的等价环境
python run_fixed_convergence.py \
    --data-dir ../deviceA_fixed_data \
    --output ../deviceA_fixed_results
```
- `--workers N` 可选,默认"可用CPU数-1";
- 结果写到 `../deviceA_fixed_results/tasks/<task_id>/<method>/result.json` 等;
- 断点续跑:脚本会跳过已有 `result.json` 的任务,中断后重跑同一命令即可续上。

### 5. 回传结果
跑完后把 `../deviceA_fixed_results/tasks/` 整个目录打包回传给设备A即可(设备A侧会按 `task_id/method/result.json` 合并)。
