# 28 CPU / 64GB 配置

在已安装环境的目录执行：

```powershell
python benchmark.py stop
git pull
python benchmark.py configure --cpus 28 --ram-gb 64
python benchmark.py status
python benchmark.py start
```

若尚未安装，可直接 `python scripts/setup_env.py --profile cpu --use-current-main --cpus 28 --ram-gb 64`，随后先执行 `python benchmark.py verify`。

分配为：ID/OOD小网格2个任务；CR与高分辨率教师共享24个大网格任务名额；为训练/调度保留约2核。并非CR24＋教师24；进程池可能存在等待中的进程，但同时计算的大网格任务合计不超过24。NumPy/BLAS、PyTorch计算线程均为1，实际CPU使用随阶段变化。

按每个大任务约2GiB估算内存预算，64GiB中预留16GiB给系统、训练和其他运行开销。单进程Linux/SciPy1.12/256×256的3次训练任务FE测得峰值约1.12GiB；这不是Windows全程峰值保证。改变并发不改变Train1000、Valid100、每类Test300和OC500步上限。

本更新还修复了原生Windows大任务退出时不适用的POSIX malloc_trim清理路径。已有500步协议的唯一resources.py来源变更会自动保存旧/新协议快照；已完成/失败任务原始身份不变，未完成OC检查点与执行转换单独留档。未知代码或数据变更不会自动放行。旧模型无需因并发修改而重训。
