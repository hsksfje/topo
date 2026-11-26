# 环境与安装来源

用户设备：Windows11 AMD64，Python3.11.18，Conda `meta64-fresh`，NumPy1.26.4、SciPy1.12.0、Matplotlib3.8.4。独立目录不使用LLM/API，因此不要求升级或使用openai SDK。

主环境固定前三项并安装PyTorch2.5.1。默认CPU命令来自[PyTorch官方历史版本](https://pytorch.org/get-started/previous-versions/)：

```powershell
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

`scripts/setup_env.py`执行这些步骤并创建隔离的JAX环境。JAX0.6.2/jaxlib0.6.2、Equinox0.13.4、Optax0.2.6、CVXOPT1.3.3及相应NumPy/SciPy版本固定在requirements中；独立JAX环境的NumPy2.4.2不会覆盖主环境1.26.4。绘图库的解析结果和全部实际安装版本写入 `.runtime/*_environment.json`。

官方资料：[JAX安装与平台支持](https://docs.jax.dev/en/latest/installation.html)、[CVXOPT安装](https://cvxopt.org/install/)。已检查官方PyPI的Python3.11 WindowsAMD64分发：存在jaxlib0.6.2和cvxopt1.3.3 wheel，后者包含 `cvxopt/cholmod.cp311-win_amd64.pyd`。实际DLL加载、JIT和数值验证由预检执行，安装包存在不等于已经在用户设备跑完实验。

原生Windows不使用Linux `.so`、`fcntl`、CUDA稀疏求解器或需要管理员权限的符号链接。使用跨平台文件锁和原子提交指针；路径统一保存为POSIX相对形式，便于把结果带回Linux分析。

默认CPU公共后端统一为SuperLU-float64；Windows的longdouble通常没有Linux平台的额外尾数位，预检记录真实值。保留残差<1e-6、体积分数和停止条件，数值拒绝仍记为失败。

Linux/WSL2的CUDA选项额外固定Torch-cu124、cuDSS、CuPy和nvmath版本，依赖安装报告也纳入协议SHA。CPU与CUDA是两次独立的固定后端实验；不对跨设备、跨后端或不同并发负载的原始墙钟时间做受控排名。
