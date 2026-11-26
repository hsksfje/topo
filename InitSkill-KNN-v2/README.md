# InitSkill KNN v2

Independent CPU benchmark: Uniform, original1NN retrieval, and repaired3NN initialization. Includes a platform-independent Decimal40 finite-element refinement fallback, sign-invariant smoothed descriptors and explicit softening.

See [完整中文运行说明](README_zh.md).

```powershell
python setup.py
python run.py verify
python run.py start --cpus 28 --ram-gb 64
python run.py export
```

Train1000, Valid100, ID/OOD/CR each300; all3 methods run the full1000 validation/test tasks (3000 optimization runs), up to500 OC steps each. This is an exploratory repair after v1 test auditing. Numerical preflight is not a full benchmark result.
