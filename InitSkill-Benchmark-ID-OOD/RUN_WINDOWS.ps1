$ErrorActionPreference = "Stop"
python scripts/setup_env.py --profile cpu --cpus 28 --ram-gb 64
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python benchmark.py plan
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python benchmark.py verify
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python benchmark.py start
exit $LASTEXITCODE
