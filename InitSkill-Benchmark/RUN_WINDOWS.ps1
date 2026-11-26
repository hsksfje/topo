$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
Set-Location $PSScriptRoot
python scripts/setup_env.py --profile cpu --use-current-main
if ($LASTEXITCODE -ne 0) { throw "Environment setup failed" }
python benchmark.py verify
if ($LASTEXITCODE -ne 0) { throw "Preflight failed; inspect its output before starting" }
python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Platform tests failed" }
python benchmark.py start
if ($LASTEXITCODE -ne 0) { throw "Benchmark interrupted or failed; use export_results.py --allow-partial" }
python scripts/export_results.py
if ($LASTEXITCODE -ne 0) { throw "Result export failed" }
