#!/usr/bin/env python3
import os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if any(a=='--profile' or a.startswith('--profile=') for a in sys.argv[1:]):raise SystemExit('Use the launcher for the intended device; do not override its profile')
os.chdir(ROOT)
raise SystemExit(subprocess.call([sys.executable,'-u',str(ROOT/'code/run.py'),'--profile','local200','--stage','run','--execute-api',*sys.argv[1:]]))
