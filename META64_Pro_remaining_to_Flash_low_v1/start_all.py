#!/usr/bin/env python3
"""Explicit paid generation followed by all CPU jobs, with shared CLI settings."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--execute-api',action='store_true',required=True)
p.add_argument('--env-file',type=Path,required=True)
p.add_argument('--workers',type=int)
p.add_argument('--api-workers',type=int,default=4)
p.add_argument('--max-step',type=int,choices=[500],default=500)
p.add_argument('--resume',action='store_true')
a=p.parse_args()
generation=subprocess.call([sys.executable,'-u',str(ROOT/'generate_missing.py'),'--execute-api','--env-file',str(a.env_file.resolve()),'--api-workers',str(a.api_workers)])
if generation:
    print('Some generation jobs remain unresolved; continuing all available numerical jobs.',flush=True)
cmd=[sys.executable,'-u',str(ROOT/'run.py'),'--max-step',str(a.max_step),'--retry-failed']
if a.workers is not None:cmd+=['--workers',str(a.workers)]
if a.resume:cmd+=['--resume']
numerical=subprocess.call(cmd)
raise SystemExit(numerical or generation)
