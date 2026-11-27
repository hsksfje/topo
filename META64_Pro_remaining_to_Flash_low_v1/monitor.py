#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'outputs/flash_remaining_500_run01')
p.add_argument('--interval',type=int,default=10)
p.add_argument('--once',action='store_true')
a=p.parse_args()
while True:
    f=a.output/'progress.json'
    if f.exists():
        try:
            r=json.loads(f.read_text())
            print(time.strftime('%H:%M:%S'),'state=',r['status'],'settled=',str(r['terminal'])+'/'+str(r['planned']),'counts=',r['counts'],'workers=',r['workers'],'active=',[(x['task_id'],x['method'],x['step']) for x in r['active']],flush=True)
        except (OSError,ValueError) as exc:print('Waiting for status:',exc,flush=True)
    else:print('Waiting for',f,flush=True)
    budget=Path(__file__).resolve().parent/'generated/budget_status.json'
    if budget.exists():
        try:
            b=json.loads(budget.read_text())
            print('API attempts=',b['attempts'],'received=',b['received'],'cap=',b['max_api_calls'],flush=True)
        except (OSError,ValueError):pass
    if a.once:break
    time.sleep(max(1,a.interval))
