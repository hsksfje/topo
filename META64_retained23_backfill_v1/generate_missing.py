#!/usr/bin/env python3
"""Optional PAID Flash-low generation for jobs without a historical start."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/vendor')]
from initskill.io import dump, save_npz
from meta64_effort_client import EffortClient
import llm_six as base
import portable_lock
sys.path.insert(0,str(ROOT))
from run import prepare_payload, read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-api',action='store_true')
    p.add_argument('--env-file',type=Path,required=True)
    p.add_argument('--model',default='deepseek-v4-1-flash-261017')
    p.add_argument('--api-workers',type=int,default=4)
    p.add_argument('--max-api-calls',type=int,default=200,help='hard cap including retry attempts')
    a=p.parse_args()
    if not a.execute_api:p.error('--execute-api is required to enable paid calls')
    assert 1<=a.api_workers<=8 and a.max_api_calls>0
    prepare_payload()
    out=ROOT/'generated';out.mkdir(exist_ok=True)
    lock=(out/'.generation.lock').open('a+b')
    portable_lock.flock(lock,portable_lock.LOCK_EX|portable_lock.LOCK_NB)
    cfg=dict(model=a.model,api_reasoning_effort='low',provider_reasoning_mapping='low',provider_host='ark.cn-beijing.volces.com',reasoning_max_tokens=32768,max_input_bytes=300000,max_api_calls=a.max_api_calls,thinking='enabled',lineage='NEW_PROPOSAL_not_historical_recovery')
    if (out/'config.json').exists():
        assert read(out/'config.json')==cfg,'Generation identity changed; use a new package copy for another model or budget'
    else:dump(out/'config.json',cfg)
    client=EffortClient(out,cfg,a.env_file.resolve())
    groups=defaultdict(set)
    for j in read(ROOT/'plan.json')['jobs']:
        if j['initialization']=='requires_new_LLM':groups[j['task_id']].add(j['method'])
    def one(tid,methods):
        task=base.explicit_task(read(ROOT/'data/inputs'/tid/'physical_input.json'))
        record=dict(task_id=tid)
        if methods & {'A1','A3'}:
            suffixes=('1','3') if 'A3' in methods else ('1',)
            base.generate_pair(out,client,record,task,'A',suffixes=suffixes)
        if 'B1' in methods:
            folder=out/'tasks'/tid
            if not (folder/'observation.json').exists():
                m=base.SIMP(ROOT/'data/inputs'/tid)
                ev=m.evaluate(m.initial())
                dump(folder/'observation.json',base.observation(m,ev))
                dump(folder/'observation_cost.json',dict(attempted_FE=1,source='new same-task Uniform FE'))
            base.generate_pair(out,client,record,task,'B',obs=read(folder/'observation.json'),suffixes=('1',))
        statuses={m:read(out/'tasks'/tid/m/'generation.json')['status'] for m in methods}
        print(tid,statuses,flush=True)
        return statuses
    bad=0
    with ThreadPoolExecutor(max_workers=a.api_workers) as pool:
        futures=[pool.submit(one,t,m) for t,m in groups.items()]
        for f in as_completed(futures):
            try:bad+=sum(x!='valid' for x in f.result().values())
            except Exception as exc:
                bad+=1;print(type(exc).__name__,str(exc),flush=True)
    print('API generation finished; invalid/failed tasks:',bad,'budget:',client.budget())
    lock.close()
    if bad:raise SystemExit(2)


if __name__=='__main__':main()
