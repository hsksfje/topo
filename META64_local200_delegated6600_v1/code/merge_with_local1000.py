#!/usr/bin/env python3
"""Validate and combine delegated6600 with local1000; no FE/API."""
import argparse,csv,json,sys,time
from pathlib import Path
from storage import dump,sha
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--local',required=True,type=Path);p.add_argument('--remote',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 local=a.local.resolve();remote=a.remote.resolve();dest=a.output.resolve();assert dest not in (local,remote)
 plan=read(ROOT/'provenance/delegation.json');sets=[set(plan['remote_jobs']),set(plan['local_jobs'])];assert len(sets[0])==6600 and len(sets[1])==1000 and not sets[0]&sets[1]
 for folder in (remote,local):
  assert read(folder/'progress.json')['status']=='complete' and (folder/'COMPLETION.json').exists()
  assert sha(folder/'config.json')==read(folder/'config_identity.json')['sha256']
 marker=read(remote/'RETURN_READY.json');assert sha(remote/'return_manifest.json')==marker['manifest_sha256']
 for name,h in read(remote/'return_manifest.json')['files'].items():
  rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts;assert sha(remote/rel)==h,name
 assert read(local/'config.json')['delegation_sha256']==sha(ROOT/'provenance/delegation.json')
 rc,lc=read(remote/'config.json'),read(local/'config.json')
 for k in ('solver','convergence','max_step'):assert rc[k]==lc[k],k
 rows=[];seen=set()
 for folder,expected in ((remote,sets[0]),(local,sets[1])):
  with (folder/'reports/results.csv').open(encoding='utf-8-sig') as f:part=list(csv.DictReader(f))
  keys={r['task_id']+'/'+r['method'] for r in part};assert keys==expected and len(part)==len(expected)
  assert not keys&seen;seen|=keys
  for r in part:r['source_output']=str(folder)
  rows+=part
 assert len(rows)==7600 and seen=={r['task_id']+'/'+m for r in rc['records'] for m in rc['methods']}
 dest.mkdir(parents=True,exist_ok=True);(dest/'reports').mkdir(exist_ok=True)
 dump(dest/'config.json',dict(rc,analysis_only=True,name='META64_local200_merged7600'))
 with (dest/'reports/results.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 dump(dest/'merge_audit.json',dict(remote=str(remote),local=str(local),records=7600,tasks=200,overlap=[],remote_return_hashes_verified=True,delegation_sha256=sha(ROOT/'provenance/delegation.json'),unix=time.time()))
 import analyze
 analyze.OUT=dest;analyze.analyze();print('MERGED7600',dest/'reports/SPEED_QUALITY_zh.md')
if __name__=='__main__':main()
