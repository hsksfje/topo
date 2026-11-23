#!/usr/bin/env python3
"""Read progress without importing numerical libraries or using API credentials."""
import argparse,json,time,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--cohort',choices=['all','original100','local200','deviceB300'],default='all');p.add_argument('--output',type=Path);p.add_argument('--pause',action='store_true');p.add_argument('--once',action='store_true');a=p.parse_args();out=a.output or ROOT/'outputs'/('flash_low_'+a.cohort+'_run01')
 if a.pause:
  assert out.exists(),'Output not prepared';(out/'PAUSE_REQUESTED').touch();print('Pause requested:',out);return
 while True:
  try:
   progress=json.loads((out/'progress.json').read_text(encoding='utf-8'));budget=json.loads((out/'budget_status.json').read_text(encoding='utf-8')) if (out/'budget_status.json').exists() else {};conv=progress.get('convergence') or {}
   stamp=datetime.datetime.fromtimestamp(progress['updated_unix']).astimezone().isoformat(timespec='seconds')
   print(f"updated={stamp} stage={progress['stage']} status={progress['status']} tasks={progress['tasks']} LLM50={progress['llm50']}/{progress['llm50_planned']} API={budget.get('attempts',0)}/{progress['max_api_calls']} convergence={conv.get('settled',0)}/{progress['tasks']*13}",flush=True)
   if progress['status']=='paused' and (out/'error.json').exists():print('Error:',json.loads((out/'error.json').read_text()).get('error'),flush=True)
  except (FileNotFoundError,json.JSONDecodeError):print('Waiting for progress:',out,flush=True)
  if a.once:return
  try:time.sleep(10)
  except KeyboardInterrupt:return
if __name__=='__main__':main()
