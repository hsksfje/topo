import time,json,traceback
from pathlib import Path
import numpy as np
from initskill.io import dump,save_npz
from llm_six import SIMP
import extra_initializers as extra
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def extra50(out,r,name):
 OUT=Path(out);BASE=OUT
 folder=OUT/'extra'/r['task_id']/name;folder.mkdir(parents=True,exist_ok=True)
 if(folder/'result.json').exists():return read(folder/'result.json')
 if(OUT/'PAUSE_REQUESTED').exists():raise InterruptedError('User pause')
 m=SIMP(BASE/r['path']);started=time.monotonic();history=[];initializing=True
 cost=read(folder/'cost.json') if(folder/'cost.json').exists() else dict(attempted_fine_FE=0,initialization_fine_FE=0,attempted_coarse_FE=0)
 original=m.evaluate
 def counted(x):
  cost['attempted_fine_FE']+=1;cost['initialization_fine_FE']+=int(initializing);cost['pending_fine']=True;dump(folder/'cost.json',cost)
  try:return original(x)
  finally:cost['pending_fine']=False;dump(folder/'cost.json',cost)
 def coarse_event(event):
  if event=='start':cost['attempted_coarse_FE']+=1
  cost['pending_coarse']=event=='start';dump(folder/'cost.json',cost)
 m.evaluate=counted;m.coarse_audit_hook=coarse_event
 try:
  checkpoint=folder/'checkpoint.npz'
  if checkpoint.exists():
   with np.load(checkpoint,allow_pickle=False) as a:x=a['x'].copy();history=json.loads(str(a['history_json']))
   initializing=False;ev=m.evaluate(x)
  else:
   x,meta=extra.generate(m,name);dump(folder/'initializer.json',meta);initializing=False
   ev=m.evaluate(x);history=[dict(step=0,**{k:float(ev[k]) for k in ['compliance','volume','residual','grayness']})];save_npz(folder/'state_0000.npz',x=x,density=ev['density'],u=ev['u'])
   save_npz(checkpoint,x=x,history_json=json.dumps(history))
  for step in range(history[-1]['step']+1,51):
   if(OUT/'PAUSE_REQUESTED').exists():raise InterruptedError('User pause')
   x=m.update(x,ev);ev=m.evaluate(x)
   if abs(ev['volume']-m.target)>1e-8:raise ValueError('Volume failed')
   history.append(dict(step=step,**{k:float(ev[k]) for k in ['compliance','volume','residual','grayness']}))
   if step%10==0:save_npz(checkpoint,x=x,history_json=json.dumps(history));dump(folder/'history.json',history)
  save_npz(folder/'state_0050.npz',x=x,density=ev['density'],u=ev['u']);dump(folder/'history.json',history)
  result=dict(task_id=r['task_id'],method=name,status='complete',C0=history[0]['compliance'],C50=history[-1]['compliance'],updates=50,cost=cost,seconds=time.monotonic()-started,max_residual=max(h['residual'] for h in history))
 except InterruptedError:raise
 except Exception as exc:result=dict(task_id=r['task_id'],method=name,status='failed',C0=history[0]['compliance'] if history else None,C50=None,error=str(exc),traceback=traceback.format_exc(),cost=cost)
 dump(folder/'result.json',result);return result
