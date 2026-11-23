#!/usr/bin/env python3
"""Fresh META64 task input ->28 fixed +10 LLM OC50 -> common convergence."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,csv,hashlib,json,sys,time,shutil,traceback,threading,platform,importlib.metadata
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code/vendor'))
import numpy as np
from initskill.io import dump,sha,save_npz
from initskill.physics import mesh,SolverConfig
from initskill.methods.trajectory_skill50_v1.geometry import METHODS,teacher_pool
from initskill.methods.trajectory_skill50_v1.evaluator import evaluate
import llm_six as base
import llm_task
import extensions
import extra_initializers as extra
from extra50 import extra50
import continuation as cont
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
DEFAULT_MANIFEST=ROOT.parent/'META64_datasets_v1/metadata/manifests/train100_new100.jsonl'
DEFAULT_DATA=ROOT.parent/'META64_datasets_v1/extracted'
DEFAULT_OUT=ROOT/'outputs/train100_new100_run01'


def verify_code():
 p=ROOT/'PACKAGE_MANIFEST.json';identity=read(ROOT/'PACKAGE_IDENTITY.json')
 assert sha(p)==identity['sha256'],'Package manifest changed'
 for name,digest in read(p)['files'].items():assert sha(ROOT/name)==digest,'Package changed: '+name


def atomic_csv(path,rows):
 if not rows:return
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.tmp')
 with tmp.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 tmp.replace(path)


def prepare(args,workers):
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);verify_code()
 if (out/'config.json').exists():
  cfg=read(out/'config.json');assert sha(out/'config.json')==read(out/'config_identity.json')['sha256'];assert cfg['package_sha256']==sha(ROOT/'PACKAGE_MANIFEST.json');assert cfg['profile']==args.profile and cfg['manifest_sha256']==sha(args.manifest.resolve()),'Output directory belongs to another profile'
  for r in cfg['records']:
   for name,h in r['assets'].items():assert sha(out/r['path']/name)==h
  return cfg
 manifest=args.manifest.resolve();data=args.data_root.resolve();rows=[json.loads(l) for l in manifest.open() if l.strip()]
 assert rows and len({r['source_task_id'] for r in rows})==len(rows) and len({r['group_id'] for r in rows})==len(rows)
 assert all((r['nx'],r['ny'])==(64,64) for r in rows),'This benchmark runner supports64x64; test_cr is256x256 and is not silently resized'
 dump(out/'selection_snapshot.json',rows);records=[]
 geometry=out/'domains/64x64.npz';save_npz(geometry,**dict(zip(('nodes','cells','centers'),mesh(64,64))))
 for i,source in enumerate(rows,1):
  for a in source['assets'].values():assert sha(data/a['path'])==a['sha256'],'Raw array missing/changed: '+a['path']
  load=lambda key:np.load(data/source['assets'][key]['path'],allow_pickle=False)
  f=load('forces');bc=load('normals');vf=float(load('target_vf').item())
  assert f.shape==bc.shape==(65,65,2) and np.isfinite(f).all() and np.isfinite(bc).all();assert abs(vf-source['volume_fraction'])<1e-12
  force=f[:,::-1,:].reshape(-1).astype(float);fixed=np.flatnonzero(bc[:,::-1,:].reshape(-1));assert not np.any(force[fixed]!=0)
  tid='META64_'+source['source_task_id'];folder=out/'inputs'/tid;folder.mkdir(parents=True,exist_ok=True)
  shutil.copy2(geometry,folder/'domain.npz');save_npz(folder/'boundary_conditions.npz',force=force,fixed_dofs=fixed.astype(np.int64));save_npz(folder/'initial_state.npz',x=np.full(4096,vf))
  physical=dict(task_id=tid,elements_xy=[64,64],volume_fraction=vf,fixed_dofs=list(map(int,fixed)),forces_sparse=[[int(j),float(force[j])] for j in np.flatnonzero(force)])
  task=dict(schema='initskill-isotropic-task-v1',task_id=tid,family='META64',split=source['role'],group_id=source['group_id'],elements_xy=[64,64],volume_fraction=vf,source_key=source['source_key'],coordinate_mapping='author[x,y,axis] -> solver[x,64-y,axis]; signed force unchanged',source_optimization_results_used=False)
  dump(folder/'task.json',task);dump(folder/'physical_input.json',physical);dump(folder/'task_input.json',dict(task,**physical))
  records.append(dict(task_id=tid,group_id=source['group_id'],vf_bin=source['vf_bin'],volume_fraction=vf,path='inputs/'+tid,source_key=source['source_key'],assets={n:sha(folder/n) for n in ('domain.npz','boundary_conditions.npz','initial_state.npz','task.json','physical_input.json','task_input.json')}))
  if i%10==0:print('INPUTS',i,len(rows),flush=True)
 protocol=read(ROOT/'protocol.json')
 cfg=dict(protocol,records=records,workers=workers,manifest_sha256=sha(manifest),package_sha256=sha(ROOT/'PACKAGE_MANIFEST.json'),max_api_calls=min(args.max_api_calls,3300),profile=args.profile,new_initializers=True,created_unix=time.time(),raw_data_root=str(data),reference_labels_excluded_from_LLM=True)
 dump(out/'config.json',cfg);dump(out/'config_identity.json',dict(sha256=sha(out/'config.json')))
 dump(out/'environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=importlib.metadata.version('scipy'),matplotlib=importlib.metadata.version('matplotlib'),openai=importlib.metadata.version('openai')))
 return cfg


def api_authorized(cfg):
 auth=read(ROOT/'authorizations'/(cfg['profile']+'.json'))
 assert {r['task_id'] for r in cfg['records']}==set(auth['task_ids']),'Task set differs from the selected device profile'
 assert cfg['manifest_sha256']==auth['manifest_sha256'] and cfg['max_api_calls']<=auth['max_api_calls']
 assert cfg['provider_host']==auth['provider_host']
 return auth


def preflight(out,cfg):
 if (out/'preflight.json').exists():assert read(out/'preflight.json')['status']=='passed';return
 r=cfg['records'][0];m=base.SIMP(out/r['path']);x=m.initial();ev=m.evaluate(x);C0=ev['compliance']
 assert abs(ev['volume']-m.target)<1e-8
 for _ in range(2):x=m.update(x,ev);ev=m.evaluate(x);assert abs(ev['volume']-m.target)<1e-8
 # Verify the two decode families without making a network call.
 for a in (np.linspace(.3,.7,200).reshape(10,20),np.linspace(.3,.7,4096).reshape(64,64)):
  y=base.decode(m,a);assert abs(m.physical(y).mean()-m.target)<1e-8
 from code_runner import execute
 code='import numpy as np\ndef initialize(nx, ny, volfrac):\n    x,y=np.meshgrid(np.linspace(0,1,nx),np.linspace(1,0,ny))\n    return np.clip(volfrac+0.05*(x-y),0.01,0.99)\n'
 values=execute(code,64,64,m.target,out/'preflight_code');assert values.shape==(64,64)
 dump(out/'preflight.json',dict(status='passed',first_task=r['task_id'],Uniform_C0=C0,Uniform_C2=ev['compliance'],matrix_and_code_decode=True,AST_child_execution=True,new_API_calls=0))
 print('PREFLIGHT passed',r['task_id'],flush=True)


def fixed_job(out,r):
 out=Path(out);task=read(out/r['path']/'task_input.json');recipes=teacher_pool(task)
 for name in METHODS:
  folder=out/'corpus'/r['task_id']/name
  if (folder/'result_50.json').exists() or (folder/'failure.json').exists():continue
  evaluate(out,r,task,recipes[name],folder,50)
 for name in extra.METHODS:extra50(out,r,name)
 dump(out/'fixed_done'/(r['task_id']+'.json'),dict(task_id=r['task_id'],unix=time.time()))
 return r['task_id']


def source_folder(out,t,m,cfg):return out/('corpus' if m in cfg['fixed'] else 'tasks' if m in cfg['llm'] else 'extra')/t/m

def baseline_result(out,t,m,cfg):
 p=source_folder(out,t,m,cfg);name='result_50.json' if m in cfg['fixed'] else 'result.json'
 if (p/name).exists():return read(p/name)
 if (p/'failure.json').exists():return read(p/'failure.json')
 return None


def report(out,cfg,stage,status='running'):
 rows=[];us={}
 for r in cfg['records']:
  t=r['task_id'];v=baseline_result(out,t,'U',cfg)
  if v and v['status']=='complete':us[t]=v['C50']
  for m in cfg['methods']:
   v=baseline_result(out,t,m,cfg)
   if v is None:continue
   h=v.get('history',[]);c0=h[0]['compliance'] if h else v.get('C0');c50=v.get('C50')
   rows.append(dict(task_id=t,method=m,status=v['status'],C0=c0,C50=c50,I50=100*(1-c50/us[t]) if c50 and t in us else None,error=v.get('error')))
 atomic_csv(out/'reports/OC50_results.csv',rows)
 conv=read(out/'convergence/progress.json') if (out/'convergence/progress.json').exists() else None
 budget=read(out/'budget_status.json') if (out/'budget_status.json').exists() else dict(attempts=0,received=0)
 p=dict(stage=stage,status=status,planned_tasks=len(cfg['records']),methods=38,planned_records=len(cfg['records'])*38,fixed50_settled=sum(r['method'] not in cfg['llm'] for r in rows),fixed50_planned=len(cfg['records'])*28,llm50_settled=sum(r['method'] in cfg['llm'] for r in rows),llm50_planned=len(cfg['records'])*10,api_attempts=budget['attempts'],api_received=budget['received'],convergence=conv,updated_unix=time.time())
 dump(out/'progress.json',p)
 print(json.dumps({k:v for k,v in p.items() if k!='convergence'},ensure_ascii=False),flush=True)
 return p


def phase(pool,jobs,func,out,cfg,stage):
 pending={pool.submit(func,*args) for args in jobs}
 try:
  while pending:
   done,pending=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
   for f in done:print(stage,f.result(),flush=True)
   report(out,cfg,stage)
   if (out/'PAUSE_REQUESTED').exists():
    for f in pending:f.cancel()
    raise InterruptedError('Pause requested')
 except BaseException:
  (out/'PAUSE_REQUESTED').touch()
  for f in pending:f.cancel()
  raise


def continue_all(out,cfg,workers):
 conv=out/'convergence';conv.mkdir(exist_ok=True);cc=dict(cfg,workers=workers,fresh_output=str(out),trajectories={})
 for r in cfg['records']:
  for m in cfg['methods']:
   v=baseline_result(out,r['task_id'],m,cfg)
   if v is None:continue
   cc['trajectories'][r['task_id']+'/'+m]={'source_status':v['status']}
 if (conv/'config.json').exists():
  old=read(conv/'config.json');assert sha(conv/'config.json')==read(conv/'config_identity.json')['sha256']
  for k in ('solver','convergence','max_step','records','trajectories'):assert old[k]==cc[k],k
 else:
  dump(conv/'config.json',cc);dump(conv/'config_identity.json',dict(sha256=sha(conv/'config.json')))
 cont.OUT=conv;cont.write_csv=atomic_csv
 last=0.
 def update(status='running'):
  nonlocal last
  p=cont.numerical_report(cc,status);report(out,cfg,'convergence',status)
  if (conv/'reports/results.csv').exists() and (time.monotonic()-last>30 or status!='running'):
   import analyze
   analyze.OUT=conv;analyze.analyze();last=time.monotonic()
  return p
 update();jobs=[(r,'U') for r in cfg['records']]+[(r,m) for r in cfg['records'] for m in cfg['methods'] if m!='U'];jobs=[(r,m) for r,m in jobs if not (conv/'tasks'/r['task_id']/m/'result.json').exists()]
 try:
  with ProcessPoolExecutor(max_workers=workers) as pool:
   pending={pool.submit(cont.run_one,r,m,cc,str(conv)) for r,m in jobs}
   while pending:
    done,pending=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
    for f in done:
     v=f.result();print('CONVERGENCE',v['task_id'],v['method'],v['status'],v.get('step'),flush=True)
    update()
    if (out/'PAUSE_REQUESTED').exists() or (conv/'PAUSE_REQUESTED').exists():
     (conv/'PAUSE_REQUESTED').touch()
     for f in pending:f.cancel()
     raise InterruptedError('Pause requested')
  update('complete');dump(conv/'COMPLETION.json',dict(unix=time.time()));dump(out/'COMPLETION.json',dict(unix=time.time(),tasks=len(cfg['records'])))
  from export_results import seal_output
  seal_output(out)
 except BaseException:
  (conv/'PAUSE_REQUESTED').touch();update('paused' if (out/'PAUSE_REQUESTED').exists() else 'stopped');raise


def main():
 p=argparse.ArgumentParser();p.add_argument('--profile',choices=['local200','deviceA300','deviceB300']);p.add_argument('--stage',choices=['verify','prepare','preflight','fixed','run','report'],default='run');p.add_argument('--manifest',type=Path,default=None);p.add_argument('--data-root',type=Path,default=DEFAULT_DATA);p.add_argument('--output',type=Path,default=None);p.add_argument('--workers',type=int);p.add_argument('--api-workers',type=int,default=None);p.add_argument('--max-api-calls',type=int,default=None);p.add_argument('--execute-api',action='store_true');p.add_argument('--env-file',type=Path,default=ROOT/'.env.local');a=p.parse_args()
 if a.stage=='verify':verify_code();print('PACKAGE VERIFIED; choose local200/deviceA300/deviceB300; no API',flush=True);return
 if a.profile is None:p.error('Select --profile local200, deviceA300 or deviceB300; no default batch is silently started')
 profile=read(ROOT/'profiles.json')[a.profile]
 a.manifest=a.manifest if a.manifest is not None else ROOT/profile['manifest']
 if sha(a.manifest.resolve())!=profile['manifest_sha256']:p.error('Manifest must match the selected profile')
 a.output=a.output if a.output is not None else ROOT/'outputs'/profile['output_subdir']
 a.max_api_calls=a.max_api_calls if a.max_api_calls is not None else profile['max_api_calls']
 cpus=cont.available_cpus();workers=a.workers if a.workers is not None else max(1,len(cpus)-1)
 if not 1<=workers<=max(1,len(cpus)-1):p.error('workers must reserve one available logical CPU')
 if not 1<=a.max_api_calls<=profile['max_api_calls']:p.error('API cap exceeds the selected profile allowance')
 if a.api_workers is None:a.api_workers=min(2,workers)
 if not 1<=a.api_workers<=workers:p.error('api-workers must not exceed numerical workers; pass a smaller value on small machines')
 if a.stage=='verify':verify_code();print('PACKAGE VERIFIED; no API',flush=True);return
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 with (out/'.run.lock').open('a+b') as lock:
  cont.acquire_lock(lock);cfg=prepare(a,workers);cfg=dict(cfg,workers=workers,max_api_calls=min(cfg["max_api_calls"],a.max_api_calls))
  if a.stage=='prepare':report(out,cfg,'prepared','prepared');return
  if a.stage=='report':report(out,cfg,'manual_report');return
  if a.stage=='run' and a.execute_api:
   api_authorized(cfg)
   from api_client import read_credentials
   from urllib.parse import urlparse
   credential=read_credentials(a.env_file.resolve());endpoint=urlparse(credential['MODEL_API_BASE_URL'])
   assert endpoint.scheme=='https' and endpoint.hostname==cfg['provider_host'],'Provider URL must use the authorized HTTPS host'
   assert credential['MODEL_API_KEY']!='REPLACE_WITH_YOUR_LOCAL_KEY','Fill your local API key before starting the full run'
  preflight(out,cfg)
  if a.stage=='preflight':return
  if (out/'PAUSE_REQUESTED').exists() or (out/'convergence/PAUSE_REQUESTED').exists():raise InterruptedError('Pause requested; remove both flags only after the process exits to resume')
  allowed=cont.reserve_one_cpu(cpus);dump(out/'process.json',dict(pid=os.getpid(),started_unix=time.time(),workers=workers,api_workers=a.api_workers,available_cpus=cpus,affinity=allowed,reserved_cpu=cpus[0] if len(cpus)>1 else None))
  print('CPU available',len(cpus),'numerical workers',workers,'API concurrency',a.api_workers,flush=True)
  try:
   if not (out/'fixed_complete.json').exists():
    jobs=[(str(out),r) for r in cfg['records'] if not (out/'fixed_done'/(r['task_id']+'.json')).exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:phase(pool,jobs,fixed_job,out,cfg,'fixed28_OC50')
    dump(out/'fixed_complete.json',dict(unix=time.time()))
   if a.stage=='fixed' or not a.execute_api:
    report(out,cfg,'fixed_complete_waiting_api','waiting');return
   auth=api_authorized(cfg);dump(out/'USER_AUTHORIZATION.json',auth)
   from api_client import Client,read_credentials
   cred=read_credentials(a.env_file.resolve())
   from urllib.parse import urlparse
   assert urlparse(cred['MODEL_API_BASE_URL']).hostname==cfg['provider_host']
   client=Client(out,cfg,a.env_file.resolve());extensions.NUMERIC=threading.BoundedSemaphore(workers)
   if not (out/'llm_complete.json').exists():
    jobs=[(out,cfg,client,r) for r in cfg['records'] if not (out/'tasks'/r['task_id']/'complete.json').exists()]
    with ThreadPoolExecutor(max_workers=a.api_workers) as pool:phase(pool,jobs,llm_task.llm_job,out,cfg,'LLM10_OC50')
    client.budget();dump(out/'llm_complete.json',dict(unix=time.time()))
   continue_all(out,cfg,workers)
  except BaseException as exc:
   dump(out/'error.json',dict(error=str(exc),traceback=traceback.format_exc(),unix=time.time()));report(out,cfg,'paused' if (out/'PAUSE_REQUESTED').exists() else 'stopped','paused' if (out/'PAUSE_REQUESTED').exists() else 'stopped');raise
if __name__=='__main__':main()
