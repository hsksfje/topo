#!/usr/bin/env python3
"""Independent Flash-low rerun, no reuse of old model proposals or FE feedback."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,csv,hashlib,json,math,multiprocessing,shutil,statistics,sys,time,traceback,zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,wait,FIRST_COMPLETED
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/vendor')]
import run as original
import continuation as cont
import llm_six as base
import meta64_parallel_pipeline as pipeline
from meta64_effort_client import EffortClient
from initskill.io import dump,sha

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csvread(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
 if rows:original.atomic_csv(p,rows)
def verify():
 m=read(ROOT/'PACKAGE_MANIFEST.json')
 assert sha(ROOT/'PACKAGE_MANIFEST.json')==read(ROOT/'PACKAGE_IDENTITY.json')['sha256']
 for name,h in m['files'].items():assert sha(ROOT/name)==h, name
 print('PACKAGE VERIFIED; no API',flush=True)

def prepare(a):
 template=read(ROOT/'selection.json');records=[r for r in template['records'] if a.cohort=='all' or r['source_cohort']==a.cohort]
 if a.limit:records=records[:a.limit]
 assert records
 workers=a.workers if a.workers is not None else max(1,len(cont.available_cpus())-1)
 assert 1<=workers<=max(1,len(cont.available_cpus())-1),'Reserve one available logical CPU'
 api_workers=a.api_workers if a.api_workers is not None else min(4,workers)
 assert 1<=api_workers<=workers
 out=a.output.resolve() if a.output else ROOT/'outputs'/('flash_low_'+a.cohort+(f'_first{a.limit}' if a.limit else '')+'_run01')
 cap=a.max_api_calls if a.max_api_calls else math.ceil(11*len(records)*1.2)
 assert cap>0
 cfg=dict(template,records=records,workers=workers,api_workers=api_workers,max_api_calls=cap,package_sha256=sha(ROOT/'PACKAGE_MANIFEST.json'),cohort=a.cohort)
 out.mkdir(parents=True,exist_ok=True)
 if (out/'config.json').exists():
  old=read(out/'config.json');assert sha(out/'config.json')==read(out/'config_identity.json')['sha256']
  for k in ('records','model','api_reasoning_effort','solver','convergence','max_step','max_api_calls','package_sha256'):assert old[k]==cfg[k],f'Changed frozen setting: {k}'
 else:
  dump(out/'config.json',cfg);dump(out/'config_identity.json',dict(sha256=sha(out/'config.json')))
 hashes=read(ROOT/'provenance/input_hashes.json')
 with zipfile.ZipFile(ROOT/'physical_inputs.zip') as z:
  for r in records:
   for name,h in r['assets'].items():
    rel=r['path']+'/'+name;assert hashes[rel]==h
    target=out/rel
    if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(rel))
    assert sha(target)==h,rel
 return out,cfg,workers,api_workers

def fixed_one(out,r):
 out=Path(out);recipes=original.teacher_pool(read(out/r['path']/'task_input.json'))
 for m in ('U','PG2','H4_B035'):
  folder=out/'corpus'/r['task_id']/m
  if not (folder/'result_50.json').exists() and not (folder/'failure.json').exists():original.evaluate(out,r,read(out/r['path']/'task_input.json'),recipes[m],folder,50)
 return r['task_id']

def progress(out,cfg,stage,status='running'):
 rows=[];uniform={};settled=0
 for r in cfg['records']:
  tid=r['task_id'];v=original.baseline_result(out,tid,'U',cfg)
  if v and v.get('status')=='complete':uniform[tid]=v['C50']
  for m in cfg['methods']:
   v=original.baseline_result(out,tid,m,cfg)
   if not v:continue
   settled+=1;c=v.get('C50')
   rows.append(dict(task_id=tid,method=m,status=v['status'],C50=c,I50=100*(1-c/uniform[tid]) if c and tid in uniform else None,error=v.get('error')))
 csvwrite(out/'reports/OC50_results.csv',rows)
 b=read(out/'budget_status.json') if (out/'budget_status.json').exists() else {}
 cp=read(out/'convergence/progress.json') if (out/'convergence/progress.json').exists() else None
 p=dict(stage=stage,status=status,tasks=len(cfg['records']),model=cfg['model'],reasoning_effort='low',baseline50=sum(r['method'] not in cfg['llm'] for r in rows),llm50=sum(r['method'] in cfg['llm'] for r in rows),llm50_planned=len(cfg['records'])*10,api_attempts=b.get('attempts',0),max_api_calls=cfg['max_api_calls'],convergence=cp,updated_unix=time.time())
 dump(out/'progress.json',p);print(json.dumps({k:v for k,v in p.items() if k!='convergence'}),flush=True)
 return p

def phase(pool,jobs,fn,out,cfg,stage):
 pending={pool.submit(fn,*job) for job in jobs}
 try:
  while pending:
   done,pending=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
   for f in done:print(stage,f.result(),flush=True)
   progress(out,cfg,stage)
   if (out/'PAUSE_REQUESTED').exists():raise InterruptedError('Pause requested')
 except BaseException:
  (out/'PAUSE_REQUESTED').touch()
  for f in pending:f.cancel()
  raise

def llm_one(out,cfg,client,r):
 pipeline.llm_job(out,cfg,client,r)
 dump(Path(out)/'tasks'/r['task_id']/'complete.json',dict(task_id=r['task_id'],finished_unix=time.time(),model=cfg['model'],effort='low'))
 return r['task_id']

def reports(out,cfg):
 report=out/'reports/REPORT_zh.md';lines=['# Flash 4.1 low独立重跑','',f'任务数：{len(cfg["records"])}；10种LLM＋Uniform、PG2、H4_B035对照。模型：`{cfg["model"]}`；思考：`low`。','', '正改善表示柔度更低。各行只比较同一任务双方均收敛的记录；停止步不含API等待，不能作端到端耗时排名。','']
 cp=out/'convergence/reports/results.csv'
 if cp.exists():
  rows=csvread(cp);u={r['task_id']:r for r in rows if r['method']=='U' and r['status']=='converged'}
  fields=['方法','配对数','平均I50%','平均最终改善%','中位停止步','更快且改善>1%','更快且接近±1%','更快但变差>1%','更慢但改善>1%','更慢但改善>5%']
  lines+=['| '+' | '.join(fields)+' |','| '+' | '.join(['---']*len(fields))+' |'];summaries=[]
  for m in cfg['methods']:
   rr=[r for r in rows if r['method']==m and r['status']=='converged' and r['task_id'] in u];g=[100*(1-float(r['Cend'])/float(u[r['task_id']]['Cend'])) for r in rr];steps=[float(r['stop_step']) for r in rr];f=[float(r['stop_step'])<float(u[r['task_id']]['stop_step']) for r in rr];slow=[float(r['stop_step'])>float(u[r['task_id']]['stop_step']) for r in rr]
   vals=[m,len(rr),statistics.mean([float(r['I50']) for r in rr]) if rr else None,statistics.mean(g) if rr else None,statistics.median(steps) if rr else None,sum(x and y>1 for x,y in zip(f,g)),sum(x and abs(y)<=1 for x,y in zip(f,g)),sum(x and y< -1 for x,y in zip(f,g)),sum(x and y>1 for x,y in zip(slow,g)),sum(x and y>5 for x,y in zip(slow,g))]
   summaries.append(dict(zip(fields,vals)));lines.append('| '+' | '.join('—' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v) for v in vals)+' |')
  csvwrite(out/'reports/summary.csv',summaries)
  old={(r['task_id'],r['method']):r for r in csvread(ROOT/'provenance/historical_convergence.csv')};lineage={(r['task_id'],r['method']):r for r in csvread(ROOT/'provenance/source_method_lineage.csv')};comparisons=[]
  for r in rows:
   key=r['task_id'],r['method'];o=old.get(key);l=lineage.get(key)
   if o and l and r['status']=='converged' and o['status']=='converged':
    gain=100*(1-float(r['Cend'])/float(o['Cend']));fast=float(r['stop_step'])<float(o['N'])
    comparisons.append(dict(task_id=key[0],method=key[1],historical_lineage=l['lineage'],Flash_Cend=r['Cend'],historical_Cend=o['Cend'],final_gain_vs_historical_pct=gain,Flash_steps=r['stop_step'],historical_steps=o['N'],faster_and_better1=fast and gain>1))
  csvwrite(out/'reports/Flash_vs_historical_paired.csv',comparisons)
  if not comparisons:
   with (out/'reports/Flash_vs_historical_paired.csv').open('w',encoding='utf-8-sig',newline='') as f:
    csv.writer(f).writerow(['task_id','method','historical_lineage','Flash_Cend','historical_Cend','final_gain_vs_historical_pct','Flash_steps','historical_steps','faster_and_better1'])
  lines+=['','`Flash_vs_historical_paired.csv`按相同任务/方法匹配历史结果，`historical_lineage=pure_Pro`才可用于纯Pro与Flash比较；混合和其他历史方法分开。它们是两次生成的单次样本，不是重复随机试验的显著性结论。']
 report.parent.mkdir(exist_ok=True,parents=True);report.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def convergence(out,cfg,workers):
 conv=out/'convergence';conv.mkdir(exist_ok=True);cc=dict(cfg,fresh_output=str(out),trajectories={})
 for r in cfg['records']:
  for m in cfg['methods']:
   v=original.baseline_result(out,r['task_id'],m,cfg);assert v is not None,(r['task_id'],m)
   cc['trajectories'][r['task_id']+'/'+m]=dict(source_status=v['status'])
 if (conv/'config.json').exists():
  previous=read(conv/'config.json');assert sha(conv/'config.json')==read(conv/'config_identity.json')['sha256']
  for k in ('records','methods','trajectories','model','solver','convergence','max_step'):assert previous[k]==cc[k],k
 else:dump(conv/'config.json',cc);dump(conv/'config_identity.json',dict(sha256=sha(conv/'config.json')))
 cont.OUT=conv;cont.write_csv=original.atomic_csv
 def update(status='running'):
  cont.numerical_report(cc,status);progress(out,cfg,'convergence',status);reports(out,cfg)
 update()
 jobs=[(r,m,cc,str(conv)) for r in cfg['records'] for m in cfg['methods'] if not (conv/'tasks'/r['task_id']/m/'result.json').exists()]
 try:
  with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
   pending={pool.submit(cont.run_one,*job) for job in jobs}
   while pending:
    done,pending=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
    for f in done:
     v=f.result();print('CONVERGENCE',v['task_id'],v['method'],v['status'],v.get('step'),flush=True)
    update()
    if (out/'PAUSE_REQUESTED').exists():
     (conv/'PAUSE_REQUESTED').touch()
     for f in pending:f.cancel()
     raise InterruptedError('Pause requested')
  update('complete');dump(out/'COMPLETION.json',dict(tasks=len(cfg['records']),methods=len(cfg['methods']),unix=time.time()))
 except BaseException:
  (conv/'PAUSE_REQUESTED').touch();update('paused');raise

def export(out,cfg):
 conv=read(out/'convergence/progress.json');assert (out/'COMPLETION.json').exists() and conv['status']=='complete' and conv['settled']==len(cfg['records'])*len(cfg['methods'])
 files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('.run.lock','.api.lock','return_manifest.json','RETURN_READY.json') and not p.name.startswith('.env')}
 dump(out/'return_manifest.json',dict(schema='META64-Flash-low-return-v1',tasks=len(cfg['records']),model=cfg['model'],effort='low',files=files,config_sha256=sha(out/'config.json'),package_sha256=cfg['package_sha256']))
 dump(out/'RETURN_READY.json',dict(manifest_sha256=sha(out/'return_manifest.json')))
 dest=ROOT/'exports'/(out.name+'_results.zip');dest.parent.mkdir(exist_ok=True)
 with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
  for rel,h in files.items():assert sha(out/rel)==h;z.write(out/rel,out.name+'/'+rel)
  for rel in ('return_manifest.json','RETURN_READY.json'):z.write(out/rel,out.name+'/'+rel)
 dest.with_name(dest.name+'.sha256').write_text(sha(dest)+'  '+dest.name+'\n')
 print('RETURN ZIP',dest,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['verify','prepare','preflight','baselines','run','report','export'],default='run');p.add_argument('--cohort',choices=['all','original100','local200','deviceB300'],default='all');p.add_argument('--limit',type=int);p.add_argument('--output',type=Path);p.add_argument('--workers',type=int);p.add_argument('--api-workers',type=int);p.add_argument('--max-api-calls',type=int);p.add_argument('--env-file',type=Path,default=ROOT/'.env.local');p.add_argument('--execute-api',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after50',action='store_true');a=p.parse_args()
 verify()
 if a.stage=='verify':return
 out,cfg,workers,api_workers=prepare(a)
 with (out/'.run.lock').open('a+b') as lock:
  cont.acquire_lock(lock)
  if a.stage=='report':progress(out,cfg,'manual_report');reports(out,cfg);return
  if a.stage=='export':export(out,cfg);return
  if a.stage=='prepare':progress(out,cfg,'prepared','prepared');return
  original.preflight(out,cfg)
  if a.stage=='preflight':return
  if a.stage=='baselines':
   assert not (out/'PAUSE_REQUESTED').exists(),'Paused output; resume explicitly before computing'
   with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:phase(pool,[(str(out),r) for r in cfg['records']],fixed_one,out,cfg,'baselines50')
   progress(out,cfg,'baselines50_complete','waiting_API');return
  if not a.execute_api:p.error('Run needs --execute-api: this explicitly enables paid Flash calls for this selected cohort')
  if a.resume:
   for f in (out/'PAUSE_REQUESTED',out/'convergence/PAUSE_REQUESTED'):f.unlink(missing_ok=True)
  if (out/'PAUSE_REQUESTED').exists() or (out/'convergence/PAUSE_REQUESTED').exists():p.error('Paused; use same command with --resume after the old process exits')
  from api_client import read_credentials
  from urllib.parse import urlparse
  credential=read_credentials(a.env_file.resolve());url=urlparse(credential['MODEL_API_BASE_URL'])
  assert url.scheme=='https' and url.hostname==cfg['provider_host'],'Authorized provider host only'
  allowed=cont.reserve_one_cpu(cont.available_cpus());dump(out/'process.json',dict(pid=os.getpid(),workers=workers,api_workers=api_workers,affinity=allowed,started_unix=time.time()))
  dump(out/'USER_AUTHORIZATION.json',dict(explicit_execute_api=True,tasks=[r['task_id'] for r in cfg['records']],model=cfg['model'],effort='low',max_api_calls=cfg['max_api_calls'],provider_host=cfg['provider_host'],sent='same-task physical conditions, generated starts and freshly computed FE summaries; no source labels',unix=time.time()))
  try:
   with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:phase(pool,[(str(out),r) for r in cfg['records']],fixed_one,out,cfg,'baselines50')
   client=EffortClient(out,cfg,a.env_file.resolve());pipeline.configure(out,workers,api_workers)
   try:
    jobs=[(out,cfg,client,r) for r in cfg['records'] if not (out/'tasks'/r['task_id']/'complete.json').exists()]
    with ThreadPoolExecutor(max_workers=api_workers) as pool:phase(pool,jobs,llm_one,out,cfg,'LLM50')
   finally:pipeline.shutdown()
   client.budget();dump(out/'llm_complete.json',dict(unix=time.time()))
   if a.stop_after50:progress(out,cfg,'LLM50_complete','waiting_convergence');reports(out,cfg);return
   convergence(out,cfg,workers);export(out,cfg)
  except BaseException as exc:
   (out/'PAUSE_REQUESTED').touch();dump(out/'error.json',dict(error=str(exc),traceback=traceback.format_exc(),unix=time.time()));progress(out,cfg,'paused','paused');raise

if __name__=='__main__':main()
