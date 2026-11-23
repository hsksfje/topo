#!/usr/bin/env python3
"""Offline transport, retry, numerical checkpoint and package regression checks."""
import importlib.util,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('flash_runner',ROOT/'run.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
from initskill.io import sha

class FakeBackend:
 def __init__(self,transient=False):self.chat=types.SimpleNamespace(completions=self);self.payloads=[];self.transient=transient
 def create(self,**payload):
  self.payloads.append(payload)
  if self.transient and len(self.payloads)==1:
   e=RuntimeError('rate limit');e.status_code=429;raise e
  raw=dict(model=payload['model'],id='offline',usage=dict(prompt_tokens=10,completion_tokens=20,total_tokens=30),choices=[])
  return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason='stop',message=types.SimpleNamespace(content='{"values":[1]}'))],model_dump=lambda **kw:raw)

class OfflineChecks(unittest.TestCase):
 def test_same_task_historical_pairing(self):
  old=next(r for r in runner.csvread(ROOT/'provenance/historical_convergence.csv') if r['method']=='A1' and r['status']=='converged')
  cfg=runner.read(ROOT/'selection.json');cfg['records']=[r for r in cfg['records'] if r['task_id']==old['task_id']]
  tid=old['task_id'];c=float(old['Cend']);n=float(old['N'])
  with tempfile.TemporaryDirectory() as folder:
   out=Path(folder);(out/'convergence/reports').mkdir(parents=True)
   runner.csvwrite(out/'convergence/reports/results.csv',[dict(task_id=tid,method='U',status='converged',Cend=c,C50=c,I50=0,stop_step=n+100),dict(task_id=tid,method='A1',status='converged',Cend=c*.9,C50=c,I50=0,stop_step=n-1)])
   runner.reports(out,cfg);paired=runner.csvread(out/'reports/Flash_vs_historical_paired.csv')
   a1=next(r for r in paired if r['method']=='A1');self.assertAlmostEqual(float(a1['final_gain_vs_historical_pct']),10);self.assertEqual(a1['historical_lineage'],'pure_Pro');self.assertEqual(a1['faster_and_better1'],'True')
 def test_low_wire_cache_and_hard_cap(self):
  with tempfile.TemporaryDirectory() as folder:
   cfg=runner.read(ROOT/'selection.json');cfg['max_api_calls']=1;b=FakeBackend();clock=[0.]
   client=runner.EffortClient(folder,cfg,backend=b,clock=lambda:clock[0],sleep=lambda x:clock.__setitem__(0,clock[0]+x))
   self.assertEqual(client.call('one',[],{}),{'values':[1]})
   self.assertEqual(client.call('one',[],{}),{'values':[1]})
   self.assertEqual(len(b.payloads),1);self.assertEqual(b.payloads[0]['reasoning_effort'],'low');self.assertEqual(b.payloads[0]['extra_body']['thinking']['type'],'enabled')
   with self.assertRaisesRegex(RuntimeError,'Request cap'):client.call('two',[],{})
   self.assertEqual(len(b.payloads),1)
 def test_transient_retry_is_counted(self):
  with tempfile.TemporaryDirectory() as folder:
   cfg=runner.read(ROOT/'selection.json');cfg['max_api_calls']=2;b=FakeBackend(True);clock=[0.]
   client=runner.EffortClient(folder,cfg,backend=b,clock=lambda:clock[0],sleep=lambda x:clock.__setitem__(0,clock[0]+x))
   self.assertEqual(client.call('retry',[],{}),{'values':[1]});self.assertEqual(client.budget()['attempts'],2);self.assertEqual(len(b.payloads),2)
 def test_temporary_windows_write_error(self):
  import initskill.io as io
  real=io.os.replace;count=[0]
  def fail_then_replace(a,b):
   count[0]+=1
   if count[0]<3:raise PermissionError('simulated temporary WinError5')
   return real(a,b)
  with tempfile.TemporaryDirectory() as folder,patch.object(io.os,'replace',side_effect=fail_then_replace),patch.object(io.time,'sleep'):
   target=Path(folder)/'file.json';io.dump(target,{'ok':True});self.assertEqual(runner.read(target),{'ok':True});self.assertEqual(count[0],3)
 def test_numeric_oc50_and_checkpoint_resume(self):
  with tempfile.TemporaryDirectory() as folder:
   a=types.SimpleNamespace(cohort='original100',limit=1,workers=1,api_workers=1,output=Path(folder),max_api_calls=None)
   out,cfg,_,_=runner.prepare(a);r=cfg['records'][0];runner.original.preflight(out,cfg);runner.fixed_one(str(out),r)
   for m in cfg['fixed']:
    value=runner.original.baseline_result(out,r['task_id'],m,cfg);self.assertEqual(value['status'],'complete');self.assertGreater(value['C50'],0)
   cc=dict(cfg,fresh_output=str(out),max_step=60,trajectories={r['task_id']+'/U':dict(source_status='complete')});conv=out/'convergence'
   result=runner.cont.run_one(r,'U',cc,str(conv));self.assertIn(result['status'],('converged','max_steps'));self.assertEqual(result['step'],60)
   final=conv/'tasks'/r['task_id']/'U/state_final.npz';h=sha(final);self.assertEqual(runner.cont.run_one(r,'U',cc,str(conv)),result);self.assertEqual(sha(final),h)

if __name__=='__main__':unittest.main(verbosity=2)
