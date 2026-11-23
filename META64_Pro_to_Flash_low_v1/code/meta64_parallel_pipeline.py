"""Concurrent per-task LLM generation and shared process pool for unchanged OC50 FE."""
import json,os,time,threading,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import llm_six as base
import extensions as ext
from initskill.io import dump,save_npz

_POOL=None
_STATE_LOCK=threading.Lock()
_STATE={}
_OUT=None

def _running(out):
    if (Path(out)/'PAUSE_REQUESTED').exists():raise RuntimeError('Pause requested; keep generation resumable')

def numeric_job(kind,out,r,arm=None):
    out=Path(out);_running(out);tid=r['task_id'];root=out/'tasks'/tid
    if kind=='arm':
        result=base.evaluate(out,r,arm)
    elif kind=='extension':
        result=ext.trajectory(out,r,arm)
        if result is None:
            result=dict(status='invalid',C0=None,C50=None,error='No valid start; no fallback')
            dump(root/arm/'result.json',result)
    elif kind=='uniform_observation':
        if not (root/'observation.json').exists():
            m=base.SIMP(out/r['path']);dump(root/'observation_cost.json',dict(attempted_fe=1,pending=True))
            ev=m.evaluate(m.initial());dump(root/'observation.json',base.observation(m,ev))
            save_npz(root/'observation_raw.npz',density=ev['density'],energy=ev['energy'],u=ev['u'])
            dump(root/'observation_cost.json',dict(attempted_fe=1,pending=False))
        result=dict(status='complete')
    elif kind=='candidate_feedback':
        p=root/'feedback.json'
        if not p.exists():
            m=base.SIMP(out/r['path']);x=ext.vector(root/'L1_PG/start.npz')
            dump(root/'feedback_cost.json',dict(attempted_fe=1,pending=True));ev=m.evaluate(x);ob=base.observation(m,ev);ob.pop('full_element_relative_strain_energy',None)
            ob.update(source='Candidate L1_PG own step-zero FE; no OC update',C0=float(ev['compliance']),volume=float(ev['volume']),residual=float(ev['residual']),density=ext.coarse(ev['density']))
            dump(p,ob);dump(root/'feedback_cost.json',dict(attempted_fe=1,pending=False))
        result=dict(status='complete')
    else:raise ValueError(kind)
    return dict(task_id=tid,kind=kind,method=arm,status=result['status'],pid=os.getpid())

def _submit(kind,out,r,arm=None):
    if _POOL is None:raise RuntimeError('Parallel numeric pool is not configured')
    _running(out)
    with _STATE_LOCK:
        f=_POOL.submit(numeric_job,kind,str(out),r,arm)
        _STATE['submitted']+=1
        _save_state()
    def finished(future):
        with _STATE_LOCK:
            _STATE['finished']+=1
            if future.cancelled():_STATE['cancelled']+=1
            else:
                try:
                    v=future.result();_STATE['worker_pids']=sorted(set(_STATE['worker_pids'])|{v['pid']})
                except BaseException:_STATE['interrupted_or_failed']+=1
            _save_state()
    f.add_done_callback(finished)
    return f

def _save_state():
    if _OUT is not None:dump(_OUT/'pipeline_progress.json',dict(_STATE,pending_or_running=_STATE['submitted']-_STATE['finished'],updated_unix=time.time()))

def configure(out,cpu_workers,api_workers):
    global _POOL,_OUT,_STATE
    assert _POOL is None
    _OUT=Path(out);_STATE=dict(api_workers=api_workers,numerical_processes=cpu_workers,submitted=0,finished=0,cancelled=0,interrupted_or_failed=0,worker_pids=[],source='shared process pool; all actual FE restricted to numerical workers')
    _POOL=ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context('spawn'))
    with _STATE_LOCK:_save_state()

def shutdown():
    global _POOL
    if _POOL is not None:_POOL.shutdown(wait=True,cancel_futures=True);_POOL=None

def llm_job(out,cfg,client,r):
    out=Path(out);_running(out);tid=r['task_id'];root=out/'tasks'/tid;root.mkdir(parents=True,exist_ok=True)
    task=base.explicit_task(base.read(out/r['path']/'physical_input.json'));task.pop('source_root',None)
    # Per-task dependencies and API prompts are identical to the frozen protocol.
    futures=[]
    for suffix in ('1','2','3'):
        base.generate_pair(out,client,r,task,'A',suffixes=(suffix,))
        futures.append(_submit('arm',out,r,'A'+suffix))
    if not (root/'observation.json').exists():_submit('uniform_observation',out,r).result()
    for suffix in ('1','2','3'):
        base.generate_pair(out,client,r,task,'B',base.read(root/'observation.json'),suffixes=(suffix,))
        futures.append(_submit('arm',out,r,'B'+suffix))
    # These trajectories supply no input to subsequent LLM prompts: overlap FE with API waiting.
    m=base.SIMP(out/r['path'])
    bases={a:ext.vector(out/'corpus'/tid/name/'state_0000.npz') for a,name in [('PG','PG2'),('H4','H4_B035')]}
    for kind,payload in [
        ('L1',dict(task=task,base_density={k:ext.coarse(v) for k,v in bases.items()},instruction='Propose ONE bounded logit-field edit for EACH base PG2 and H4_B035. Multiply existing logit field by strength, add up to4 circular Gaussian patches amplitude*exp(-distance_normalized^2/(2*radius^2)); exact target volume is recalibrated. Keep edits small and improve load paths.')),
        ('L2',dict(task=task,instruction='Propose 2..12 solid capsule members. Centers cx,cy in uW,vH; length and full width measured in H units, angles physical degrees counterclockwise from x. Host takes max of (width*H/2-distance_to_segment)/(.04H), scales by alpha and sigmoid-calibrates filtered volume. Avoid isolated rods.'))]:
        try:
            _running(out);result=ext.ask(out,client,tid+'_'+kind,kind,payload)
            if kind=='L1':
                for key,arm in [('PG','L1_PG'),('H4','L1_H4')]:ext.save_start(out,tid,arm,ext.edit(m,bases[key],result[key]),dict(proposal=tid+'_L1',base=key,edit=result[key]))
            else:ext.save_start(out,tid,'L2_GEOM',ext.geometry_decode(m,dict(representation='components',parameters=dict(items=result['items'],alpha=result['alpha'],transition=.04))),result)
        except ValueError as e:dump(root/(kind+'_invalid.json'),dict(error=str(e)))
    parent=root/'L1_PG/start.npz'
    if parent.exists():
        if not (root/'feedback.json').exists():_submit('candidate_feedback',out,r).result()
        try:
            _running(out)
            result=ext.ask(out,client,tid+'_L3','L3',dict(task=task,parent_initializer=base.read(parent.parent/'initializer.json'),candidate_FE=base.read(root/'feedback.json'),instruction='Modify THIS candidate using its own FE feedback, not uniform FE. Return a bounded logit edit relative to the actual parent start. Inspect energy/deflection maps to decide where to redistribute density. Objective is C50; C0 alone is not a reliable proxy. Same exact-volume calibration. No OC steps were used.'))
            ext.save_start(out,tid,'L3_FE',ext.edit(m,ext.vector(parent),result['edit']),result)
        except ValueError as e:dump(root/'L3_invalid.json',dict(error=str(e)))
    futures += [_submit('extension',out,r,arm) for arm in cfg['llm'][6:]]
    for future in futures:
        value=future.result();print('PARALLEL_OC50',value['task_id'],value['method'],value['status'],'pid',value['pid'],flush=True)
    dump(root/'complete.json',dict(task_id=tid,finished_unix=time.time()))
    return tid
