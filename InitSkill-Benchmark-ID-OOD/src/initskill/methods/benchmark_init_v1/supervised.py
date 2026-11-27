"""Shared supervised label acquisition, never uses held-out test targets."""
import json
from pathlib import Path
import time
import numpy as np
from .common import ROOT,DATA,dump,sha,rows,features,solver,trajectory

def build_labels(out,subset='train1000',reuse_archive=False,steps=300,limit=None):
    if subset not in {'train1000','valid100'}:raise ValueError('Teacher building is restricted to Train/Valid')
    source=rows(subset)
    if limit is not None:source=source[:limit]
    dest=Path(out)/'labels'/subset;dest.mkdir(parents=True,exist_ok=True)
    archive={}
    if reuse_archive:
        if subset!='train1000':raise ValueError('Archive pool is training only')
        for line in (ROOT/'catalog/trajectory_index.jsonl').open():
            r=json.loads(line)
            if r['variant']=='default' and r['method']=='U':archive[r['task_id']]=r
    summary=[]
    for index,r in enumerate(source):
        p=dest/(r['source_task_id']+'.npz');meta=p.with_suffix('.json')
        if p.exists() and meta.exists():
            record=json.loads(meta.read_text())
            if record['sha256']!=sha(p) or record['task_key']!=r['task_key']:raise ValueError('Label changed')
            summary.append(record);continue
        m=solver(out,subset,r);start=time.perf_counter()
        if reuse_archive:
            a=archive.get('META64_'+r['source_task_id'])
            if not a or not a['final_state']:raise ValueError('Missing Uniform label: '+r['task_key'])
            state=ROOT/a['final_state']
            with np.load(state,allow_pickle=False) as z:x=z['x'].copy()
            ev=m.evaluate(x)
            if abs(ev['compliance']/a['Cend']-1)>1e-6:raise ValueError('Archive physics mismatch')
            if abs(ev['volume']-m.target)>1e-8:raise ValueError('Archive volume mismatch')
            origin=dict(type='audited_uniform_archive',path=a['final_state'],sha256=sha(state),
                        status=a['status'],steps=a['stop_step'],original_record=a,
                        teacher_quality='converged' if a['status']=='converged' else 'budget_limited')
        else:
            run=trajectory(m,m.initial(),steps,dest/'runs'/r['source_task_id'])
            with np.load(dest/'runs'/r['source_task_id']/f'state_{steps:04d}.npz') as z:x=z['x'].copy()
            ev=m.evaluate(x)
            origin=dict(type='new_fixed_budget_uniform_teacher',status='budget_limited',steps=steps,
                        teacher_quality='fixed_budget_not_claimed_optimal')
        np.savez_compressed(p,x=x,density=ev['density'])
        record=dict(task_key=r['task_key'],group_id=r['group_id'],sha256=sha(p),
            physical_inputs={k:r['assets'][k]['sha256'] for k in ('forces','normals','target_vf')},
            compliance=ev['compliance'],volume=ev['volume'],FE=m.fe_calls,
            seconds=time.perf_counter()-start,origin=origin)
        dump(meta,record);summary.append(record)
        if index%25==0:print('labels',subset,index+1,len(source),flush=True)
    dump(dest/'index.json',dict(subset=subset,count=len(summary),records=summary,
        manifest_sha256=sha(DATA/'manifests'/f'{subset}.jsonl'),
        quality_policy='uniform only; converged and max_steps preserved, no per-task method selection'))
    return summary

def arrays(out,subset):
    r=rows(subset);x=[]
    for row in r:
        p=Path(out)/'labels'/subset/(row['source_task_id']+'.npz')
        meta=json.loads(p.with_suffix('.json').read_text())
        if meta['task_key']!=row['task_key'] or meta['group_id']!=row['group_id'] or sha(p)!=meta['sha256']:
            raise ValueError('Teacher identity mismatch')
        with np.load(p,allow_pickle=False) as z:x.append(z['x'].copy())
    return r,np.stack([features(a) for a in r]),np.stack(x)
