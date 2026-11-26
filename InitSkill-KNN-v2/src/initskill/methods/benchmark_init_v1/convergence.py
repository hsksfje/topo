"""Resumable full-task optimization and two-goal metrics, shared across methods."""
import json
from pathlib import Path
import time
import uuid
import numpy as np
from .common import dump,sha,atomic_savez


def run_to_convergence(m,x,max_steps,folder,identity=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);identity=identity or {}
    if (folder/'identity.json').exists():
        if json.loads((folder/'identity.json').read_text())!=identity:raise ValueError('Run identity changed')
    else:dump(folder/'identity.json',identity)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    history=[];start_step=0;streak=0;seconds=0.
    if (folder/'checkpoint.json').exists():
        saved=json.loads((folder/'checkpoint.json').read_text())
        state=folder/saved.get('state_file','checkpoint.npz')
        if 'state_sha256' in saved:
            if sha(state)!=saved['state_sha256']:raise ValueError('Changed coherent OC checkpoint')
        elif state.stat().st_mtime_ns>(folder/'checkpoint.json').stat().st_mtime_ns:
            raise ValueError('Legacy OC state is newer than committed history')
        with np.load(state,allow_pickle=False) as a:x=a['x'].copy()
        history=saved['history'];start_step=saved['step'];streak=saved['streak'];seconds=saved['seconds']
    ledger=folder/'FE_attempts.jsonl';attempts=0
    if ledger.exists():
        with ledger.open() as stream:attempts=sum(1 for _ in stream)
    start=time.perf_counter();last_x=None;status='max_steps'
    try:
        for step in range(start_step,max_steps+1):
            attempts+=1
            with ledger.open('a') as stream:stream.write(json.dumps(dict(attempt=attempts,step=step))+'\n')
            ev=m.evaluate(x)
            replay=bool(history and history[-1]['step']==step)
            # A resumed checkpoint has already evaluated this exact state.
            # Re-evaluation is charged, but its saved change/stability counter
            # must neither be erased nor counted as a new optimization step.
            change=(history[-1]['max_dx'] if replay and last_x is None else
                    float(np.max(np.abs(x-last_x))) if last_x is not None else None)
            record=dict(step=step,compliance=ev['compliance'],volume=ev['volume'],residual=ev['residual'],max_dx=change)
            if ev.get('linear_solver'):record['linear_solver']=ev['linear_solver']
            if ev.get('precision'):
                record.update(precision=ev['precision'],raw_residual=ev['raw_residual'],
                              refinement_steps=ev['refinement_steps'],energy_balance_relative=ev['energy_balance_relative'])
            if history and history[-1]['step']==step:history[-1]=record
            else:history.append(record)
            if step in {0,10,30,50,100,200,500,1000,2000}:
                atomic_savez(folder/f'state_{step:04d}.npz',x=x,density=ev['density'])
            stable=False
            if step>=50 and len(history)>=11 and change is not None:
                recent=np.array([r['compliance'] for r in history[-11:]])
                stable=(change<=.01 and abs(recent[-1]-recent[-2])/max(abs(recent[-2]),1e-30)<=1e-4 and
                    np.ptp(recent)/max(abs(recent[-1]),1e-30)<=5e-4 and abs(ev['volume']-m.target)<=1e-8 and ev['residual']<1e-6)
            if not replay:streak=streak+1 if stable else 0
            # All methods retain exact fixed-budget C50/C100/C200. Continue
            # through 200 even if the stopping criterion first holds earlier.
            if step>=200 and streak>=10:status='converged';break
            if step==max_steps:break
            if step%10==0:
                state=folder/'oc_checkpoint_slots'/str((step//10)%2)/'state.npz'
                state.parent.mkdir(parents=True,exist_ok=True);atomic_savez(state,x=x)
                dump(folder/'checkpoint.json',dict(step=step,streak=streak,history=history,
                    seconds=seconds+time.perf_counter()-start,state_file=state.relative_to(folder).as_posix(),state_sha256=sha(state)))
                # Restoration reads state_file; no privileged Windows symlink is needed.
            last_x=x.copy();x=m.update(x,ev)
        atomic_savez(folder/'state_final.npz',x=x,density=ev['density'])
        result=dict(status=status,stop_step=step,final_compliance=ev['compliance'],FE=attempts,
            seconds=seconds+time.perf_counter()-start,history=history,identity=identity,
            converged=status=='converged',budget_points={str(r['step']):r['compliance'] for r in history if r['step'] in {0,10,30,50,100,200,500,1000,2000}})
        dump(folder/'result.json',result);return result
    except Exception as exc:
        dump(folder/'failure.json',dict(status='failed',error=str(exc),FE=attempts,last_step=history[-1]['step'] if history else None,
                                       seconds=seconds+time.perf_counter()-start))
        raise


def goal_metrics(method,uniform):
    target=uniform['final_compliance']*1.01
    hits=[r['step'] for r in method['history'] if r['compliance']<=target]
    return dict(target=target,reference_status=uniform['status'],steps_to_target=min(hits) if hits else None,
        reached=bool(hits),final_ratio=method['final_compliance']/uniform['final_compliance'],
        budget_ratios={k:v/uniform['budget_points'][k] for k,v in method['budget_points'].items() if k in uniform['budget_points']},
        final_status=method['status'],FE=method['FE'],seconds=method['seconds'])
