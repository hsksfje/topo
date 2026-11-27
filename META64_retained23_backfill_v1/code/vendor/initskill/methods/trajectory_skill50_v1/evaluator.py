"""Exact 30→50 continuation; continuous improvements and durable FE accounting."""
import json
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
from initskill.io import dump,save_npz,digest
from initskill.physics import SIMP as BaseSIMP, SolverConfig
from .geometry import decode

SOLVER = asdict(SolverConfig())
SCALARS = ('compliance','volume','grayness','residual')
ARRAYS = ('density','gradient','energy','u')


class SIMP(BaseSIMP):
    def update(self,x,evaluation):
        c=self.config
        dc=np.minimum(evaluation['gradient'],0.)
        low,high,feasible=0.,1e9,None
        for _ in range(250):
            mid=.5*(low+high)
            new=np.clip(x*np.sqrt(-dc/self.dv/mid),np.maximum(c.xmin,x-c.move),np.minimum(1.,x+c.move))
            if self.physical(new).mean()>self.target:
                low=mid
            else:
                high,feasible=mid,new
            if (high-low)/(high+low)<=1e-8 and feasible is not None:
                return feasible
        raise RuntimeError('Feasible-side OC bisection failed')


def improvement(compliance,uniform):
    if compliance is None or uniform is None:
        return None
    if not np.isfinite([compliance,uniform]).all() or compliance<=0 or uniform<=0:
        raise ValueError('Positive finite compliance required')
    return 100.*(uniform-compliance)/uniform


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def evaluate(output,record,task,recipe,folder,stop):
    output,folder=Path(output),Path(folder)
    if stop not in (30,50):
        raise ValueError('Only 30/50 endpoints')
    folder.mkdir(parents=True,exist_ok=True)
    identity=digest(dict(assets=record['assets'],recipe=recipe,solver=SOLVER,decoder='trajectory-skill50-v1',oc_tolerance=1e-8))
    if (folder/'identity.json').exists():
        if read(folder/'identity.json')['identity']!=identity:
            raise ValueError('Evaluation identity changed')
    else:
        dump(folder/'identity.json',dict(identity=identity,recipe=recipe,task_id=record['task_id']))
    if (folder/'failure.json').exists():
        return read(folder/'failure.json')
    model=SIMP(output/record['path'],SolverConfig(**SOLVER))
    checkpoint=folder/'checkpoint.npz'
    history=[]
    elapsed=0.
    old_cost=read(folder/'cost.json') if (folder/'cost.json').exists() else {}
    attempts=old_cost.get('attempted_fe',0)
    init_attempts=old_cost.get('initialization_fe',0)
    initializing=True
    original=model.evaluate
    def counted(x):
        nonlocal attempts,init_attempts
        attempts+=1
        init_attempts+=int(initializing)
        cost=dict(attempted_fe=attempts,initialization_fe=init_attempts,pending=True)
        dump(folder/'cost.json',cost)
        try:
            result=original(x)
        finally:
            cost['pending']=False
            dump(folder/'cost.json',cost)
        return result
    model.evaluate=counted
    started=time.monotonic()
    try:
        if checkpoint.exists():
            with np.load(checkpoint,allow_pickle=False) as a:
                x=a['x'].copy()
                ev={k:a[k].copy() for k in ARRAYS}
                ev.update(zip(SCALARS,map(float,a['scalars'])))
                history=json.loads(str(a['history_json']))
                elapsed=float(a['elapsed_seconds'])
        else:
            x,metadata=decode(model,task,recipe)
            dump(folder/'initializer.json',dict(**metadata,initialization_fe=init_attempts,
                initial_density_sha=digest(x.tolist()),task_id=record['task_id']))
            if abs(model.physical(x).mean()-model.target)>1e-9:
                raise ValueError('Initial volume mismatch')
            initializing=False
            ev=model.evaluate(x)
            history=[dict(step=0,**{k:float(ev[k]) for k in SCALARS})]
            save_npz(folder/'state_0000.npz',x=x,density=ev['density'],u=ev['u'],step=0)
        initializing=False
        def save():
            save_npz(checkpoint,x=x,**{k:ev[k] for k in ARRAYS},scalars=np.array([ev[k] for k in SCALARS]),
                history_json=json.dumps(history),elapsed_seconds=elapsed+time.monotonic()-started)
            dump(folder/'history.json',history)
        save()
        for step in range(history[-1]['step']+1,stop+1):
            if (output/'PAUSE_REQUESTED').exists():
                save()
                raise InterruptedError('Pause requested')
            x=model.update(x,ev)
            ev=model.evaluate(x)
            if ev['volume']>model.target+1e-8:
                raise ValueError('Volume infeasible')
            history.append(dict(step=step,**{k:float(ev[k]) for k in SCALARS}))
            if step in (30,50):
                save_npz(folder/f'state_{step:04d}.npz',x=x,density=ev['density'],u=ev['u'],step=step)
            if step%10==0 or step==stop:
                save()
        result=dict(status='complete',step=min(history[-1]['step'],stop),history=[r for r in history if r['step']<=stop],
            C30=history[30]['compliance'] if len(history)>30 else None,
            C50=history[50]['compliance'] if stop==50 and len(history)>50 else None,
            fe_calls=attempts,initialization_fe=init_attempts,wall_seconds=elapsed+time.monotonic()-started,
            states={str(s):str((folder/f'state_{s:04d}.npz').relative_to(output)) for s in (0,30,50) if s<=stop and (folder/f'state_{s:04d}.npz').exists()})
        dump(folder/f'result_{stop}.json',result)
        return result
    except InterruptedError:
        raise
    except Exception as exc:
        result=dict(status='failed',error_type=type(exc).__name__,error=str(exc),
                    C30=history[30]['compliance'] if len(history)>30 else None,
                    C50=None,history=history,fe_calls=attempts,initialization_fe=init_attempts,
                    states={str(s):str((folder/f'state_{s:04d}.npz').relative_to(output))
                            for s in (0,30) if (folder/f'state_{s:04d}.npz').exists()})
        dump(folder/'failure.json',result)
        return result


def choose(rows):
    eligible=[(r['C30'],i) for i,r in enumerate(rows) if r.get('status')=='complete' and r.get('C30') is not None]
    return min(eligible)[1] if eligible else None
