"""OC50 and continuation share the exact same initial state and checkpoint."""
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
from physics import SolverConfig
from solver import SIMP
from initializers import generate
from storage import dump, save_npz


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def converged(history, cfg):
    c = cfg['convergence']; w = c['window']
    if history[-1]['step'] < cfg['earliest_certifiable_step'] or len(history) < w+1:
        return False
    tail = history[-w:]
    if any(r['max_design_change'] is None for r in tail):
        return False
    costs = [r['compliance'] for r in history[-w-1:]]
    return (max(r['max_design_change'] for r in tail) <= c['max_design_change']
            and max(r['relative_compliance_change'] for r in tail) <= c['max_relative_compliance_change']
            and (max(costs)-min(costs))/max(abs(costs[-1]), 1e-30) <= c['relative_compliance_span']
            and max(r['residual'] for r in tail) < c['residual_tolerance']
            and max(abs(r['volume']-cfg['_target']) for r in tail) <= c['volume_tolerance'])


def run_one(out, record, method, cfg, stage):
    out = Path(out); dest = out/'tasks'/record['task_id']/method
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'result.json').exists():
        return read(dest/'result.json')
    if stage == '50' and (dest/'result_50.json').exists():
        return read(dest/'result_50.json')
    if (dest/'failure.json').exists():
        return read(dest/'failure.json')
    if (out/'PAUSE_REQUESTED').exists():
        return dict(status='paused')
    if stage == 'converge' and not (dest/'result_50.json').exists():
        raise ValueError('OC50 missing; run --stage 50 or --stage all first')
    model = SIMP(out/record['path'], SolverConfig(**cfg['solver']))
    cfg = dict(cfg, _target=model.target)
    cost = read(dest/'cost.json') if (dest/'cost.json').exists() else dict(
        attempted_FE=0, recovery_FE=0, initialization_FE=0, update_FE=0,
        seconds=0., pending=False)
    history = []; started = time.monotonic()

    def evaluate(x, kind):
        cost['attempted_FE'] += 1; cost[kind+'_FE'] += 1
        cost['pending'] = True; dump(dest/'cost.json', cost)
        ev = model.evaluate(x)
        if abs(ev['volume']-model.target) > cfg['convergence']['volume_tolerance']:
            raise ValueError('Physical volume violated')
        cost['pending'] = False
        return ev

    def save(x, ev):
        cost['seconds'] += time.monotonic()-clock[0]; clock[0] = time.monotonic()
        # Checkpoint is authoritative after an interruption; history is a readable mirror.
        save_npz(dest/'checkpoint.npz', x=x, history_json=json.dumps(history))
        dump(dest/'history.json', history)
        dump(dest/'cost.json', cost)
        dump(dest/'progress.json', dict(task_id=record['task_id'], method=method,
             step=history[-1]['step'], compliance=ev['compliance'], pid=os.getpid(),
             status='running', updated_unix=time.time()))

    clock = [started]
    try:
        if (dest/'checkpoint.npz').exists():
            with np.load(dest/'checkpoint.npz', allow_pickle=False) as a:
                x = a['x'].copy(); history = json.loads(str(a['history_json']))
            ev = evaluate(x, 'recovery')
            if abs(ev['compliance']/history[-1]['compliance']-1) > 1e-8:
                raise ValueError('Checkpoint FE mismatch')
            # A hard interruption may have happened between the step-50 checkpoint
            # and its public artifacts. Reconstruct only from that exact state.
            if history[-1]['step'] == 50 and not (dest/'result_50.json').exists():
                save_npz(dest/'state_0050.npz', x=x, density=ev['density'])
                dump(dest/'result_50.json', dict(task_id=record['task_id'], method=method,
                     status='complete50', C0=history[0]['compliance'], C50=ev['compliance'],
                     step=50, attempted_FE_so_far=cost['attempted_FE']))
        else:
            x, meta = generate(model, method)
            dump(dest/'initializer.json', meta)
            ev = evaluate(x, 'initialization')
            history = [dict(step=0, **{k:float(ev[k]) for k in
                       ('compliance','volume','residual','grayness')},
                       max_design_change=None, relative_compliance_change=None)]
            save_npz(dest/'state_0000.npz', x=x, density=ev['density'])
            save(x, ev)
        limit = 50 if stage == '50' else cfg['max_step']
        while history[-1]['step'] < limit:
            if history[-1]['step'] >= 50 and stage != '50' and converged(history, cfg):
                break
            if (out/'PAUSE_REQUESTED').exists():
                save(x, ev)
                return dict(task_id=record['task_id'], method=method, status='paused')
            oldx = x; oldc = ev['compliance']
            x = model.update(x, ev); ev = evaluate(x, 'update')
            step = history[-1]['step']+1
            history.append(dict(step=step, **{k:float(ev[k]) for k in
                ('compliance','volume','residual','grayness')},
                max_design_change=float(np.max(np.abs(x-oldx))),
                relative_compliance_change=abs(ev['compliance']-oldc)/max(abs(oldc),1e-30)))
            if step % 10 == 0:
                save(x, ev)
            if step == 50:
                save_npz(dest/'state_0050.npz', x=x, density=ev['density'])
                dump(dest/'result_50.json', dict(task_id=record['task_id'], method=method,
                     status='complete50', C0=history[0]['compliance'], C50=ev['compliance'],
                     step=50, attempted_FE_so_far=cost['attempted_FE']))
        save(x, ev)
        if stage == '50':
            return read(dest/'result_50.json')
        save_npz(dest/'state_final.npz', x=x, density=ev['density'], u=ev['u'])
        result = dict(task_id=record['task_id'], method=method,
            status='converged' if converged(history, cfg) else 'max_steps',
            C0=history[0]['compliance'], C50=history[50]['compliance'],
            Cend=ev['compliance'], step=history[-1]['step'], cost=cost,
            completed_unix=time.time())
        dump(dest/'result.json', result)
        return result
    except Exception as exc:
        cost['seconds'] += time.monotonic()-clock[0]
        dump(dest/'cost.json', cost)
        result = dict(task_id=record['task_id'], method=method, status='failed',
                      last_step=history[-1]['step'] if history else None,
                      error=str(exc), traceback=traceback.format_exc(), cost=cost)
        dump(dest/'failure.json', result)
        return result
