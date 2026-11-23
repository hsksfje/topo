"""Two-stage parameter experiment, independent of API and initializer packages."""
import json
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from physics import SolverConfig, mesh
from solver import SIMP
from storage import digest, dump, save_npz, sha

SCALARS = ('compliance', 'volume', 'grayness', 'residual')
ARRAYS = ('density', 'gradient', 'energy', 'u')
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def array_digest(value):
    a = np.ascontiguousarray(value)
    import hashlib
    return hashlib.sha256(a.tobytes()).hexdigest()


def prepare_inputs(out, records):
    """Reconstruct exactly the audited mesh, sparse force and constrained DOFs."""
    domain = out / 'domains/64x64.npz'
    if not domain.exists():
        save_npz(domain, **dict(zip(('nodes', 'cells', 'centers'), mesh(64, 64))))
    result = []
    for source in records:
        r = dict(source)
        folder = out / 'inputs' / r['task_id']
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / 'domain.npz').exists():
            # Regular files work on Windows and when moving the output folder.
            import shutil
            shutil.copyfile(domain, folder / 'domain.npz')
        physical = r['physical']
        force = np.zeros(2 * 65 * 65)
        for dof, value in physical['forces_sparse']:
            force[int(dof)] = value
        fixed = np.asarray(physical['fixed_dofs'], dtype=np.int64)
        if not (folder / 'boundary_conditions.npz').exists():
            save_npz(folder / 'boundary_conditions.npz', force=force, fixed_dofs=fixed)
        if not (folder / 'task.json').exists():
            dump(folder / 'task.json', r['task'])
        starts = PACKAGE_ROOT / r['starts_file']
        if sha(starts) != r['starts_file_sha256']:
            raise ValueError('Packaged start fields changed: ' + r['task_id'])
        if not (folder / 'starts.npz').exists():
            import shutil
            shutil.copyfile(starts, folder / 'starts.npz')
        r['path'] = 'inputs/' + r['task_id']
        r['assets'] = {name: sha(folder / name) for name in
                       ('domain.npz', 'boundary_conditions.npz', 'task.json', 'starts.npz')}
        result.append(r)
    return result


def model_for(out, record, cfg, profile, common=False):
    config = SolverConfig(**cfg['solver'])
    if not common:
        config = replace(config, **{k: profile[k] for k in ('penal', 'rmin', 'move')})
    return SIMP(Path(out) / record['path'], config)


def load_start(out, record, model, initializer):
    if initializer == 'U':
        return model.initial()
    if initializer not in ('PG1', 'L1_PG'):
        raise ValueError('Unknown initializer: ' + initializer)
    with np.load(Path(out) / record['path'] / 'starts.npz', allow_pickle=False) as a:
        x = a[initializer].copy()
    if x.shape != (len(model.cells),) or not np.isfinite(x).all():
        raise ValueError('Invalid frozen start shape/values')
    if np.any(x < model.config.xmin - 1e-12) or np.any(x > 1 + 1e-12):
        raise ValueError('Frozen start outside density bounds')
    if array_digest(x) != record['starts'][initializer]['array_sha256']:
        raise ValueError('Frozen initial field identity mismatch')
    return x


def repair_volume(model, x, cfg):
    """Smallest feasible uniform L-infinity shift, with density-bound clipping.

    This acts on DESIGN density, not on physical density or a logit field.
    The filtered-volume functional is linear before clipping.
    """
    old_volume = float(model.physical(x).mean())
    if abs(old_volume - model.target) <= cfg['convergence']['volume_tolerance']:
        return x.copy(), dict(applied=False, old_volume=old_volume,
                              new_volume=old_volume, max_design_change=0., shift=0.)
    lo, hi = -1., 1.
    for _ in range(100):
        shift = (lo + hi) / 2
        new = np.clip(x + shift, model.config.xmin, 1.)
        volume = float(model.physical(new).mean())
        if abs(volume - model.target) <= cfg['switch_volume_repair']['repair_tolerance']:
            return new, dict(applied=True, old_volume=old_volume, new_volume=volume,
                             max_design_change=float(np.max(np.abs(new - x))), shift=shift)
        if volume > model.target:
            hi = shift
        else:
            lo = shift
    raise RuntimeError('Switch volume repair did not converge')


def is_converged(history, cfg):
    c = cfg['convergence']
    w = c['window']
    if len(history) < w + 1 or history[-1]['step'] < cfg['earliest_certifiable_step']:
        return False
    tail = history[-w:]
    if any(r.get('phase') != 'common' or r.get('max_design_change') is None
           or r.get('relative_compliance_change') is None for r in tail):
        return False
    costs = [history[-w - 1]['compliance']] + [r['compliance'] for r in tail]
    return (max(r['max_design_change'] for r in tail) <= c['max_design_change']
            and max(r['relative_compliance_change'] for r in tail) <= c['max_relative_compliance_change']
            and (max(costs) - min(costs)) / max(abs(costs[-1]), 1e-30) <= c['relative_compliance_span']
            and max(r['volume_error'] for r in tail) <= c['volume_tolerance']
            and max(r['residual'] for r in tail) < c['residual_tolerance'])


def observation(step, ev, model, phase, oldx=None, x=None, oldc=None):
    return dict(step=step, phase=phase, **{k: float(ev[k]) for k in SCALARS},
                volume_error=abs(float(ev['volume']) - model.target),
                max_design_change=None if oldx is None else float(np.max(np.abs(x - oldx))),
                relative_compliance_change=None if oldc is None else
                abs(float(ev['compliance']) - oldc) / max(abs(oldc), 1e-30))


def run_one(out, record, profile, cfg, pause_after=None):
    """Resume immutable x+FE checkpoints. pause_after is only for offline tests."""
    out = Path(out)
    folder = out / 'tasks' / record['task_id'] / profile['id']
    folder.mkdir(parents=True, exist_ok=True)
    identity = digest(dict(task_assets=record['assets'], profile=profile,
                           protocol=cfg['protocol_sha256'], package=cfg['package_sha256']))
    if (folder / 'identity.json').exists():
        if read(folder / 'identity.json')['identity'] != identity:
            raise ValueError('Trajectory identity mismatch: ' + str(folder))
    else:
        dump(folder / 'identity.json', dict(identity=identity))
    if (folder / 'result.json').exists():
        return read(folder / 'result.json')
    cost = read(folder / 'cost.json') if (folder / 'cost.json').exists() else dict(
        attempted_FE=0, completed_FE=0, launch_FE=0, common_FE=0, switch_FE=0, pending=False)
    history, transition, initialization = [], None, None
    elapsed = 0.
    start = time.monotonic()
    step = 0
    common = False
    model = model_for(out, record, cfg, profile)

    def evaluate(x, category):
        cost['attempted_FE'] += 1
        cost[category + '_FE'] += 1
        cost['pending'] = True
        dump(folder / 'cost.json', cost)
        result = model.evaluate(x)
        if abs(result['volume'] - model.target) > cfg['convergence']['volume_tolerance']:
            raise ValueError('Physical volume outside common tolerance')
        cost['completed_FE'] += 1
        cost['pending'] = False
        dump(folder / 'cost.json', cost)
        return result

    def save():
        save_npz(folder / 'checkpoint.npz', x=x, **{k: ev[k] for k in ARRAYS},
                 scalars=np.array([ev[k] for k in SCALARS]),
                 history_json=json.dumps(history), step=step, common=common,
                 transition_json=json.dumps(transition),
                 initialization_json=json.dumps(initialization),
                 elapsed_seconds=elapsed + time.monotonic() - start, identity=identity)
        dump(folder / 'history.json', history)
        dump(folder / 'progress.json', dict(step=step, phase='common' if common else 'launch',
             compliance=float(ev['compliance']), updated_unix=time.time()))

    try:
        if (folder / 'checkpoint.npz').exists():
            with np.load(folder / 'checkpoint.npz', allow_pickle=False) as a:
                if str(a['identity']) != identity:
                    raise ValueError('Checkpoint identity mismatch')
                x = a['x'].copy()
                ev = {k: a[k].copy() for k in ARRAYS}
                ev.update(zip(SCALARS, map(float, a['scalars'])))
                history = json.loads(str(a['history_json']))
                transition = json.loads(str(a['transition_json']))
                initialization = json.loads(str(a['initialization_json']))
                step, common = int(a['step']), bool(a['common'])
                elapsed = float(a['elapsed_seconds'])
            model = model_for(out, record, cfg, profile, common=common)
            if history[-1]['step'] != step or len(history) != step + 1:
                raise ValueError('Checkpoint/history mismatch')
        else:
            source_x = load_start(out, record, model, profile.get('initializer', 'U'))
            x, initial_repair = repair_volume(model, source_x, cfg)
            initialization = dict(initializer=profile.get('initializer', 'U'),
                strategy=profile.get('strategy', profile['id']),
                source_array_sha256=array_digest(source_x),
                launch_array_sha256=array_digest(x),volume_repair=initial_repair,
                new_LLM_calls=0,extra_FE_for_volume_repair=0)
            save_npz(folder / 'state_source.npz', x=source_x)
            dump(folder / 'initialization.json', initialization)
            ev = evaluate(x, 'launch')
            history = [observation(0, ev, model, 'launch')]
            save_npz(folder / 'state_0000.npz', x=x, density=ev['density'])
            save()
        while True:
            # Always complete the common-boundary transition before certifying or stopping.
            if step == cfg['launch_updates'] and not common:
                save_npz(folder / 'state_0050_launch.npz', x=x, density=ev['density'])
                before = dict(history[-1])
                previous_x = x.copy()
                model = model_for(out, record, cfg, profile, common=True)
                x, repair = repair_volume(model, x, cfg)
                changed = (profile['penal'] != cfg['solver']['penal'] or
                           profile['rmin'] != cfg['solver']['rmin'] or repair['applied'])
                if changed:
                    ev = evaluate(x, 'switch')
                common = True
                transition = dict(step=step, launch_observation=before, volume_repair=repair,
                                  common_compliance=float(ev['compliance']), extra_FE=int(changed),
                                  x_before_sha256=array_digest(previous_x), x_after_sha256=array_digest(x))
                history[-1] = observation(step, ev, model, 'common_boundary')
                dump(folder / 'transition.json', transition)
                save_npz(folder / 'state_0050_common.npz', x=x, density=ev['density'])
                save()
            if common and is_converged(history, cfg):
                status = 'converged'
                break
            if step >= cfg['max_step']:
                status = 'max_steps'
                break
            if (out / 'PAUSE_REQUESTED').exists() or (pause_after is not None and step >= pause_after):
                save()
                return dict(status='paused', task_id=record['task_id'], profile=profile['id'], step=step)
            oldx, oldc = x, float(ev['compliance'])
            x = model.update(x, ev)
            ev = evaluate(x, 'common' if common else 'launch')
            step += 1
            history.append(observation(step, ev, model, 'common' if common else 'launch', oldx, x, oldc))
            if step % 10 == 0 or step in (cfg['launch_updates'], cfg['max_step']):
                save()
        save()
        save_npz(folder / 'state_final.npz', x=x, density=ev['density'], u=ev['u'])
        result = dict(status=status, task_id=record['task_id'], group_id=record['group_id'],
                      profile=profile['id'], launch_parameters=profile, step=step,
                      initializer=profile.get('initializer', 'U'),
                      strategy=profile.get('strategy', profile['id']),
                      initialization=initialization,
                      Cend=float(ev['compliance']), C50_common=history[cfg['launch_updates']]['compliance'],
                      C50_launch=transition['launch_observation']['compliance'],
                      cost=cost, transition=transition,
                      elapsed_seconds=elapsed + time.monotonic() - start,
                      identity=identity, completed_unix=time.time())
        dump(folder / 'result.json', result)
        return result
    except KeyboardInterrupt:
        # The last atomic checkpoint remains valid; durable FE attempts include discarded work.
        raise
    except Exception as exc:
        result = dict(status='failed', task_id=record['task_id'], profile=profile['id'],
                      initializer=profile.get('initializer', 'U'),
                      strategy=profile.get('strategy', profile['id']),
                      initialization=initialization, transition=transition,
                      step=step, error=str(exc), traceback=traceback.format_exc(), cost=cost,
                      identity=identity, elapsed_seconds=elapsed + time.monotonic() - start)
        dump(folder / 'result.json', result)
        return result
