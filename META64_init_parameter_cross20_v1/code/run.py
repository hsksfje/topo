#!/usr/bin/env python3
"""Relocatable, restartable offline runner. No credentials and no network code."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import contextlib
import platform
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from experiment import array_digest, load_start, model_for, prepare_inputs, read, repair_volume, run_one
from storage import digest, dump, sha


@contextlib.contextmanager
def output_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.run.lock').open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def verify_package():
    expected = (ROOT / 'MANIFEST.sha256').read_text().split()[0]
    if sha(ROOT / 'MANIFEST.json') != expected:
        raise ValueError('Package manifest changed')
    manifest = read(ROOT / 'MANIFEST.json')
    for name, value in manifest['files'].items():
        if sha(ROOT / name) != value:
            raise ValueError('Package file missing or changed: ' + name)
    return expected


def select_records(records, count):
    if count == 20:
        return records
    if count != 10:
        raise ValueError('--task-count supports10 or20; full protocol uses20 tasks')
    chosen = {next(r['task_id'] for r in records if r['vf_bin']==b) for b in range(10)}
    return [r for r in records if r['task_id'] in chosen]




def prepare(out, args, package_hash):
    protocol = read(ROOT / 'protocol.json')
    all_records = read(ROOT / 'data/tasks.json')
    records = all_records[:1] if args.stage == 'smoke' else select_records(all_records, args.task_count)
    profiles = protocol['profiles']
    if args.profiles:
        requested = set(args.profiles)
        if requested - {p['id'] for p in profiles}:
            raise ValueError('Unknown profile ID')
        profiles = [p for p in profiles if p['id'] in requested]
    if args.stage == 'smoke':
        records = records[:1]
        profiles = protocol['profiles']
        protocol['max_step'] = 70
        protocol['validation_only'] = True
    protocol['profiles'] = profiles
    protocol['trajectory_count'] = len(records) * len(profiles)
    protocol['reference_cell_count'] = len(records) * 6
    protocol['comparison_cell_count'] = len(records) * 12
    expected = dict(protocol_sha256=digest(protocol), package_sha256=package_hash,
                    task_ids=[r['task_id'] for r in records])
    if (out / 'config.json').exists():
        cfg = read(out / 'config.json')
        if sha(out / 'config.json') != read(out / 'config_identity.json')['sha256']:
            raise ValueError('Output config changed')
        if any(cfg[k] != v for k, v in expected.items()):
            raise ValueError('Output belongs to a different package/selection/protocol; use its original command or a new output')
        for r in cfg['records']:
            for name, value in r['assets'].items():
                if sha(out / r['path'] / name) != value:
                    raise ValueError('Prepared input changed: ' + r['task_id'] + '/' + name)
        return cfg
    records = prepare_inputs(out, records)
    references = [r for r in read(ROOT/'data/references.json') if r['task_id'] in expected['task_ids']]
    cfg = dict(protocol, **expected, records=records, references=references,
               source_snapshot_required=True, created_unix=time.time())
    dump(out / 'config.json', cfg)
    dump(out / 'config_identity.json', dict(sha256=sha(out / 'config.json')))
    dump(out / 'environment.json', dict(python=sys.version, platform=platform.platform(),
         numpy=__import__('numpy').__version__, scipy=__import__('scipy').__version__,
         BLAS_threads=1, new_API_calls=0))
    # Return the actual executed source, not just a package hash.
    import shutil
    for source in list((ROOT/'code').glob('*.py')) + [ROOT/'run.py', ROOT/'protocol.json']:
        relative=source.relative_to(ROOT)
        target=out/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    shutil.copy2(ROOT/'MANIFEST.json',out/'source_snapshot/package_MANIFEST.json')
    shutil.copy2(ROOT/'MANIFEST.sha256',out/'source_snapshot/package_MANIFEST.sha256')
    return cfg


def preflight(out, cfg):
    if (out/'preflight.json').exists():
        audit=read(out/'preflight.json')
        if audit['config_sha256'] != sha(out/'config.json'):
            raise ValueError('Preflight belongs to a different config')
        return audit
    default=dict(id='U__P34',initializer='U',strategy='P34',penal=3.,rmin=2.5,move=.2)
    rows=[];attempts=0;initial_repairs=[]
    for record in cfg['records']:
        model=model_for(out,record,cfg,default,common=True)
        for initializer in ('U','PG1','L1_PG'):
            x=load_start(out,record,model,initializer)
            ev=model.evaluate(x);attempts+=1
            target=record['uniform_C0'] if initializer=='U' else record['starts'][initializer]['default_C0']
            relative=abs(ev['compliance']-target)/target
            if relative>1e-8 or abs(ev['volume']-model.target)>1e-8:
                raise ValueError('Original step-zero FE mismatch: '+record['task_id']+'/'+initializer)
            rows.append(dict(task_id=record['task_id'],initializer=initializer,relative_C0_error=relative,volume_error=abs(ev['volume']-model.target)))
        # Check start-field repair under every launch strategy WITHOUT adding FE.
        for profile in cfg['profiles']:
            launch=model_for(out,record,cfg,profile)
            original=load_start(out,record,launch,profile['initializer'])
            adjusted,repair=repair_volume(launch,original,cfg)
            if abs(launch.physical(adjusted).mean()-launch.target)>1e-8:
                raise ValueError('Launch volume repair failed')
            initial_repairs.append(dict(task_id=record['task_id'],job=profile['id'],repair=repair,
                original_array_sha256=array_digest(original),launch_array_sha256=array_digest(adjusted)))
        dump(out/'preflight_progress.json',dict(completed_tasks=len(rows)//3,planned_tasks=len(cfg['records'])))
    # One whole default OC50 for ALL three initializers verifies frozen source fields.
    endpoint=[];record=cfg['records'][0]
    for initializer in ('U','PG1','L1_PG'):
        model=model_for(out,record,cfg,default,common=True)
        x=load_start(out,record,model,initializer);ev=model.evaluate(x);attempts+=1
        for _ in range(50):
            x=model.update(x,ev);ev=model.evaluate(x);attempts+=1
        target=record['uniform_C50'] if initializer=='U' else record['starts'][initializer]['default_C50']
        rel=abs(ev['compliance']-target)/target
        if rel>1e-8:raise ValueError('Original OC50 mismatch: '+initializer)
        endpoint.append(dict(task_id=record['task_id'],initializer=initializer,relative_C50_error=rel))
    audit=dict(status='passed',config_sha256=sha(out/'config.json'),rows=rows,
        OC50_checks=endpoint,launch_volume_checks=initial_repairs,attempted_FE=attempts,
        cost_scope='Preflight only; separate from120 new trajectory costs',new_API_calls=0)
    dump(out/'preflight.json',audit)
    print('PREFLIGHT',len(cfg['records']),'tasks;3 frozen starts;all launch volumes checked',flush=True)
    return audit




def available_cpus():
    if hasattr(os, 'sched_getaffinity'):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def execute(out, cfg, workers):
    from analyze import report
    if (out / 'PAUSE_REQUESTED').exists():
        print('PAUSE_REQUESTED exists. Remove it, then rerun the same command.', flush=True)
        report(out, cfg, 'paused')
        return
    preflight(out, cfg)
    # Round-robin over tasks within each profile; deterministic and independent of results.
    jobs = [(r, p) for p in cfg['profiles'] for r in cfg['records']
            if not (out / 'tasks' / r['task_id'] / p['id'] / 'result.json').exists()]
    iterator = iter(jobs)
    report(out, cfg, 'running')
    if (out / 'RETURN_READY.json').exists():
        raise ValueError('Already sealed; use --stage verify-return or --stage status')
    last_report = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}

        def submit():
            if (out / 'PAUSE_REQUESTED').exists():
                return False
            item = next(iterator, None)
            if item is None:
                return False
            r, p = item
            pending[pool.submit(run_one, str(out), r, p, cfg)] = (r['task_id'], p['id'])
            return True

        for _ in range(workers):
            if not submit():
                break
        try:
            while pending:
                done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    key = pending.pop(future)
                    result = future.result()
                    print(*key, result['status'], 'step', result.get('step'), flush=True)
                    submit()
                if time.monotonic() - last_report >= 30 or not pending:
                    report(out, cfg, 'paused' if (out/'PAUSE_REQUESTED').exists() else 'running')
                    last_report = time.monotonic()
        except BaseException:
            (out / 'PAUSE_REQUESTED').touch()
            for future in pending:
                future.cancel()
            raise
    status = 'paused' if (out / 'PAUSE_REQUESTED').exists() else 'complete'
    progress = report(out, cfg, status)
    if status == 'complete' and progress['settled'] == progress['planned']:
        dump(out / 'COMPLETION.json', dict(settled=progress['settled'],
             failed=progress['failed'], max_steps=progress['max_steps'],
             note='Complete means all outcomes settled, not all converged.', unix=time.time()))
        from export_results import seal_output
        seal_output(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['verify','prepare','preflight','smoke','run','report','status','pause','resume','export','verify-return'], default='run')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--task-count', type=int, default=20)
    parser.add_argument('--profiles', nargs='+')
    args = parser.parse_args()
    if args.stage == 'verify':
        print('PACKAGE VERIFIED', verify_package(), '20 tasks,40 frozen fields,120 reused cells,120 new trajectories')
        return
    default_name = 'smoke_cross_validation' if args.stage == 'smoke' else 'train20_init_parameter_cross_run01'
    out = (args.output or ROOT / 'outputs' / default_name).resolve()
    # Do not allow outputs to overwrite source, sealed data, or package identity.
    if out == ROOT or any(out == ROOT / n or ROOT / n in out.parents for n in ('code','data','tests','provenance','validation')):
        parser.error('Choose an output directory outside package source/data')
    if args.stage in ('status','pause','resume','report','export','verify-return'):
        if not (out / 'config.json').exists():
            parser.error('Output is not prepared: ' + str(out))
        if args.stage == 'pause':
            (out / 'PAUSE_REQUESTED').touch()
            print('Pause requested; in-flight workers save at their next OC boundary.')
            return
        if args.stage == 'status':
            print((out / 'progress.json').read_text(encoding='utf-8') if (out/'progress.json').exists() else 'Prepared; no run yet')
            return
        with output_lock(out):
            if args.stage == 'verify-return':
                from export_results import verify_return
                print('RETURN VERIFIED', verify_return(out)['trajectory_count'])
                return
            package_hash = verify_package()
            cfg = read(out / 'config.json')
            if sha(out/'config.json') != read(out/'config_identity.json')['sha256'] or cfg['package_sha256'] != package_hash:
                raise ValueError('Output/package identity mismatch')
            if args.stage == 'export':
                from export_results import export_zip
                print(export_zip(out))
                return
            if args.stage == 'report':
                if (out/'RETURN_READY.json').exists():
                    print('Sealed reports are already final. View reports/REPORT_zh.md.')
                    return
                from analyze import report
                report(out, cfg, 'manual_snapshot')
                return
            (out / 'PAUSE_REQUESTED').unlink(missing_ok=True)
            # Restore the frozen selection for resume instead of CLI defaults.
            args.task_count = len(cfg['records'])
            args.profiles = [p['id'] for p in cfg['profiles']]
            args.stage = 'smoke' if cfg.get('validation_only') else 'run'
    elif (args.task_count != 20 or args.profiles) and args.output is None:
        parser.error('Partial runs require an explicit --output to keep the full20 run separate')
    cpus = available_cpus()
    limit = max(1, len(cpus) - 1)
    workers = args.workers if args.workers is not None else limit
    if workers < 1 or workers > limit:
        parser.error('--workers must be between1 and available CPUs minus one ('+str(limit)+')')
    with output_lock(out):
        package_hash = verify_package()
        cfg = prepare(out, args, package_hash)
        if args.stage == 'prepare':
            print('PREPARED', len(cfg['records']), 'tasks', len(cfg['profiles']), 'profiles', out)
            return
        if args.stage == 'preflight':
            preflight(out, cfg)
            return
        if (out/'RETURN_READY.json').exists():
            from export_results import verify_return
            verify_return(out)
            print('Already complete and sealed:',out)
            return
        if hasattr(os, 'sched_setaffinity') and len(cpus) > 1:
            os.sched_setaffinity(0, cpus[1:])
        dump(out / 'process.json', dict(pid=os.getpid(), workers=workers,
             available_cpus=cpus, reserved_logical_cpus=1 if len(cpus)>1 else 0, started_unix=time.time()))
        print('RUN', len(cfg['records'])*len(cfg['profiles']), 'trajectories;',workers,'workers;',out,flush=True)
        execute(out, cfg, workers)


if __name__ == '__main__':
    main()
