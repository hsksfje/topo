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
from experiment import model_for, prepare_inputs, read, run_one
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
    if count == 100:
        return records
    if count <= 0 or count > 100 or count % 10:
        raise ValueError('--task-count must be10,20,...,100; selection is stratified, not outcome-based')
    selected = {r['task_id'] for b in range(10) for r in
                [r for r in records if r['vf_bin'] == b][:count // 10]}
    return [r for r in records if r['task_id'] in selected]


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
        profiles = [p for p in protocol['profiles'] if p['id'] in ('P01', 'P34', 'P54', 'P60')]
        protocol['max_step'] = 70
        protocol['validation_only'] = True
    protocol['profiles'] = profiles
    protocol['trajectory_count'] = len(records) * len(profiles)
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
    cfg = dict(protocol, **expected, records=records, created_unix=time.time())
    dump(out / 'config.json', cfg)
    dump(out / 'config_identity.json', dict(sha256=sha(out / 'config.json')))
    dump(out / 'environment.json', dict(python=sys.version, platform=platform.platform(),
         numpy=__import__('numpy').__version__, scipy=__import__('scipy').__version__,
         BLAS_threads=1, new_API_calls=0))
    return cfg


def preflight(out, cfg):
    if (out / 'preflight.json').exists():
        audit = read(out / 'preflight.json')
        if audit['config_sha256'] != sha(out / 'config.json'):
            raise ValueError('Preflight belongs to a different config')
        return audit
    profile = next(p for p in read(ROOT / 'protocol.json')['profiles'] if p['id'] == 'P34')
    rows = []
    for record in cfg['records']:
        model = model_for(out, record, cfg, profile, common=True)
        ev = model.evaluate(model.initial())
        relative = abs(ev['compliance'] - record['uniform_C0']) / record['uniform_C0']
        if relative > 1e-8 or abs(ev['volume'] - model.target) > 1e-8:
            raise ValueError('Original Uniform C0 mismatch: ' + record['task_id'])
        rows.append(dict(task_id=record['task_id'],relative_C0_error=relative,residual=ev['residual']))
        dump(out / 'preflight_progress.json', dict(completed=len(rows), planned=len(cfg['records'])))
    # A nontrivial OC trajectory confirms coordinate mapping AND feasible-side update semantics.
    record = cfg['records'][0]
    model = model_for(out, record, cfg, profile, common=True)
    x = model.initial()
    ev = model.evaluate(x)
    for _ in range(50):
        x = model.update(x, ev)
        ev = model.evaluate(x)
    relative50 = abs(ev['compliance'] - record['uniform_C50']) / record['uniform_C50']
    if relative50 > 1e-8:
        raise ValueError('Original Uniform OC50 mismatch')
    audit = dict(status='passed', config_sha256=sha(out / 'config.json'), rows=rows,
                 Uniform50_relative_error=relative50, attempted_FE=len(rows)+51,
                 cost_scope='preflight only, separate from experiment trajectories',new_API_calls=0)
    dump(out / 'preflight.json', audit)
    print('PREFLIGHT', len(rows), 'tasks; Uniform50 relative error', relative50, flush=True)
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
    parser.add_argument('--task-count', type=int, default=100)
    parser.add_argument('--profiles', nargs='+')
    args = parser.parse_args()
    if args.stage == 'verify':
        print('PACKAGE VERIFIED', verify_package(), '100 tasks,60 profiles,3800 archived outcomes')
        return
    default_name = 'smoke_validation' if args.stage == 'smoke' else 'train100_parameter60_run01'
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
    elif (args.task_count != 100 or args.profiles) and args.output is None:
        parser.error('Partial runs require an explicit --output to keep the full100 run separate')
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
