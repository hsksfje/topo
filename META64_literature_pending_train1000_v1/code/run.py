"""Cross-platform, bounded process queue with OS locks and safe checkpoints."""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import json
from pathlib import Path, PurePosixPath
import platform
import shutil
import signal
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
import numpy as np
import scipy
from storage import dump, sha, digest
from experiment import read, run_one
from report import report


@contextmanager
def lock(out):
    path = Path(out)/'.run.lock'; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        f.seek(0, 2)
        if not f.tell():
            f.write(b'0'); f.flush()
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Output is already locked by another process') from exc
        else:
            import fcntl
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('Output is already locked by another process') from exc
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def verify_package():
    manifest = read(ROOT/'MANIFEST.json')
    if sha(ROOT/'MANIFEST.json') != (ROOT/'MANIFEST.sha256').read_text().split()[0]:
        raise ValueError('Package manifest identity changed')
    for name, expected in manifest['files'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Package file changed: '+name)
    return sha(ROOT/'MANIFEST.json')


def prepare(args, out):
    package_hash = verify_package()
    protocol = read(ROOT/'protocol.json')
    records = read(ROOT/'data_identity.json')['records']
    if not 0 <= args.start < len(records) or args.count < 1 or args.start+args.count > len(records):
        raise ValueError('Invalid task range: --start is zero-based, count must fit train1000')
    records = records[args.start:args.start+args.count]
    if args.stage == 'smoke':
        records = records[:1]
    cfg = dict(protocol, records=records, package_sha256=package_hash,
               validation_only=args.stage=='smoke', range_start=args.start, range_count=len(records))
    if cfg['validation_only']:
        cfg['max_step'] = 70
    if (out/'config.json').exists():
        if read(out/'config.json') != cfg:
            raise ValueError('Output identity differs; use original range/stage or a new output')
        if read(out/'config_identity.json')['sha256'] != sha(out/'config.json'):
            raise ValueError('Output config hash mismatch')
    else:
        if any(out.iterdir()):
            allowed = {'.run.lock'}
            if {p.name for p in out.iterdir()}-allowed:
                raise ValueError('Nonempty output has no matching config; choose an empty output')
        selected = {r['task_id'] for r in records}
        with zipfile.ZipFile(ROOT/'physical_inputs.zip') as z:
            for info in z.infolist():
                parts = PurePosixPath(info.filename).parts
                if info.is_dir():
                    continue
                if len(parts)!=3 or parts[0]!='inputs' or '..' in parts:
                    raise ValueError('Unexpected input archive path')
                if parts[1] in selected:
                    dest = out.joinpath(*parts); dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(z.read(info))
        # A source snapshot makes returned results independently auditable.
        files = read(ROOT/'MANIFEST.json')['files']
        for name in files:
            target = out/'source_snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, target)
        for name in ('MANIFEST.json','MANIFEST.sha256'):
            shutil.copy2(ROOT/name, out/'source_snapshot'/name)
        dump(out/'config.json', cfg)
        dump(out/'config_identity.json', dict(sha256=sha(out/'config.json')))
        dump(out/'environment.json', dict(python=sys.version, platform=platform.platform(),
             numpy=np.__version__, scipy=scipy.__version__, threads_per_worker=1))
    for r in records:
        for name, expected in r['assets'].items():
            if sha(out/r['path']/name) != expected:
                raise ValueError('Output input changed: '+r['task_id']+'/'+name)
    snapshot = out/'source_snapshot'
    if sha(snapshot/'MANIFEST.json') != package_hash:
        raise ValueError('Execution source snapshot mismatch')
    for name, expected in read(snapshot/'MANIFEST.json')['files'].items():
        if sha(snapshot/name) != expected:
            raise ValueError('Execution snapshot changed: '+name)
    return cfg


def run_phase(out, cfg, workers, stage):
    def jobs():
        for record in cfg['records']:
            for method in cfg['methods']:
                dest = out/'tasks'/record['task_id']/method
                if (dest/'failure.json').exists() or (dest/'result.json').exists():
                    continue
                if stage == '50' and (dest/'result_50.json').exists():
                    continue
                if stage == 'converge' and not (dest/'result_50.json').exists():
                    raise ValueError('Run stage 50 for all tasks before convergence')
                yield record, method
    queue = iter(jobs()); pending = {}; exhausted = False; last = 0.
    def pause_handler(*_):
        (out/'PAUSE_REQUESTED').touch()
        print('Pause requested; saving at the next OC boundary...', flush=True)
    previous = signal.signal(signal.SIGINT, pause_handler)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            while pending or not exhausted:
                while not exhausted and len(pending)<2*workers and not (out/'PAUSE_REQUESTED').exists():
                    try:
                        record, method = next(queue)
                    except StopIteration:
                        exhausted = True; break
                    future = pool.submit(run_one, str(out), record, method, cfg, stage)
                    pending[future] = (record, method)
                if (out/'PAUSE_REQUESTED').exists():
                    exhausted = True
                if pending:
                    done, _ = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
                    for future in done:
                        record, method = pending.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            # Construction/I/O exceptions outside the numerical worker
                            # are explicit failures, never a silent Uniform fallback.
                            result = dict(task_id=record['task_id'],method=method,status='failed',
                                          error=str(exc),traceback=traceback.format_exc())
                            dump(out/'tasks'/record['task_id']/method/'failure.json',result)
                        print(record['task_id'], method, result['status'], result.get('step',''), flush=True)
                if time.monotonic()-last >= 30:
                    p = report(out,cfg,'running_'+stage)
                    print('PROGRESS',p['complete50'], '/', p['planned'], 'OC50;',
                          p['converged'], 'converged;', p['failed'], 'failed',flush=True)
                    last = time.monotonic()
    finally:
        signal.signal(signal.SIGINT, previous)
    return not (out/'PAUSE_REQUESTED').exists()


EXCLUDED = {'.run.lock','PAUSE_REQUESTED','return_manifest.json','RETURN_READY.json'}


def verify_return(out):
    manifest = read(out/'return_manifest.json'); ready = read(out/'RETURN_READY.json')
    if sha(out/'return_manifest.json') != ready['manifest_sha256']:
        raise ValueError('Return manifest hash mismatch')
    actual = {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and p.name not in EXCLUDED}
    if actual != set(manifest['files']):
        raise ValueError('Return file set differs from sealed manifest')
    for name, expected in manifest['files'].items():
        if sha(out/name) != expected:
            raise ValueError('Return file hash mismatch: '+name)
    cfg = read(out/'config.json'); snap = out/'source_snapshot'
    if sha(snap/'MANIFEST.json') != cfg['package_sha256']:
        raise ValueError('Source manifest mismatch')
    for name, expected in read(snap/'MANIFEST.json')['files'].items():
        if sha(snap/name) != expected:
            raise ValueError('Returned source changed: '+name)
    return manifest


def export(out):
    cfg = read(out/'config.json'); p = report(out,cfg,'export_snapshot')
    files = {v.relative_to(out).as_posix():sha(v) for v in sorted(out.rglob('*'))
             if v.is_file() and v.name not in EXCLUDED}
    manifest = dict(schema='meta64-literature-pending-return-v1', files=files,
        config_sha256=sha(out/'config.json'), package_sha256=cfg['package_sha256'],
        validation_only=cfg['validation_only'], planned=p['planned'], complete50=p['complete50'],
        converged=p['converged'], max_steps=p['max_steps'], failed=p['failed'],
        all_settled=p['converged']+p['max_steps']+p['failed']==p['planned'],
        all_converged=p['converged']==p['planned'], created_unix=time.time())
    dump(out/'return_manifest.json',manifest)
    dump(out/'RETURN_READY.json',dict(manifest_sha256=sha(out/'return_manifest.json')))
    verify_return(out)
    target = out.with_name(out.name+'_return.zip'); temporary = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in list(files)+['return_manifest.json','RETURN_READY.json']:
            z.write(out/name,arcname=out.name+'/'+name)
    os.replace(temporary,target)
    target.with_suffix('.zip.sha256').write_text(sha(target)+'  '+target.name+'\n',encoding='utf-8')
    print('RETURN',target,flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['verify','all','50','converge','resume','smoke','status','pause','export','verify-return'],default='all')
    parser.add_argument('--workers',type=int)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--count',type=int,default=1000)
    parser.add_argument('--retry-failed',action='store_true')
    args = parser.parse_args()
    out = (args.output or ROOT/'outputs'/('smoke_validation' if args.stage=='smoke' else 'train1000_literature_pending_run01')).resolve()
    if args.stage=='status':
        print(json.dumps(read(out/'progress.json'),ensure_ascii=False,indent=2) if (out/'progress.json').exists() else 'Not started: '+str(out))
        return
    if args.stage=='pause':
        if not (out/'config.json').exists():
            raise ValueError('Output has not been initialized')
        (out/'PAUSE_REQUESTED').touch(); print('Pause requested:',out); return
    if args.stage=='verify':
        print('PACKAGE SHA256',verify_package())
        print('INPUTS',len(read(ROOT/'data_identity.json')['records']),'tasks; API calls: 0'); return
    if out==ROOT or ROOT in out.parents and any(p=='.git' for p in out.parts):
        raise ValueError('Unsafe output directory')
    with lock(out):
        if args.stage=='verify-return':
            verify_return(out); print('RETURN VERIFIED'); return
        if args.stage=='export':
            export(out); return
        cfg = prepare(args,out)
        if args.stage=='resume':
            (out/'PAUSE_REQUESTED').unlink(missing_ok=True)
        elif (out/'PAUSE_REQUESTED').exists():
            raise ValueError('Paused: use --stage resume with the same task range')
        if args.retry_failed:
            for path in sorted((out/'tasks').glob('*/*/failure.json')):
                dest=path.parent/'failure_archive'/str(time.time_ns()); dest.mkdir(parents=True)
                shutil.move(str(path),str(dest/'failure.json'))
        for name in ('return_manifest.json','RETURN_READY.json'):
            (out/name).unlink(missing_ok=True)
        available = len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
        workers = args.workers if args.workers is not None else max(1,available-1)
        if not 1 <= workers <= max(1,available-1):
            raise ValueError('Workers must reserve one available logical CPU')
        # Linux processes inherit this mask; Windows limits workers only.
        if hasattr(os,'sched_getaffinity') and available>1:
            mask=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,mask[:-1])
        dump(out/'runtime.json',dict(parent_pid=os.getpid(),workers=workers,
             available_logical_CPUs=available,reserved_logical_CPUs=1,started_unix=time.time(),stage=args.stage))
        print('OUTPUT',out,'TASKS',len(cfg['records']),'METHODS',len(cfg['methods']),'WORKERS',workers,flush=True)
        stage='all' if args.stage in ('resume','smoke') else args.stage
        ok=True
        if stage in ('all','50'):
            ok=run_phase(out,cfg,workers,'50')
        if ok and stage in ('all','converge'):
            ok=run_phase(out,cfg,workers,'converge')
        p=report(out,cfg,'paused' if not ok else 'phase_finished')
        if ok and p['converged']+p['max_steps']+p['failed']==p['planned']:
            dump(out/'COMPLETION.json',dict(all_settled=True,all_converged=p['converged']==p['planned'],
                 validation_only=cfg['validation_only'],progress=p))
        print(json.dumps({k:v for k,v in p.items() if k!='recent_progress'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
