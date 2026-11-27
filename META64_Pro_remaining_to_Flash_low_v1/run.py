#!/usr/bin/env python3
"""Independent Flash rerun of confirmed Pro methods missing Flash coverage; CPU entry point."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import sys
import time
import traceback
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'code/vendor')]
import numpy as np
from initskill.io import dump, save_npz, sha
from initskill.physics import SolverConfig
from initskill.methods.trajectory_skill50_v1.geometry import METHODS, teacher_pool, decode
from llm_six import SIMP, decode as matrix_decode
import continuation as cont
import extra_initializers as extra
import portable_lock


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare_payload():
    from verify import verify_package
    verify_package()
    hashes = read(ROOT / 'PAYLOAD_HASHES.json')
    dest = ROOT / 'data'
    with zipfile.ZipFile(ROOT / 'payload.zip') as z:
        assert set(z.namelist()) == set(hashes)
        for name, expected in hashes.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            path = dest / name
            if not path.exists():
                data = z.read(name)
                assert hashlib.sha256(data).hexdigest() == expected
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(path.suffix + '.extracting')
                temp.write_bytes(data)
                temp.replace(path)
            assert sha(path) == expected, 'Payload modified: ' + name
    return dest


def check_eval(model, x, plan):
    ev = model.evaluate(x)
    if abs(ev['volume'] - model.target) > plan['convergence']['volume_tolerance']:
        raise ValueError('Physical volume constraint failed')
    if ev['residual'] >= plan['convergence']['residual_tolerance']:
        raise ValueError('FE residual failed')
    if not np.isfinite(ev['compliance']) or ev['compliance'] <= 0:
        raise ValueError('Invalid compliance')
    return ev


def scalar_row(step, ev):
    return dict(step=step, **{k:float(ev[k]) for k in ('compliance','volume','residual','grayness')})


def worker(job, out_s, plan, max_step):
    if not 50 <= max_step <= min(500, plan['max_step']):
        raise ValueError('This package has a hard 500-step OC limit')
    out = Path(out_s)
    tid, method = job['task_id'], job['method']
    folder = out / 'tasks' / tid / method
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / '.job.lock').open('a+b')
    portable_lock.flock(lock, portable_lock.LOCK_EX | portable_lock.LOCK_NB)
    begin = time.monotonic()
    history = []
    try:
        if (folder / 'result.json').exists():
            old = read(folder / 'result.json')
            if old['status'] in ('converged','max_steps'):
                return old
        model = SIMP(ROOT / 'data' / job['path'], SolverConfig(**plan['solver']))
        lineage = job['initialization']
        fe_count = 0
        initialization_fe = 0
        initialization_coarse_fe = 0
        def evaluate(x):
            nonlocal fe_count
            fe_count += 1
            return check_eval(model, x, plan)
        checkpoint = folder / 'checkpoint.npz'
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as a:
                x = a['x'].copy()
                history = json.loads(str(a['history_json']))
                lineage = str(a['lineage'])
                fe_count = int(a['fe_count'])
                initialization_fe = int(a['initialization_fe'])
                initialization_coarse_fe = int(a['initialization_coarse_fe'])
        elif job.get('seed'):
            with np.load(ROOT / 'data' / job['seed'], allow_pickle=False) as a:
                if 'x' in a:
                    x = a['x'].copy()
                    history = json.loads(str(a['history_json']))
                else:
                    x = matrix_decode(model, a['values'])
        elif method in ('A1','A3','B1'):
            generated = ROOT / 'generated' / 'tasks' / tid / method / 'raw_matrix.npz'
            receipt = generated.parent / 'generation.json'
            if not generated.exists() or not receipt.exists() or read(receipt)['status'] != 'valid':
                result = dict(task_id=tid, method=method, status='blocked_missing_LLM', error='No historical initializer; optional generate_missing.py creates a separately identified Flash-low proposal')
                dump(folder / 'result.json', result)
                return result
            with np.load(generated, allow_pickle=False) as a:
                x = matrix_decode(model, a['values'])
            lineage = 'new_Flash_low:' + read(ROOT / 'generated/config.json')['model']
        else:
            task = read(ROOT / 'data' / job['path'] / 'task_input.json')
            if method in METHODS:
                old = model.evaluate
                def counted(x):
                    nonlocal initialization_fe
                    initialization_fe += 1
                    return old(x)
                model.evaluate = counted
                x, meta = decode(model, task, teacher_pool(task)[method])
                model.evaluate = old
            else:
                old = model.evaluate
                def counted(x):
                    nonlocal initialization_fe
                    initialization_fe += 1
                    return old(x)
                def coarse(event):
                    nonlocal initialization_coarse_fe
                    initialization_coarse_fe += int(event == 'start')
                model.evaluate = counted
                model.coarse_audit_hook = coarse
                x, meta = extra.generate(model, method)
                model.evaluate = old
            dump(folder / 'initializer.json', meta)
        ev = evaluate(x)
        if history:
            relative = abs(ev['compliance'] - history[-1]['compliance']) / max(abs(ev['compliance']), 1e-30)
            if relative > 1e-8:
                raise ValueError(f'Saved state/history compliance mismatch: {relative}')
        else:
            history = [scalar_row(0, ev)]
            save_npz(folder / 'state_0000.npz', x=x, density=ev['density'])
        def save():
            save_npz(checkpoint, x=x, history_json=json.dumps(history), lineage=lineage, fe_count=fe_count, initialization_fe=initialization_fe, initialization_coarse_fe=initialization_coarse_fe)
            dump(folder / 'history.json', history)
            dump(folder / 'progress.json', dict(task_id=tid, method=method, step=history[-1]['step'], compliance=ev['compliance'], pid=os.getpid(), updated_unix=time.time(), lineage=lineage))
        def converged():
            return history[-1]['step'] >= 60 and cont.converged(history, plan)
        save()
        while history[-1]['step'] < max_step and not converged():
            if (out / 'PAUSE_REQUESTED').exists():
                save()
                return dict(task_id=tid, method=method, status='paused')
            oldx, oldc = x, ev['compliance']
            x = model.update(x, ev)
            ev = evaluate(x)
            history.append(dict(scalar_row(history[-1]['step'] + 1, ev), max_design_change=float(np.max(np.abs(x-oldx))), relative_compliance_change=abs(ev['compliance']-oldc)/max(abs(oldc),1e-30)))
            if history[-1]['step'] % 10 == 0:
                save()
            if history[-1]['step'] in (50,100,200,500):
                save_npz(folder / f'state_{history[-1]["step"]:04d}.npz', x=x, density=ev['density'])
        save()
        save_npz(folder / 'state_final.npz', x=x, density=ev['density'], u=ev['u'])
        result = dict(task_id=tid, method=method, status='converged' if converged() else 'max_steps', step=history[-1]['step'], C50=history[50]['compliance'], Cend=ev['compliance'], lineage=lineage, max_step=max_step, new_FE=fe_count, initialization_FE=initialization_fe, initialization_coarse_FE=initialization_coarse_fe, historical_initialization_cost_included=False, new_API=0, seconds_this_invocation=time.monotonic()-begin, completed_unix=time.time())
        dump(folder / 'result.json', result)
        return result
    except Exception as exc:
        result = dict(task_id=tid, method=method, status='failed', error=str(exc), traceback=traceback.format_exc())
        dump(folder / 'result.json', result)
        return result
    finally:
        lock.close()


def report(out, jobs, plan, state='running'):
    results = []
    active = []
    for job in jobs:
        folder = out / 'tasks' / job['task_id'] / job['method']
        if (folder / 'result.json').exists():
            r = read(folder / 'result.json')
            if r['status'] in ('converged','max_steps'):
                h = read(folder / 'history.json')
                cap = next((v for v in h if v['step'] == 500), h[-1])
                u = plan['uniform_references'][r['task_id']]
                better = 100 * (1 - cap['compliance'] / float(u['Ccap']))
                stop = min(r['step'], 500)
                both = r['status']=='converged' and r['step']<=500 and u['uniform_converged500'].lower()=='true'
                r.update(I50=100*(1-r['C50']/float(u['Uniform50'])), C500=cap['compliance'], stop500=stop, gain500_pct=better, both_converged500=both, strict_double_win500=both and stop<int(u['Uniform_Ncap']) and better>1, faster_similar500=both and stop<int(u['Uniform_Ncap']) and abs(better)<1)
            if r['status'] in ('converged','max_steps'):
                pro=plan['pro_references'][r['task_id']+'/'+r['method']]
                both_pro=r['status']=='converged' and r['step']<=500 and pro['method_converged500'].lower()=='true'
                gain_pro=100*(1-r['C500']/float(pro['Ccap']))
                r.update(Pro_C50=float(pro['C50']),Pro_C500=float(pro['Ccap']),Flash_vs_Pro_I50=100*(1-r['C50']/float(pro['C50'])),Flash_vs_Pro_gain500=gain_pro,both_Pro_converged500=both_pro,Flash_vs_Pro_strict_win500=both_pro and r['stop500']<int(pro['Ncap']) and gain_pro>1,Flash_vs_Pro_faster_similar500=both_pro and r['stop500']<int(pro['Ncap']) and abs(gain_pro)<1)
            results.append(r)
        elif (folder / 'progress.json').exists():
            active.append(read(folder / 'progress.json'))
    status_counts = Counter(r['status'] for r in results)
    progress = dict(status=state, planned=len(jobs), terminal=len(results), counts=dict(status_counts), active=active, workers=read(out/'config.json')['workers'], updated_unix=time.time(), new_API_calls=0)
    dump(out / 'progress.json', progress)
    target = out / 'reports'
    target.mkdir(exist_ok=True)
    if results:
        fields = sorted(set().union(*(r.keys() for r in results)))
        with (target / 'results.csv').open('w', encoding='utf-8-sig', newline='') as f:
            w=csv.DictWriter(f, fieldnames=fields);w.writeheader();w.writerows(results)
    lines = ['# 已确认Pro方法的Flash-low独立重跑', '', f"状态：{state}；已结算 {len(results)}/{len(jobs)}；{dict(status_counts)}。", '', '仅统计这批补跑任务，不能当作全1000任务的收益率；返回主机后按任务ID和方法合并。新Flash提案与历史初态按lineage分开。500步双方真正收敛才计严格双赢或更快且差异<1%；达到上限不是收敛。', '', '|方法|初态来源|成功轨迹|500步严格双赢>1%|500步更快且差异<1%|', '|---|---|---:|---:|---:|']
    groups = sorted({(r['method'],r.get('lineage','')) for r in results if r['status'] in ('converged','max_steps')})
    for method,lineage in groups:
        rows=[r for r in results if r['method']==method and r.get('lineage')==lineage]
        lines.append(f"|{method}|{lineage}|{len(rows)}|{sum(r['strict_double_win500'] for r in rows)}|{sum(r['faster_similar500'] for r in rows)}|")
    lines+=['','## 与同任务纯Pro结果比较','','|方法|已完成配对|50步柔度相对Pro改善%|500步平均柔度改善%|500步双方真收敛|500步更快且改善>1%|500步更快且差异<1%|','|---|---:|---:|---:|---:|---:|---:|']
    for method in sorted({r['method'] for r in results}):
        rs=[r for r in results if r['method']==method and r['status'] in ('converged','max_steps')]
        if rs:
            lines.append(f"|{method}|{len(rs)}|{sum(r['Flash_vs_Pro_I50'] for r in rs)/len(rs):+.4f}|{sum(r['Flash_vs_Pro_gain500'] for r in rs)/len(rs):+.4f}|{sum(r['both_Pro_converged500'] for r in rs)}|{sum(r['Flash_vs_Pro_strict_win500'] for r in rs)}|{sum(r['Flash_vs_Pro_faster_similar500'] for r in rs)}|")
    (target / 'REPORT_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in progress.items() if k!='active'}, ensure_ascii=False), flush=True)
    return progress


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['prepare','run','status','export'], default='run')
    p.add_argument('--workers', type=int, help='default: available logical CPUs minus one')
    p.add_argument('--max-step', type=int, choices=[52,500], default=500, help='Production hard limit: 500; 52 is validation only')
    p.add_argument('--limit', type=int)
    p.add_argument('--methods', help='comma separated method names')
    p.add_argument('--output', type=Path)
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--resume', action='store_true', help='remove this run PAUSE_REQUESTED flag')
    a=p.parse_args()
    out=(a.output or ROOT/'outputs'/f'flash_remaining_{a.max_step}_run01').resolve()
    plan=read(ROOT/'plan.json')
    jobs=plan['jobs']
    if a.methods:
        selected=set(a.methods.split(','));assert selected<=set(j['method'] for j in jobs), 'Unknown method'
        jobs=[j for j in jobs if j['method'] in selected]
    if a.limit is not None:
        assert a.limit>0;jobs=jobs[:a.limit]
    assert jobs
    if a.stage=='status':
        print(json.dumps(read(out/'progress.json'), ensure_ascii=False, indent=2));return
    if a.stage=='export':
        from export_backfill import export
        export(out);return
    prepare_payload()
    if a.stage=='prepare':
        print('Payload verified; no FE/API. Jobs:',len(jobs));return
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.run.lock').open('a+b')
    portable_lock.flock(lock,portable_lock.LOCK_EX|portable_lock.LOCK_NB)
    available=cont.available_cpus()
    quota=len(available)
    try:
        q,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if q!='max':quota=min(quota,max(1,int(q)//int(period)))
    except (OSError,ValueError):pass
    worker_limit=max(1,quota-1)
    workers=a.workers if a.workers is not None else worker_limit
    assert 1<=workers<=worker_limit, f'workers must be 1..{worker_limit} (reserve one CPU)'
    # Affinity reservation applies to the controller and all spawned workers.
    cont.reserve_one_cpu(available)
    config=dict(plan_sha256=sha(ROOT/'plan.json'), max_step=a.max_step, jobs=jobs, workers=workers, available_cpu_ids=available, cpu_quota=quota, threads_per_worker=1, validation_only=a.max_step==52)
    if (out/'config.json').exists():
        old=read(out/'config.json')
        for key in ('plan_sha256','max_step','jobs'):
            assert old[key]==config[key], 'Output belongs to another selection/protocol; choose another --output'
    dump(out/'config.json',config)
    if a.resume:(out/'PAUSE_REQUESTED').unlink(missing_ok=True)
    if (out/'PAUSE_REQUESTED').exists():raise SystemExit('Paused. Use --resume when ready.')
    pending_jobs=[]
    for job in jobs:
        path=out/'tasks'/job['task_id']/job['method']/'result.json'
        if path.exists():
            status=read(path)['status']
            if status in ('converged','max_steps'):continue
            if status=='failed' and not a.retry_failed:continue
            if status=='blocked_missing_LLM' and not (ROOT/'generated/tasks'/job['task_id']/job['method']/'raw_matrix.npz').exists():continue
            archive=path.with_name('previous_result_'+str(time.time_ns())+'.json');path.replace(archive)
        pending_jobs.append(job)
    report(out,jobs,plan)
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        # Keep the queue bounded so PAUSE can stop scheduling promptly.
        iterator=iter(pending_jobs)
        pending={}
        def submit():
            for _ in range(max(0,workers-len(pending))):
                try:job=next(iterator)
                except StopIteration:break
                pending[pool.submit(worker,job,str(out),plan,a.max_step)]=job
        submit()
        while pending:
            done,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
            for future in done:
                job=pending.pop(future)
                try:r=future.result()
                except Exception:
                    dump(out/'controller_failure.json',dict(job=job,error=traceback.format_exc()))
                    (out/'PAUSE_REQUESTED').touch();raise
                print(job['task_id'],job['method'],r['status'],flush=True)
            report(out,jobs,plan,'paused' if (out/'PAUSE_REQUESTED').exists() else 'running')
            if not (out/'PAUSE_REQUESTED').exists():submit()
    final=report(out,jobs,plan,'paused' if (out/'PAUSE_REQUESTED').exists() else 'finished')
    lock.close()
    if final['counts'].get('failed') or final['counts'].get('blocked_missing_LLM'):
        raise SystemExit(2)


if __name__=='__main__':
    mp.freeze_support()
    main()
