from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import hashlib
import json
import math
import os
import time
import traceback
import sys
import platform
import scipy
import numpy as np
import psutil
from initskill import locking
from initskill.methods.benchmark_init_v1 import common
from initskill.methods.benchmark_init_v1.convergence import run_to_convergence,goal_metrics
from scripts.knn import CONFIG,build_bank,prediction,resize_design

ROOT=common.ROOT
OUT=ROOT/'results'
METHODS=['uniform','knn_legacy','knn_v2']
SPLITS=['valid100','test_id300','test_ood300','test_cr300']


def protocol():
    values=dict(schema='knn-v2-independent',methods=METHODS,task_counts={s:len(common.rows(s)) for s in ['train1000']+SPLITS},
        manifests={s:common.sha(common.DATA/'manifests'/f'{s}.jsonl') for s in ['train1000']+SPLITS},
        bundle_sha256=common.sha(ROOT/'data/BUNDLE_MANIFEST.json'),solver=asdict(common.SolverConfig()),
        max_steps=500,min_steps=200,knn=CONFIG,decimal_digits=40,filter='rmin=2.5*ny/64',
        original_legacy_metric_preserved=True,validation_role='full audit of fixed variant; no fit or search',
        test_status='Exploratory repair after viewing v1 test results; not blind confirmation',
        sources={p.relative_to(ROOT).as_posix():common.sha(p) for p in sorted(list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').rglob('*.py'))+list((ROOT/'tests').rglob('*.py'))+[ROOT/'run.py',ROOT/'setup.py'])})
    path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=values:raise ValueError('Numerical/source/data protocol changed. Use a fresh package/results folder.')
    if not path.exists():common.dump(path,values)
    return values


def allocation(cpus,ram):
    cpus=int(cpus);ram=float(ram)
    if cpus<2 or ram<8:raise ValueError('At least2 CPU slots and8GiB RAM required')
    detected=psutil.virtual_memory().total/1024**3;usable=min(ram,detected)
    reserve=min(16.,max(6.,usable/4));small=min(12,max(1,cpus//2));large=min(cpus-small,max(1,int((usable-reserve-small*.35)//2)))
    return dict(cpu_budget=cpus,requested_ram_gib=ram,detected_ram_gib=detected,planning_ram_gib=usable,
        small_workers=small,large_workers=large,total_worker_limit=small+large,
        reserved_ram_gib=reserve,large_job_estimate_gib=2.,threads_per_worker=1,
        note='Separate small/large pools. Actual utilization varies by stage; RAM figures are planning estimates.')


def worker(job):
    out,subset,row,budget=job
    out=Path(out)
    folder=out/'full_evaluation'/subset
    # One solver per task; filter and input preparation reused among3 methods.
    m=common.solver(out,subset,row)
    with np.load(out/'models/knn/bank.npz',allow_pickle=False) as z:bank={k:z[k] for k in z.files}
    answer=[]
    for method in METHODS:
        dest=folder/method/row['source_task_id']
        if (dest/'result.json').exists() or (dest/'failure.json').exists():answer.append(dict(method=method,cached=True));continue
        dest.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter()
        try:
            meta_path=dest/'initialization.json';state=dest/'initialization.npz'
            if state.exists() and meta_path.exists():
                meta=json.loads(meta_path.read_text());assert common.sha(state)==meta['state_sha256']
                with np.load(state,allow_pickle=False) as z:x=z['x'].copy()
            else:
                raw,meta=prediction(bank,row,method)
                if raw is None:raw=m.initial()
                if raw.size!=m.nx*m.ny:
                    # numpy equivalent of align_corners=False bilinear with
                    # edge replication, in the original x-major cell order.
                    raw=resize_design(raw,m.nx,m.ny)
                    meta['grid_transfer']='64 -> target; half-pixel bilinear, replicate edges'
                calibrated=common.calibrate(raw,m)
                alpha=CONFIG['uniform_mix'] if method=='knn_v2' else 0.
                x=(1-alpha)*calibrated+alpha*m.target
                if abs(m.physical(x).mean()-m.target)>1e-10:raise ValueError('Initial volume violation')
                common.atomic_savez(state,raw=raw,calibrated=calibrated,x=x,density=m.physical(x))
                meta.update(initialization_seconds=time.perf_counter()-start,uniform_mix=alpha,state_sha256=common.sha(state),
                    physical_volume=float(m.physical(x).mean()),calibration_max_change=float(np.max(abs(calibrated-raw))))
                common.dump(meta_path,meta)
            identity=dict(method=method,subset=subset,task_key=row['task_key'],group_id=row['group_id'],max_steps=budget,protocol_sha256=common.sha(out/'protocol.json'))
            result=run_to_convergence(m,x,budget,dest,identity)
            answer.append(dict(method=method,status=result['status'],step=result['stop_step']))
        except Exception as exc:
            old=json.loads((dest/'failure.json').read_text()) if (dest/'failure.json').exists() else {}
            old.update(status='failed',task_key=row['task_key'],method=method,error=str(exc),traceback=traceback.format_exc())
            common.dump(dest/'failure.json',old);answer.append(dict(method=method,status='failed',error=str(exc)))
    return dict(subset=subset,task_key=row['task_key'],methods=answer)


def summarize(strict=False):
    summary={};cases=[];coverage={};done=True
    for subset in SPLITS:
        block={};coverage[subset]={}
        rows=common.rows(subset)
        for method in METHODS:
            runs=[];failed=0;pending=0;reached=0;penalties=[];ratios=[];budgets={k:[] for k in ['50','100','200']};converged=0;improved=0;defined=0;inline=0;fe=0;seconds=0.
            for row in rows:
                folder=OUT/'full_evaluation'/subset/method/row['source_task_id'];rp=folder/'result.json';fp=folder/'failure.json';up=OUT/'full_evaluation'/subset/'uniform'/row['source_task_id']/'result.json'
                if not rp.exists() and not fp.exists():pending+=1;continue
                attempt=folder/'FE_attempts.jsonl'
                if attempt.exists():fe+=len(attempt.read_text().splitlines())
                if fp.exists():
                    failed+=1
                    if up.exists():defined+=1;penalties.append(500)
                    cases.append(dict(subset=subset,method=method,task=row['task_key'],status='failed',error=json.loads(fp.read_text())['error']));continue
                r=json.loads(rp.read_text());runs.append(r);converged+=r['status']=='converged';seconds+=r['seconds'];inline+=sum('precision' in h for h in r['history'])
                init=json.loads((folder/'initialization.json').read_text());seconds+=init['initialization_seconds']
                if up.exists():
                    u=json.loads(up.read_text());metrics=goal_metrics(r,u);defined+=1;reached+=metrics['reached'];penalties.append(metrics['steps_to_target'] if metrics['reached'] else 500)
                    ratios.append(metrics['final_ratio']);improved+=metrics['final_ratio']<1-1e-6
                    for k,v in metrics['budget_ratios'].items():
                        if k in budgets:budgets[k].append(v)
                    cases.append(dict(subset=subset,method=method,task=row['task_key'],status=r['status'],final_ratio=metrics['final_ratio'],steps_to_target=metrics['steps_to_target'],stop_step=r['stop_step'],C50=r['budget_points'].get('50'),C100=r['budget_points'].get('100'),C200=r['budget_points'].get('200'),Cfinal=r['final_compliance']))
            coverage[subset][method]=dict(planned=len(rows),completed=len(runs),failed=failed,pending=pending)
            done=done and pending==0
            avg=lambda a:float(np.mean(a)) if a else None
            block[method]=dict(**coverage[subset][method],reached=reached,target_defined=defined,penalized_steps_mean=avg(penalties),final_ratio_mean=avg(ratios),final_ratio_median=float(np.median(ratios)) if ratios else None,improved=improved,converged=converged,capped=len(runs)-converged,budget_ratios={k:avg(a) for k,a in budgets.items()},FE_attempts=fe,accepted_decimal_FE=inline,measured_success_seconds=seconds)
        summary[subset]=block
    dest=OUT/'reports';dest.mkdir(exist_ok=True)
    common.dump(dest/'summary.json',summary);common.dump(dest/'coverage.json',coverage)
    keys=sorted({k for r in cases for k in r})
    with (dest/'paired_tasks.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows(cases)
    fmt=lambda v:'—' if v is None else f'{v:.4f}'
    text=['# 独立KNN v2：双目标报告','','状态：'+('所有计划任务均已结束（含明确记录的失败）。' if done else '中间结果，尚未全量结束。'),'',
        'Train1000建库；Valid100用于固定方案的完整验证审计，不进行测试集选参。三类Test各300任务。该修正版在查看v1测试错误后设计，属于探索性修复比较，不能称全盲确认。',
        '','三种方法均使用新的Decimal40公共数值后备路径。knn_legacy保留旧1NN检索/初态，属于新求解器下的重跑，不等于旧Windows结果。knn_v2为平滑/反向对称检索＋3NN加权＋10%均匀混合。',
        '','## 表1：达到同任务Uniform终态柔度×1.01的步数','','未达到或失败按500步封顶；参考失败时不定义目标。此达标步数不是实际收敛停止步数。',
        '','| 测试集 | 方法 | 成功/失败/待运行 | 达到/有参考目标 | 封顶平均步数 |','| --- | --- | --- | --- | --- |']
    for s in SPLITS[1:]:
        for method,b in summary[s].items():text.append(f"| {s} | {method} | {b['completed']}/{b['failed']}/{b['pending']} | {b['reached']}/{b['target_defined']} | {fmt(b['penalized_steps_mean'])} |")
    text+=['','## 表2：相同预算和终态柔度比','','逐任务相对Uniform后再汇总；柔度比仅含两边都有成功结果的样本，需连同失败率解读。至少200步、最多500步，到上限不称收敛。',
        '','| 测试集 | 方法 | C50比 | C100比 | C200比 | 终态比均值/中位数 | 收敛/封顶 |','| --- | --- | --- | --- | --- | --- | --- |']
    for s in SPLITS[1:]:
        for method,b in summary[s].items():text.append(f"| {s} | {method} | {fmt(b['budget_ratios']['50'])} | {fmt(b['budget_ratios']['100'])} | {fmt(b['budget_ratios']['200'])} | {fmt(b['final_ratio_mean'])}/{fmt(b['final_ratio_median'])} | {b['converged']}/{b['capped']} |")
    text+=['','Valid100单独保存于summary.json，不与测试混合排名。逐任务数据见paired_tasks.csv。',
        '','FE账本包含失败和恢复重算；Decimal内层修正不冒称额外OC步。成功耗时不含公共网格准备/建库，不能当作全部冷启动墙钟成本；中断后未提交的耗时可能丢失。',
        '','共享标签来自旧Uniform优化库（997收敛、3到原预算上限），不是全局最优证明。标签生成成本快照见data/historical_teacher_costs.json；新建库不需要FE。',
        '','此次只比较初始化后公共OC，不涉及原始Meta/CNN等模型。']
    (dest/'REPORT_zh.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    if strict and not done:raise ValueError('Run unfinished; see coverage.json')
    return done


def start(cpus,ram):
    OUT.mkdir(exist_ok=True);control=OUT/'campaign';control.mkdir(exist_ok=True)
    with (control/'run.lock').open('a') as lock:
        try:locking.flock(lock,locking.LOCK_EX|locking.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This KNN package is already running')
        from scripts.prepare_data import prepare
        prepare();protocol();build_bank(OUT);settings=allocation(cpus,ram)
        run=dict(pid=os.getpid(),created=psutil.Process().create_time(),root=str(ROOT.resolve()),started_unix=time.time(),concurrency=settings,environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()))
        previous=control/'execution.json'
        if previous.exists():
            history=control/'execution_history';history.mkdir(exist_ok=True)
            (history/(str(time.time_ns())+'.json')).write_bytes(previous.read_bytes())
        common.dump(control/'execution.json',run);print(json.dumps(run),flush=True)
        try:
            for subsets in [['valid100'],SPLITS[1:]]:
                with ProcessPoolExecutor(max_workers=settings['small_workers']) as small,ProcessPoolExecutor(max_workers=settings['large_workers']) as large:
                    futures=[]
                    for subset in subsets:
                        for row in common.rows(subset):
                            if all((OUT/'full_evaluation'/subset/m/row['source_task_id']/'result.json').exists() or (OUT/'full_evaluation'/subset/m/row['source_task_id']/'failure.json').exists() for m in METHODS):continue
                            futures.append((large if row['nx']>64 else small).submit(worker,(str(OUT),subset,row,500)))
                    for i,future in enumerate(as_completed(futures),1):
                        print(json.dumps(dict(task_finished=i,**future.result())),flush=True)
                        if i%10==0:summarize()
                summarize()
            summarize(strict=True);common.dump(control/'COMPLETE.json',dict(finished_unix=time.time(),all_planned_runs_settled=True,methods=METHODS,expected_runs=3000,numerical_failures_not_disguised_as_success=True))
        except BaseException:
            for child in psutil.Process().children(recursive=True):
                try:child.terminate()
                except psutil.NoSuchProcess:pass
            raise


def stop():
    path=OUT/'campaign/execution.json'
    if not path.exists():print('No KNN execution record');return
    r=json.loads(path.read_text())
    try:p=psutil.Process(r['pid'])
    except psutil.NoSuchProcess:print('KNN process already stopped');return
    if abs(p.create_time()-r['created'])>.001 or r['root']!=str(ROOT.resolve()):raise ValueError('PID identity mismatch; refusing stop')
    children=p.children(recursive=True)
    for child in children:
        try:child.terminate()
        except psutil.NoSuchProcess:pass
    p.terminate();psutil.wait_procs(children+[p],timeout=10);print('Stopped this independent KNN run. Start resumes coherent checkpoints; recorded failures remain recorded.')
