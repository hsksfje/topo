"""Full authorized campaign; persistent state, distinct method folders and tables."""
import argparse
from initskill import locking as fcntl
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from .common import ROOT,DEFAULT_OUT,DATA,rows,solver,calibrate,dump,sha,protocol,MAX_OC_STEPS
from .convergence import run_to_convergence,goal_metrics

from .scope import current_scope
METHODS=current_scope()['methods']
SUBSETS=current_scope()['test_subsets']


def predictor(out,method,row):
    if method=='uniform':return None,dict(initialization_FE=0)
    if method=='knn':from .knn.core import predict_knn as fn
    elif method=='ulu':from .ulu_pca.core import predict as fn
    elif method=='wang':from .wang_pca.core import predict as fn
    elif method=='cnn':from .cnn_encoder_decoder.core import predict_cnn as fn
    elif method in {'meta','early'}:
        from .meta_neural.adapter import predict
        return predict(out,row,method)
    elif method=='cgan':from .cnn_cgan.core import predict_cgan as fn
    else:raise ValueError(method)
    return fn(out,row)


def evaluation_job(job):
    from .resources import high_resolution_slot
    out,method,subset,row,max_steps=job
    with high_resolution_slot(out,row['nx'],row['ny']):
        return _evaluation_job_unleased(job)


def _evaluation_job_unleased(job):
    out,method,subset,row,max_steps=job;out=Path(out)
    folder=out/'full_evaluation'/subset/method/row['source_task_id']
    if (folder/'result.json').exists():return dict(task_key=row['task_key'],status='cached')
    if (folder/'failure.json').exists():return dict(task_key=row['task_key'],status='previous_failure')
    try:
        # Prediction has no access to uniform final reference or test targets.
        m=solver(out,subset,row);started=time.perf_counter()
        if (folder/'initialization.npz').exists():
            meta=json.loads((folder/'initialization.json').read_text())
            with np.load(folder/'initialization.npz',allow_pickle=False) as a:x=a['x'].copy()
        else:
            raw,meta=predictor(out,method,row)
            if raw is None:raw=m.initial()
            if raw.size!=m.nx*m.ny:
                import torch
                a=torch.tensor(raw.reshape((64,64),order='F'))[None,None]
                raw=torch.nn.functional.interpolate(a,size=(m.ny,m.nx),mode='bilinear',align_corners=False)[0,0].numpy().ravel(order='F')
                meta['grid_transfer']='64 -> target bilinear design interpolation'
            x=calibrate(raw,m);folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'initialization.npz',raw=raw,x=x,density=m.physical(x))
            meta.update(initialization_seconds=time.perf_counter()-started+meta.get('external_initialization_seconds',0.),
                initialization_FE=meta.get('initialization_FE',0),calibration_max_change=float(np.max(np.abs(x-raw))),
                raw_mean=float(np.mean(raw)),raw_std=float(np.std(raw)),calibrated_std=float(np.std(x)),
                raw_nearly_constant=bool(np.std(raw)<1e-8))
            dump(folder/'initialization.json',meta)
        identity=dict(method=method,subset=subset,task_key=row['task_key'],group_id=row['group_id'],
            max_steps=max_steps,protocol_sha256=sha(out/'protocol.json'))
        result=run_to_convergence(m,x,max_steps,folder,identity)
        return dict(task_key=row['task_key'],status=result['status'],step=result['stop_step'])
    except Exception as exc:
        saved=json.loads((folder/'failure.json').read_text()) if (folder/'failure.json').exists() else {}
        saved.update(status='failed',task_key=row['task_key'],error=str(exc),traceback=traceback.format_exc())
        dump(folder/'failure.json',saved)
        return dict(task_key=row['task_key'],status='failed',error=str(exc))


def label_job(job):
    out,subset,row,max_steps=job;out=Path(out);dest=out/'labels'/subset
    p=dest/(row['source_task_id']+'.npz')
    if p.exists():return row['task_key']
    m=solver(out,subset,row);runfolder=dest/'runs'/row['source_task_id']
    result=run_to_convergence(m,m.initial(),max_steps,runfolder)
    with np.load(runfolder/'state_final.npz',allow_pickle=False) as a:x=a['x'].copy();rho=a['density'].copy()
    dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,x=x,density=rho)
    dump(p.with_suffix('.json'),dict(task_key=row['task_key'],group_id=row['group_id'],sha256=sha(p),
        physical_inputs={k:row['assets'][k]['sha256'] for k in ('forces','normals','target_vf')},
        compliance=result['final_compliance'],volume=float(rho.mean()),FE=result['FE'],seconds=result['seconds'],
        origin=dict(type='new_uniform_teacher',status=result['status'],steps=result['stop_step'])))
    return row['task_key']


def full_evaluate(out,method,workers=6,max_steps=MAX_OC_STEPS):
    leases=Path(out)/'campaign/method_locks';leases.mkdir(parents=True,exist_ok=True)
    with (leases/(method+'.lock')).open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        return _full_evaluate_unlocked(out,method,workers,max_steps)


def evaluation_plan(out,method,subset,max_steps=MAX_OC_STEPS):
    jobs=[];settled=[]
    for row in rows(subset):
        folder=Path(out)/'full_evaluation'/subset/method/row['source_task_id']
        if (folder/'result.json').exists() or (folder/'failure.json').exists():
            settled.append(row['task_key'])
        else:jobs.append((str(out),method,subset,row,max_steps))
    return jobs,settled


def _full_evaluate_unlocked(out,method,workers=6,max_steps=MAX_OC_STEPS):
    for subset in SUBSETS:
        jobs,settled=evaluation_plan(out,method,subset,max_steps)
        print('full-evaluation-resume',method,subset,'settled',len(settled),'pending',len(jobs),flush=True)
        # Avoid retaining many large CUDA/library contexts while waiting for
        # the shared slots. Settled+pending always covers the full manifest.
        from .resources import HIGH_RESOLUTION_SLOTS
        large=any(r['nx']*r['ny']>64*64 for r in rows(subset))
        count=min(workers,HIGH_RESOLUTION_SLOTS) if large else workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=count) as pool:
            futures=[pool.submit(evaluation_job,j) for j in jobs]
            for i,f in enumerate(concurrent.futures.as_completed(futures)):
                result=f.result();print('full-evaluation',method,subset,i+1,result,flush=True)
                if i%25==0:report(out)
        report(out)


def first_methods_full(out,workers=6):
    """Training-free/frozen retrieval methods can evaluate before NN training."""
    full_evaluate(out,'uniform',workers)
    full_evaluate(out,'knn',workers)


def report(out):
    out=Path(out);summary=[]
    for subset in SUBSETS:
        records=rows(subset)
        for method in METHODS:
            valid=[];fail=0;done=0;stopped=0
            for row in records:
                folder=out/'full_evaluation'/subset/method/row['source_task_id']
                p=folder/'result.json';u=out/'full_evaluation'/subset/'uniform'/row['source_task_id']/'result.json'
                if (folder/'failure.json').exists():fail+=1
                if not p.exists():continue
                done+=1;r=json.loads(p.read_text());stopped+=r['status']=='converged'
                if u.exists():valid.append(goal_metrics(r,json.loads(u.read_text())))
            reached=[v['steps_to_target'] for v in valid if v['reached']]
            ratios={k:[v['budget_ratios'][k] for v in valid if k in v['budget_ratios']] for k in ['50','100','200']}
            summary.append(dict(subset=subset,method=method,planned=len(records),complete=done,failed=fail,converged=stopped,
                paired=len(valid),target_reached=len(reached),mean_steps_reached=float(np.mean(reached)) if reached else None,
                median_steps_reached=float(np.median(reached)) if reached else None,
                mean_final_ratio=float(np.mean([v['final_ratio'] for v in valid])) if valid else None,
                budget_mean_ratios={k:float(np.mean(v)) if v else None for k,v in ratios.items()}))
    dump(out/'reports/summary.json',summary)
    fmt=lambda x:'—' if x is None else f'{x:.4f}'
    lines=['# 全量初始化 Benchmark：两个目标','',
        'Train1000、Valid100；ID/OOD/CR 各 Test300。此报告随队列更新，未完成的格不表示最终成绩。',
        '','## 表 1：达到相同柔度目标的速度','',
        '同任务共同目标：Uniform 终态柔度 × 1.01。报告预算内达到率；已达到均值只作描述，未达到与失败不能从分母删除。完整配对后再生成排名与区间。',
        '','| 测试集 | 方法 | 已完成/计划 | 达到/有配对 | 已达到平均步数 | 已达到中位步数 | 失败 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in summary:lines.append(f'| {r["subset"]} | {r["method"]} | {r["complete"]}/{r["planned"]} | {r["target_reached"]}/{r["paired"]} | {fmt(r["mean_steps_reached"])} | {fmt(r["median_steps_reached"])} | {r["failed"]} |')
    lines+=['','## 表 2：同预算与充分优化的柔度','',
        '柔度按同任务 Uniform 同预算/终态归一化，越低越好。终态仅代表共同停止规则或 500 步上限，并非全局最优。',
        '','| 测试集 | 方法 | C50 比 | C100 比 | C200 比 | 终态柔度比 | 已收敛/已完成 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in summary:lines.append(f'| {r["subset"]} | {r["method"]} | {fmt(r["budget_mean_ratios"]["50"])} | {fmt(r["budget_mean_ratios"]["100"])} | {fmt(r["budget_mean_ratios"]["200"])} | {fmt(r["mean_final_ratio"])} | {r["converged"]}/{r["complete"]} |')
    lines+=['','原生元学习步数与 OC 步数、参考 FE/热身/多分辨率教师开销在各方法账本中保留，不用接续 OC 的局部步数冒充总计算量。实现差异与未明确细节见方法审计。']
    (out/'reports/REPORT_zh.md').write_text('\n'.join(lines)+'\n')


def stage(out,name,command):
    marker=out/'campaign'/f'{name}.complete.json'
    if marker.exists():return
    logs=out/'campaign/logs';logs.mkdir(parents=True,exist_ok=True)
    dump(out/'campaign/progress.json',dict(stage=name,status='running',command=command,started_unix=time.time()))
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             MPLCONFIGDIR=str(out/'cache/matplotlib'),XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
    with (logs/f'{name}.log').open('a') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
    if result.returncode:
        dump(out/'campaign'/f'{name}.failure.json',dict(command=command,returncode=result.returncode))
        raise RuntimeError(f'Campaign stage failed: {name}; see {logs/name}.log')
    dump(marker,dict(command=command,finished_unix=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--stage',default='all',choices=['all','labels-valid','pairs-train','pairs-valid','evaluate','report'])
    p.add_argument('--method',choices=METHODS,default='uniform');a=p.parse_args();out=a.out
    if a.stage=='all':raise SystemExit('Use benchmark.py start for the scope-aware complete workflow')
    if a.stage=='report':report(out);return
    if a.stage=='labels-valid':
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
            for i,key in enumerate(pool.map(label_job,[(str(out),'valid100',r,MAX_OC_STEPS) for r in rows('valid100')])):print('valid-label',i+1,key,flush=True)
        return
    if a.stage.startswith('pairs-'):
        from .cnn_cgan.pairs import build
        build(out,'train1000' if a.stage=='pairs-train' else 'valid100',workers=a.workers);return
    if a.stage=='evaluate':full_evaluate(out,a.method,a.workers);return
    protocol(out);report(out)
    python=sys.executable;run=[python,'-m','initskill.methods.benchmark_init_v1.run','--out',str(out)]
    # argparse positional must precede flags in the concrete commands below.
    def r(stage_name,*flags):return [python,'-m','initskill.methods.benchmark_init_v1.run',stage_name,'--out',str(out),*map(str,flags)]
    candidate_workers=max(1,a.workers-2)
    def c(stage_name,*flags,python_bin=None):return [python_bin or python,'-m','initskill.methods.benchmark_init_v1.campaign','--out',str(out),'--stage',stage_name,'--workers',str(candidate_workers),*map(str,flags)]
    stage(out,'verification',[python,'-m','initskill.methods.benchmark_init_v1.verification','--out',str(out)])
    stage(out,'train_labels',r('labels','--reuse-archive'))
    stage(out,'valid_labels',c('labels-valid'))
    stage(out,'knn_fit',r('fit-knn'))
    # Reference optimization does not supply model inputs/selection. It can
    # proceed independently, while each candidate method runs sequentially.
    # A method-level lease joins any already live Uniform run without repeating
    # its in-flight tasks. Reserve two workers for the common reference.
    baseline_command=c('evaluate','--method','uniform','--workers',2)
    baseline_log=out/'campaign/logs/uniform_background.log'
    baseline_log.parent.mkdir(parents=True,exist_ok=True)
    baseline_stream=baseline_log.open('a')
    baseline_env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    baseline=subprocess.Popen(baseline_command,stdout=baseline_stream,stderr=subprocess.STDOUT,env=baseline_env)
    stage(out,'knn_full',c('evaluate','--method','knn'))
    for method in ['ulu','wang']:
        stage(out,method+'_train',r('train-pca','--method',method,'--epochs',1000))
        stage(out,method+'_full',c('evaluate','--method',method))
    stage(out,'cnn_train',r('train-cnn','--epochs',500,'--batch-size',16))
    stage(out,'cnn_full',c('evaluate','--method','cnn'))
    jax_python=str(out/'jax_environment/bin/python')
    stage(out,'native_fem_verification',[jax_python,'-m','initskill.methods.benchmark_init_v1.meta_neural.verify','--out',str(out)])
    stage(out,'meta_smoke',[jax_python,'-m','initskill.methods.benchmark_init_v1.run','smoke-meta','--out',str(out)])
    stage(out,'meta_train',[jax_python,'-m','initskill.methods.benchmark_init_v1.run','train-meta','--out',str(out)])
    stage(out,'meta_export',[jax_python,'-m','initskill.methods.benchmark_init_v1.meta_neural.export','--method','meta','--out',str(out)])
    stage(out,'meta_full',c('evaluate','--method','meta'))
    stage(out,'early_train',[jax_python,'-m','initskill.methods.benchmark_init_v1.early_trajectories.run','--out',str(out)])
    stage(out,'early_export',[jax_python,'-m','initskill.methods.benchmark_init_v1.meta_neural.export','--method','early','--out',str(out)])
    stage(out,'early_full',c('evaluate','--method','early'))
    stage(out,'paired_train',c('pairs-train'));stage(out,'paired_valid',c('pairs-valid'))
    stage(out,'cgan_train',r('train-cgan','--epochs',20,'--batch-size',16))
    stage(out,'cgan_full',c('evaluate','--method','cgan'))
    baseline_returncode=baseline.wait();baseline_stream.close()
    if baseline_returncode:raise RuntimeError('Common Uniform reference run failed; inspect uniform_background.log')
    dump(out/'campaign/uniform_full.complete.json',dict(command=baseline_command,finished_unix=time.time()))
    report(out);dump(out/'campaign/COMPLETE.json',dict(methods=METHODS,test_counts={k:len(rows(k)) for k in SUBSETS}))


if __name__=='__main__':main()
