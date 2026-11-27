import os,sys,json,time,traceback,csv,statistics
from pathlib import Path
import numpy as np
from initskill.io import dump,sha,save_npz
from initskill.physics import SolverConfig
from llm_six import SIMP
OUT=Path('.')
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def source_folder(task, method, cfg):
    root=Path(cfg['fresh_output'])
    return root/('corpus' if method in cfg['fixed'] else 'tasks' if method in cfg['llm'] else 'extra')/task/method

def new_model(record, cfg):
    return SIMP(Path(cfg['fresh_output'])/record['path'], SolverConfig(**cfg['solver']))

def checked_ev(model, x, cfg):
    ev = model.evaluate(x)
    if abs(ev['volume']-model.target) > cfg['convergence']['volume_tolerance']:
        raise ValueError('Physical volume constraint failed')
    return ev

def converged(history, cfg):
    c = cfg['convergence']; w = c['window']; tail = history[-w:]
    if len(tail) < w or any(r.get('max_design_change') is None for r in tail):
        return False
    previous = history[-w-1]['compliance']
    costs = [previous]+[r['compliance'] for r in tail]
    return (max(r['max_design_change'] for r in tail) <= c['max_design_change']
            and max(r['relative_compliance_change'] for r in tail) <= c['max_relative_compliance_change']
            and (max(costs)-min(costs))/max(abs(costs[-1]), 1e-30) <= c['relative_compliance_span']
            and max(r['residual'] for r in tail) < c['residual_tolerance'])

def run_one(record, method, cfg, output=None):
    global OUT
    if output is not None:
        OUT=Path(output)
    task = record['task_id']; folder = OUT/'tasks'/task/method
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'result.json').exists():
        return read(folder/'result.json')
    source = source_folder(task, method, cfg)
    identity = cfg['trajectories'][task+'/'+method]
    if identity['source_status'] != 'complete':
        result = dict(task_id=task, method=method, status='source_unavailable',
                      error='Original OC50 not valid: '+identity['source_status'])
        dump(folder/'result.json', result)
        return result
    if (OUT/'PAUSE_REQUESTED').exists():
        raise InterruptedError('Pause requested')
    cost = read(folder/'cost.json') if (folder/'cost.json').exists() else dict(
        attempted_FE=0, recovery_FE=0, update_FE=0, continuation_seconds=0., pending=False)
    tic = time.monotonic(); history = []; model = new_model(record, cfg)
    def fe(x, recovery=False):
        cost['attempted_FE'] += 1; cost['recovery_FE' if recovery else 'update_FE'] += 1
        cost['pending'] = True; dump(folder/'cost.json', cost)
        ev = checked_ev(model, x, cfg)
        cost['pending'] = False
        return ev
    def save(x, ev):
        save_npz(folder/'checkpoint.npz', x=x, history_json=json.dumps(history))
        dump(folder/'history.json', history)
        dump(folder/'progress.json', dict(task_id=task, method=method, step=history[-1]['step'],
             compliance=ev['compliance'], max_design_change=history[-1].get('max_design_change'),
             pid=os.getpid(), updated_unix=time.time(), status='running'))
        dump(folder/'cost.json', cost)
    try:
        if (folder/'checkpoint.npz').exists():
            with np.load(folder/'checkpoint.npz', allow_pickle=False) as a:
                x = a['x'].copy(); history = json.loads(str(a['history_json']))
        else:
            with np.load(source/'state_0050.npz', allow_pickle=False) as a:
                x = a['x'].copy()
            history = read(source/'history.json')
        ev = fe(x, recovery=True)
        assert abs(ev['compliance']-history[-1]['compliance'])/ev['compliance'] < 1e-8
        save(x, ev)
        while history[-1]['step'] < cfg['max_step'] and not converged(history, cfg):
            if (OUT/'PAUSE_REQUESTED').exists():
                save(x, ev)
                raise InterruptedError('Pause requested')
            oldx = x; oldc = ev['compliance']; x = model.update(x, ev); ev = fe(x)
            history.append(dict(step=history[-1]['step']+1,
                **{k:float(ev[k]) for k in ('compliance','volume','residual','grayness')},
                max_design_change=float(np.max(np.abs(x-oldx))),
                relative_compliance_change=abs(ev['compliance']-oldc)/max(abs(oldc),1e-30)))
            if history[-1]['step'] % 10 == 0:
                save(x, ev)
            if history[-1]['step'] in (100,200,500,1000):
                save_npz(folder/f'state_{history[-1]["step"]:04d}.npz', x=x,
                         density=ev['density'])
        save(x, ev)
        save_npz(folder/'state_final.npz', x=x, density=ev['density'], u=ev['u'])
        cost['continuation_seconds'] += time.monotonic()-tic
        result = dict(task_id=task, method=method,
                      status='converged' if converged(history,cfg) else 'max_steps',
                      step=history[-1]['step'], C50=history[50]['compliance'],
                      Cend=ev['compliance'], additional_OC=history[-1]['step']-50,
                      end_density_change=history[-1].get('max_design_change'),
                      end_relative_compliance_change=history[-1].get('relative_compliance_change'),
                      cost=cost, source=str(source), completed_unix=time.time())
        dump(folder/'cost.json', cost)
        dump(folder/'result.json', result)
        return result
    except InterruptedError:
        cost['continuation_seconds'] += time.monotonic()-tic
        dump(folder/'cost.json', cost)
        raise
    except Exception as exc:
        cost['continuation_seconds'] += time.monotonic()-tic
        result = dict(task_id=task, method=method, status='failed', error=str(exc),
                      last_step=history[-1]['step'] if history else None,
                      traceback=traceback.format_exc(), cost=cost)
        dump(folder/'result.json', result)
        return result

def first_sustained(history, threshold, window=5):
    for i in range(len(history)-window+1):
        if all(r['compliance'] <= threshold for r in history[i:i+window]):
            return history[i]['step']
    return None

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def numerical_report(cfg, status='running'):
    rows = []; active = []; results = {}; histories = {}
    for r in cfg['records']:
        for m in cfg['methods']:
            p = OUT/'tasks'/r['task_id']/m
            if (p/'result.json').exists():
                results[r['task_id'],m] = read(p/'result.json')
            elif (p/'progress.json').exists():
                active.append(read(p/'progress.json'))
    for (t,m), v in results.items():
        if v['status'] in ('converged','max_steps'):
            histories[t,m] = read(OUT/'tasks'/t/m/'history.json')
    for r in cfg['records']:
        t = r['task_id']; u = results.get((t,'U'),{})
        upath=source_folder(t,'U',cfg)/'result_50.json'
        u50=read(upath).get('C50') if upath.exists() else None
        for m in cfg['methods']:
            v = results.get((t,m))
            if v is None:
                continue
            both = v['status'] == u.get('status') == 'converged'
            h = histories.get((t,m),[])
            rows.append(dict(task_id=t, vf_bin=r['vf_bin'], volume_fraction=r['volume_fraction'],
                method=m, status=v['status'], stop_step=v.get('step'), C50=v.get('C50'),
                Cend=v.get('Cend'), I50=100*(1-v['C50']/u50) if v.get('C50') and u50 else None,
                Iend_vs_Uniform_converged=100*(1-v['Cend']/u['Cend']) if both else None,
                convergence_step_saving_pct=100*(1-v['step']/u['step']) if both else None,
                step_reach_Uniform50=first_sustained(h,u50) if h else None,
                step_reach_Uniform_end_plus1pct=first_sustained(h,u['Cend']*1.01) if h and u.get('status')=='converged' else None,
                continued_FE=v.get('cost',{}).get('attempted_FE'),
                continuation_seconds=v.get('cost',{}).get('continuation_seconds'),
                error=v.get('error')))
    summary=[]
    common=[r['task_id'] for r in cfg['records'] if all(results.get((r['task_id'],m),{}).get('status')=='converged' for m in cfg['methods'])]
    mean=lambda vv:statistics.mean(vv) if vv else None
    med=lambda vv:statistics.median(vv) if vv else None
    for m in cfg['methods']:
        rr=[r for r in rows if r['method']==m];paired=[r for r in rr if r['Iend_vs_Uniform_converged'] is not None]
        co=[r for r in paired if r['task_id'] in common]
        summary.append(dict(method=m,settled=len(rr),converged=sum(r['status']=='converged' for r in rr),
            max_steps=sum(r['status']=='max_steps' for r in rr),
            failed=sum(r['status'] not in ('converged','max_steps') for r in rr),
            paired_n=len(paired),mean_Iend=mean([r['Iend_vs_Uniform_converged'] for r in paired]),
            median_Iend=med([r['Iend_vs_Uniform_converged'] for r in paired]),
            median_stop_step=med([r['stop_step'] for r in paired]),
            median_step_saving_pct=med([r['convergence_step_saving_pct'] for r in paired]),
            wins_over1pct=sum(r['Iend_vs_Uniform_converged']>1 for r in paired),
            common_n=len(common),common_mean_Iend=mean([r['Iend_vs_Uniform_converged'] for r in co]),
            common_median_stop_step=med([r['stop_step'] for r in co])))
    progress=dict(status=status,planned=len(cfg['records'])*len(cfg['methods']),settled=len(rows),
        converged=sum(r['status']=='converged' for r in rows),max_steps=sum(r['status']=='max_steps' for r in rows),
        failed=sum(r['status'] not in ('converged','max_steps') for r in rows),
        common_converged_tasks=len(common),workers=cfg['workers'],active=active,
        new_API_calls=0,updated_unix=time.time())
    dump(OUT/'progress.json',progress)
    dest=OUT/'reports';dest.mkdir(exist_ok=True)
    write_csv(dest/'results.csv',rows);write_csv(dest/'method_summary.csv',summary)
    fmt=lambda x:f'{x:+.4f}' if x is not None else '—'
    lines=['# META64 新工况：从OC50继续到收敛','',
        f'状态：{status}；已结算{len(rows)}/{len(cfg["records"])*len(cfg["methods"])}，其中收敛{progress["converged"]}、到上限未收敛{progress["max_steps"]}、失败{progress["failed"]}。全部38方法共同收敛任务数：{len(common)}。','',
        '本次独立冻结的新增训练任务，与此前100任务物理组不重叠。初态生成不重做，无新增API。最终结果是本判据下的数值稳定解，不保证全局最优。','',
        '连续10次更新均满足 max|Δx|≤0.01、逐步|ΔC|/C≤1e-4，且这10次及前一状态的柔度相对极差≤5e-4，物理体积误差≤1e-8、FE残差<1e-6。总细网格OC上限2000；到上限不能称为收敛。','',
        '历史0—50步没有逐步密度变化量，因此最早可认证停止点为60步；早期停止时间属于从50步起的认证时间，不能断言此前未收敛。报告同时比较达到共同Uniform质量目标的步数，避免把较差局部解的快速停机当成成功。目标需连续5个观测状态满足；最后不足5步时不推定持续达到。','',
        '以下均值和中位数仅使用该方法与Uniform都收敛的配对任务；运行中各行样本不同，不作正式排名。共同样本统计见method_summary.csv。','',
        '| 方法 | 收敛/任务数 | 上限未收敛 | 配对数 | 平均最终改善% | 中位最终改善% | 中位停止步数 | 中位步数节省% |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["method"]} | {r["converged"]}/{len(cfg["records"])} | {r["max_steps"]} | {r["paired_n"]} | {fmt(r["mean_Iend"])} | {fmt(r["median_Iend"])} | {r["median_stop_step"] if r["median_stop_step"] is not None else "—"} | {fmt(r["median_step_saving_pct"])} |')
    lines+=['','COARSE30/80的前置31/81次粗FE和OC5的前置5次OC不包含在这里的细网格轨迹步数中，不能当作免费初始化。LLM生成、初始化FE等历史成本仍以源目录账本为准。续跑时间和新FE是本轮增量成本，不等于完整端到端时间；并行墙钟时间不用于方法公平排名。','',
            '[逐任务数值](results.csv) · [方法汇总](method_summary.csv) · [配置](../config.json) · [运行说明](../README_zh.md)']
    (dest/'REPORT_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in progress.items() if k!='active'}),flush=True)
    return progress

def acquire_lock(handle):
    if os.name=='nt':
        import msvcrt
        handle.seek(0);handle.write(b'0');handle.flush();handle.seek(0)
        msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)

def available_cpus():
    if hasattr(os,'sched_getaffinity'):return sorted(os.sched_getaffinity(0))
    if os.name=='nt':
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        process_mask=ctypes.c_size_t();system_mask=ctypes.c_size_t()
        if kernel.GetProcessAffinityMask(ctypes.c_void_p(kernel.GetCurrentProcess()),ctypes.byref(process_mask),ctypes.byref(system_mask)):
            return [i for i in range(8*ctypes.sizeof(ctypes.c_size_t)) if process_mask.value & (1<<i)]
    return list(range(os.cpu_count() or 1))

def reserve_one_cpu(cpus):
    allowed=cpus[1:] if len(cpus)>1 else cpus
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,allowed)
    elif os.name=='nt':
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        if not kernel.SetProcessAffinityMask(ctypes.c_void_p(kernel.GetCurrentProcess()),ctypes.c_size_t(sum(1<<i for i in allowed))):
            raise OSError(ctypes.get_last_error(),'Cannot reserve one logical CPU')
    return allowed
