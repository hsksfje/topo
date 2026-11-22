"""Paired, offline summaries. Historical outcomes are scalar references only."""
import csv
import os
import statistics
import time
from pathlib import Path

from experiment import read
from storage import dump

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.tmp')
    with temp.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def mean(values):
    return statistics.mean(values) if values else None


def median(values):
    return statistics.median(values) if values else None


def first_sustained(history, target, window=5):
    # Only common-model evaluations qualify; launch compliance is incomparable.
    for i in range(len(history) - window + 1):
        block = history[i:i + window]
        if all(r['phase'] in ('common_boundary','common') and r['compliance'] <= target for r in block):
            return block[0]['step']
    return None


def report(out, cfg, status='running'):
    out = Path(out)
    records = {r['task_id']: r for r in cfg['records']}
    with (ROOT / 'data/initialization_results.csv').open(encoding='utf-8-sig') as f:
        old = [r for r in csv.DictReader(f) if r['task_id'] in records]
    uniform = {r['task_id']: r for r in old if r['method']=='U'}
    if len(uniform) != len(records):
        raise ValueError('Archived Uniform task set mismatch')
    results, per, pairs = [], {}, []
    for p in cfg['profiles']:
        for tid, task in records.items():
            folder = out / 'tasks' / tid / p['id']
            path = folder / 'result.json'
            if not path.exists():
                continue
            value = read(path)
            if value['task_id'] != tid or value['profile'] != p['id']:
                raise ValueError('Result key mismatch')
            u = uniform[tid]
            paired = value['status']=='converged' and u['status']=='converged'
            q = 100*(1-value['Cend']/float(u['Cend'])) if paired else None
            s = 100*(1-value['step']/int(u['stop_step'])) if paired else None
            target_step = None
            if value['status'] in ('converged','max_steps') and u['status']=='converged':
                target_step = first_sustained(read(folder/'history.json'),1.01*float(u['Cend']))
            row = dict(task_id=tid,vf_bin=task['vf_bin'],profile=p['id'],
                       launch_penal=p['penal'],launch_rmin=p['rmin'],launch_move=p['move'],
                       status=value['status'],stop_step=value.get('step'),Cend=value.get('Cend'),
                       C50_launch=value.get('C50_launch'),C50_common=value.get('C50_common'),
                       Iend_vs_Uniform_pct=q,OC_saving_vs_Uniform_pct=s,
                       strict_double_win=bool(paired and q>1 and s>=10),
                       common_quality_first_step=target_step,
                       attempted_FE=value.get('cost',{}).get('attempted_FE'),
                       switch_FE=value.get('cost',{}).get('switch_FE'),
                       repair_max_change=(value.get('transition') or {}).get('volume_repair',{}).get('max_design_change'),
                       elapsed_seconds=value.get('elapsed_seconds'),error=value.get('error'))
            results.append(row)
            per[tid,p['id']] = row
    dest = out/'reports'
    result_fields=['task_id','vf_bin','profile','launch_penal','launch_rmin','launch_move','status','stop_step','Cend','C50_launch','C50_common','Iend_vs_Uniform_pct','OC_saving_vs_Uniform_pct','strict_double_win','common_quality_first_step','attempted_FE','switch_FE','repair_max_change','elapsed_seconds','error']
    write_csv(dest/'results.csv',results,result_fields)
    summaries=[]
    for p in cfg['profiles']:
        rows=[r for r in results if r['profile']==p['id']]
        valid=[r for r in rows if r['Iend_vs_Uniform_pct'] is not None]
        summaries.append(dict(profile=p['id'],penal=p['penal'],rmin=p['rmin'],move=p['move'],
            planned=len(records),settled=len(rows),converged=sum(r['status']=='converged' for r in rows),
            max_steps=sum(r['status']=='max_steps' for r in rows),failed=sum(r['status']=='failed' for r in rows),
            paired_with_Uniform=len(valid),mean_Iend_pct=mean([r['Iend_vs_Uniform_pct'] for r in valid]),
            median_Iend_pct=median([r['Iend_vs_Uniform_pct'] for r in valid]),
            mean_OC_saving_pct=mean([r['OC_saving_vs_Uniform_pct'] for r in valid]),
            median_OC_saving_pct=median([r['OC_saving_vs_Uniform_pct'] for r in valid]),
            gain_gt1pct=sum(r['Iend_vs_Uniform_pct']>1 for r in valid),
            degradation_gt1pct=sum(r['Iend_vs_Uniform_pct']< -1 for r in valid),
            strict_double_win_count=sum(r['strict_double_win'] for r in rows),
            strict_double_win_pct_all_tasks=100*sum(r['strict_double_win'] for r in rows)/len(records),
            reached_common_quality=sum(r['common_quality_first_step'] is not None for r in rows),
            attempted_FE=sum(r['attempted_FE'] or 0 for r in rows)))
    write_csv(dest/'parameter_summary.csv',summaries)
    historical=[]
    for method in sorted({r['method'] for r in old}):
        rr=[r for r in old if r['method']==method]
        valid=[r for r in rr if r['status']=='converged' and uniform[r['task_id']]['status']=='converged']
        qq=[100*(1-float(r['Cend'])/float(uniform[r['task_id']]['Cend'])) for r in valid]
        ss=[100*(1-int(r['stop_step'])/int(uniform[r['task_id']]['stop_step'])) for r in valid]
        historical.append(dict(method=method,planned=len(records),settled=len(rr),
            converged=sum(r['status']=='converged' for r in rr),paired_with_Uniform=len(valid),
            mean_Iend_pct=mean(qq),median_Iend_pct=median(qq),median_OC_saving_pct=median(ss),
            strict_double_win_count=sum(q>1 and s>=10 for q,s in zip(qq,ss)),
            strict_double_win_pct_all_tasks=100*sum(q>1 and s>=10 for q,s in zip(qq,ss))/len(records),
            cost_note='OC trajectory steps only; old initialization/search FE and API costs are not reconstructed'))
    write_csv(dest/'initialization_summary.csv',historical)
    old_per={(r['task_id'],r['method']):r for r in old}
    for p in cfg['profiles']:
        for method in sorted({r['method'] for r in old}):
            qvalues,svalues=[],[]
            for tid in records:
                a=per.get((tid,p['id']))
                b=old_per[tid,method]
                if a and a['status']=='converged' and b['status']=='converged':
                    qvalues.append(100*(1-a['Cend']/float(b['Cend'])))
                    svalues.append(100*(1-a['stop_step']/int(b['stop_step'])))
            pairs.append(dict(profile=p['id'],initializer=method,planned=len(records),
                paired_converged=len(qvalues),mean_quality_gain_pct=mean(qvalues),
                median_quality_gain_pct=median(qvalues),median_OC_saving_pct=median(svalues),
                strict_double_win_count=sum(q>1 and s>=10 for q,s in zip(qvalues,svalues)),
                strict_double_win_pct_all_tasks=100*sum(q>1 and s>=10 for q,s in zip(qvalues,svalues))/len(records)))
    write_csv(dest/'parameter_vs_initialization.csv',pairs)
    progress=dict(status=status,planned=len(records)*len(cfg['profiles']),settled=len(results),
        converged=sum(r['status']=='converged' for r in results),
        max_steps=sum(r['status']=='max_steps' for r in results),failed=sum(r['status']=='failed' for r in results),
        tasks=len(records),profiles=len(cfg['profiles']),updated_unix=time.time(),new_API_calls=0)
    dump(out/'progress.json',progress)
    def fmt(value):return '—' if value is None else f'{value:+.3f}'
    lines=['# Uniform启动参数60组：原Train100配对实验','',str(progress),'',
        '前50步为固定启动参数，之后共同p=3、rmin=2.5、move=0.2至收敛；总上限2000步。',
        'Iend>0表示柔度更低；步数节省包含前50步。历史初始化额外FE/API成本没有混入OC步数。',
        '均值仅使用双方已收敛配对，必须同时看配对数；不同配对集合不作直接总体排名。',
        '严格双赢：最终柔度改善>1%且总OC步数减少≥10%；分母为全部冻结任务，未完成/失败/未收敛不计成功。',
        '共同质量首次到达只统计切换后的共同模型，需连续5个观测状态达标；切换前是否曾达到该目标未知。','',
        '|配置|p₀|r₀|move₀|已结算|收敛配对|平均最终改善%|中位步数节省%|严格双赢/全部任务|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"|{s['profile']}|{s['penal']}|{s['rmin']}|{s['move']}|{s['settled']}/{s['planned']}|{s['paired_with_Uniform']}|{fmt(s['mean_Iend_pct'])}|{fmt(s['median_OC_saving_pct'])}|{s['strict_double_win_count']}/{s['planned']}|")
    lines+=['','- `results.csv`：每任务、每配置的结果及实际FE。',
            '- `parameter_summary.csv`：60组汇总。',
            '- `initialization_summary.csv`：原38种初始化在相同任务上的汇总。',
            '- `parameter_vs_initialization.csv`：每个参数配置分别与38种初始化配对。','',
            '这100条是已有开发任务；逐任务事后挑选60组最佳不代表可部署选参器或未见测试效果。',
            'C50_launch属于各自启动模型，不能跨p/rmin直接排名；C50_common已经切换并校准，单列解释。']
    temp=dest/'.REPORT_zh.md.tmp'
    temp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    os.replace(temp,dest/'REPORT_zh.md')
    return progress
