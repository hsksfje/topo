"""Paired initialization effects UNDER THE SAME launch strategy; no FE/API."""
import csv,os,statistics,time
from pathlib import Path
from experiment import read
from storage import dump


def mean(a):return statistics.mean(a) if a else None
def median(a):return statistics.median(a) if a else None

def write_csv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name('.'+path.name+'.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    os.replace(temp,path)


def report(out,cfg,status='running'):
    out=Path(out);n=len(cfg['records']);ids=[r['task_id'] for r in cfg['records']]
    cells={}
    for r in cfg['references']:
        assert (r['task_id'],r['initializer'],r['strategy']) not in cells
        cells[r['task_id'],r['initializer'],r['strategy']]=dict(r,source='historical_reference',new_FE=0)
    new_results=[]
    for profile in cfg['profiles']:
        for tid in ids:
            key=tid,profile['initializer'],profile['strategy']
            assert key not in cells,'New trajectory overlaps a reused reference'
            path=out/'tasks'/tid/profile['id']/'result.json'
            if path.exists():
                value=read(path)
                assert value['initializer']==profile['initializer'] and value['strategy']==profile['strategy']
                row=dict(task_id=tid,initializer=profile['initializer'],strategy=profile['strategy'],
                    status=value['status'],N=value.get('step'),Cend=value.get('Cend'),
                    C50_common=value.get('C50_common'),source='new_trajectory',
                    new_FE=value.get('cost',{}).get('attempted_FE',0),error=value.get('error'),
                    initial_repair_max_change=(value.get('initialization') or {}).get('volume_repair',{}).get('max_design_change'),
                    switch_repair_max_change=(value.get('transition') or {}).get('volume_repair',{}).get('max_design_change'))
                new_results.append(row)
            else:
                row=dict(task_id=tid,initializer=profile['initializer'],strategy=profile['strategy'],
                    status='pending',N=None,Cend=None,C50_common=None,source='new_trajectory',new_FE=0)
            cells[key]=row
    rows=[]
    for key,row in cells.items():
        tid,initializer,strategy=key;u=cells[tid,'U',strategy]
        paired=row['status']=='converged' and u['status']=='converged'
        q=100*(1-row['Cend']/u['Cend']) if paired else None
        speed=100*(1-row['N']/u['N']) if paired else None
        faster=paired and row['N']<u['N'];slower=paired and row['N']>u['N']
        rows.append(dict(task_id=tid,initializer=initializer,strategy=strategy,status=row['status'],
            source=row['source'],N=row.get('N'),Cend=row.get('Cend'),C50_common=row.get('C50_common'),
            same_strategy_Uniform_N=u['N'],same_strategy_Uniform_Cend=u['Cend'],
            I_same_strategy_U=q,OC_saving_same_strategy_U=speed,
            faster_better1=bool(faster and q>1),faster_similar1=bool(faster and abs(q)<=1),
            faster_worse1=bool(faster and q< -1),slower_better1=bool(slower and q>1),
            strict_double_win=bool(paired and q>1 and speed>=10),new_FE=row['new_FE'],
            initial_repair_max_change=row.get('initial_repair_max_change'),
            switch_repair_max_change=row.get('switch_repair_max_change'),error=row.get('error')))
    summary=[]
    for initializer in cfg['initializers']:
        for strategy in cfg['strategies']:
            selected=[r for r in rows if r['initializer']==initializer and r['strategy']==strategy['id']]
            if not selected:continue
            paired=[r for r in selected if r['I_same_strategy_U'] is not None]
            q=[r['I_same_strategy_U'] for r in paired];s=[r['OC_saving_same_strategy_U'] for r in paired]
            summary.append(dict(initializer=initializer,strategy=strategy['id'],planned=len(selected),
                settled=sum(r['status']!='pending' for r in selected),paired_converged=len(paired),
                mean_I=mean(q),median_I=median(q),worst_I=min(q) if q else None,
                median_N=median([r['N'] for r in paired]),median_OC_saving=median(s),
                gain_gt1=sum(v>1 for v in q),degradation_gt1=sum(v< -1 for v in q),
                faster_better1=sum(r['faster_better1'] for r in paired),
                faster_similar1=sum(r['faster_similar1'] for r in paired),
                faster_worse1=sum(r['faster_worse1'] for r in paired),
                slower_better1=sum(r['slower_better1'] for r in paired),
                strict_double_win=sum(r['strict_double_win'] for r in paired),
                max_steps=sum(r['status']=='max_steps' for r in selected),failed=sum(r['status']=='failed' for r in selected),
                new_FE=sum(r['new_FE'] for r in selected)))
    interaction=[]
    for initializer in ('PG1','L1_PG'):
        for strategy in cfg['strategies']:
            pid=strategy['id']
            if pid==cfg['default_strategy']:continue
            valid=[]
            for tid in ids:
                keys=[(tid,initializer,pid),(tid,'U',pid),(tid,initializer,'P34'),(tid,'U','P34')]
                if any(k not in cells or cells[k]['status']!='converged' for k in keys):continue
                a,u,d,ud=[cells[k] for k in keys]
                qp=100*(1-a['Cend']/u['Cend']);qd=100*(1-d['Cend']/ud['Cend'])
                valid.append((qp,qd))
            interaction.append(dict(initializer=initializer,strategy=pid,paired_converged=len(valid),
                mean_gain_under_strategy=mean([a for a,b in valid]),
                mean_gain_under_default_on_same_tasks=mean([b for a,b in valid]),
                mean_gain_change_pp=mean([a-b for a,b in valid]),
                important='Difference in initialization gains;not a direct compliance ratio. Same task pairs only.'))
    dest=out/'reports'
    write_csv(dest/'paired_task_results.csv',rows)
    write_csv(dest/'same_strategy_summary.csv',summary)
    write_csv(dest/'initialization_strategy_interaction.csv',interaction)
    progress=dict(status=status,tasks=n,new_planned=n*len(cfg['profiles']),new_settled=len(new_results),
        new_converged=sum(r['status']=='converged' for r in new_results),
        new_max_steps=sum(r['status']=='max_steps' for r in new_results),
        new_failed=sum(r['status']=='failed' for r in new_results),
        references=len(cfg['references']),comparison_planned=len(cells),
        # Runner and return schema use these conventional fields for NEW outcomes only.
        planned=n*len(cfg['profiles']),settled=len(new_results),
        converged=sum(r['status']=='converged' for r in new_results),
        max_steps=sum(r['status']=='max_steps' for r in new_results),failed=sum(r['status']=='failed' for r in new_results),
        new_API_calls=0,updated_unix=time.time())
    dump(out/'progress.json',progress)
    def fmt(v):return '—' if v is None else f'{v:+.3f}'
    def step(v):return '—' if v is None else f'{v:g}'
    lines=['# 初始化×启动参数：原Train20交叉实验','',str(progress),'',
        '**每行都与同任务、同策略的Uniform比较。** PG1+P39只与U+P39比较，不与U+P34混比。',
        'Uniform四策略和PG1/L1_PG默认策略为历史参考，其余为新增轨迹。参考复用不代表其原始生成或求解成本为零。','',
        '|初态|启动策略|收敛配对/任务数|平均初始化收益%|中位收益%|中位停止步|更快且改善>1%|更快且接近±1%|更快但变差>1%|最终退化>1%|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"|{s['initializer']}|{s['strategy']}|{s['paired_converged']}/{n}|{fmt(s['mean_I'])}|{fmt(s['median_I'])}|{step(s['median_N'])}|{s['faster_better1']}|{s['faster_similar1']}|{s['faster_worse1']}|{s['degradation_gt1']}|")
    lines+=['','## 参数改变后，初始化收益是否保持？','',
        '|初态|策略|共同收敛配对|该策略下平均收益%|同一配对默认策略收益%|收益变化（百分点）|',
        '|---|---|---:|---:|---:|---:|']
    for s in interaction:
        lines.append(f"|{s['initializer']}|{s['strategy']}|{s['paired_converged']}|{fmt(s['mean_gain_under_strategy'])}|{fmt(s['mean_gain_under_default_on_same_tasks'])}|{fmt(s['mean_gain_change_pp'])}|")
    lines+=['','正收益表示初态在该策略下仍优于Uniform；变化为负表示相对默认策略的初始化收益减弱。均值不等于每个任务都获益，需同时看明显退化与有效配对数。',
        '每条新增轨迹：冻结初态→按启动滤波进行必要体积修正→前50步指定策略→共同p3/rmin2.5/move0.2→收敛或总2000步。两次体积修正、实际FE和原始场哈希均保留。',
        '主轨迹步数包含前50步；初态原LLM生成成本和历史参考成本另列，不用跨设备墙钟时间排名。',
        '这是已见20条任务上的开发诊断，不能据此声称任意参数或任意任务都优于Uniform。',
        '验证模式的70步上限不是正式收敛实验；若validation_only为true，仅用于工程检查。','',
        '- paired_task_results.csv：逐任务同策略配对。',
        '- same_strategy_summary.csv：完整12个组合汇总。',
        '- initialization_strategy_interaction.csv：初始化收益随策略的变化。']
    (dest/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return progress
