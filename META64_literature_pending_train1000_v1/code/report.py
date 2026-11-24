"""Task-paired quality/speed metrics; random replicates are never best-of-three."""
import csv
import os
import statistics
import time
from pathlib import Path
from experiment import read
from storage import dump


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.'+path.name+'.tmp')
    with temp.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    os.replace(temp, path)


def report(out, cfg, status='idle'):
    out = Path(out); rows = []; active = []
    for rec in cfg['records']:
        tid = rec['task_id']; values = {}
        for method in cfg['methods']:
            d = out/'tasks'/tid/method
            final = read(d/'result.json') if (d/'result.json').exists() else {}
            fifty = read(d/'result_50.json') if (d/'result_50.json').exists() else {}
            failed = read(d/'failure.json') if (d/'failure.json').exists() else {}
            values[method] = dict(fifty, **final) if final else dict(fifty)
            if failed:
                values[method].update(failed)
            if not final and not failed and (d/'progress.json').exists():
                p = read(d/'progress.json')
                # A recent update is an activity hint, not proof that a PID is still alive.
                if time.time()-p['updated_unix'] < 120:
                    active.append(p)
        u = values['U']
        for method, v in values.items():
            paired = v.get('status') == u.get('status') == 'converged'
            gain = 100*(1-v['Cend']/u['Cend']) if paired else None
            saving = 100*(1-v['step']/u['step']) if paired else None
            gain50 = 100*(1-v['C50']/u['C50']) if v.get('C50') and u.get('C50') else None
            rows.append(dict(task_id=tid, vf_bin=rec['vf_bin'], method=method,
                status=v.get('status','pending'), C0=v.get('C0'), C50=v.get('C50'),
                Cend=v.get('Cend'), stop_step=v.get('step') if v.get('Cend') else None,
                I50=gain50, Iend=gain, step_saving_pct=saving,
                double_win=bool(gain > 1 and saving > 0) if paired else None,
                faster_similar=bool(abs(gain) <= 1 and saving > 0) if paired else None,
                faster_worse=bool(gain < -1 and saving > 0) if paired else None,
                slower_better1=bool(gain > 1 and saving < 0) if paired else None,
                slower_better5=bool(gain > 5 and saving < 0) if paired else None,
                attempted_FE=v.get('cost',{}).get('attempted_FE'), error=v.get('error')))
    mean = lambda xs: statistics.mean(xs) if xs else None
    median = lambda xs: statistics.median(xs) if xs else None
    common = {r['task_id'] for r in rows if r['method']=='U' and r['status']=='converged'}
    for method in cfg['methods']:
        common &= {r['task_id'] for r in rows if r['method']==method and r['status']=='converged'}
    summary = []
    for method in cfg['methods']:
        rr = [r for r in rows if r['method']==method]
        pp = [r for r in rr if r['Iend'] is not None]
        summary.append(dict(method=method, complete50=sum(r['C50'] is not None for r in rr),
            converged=sum(r['status']=='converged' for r in rr),
            max_steps=sum(r['status']=='max_steps' for r in rr),
            failed=sum(r['status']=='failed' for r in rr), paired_n=len(pp),
            mean_I50=mean([r['I50'] for r in rr if r['I50'] is not None]),
            mean_Iend=mean([r['Iend'] for r in pp]),
            median_stop_step=median([r['stop_step'] for r in pp]),
            median_step_saving_pct=median([r['step_saving_pct'] for r in pp]),
            double_wins=sum(r['double_win'] for r in pp),
            double_win_pct=100*sum(r['double_win'] for r in pp)/len(pp) if pp else None,
            faster_similar=sum(r['faster_similar'] for r in pp),
            faster_worse=sum(r['faster_worse'] for r in pp),
            slower_better1=sum(r['slower_better1'] for r in pp),
            slower_better5=sum(r['slower_better5'] for r in pp), common_n=len(common),
            common_mean_Iend=mean([r['Iend'] for r in pp if r['task_id'] in common])))
    # One row per physical task, average all 3 repeats only when all are available.
    random_rows = []
    for rec in cfg['records']:
        rr = [r for r in rows if r['task_id']==rec['task_id'] and r['method'].startswith('RAND_')]
        if len(rr) != 3:
            continue
        full50 = all(r['I50'] is not None for r in rr)
        fullend = all(r['Iend'] is not None for r in rr)
        random_rows.append(dict(task_id=rec['task_id'], vf_bin=rec['vf_bin'],
            repeats_with_C50=sum(r['I50'] is not None for r in rr),
            converged_paired_repeats=sum(r['Iend'] is not None for r in rr),
            mean_I50_over3=mean([r['I50'] for r in rr]) if full50 else None,
            mean_Iend_over3=mean([r['Iend'] for r in rr]) if fullend else None,
            double_win_probability_over3=mean([int(r['double_win']) for r in rr]) if fullend else None))
    progress = dict(status=status, planned=len(rows), tasks=len(cfg['records']),
        complete50=sum(r['C50'] is not None for r in rows),
        converged=sum(r['status']=='converged' for r in rows),
        max_steps=sum(r['status']=='max_steps' for r in rows),
        failed=sum(r['status']=='failed' for r in rows),
        validation_only=cfg['validation_only'], new_API_calls=0,
        common_converged_tasks=len(common), recent_progress=active[:32], updated_unix=time.time())
    dest = out/'reports'; dest.mkdir(exist_ok=True)
    write_csv(dest/'results.csv', rows); write_csv(dest/'method_summary.csv', summary)
    write_csv(dest/'random_replicates_task_mean.csv', random_rows)
    f = lambda x: '—' if x is None else f'{x:.4f}'
    lines = ['# META64：有文献类别来源的新增初始化', '',
        f"状态：{status}；50步 {progress['complete50']}/{len(rows)}；收敛 {progress['converged']}；上限 {progress['max_steps']}；失败 {progress['failed']}。", '',
        '中心孔和随机密度是Alvarez2019类别的本地适配，不是其周期胞元实验复现。',
        'I=100×(1−C方法/C同任务Uniform)，正数更好；严格最终配对仅计双方均收敛。',
        '失败与到上限不填Uniform；不同方法配对范围可能不同，共同收敛组另列CSV。',
        '停止步数是累计OC更新次数，包含前50步；不等同于墙钟加速。', '',
        '|方法|配对数|平均I50%|平均最终改善%|中位停止步|更快且改善>1%|更快且接近±1%|更快但变差>1%|更慢但改善>1%|更慢但改善>5%|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['method']}|{r['paired_n']}|{f(r['mean_I50'])}|{f(r['mean_Iend'])}|{f(r['median_stop_step'])}|{r['double_wins']}|{r['faster_similar']}|{r['faster_worse']}|{r['slower_better1']}|{r['slower_better5']}|")
    paired_random = [r for r in random_rows if r['mean_Iend_over3'] is not None]
    lines += ['', '## 随机方法：任务内三次重复平均', '',
        f"三次均可最终配对的任务数：{len(paired_random)}；任务均值再平均的最终改善：{f(mean([r['mean_Iend_over3'] for r in paired_random]))}%。",
        '双赢概率按每任务三个重复的事件比例计算，不取三次中最好者，也不将重复当成三个独立物理任务。',
        '参见random_replicates_task_mean.csv；部分结果有完成速度与成功条件筛选偏差，不能作为完整1000任务结论。', '']
    text = '\n'.join(lines); temp = dest/'.REPORT_zh.md.tmp'
    temp.write_text(text, encoding='utf-8'); os.replace(temp, dest/'REPORT_zh.md')
    dump(out/'progress.json', progress)
    return progress
