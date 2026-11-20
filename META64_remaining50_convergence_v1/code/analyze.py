#!/usr/bin/env python3
"""Read-only trajectory analysis; write speed/quality artifacts, no FE/API."""
import os
import tempfile
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR',os.path.join(tempfile.gettempdir(),'meta64-convergence-mpl'))
import argparse
import csv
import json
import statistics
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/remaining50_run01'


def load_rows():
    with (OUT/'reports/results.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        for k in ('stop_step','C50','Cend','I50','Iend_vs_Uniform_converged',
                  'convergence_step_saving_pct','step_reach_Uniform50',
                  'step_reach_Uniform_end_plus1pct','continued_FE','continuation_seconds'):
            r[k]=float(r[k]) if r.get(k) not in (None,'') else None
    return rows


def write_csv(name,rows):
    if not rows:return
    with (OUT/'reports'/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def analyze():
    cfg=json.loads((OUT/'config.json').read_text());rows=load_rows()
    data={(r['task_id'],r['method']):r for r in rows}
    common=[r['task_id'] for r in cfg['records'] if all(
        data.get((r['task_id'],m),{}).get('status')=='converged' for m in cfg['methods'])]
    mean=lambda a:statistics.mean(a) if a else None
    med=lambda a:statistics.median(a) if a else None
    summary=[]
    for m in cfg['methods']:
        rr=[r for r in rows if r['method']==m and r['Iend_vs_Uniform_converged'] is not None]
        reached=[r for r in rr if r['step_reach_Uniform_end_plus1pct'] is not None]
        co=[r for r in rr if r['task_id'] in common]
        target_savings=[]
        for r in reached:
            u=data[r['task_id'],'U'];a=u['step_reach_Uniform_end_plus1pct']
            if a and a>0:target_savings.append(100*(1-r['step_reach_Uniform_end_plus1pct']/a))
        summary.append(dict(method=m,paired_converged=len(rr),
            mean_I50=mean([r['I50'] for r in rr]),
            mean_Iend=mean([r['Iend_vs_Uniform_converged'] for r in rr]),
            median_Iend=med([r['Iend_vs_Uniform_converged'] for r in rr]),
            mean_gain_change_after50_pp=mean([r['Iend_vs_Uniform_converged']-r['I50'] for r in rr]),
            median_stop_step=med([r['stop_step'] for r in rr]),
            faster_stop_than_U=sum(r['convergence_step_saving_pct']>0 for r in rr),
            faster_stop_and_better1pct=sum(r['convergence_step_saving_pct']>0 and r['Iend_vs_Uniform_converged']>1 for r in rr),
            faster_stop_and_similar1pct=sum(r['convergence_step_saving_pct']>0 and abs(r['Iend_vs_Uniform_converged'])<=1 for r in rr),
            faster_stop_but_worse1pct=sum(r['convergence_step_saving_pct']>0 and r['Iend_vs_Uniform_converged']< -1 for r in rr),
            slower_stop_and_better1pct=sum(r['convergence_step_saving_pct']<0 and r['Iend_vs_Uniform_converged']>1 for r in rr),
            slower_stop_and_better5pct=sum(r['convergence_step_saving_pct']<0 and r['Iend_vs_Uniform_converged']>5 for r in rr),
            reached_U_end_plus1pct=len(reached),
            median_updates_to_common_quality=med([r['step_reach_Uniform_end_plus1pct'] for r in reached]),
            median_common_quality_step_saving_pct=med(target_savings),
            wins_over1pct=sum(r['Iend_vs_Uniform_converged']>1 for r in rr),
            losses_over1pct=sum(r['Iend_vs_Uniform_converged']< -1 for r in rr),
            common_n=len(common),common_mean_Iend=mean([r['Iend_vs_Uniform_converged'] for r in co]),
            common_median_stop_step=med([r['stop_step'] for r in co])))
    write_csv('speed_quality_analysis.csv',summary)
    # Pareto membership uses fine OC updates and final compliance only; prep costs excluded.
    pareto=[]
    for r in rows:
        if r['status']!='converged':continue
        peers=[q for q in rows if q['task_id']==r['task_id'] and q['status']=='converged']
        dominated=any(q['stop_step']<=r['stop_step'] and q['Cend']<=r['Cend']
            and (q['stop_step']<r['stop_step'] or q['Cend']<r['Cend']) for q in peers)
        pareto.append(dict(task_id=r['task_id'],method=r['method'],stop_step=r['stop_step'],
                           Cend=r['Cend'],on_fine_OC_quality_frontier=not dominated,
                           converged_candidates_available=len(peers)))
    write_csv('task_pareto.csv',pareto)
    fmt=lambda v:f'{v:+.3f}' if v is not None else '—'
    fmtstep=lambda v:f'{v:.0f}' if v is not None else '—'
    lines=['# 收敛速度与最终效果分析','',
        f'生成时间戳：{time.time():.0f}。当前全部38方法共同收敛任务数：{len(common)}/{len(cfg["records"])}。下表为各方法与Uniform已共同收敛的配对任务，配对不足全部任务数时不作总体排名。','',
        '最终改善以该方法自己的收敛柔度对比同任务Uniform的收敛柔度；50步改善也在相同配对任务上计算。达到共同质量指柔度不高于Uniform收敛柔度的1.01倍，且连续5个观测状态满足。达标步数中位数仅针对已达标任务，必须同时看达标数量。','',
        '“更快收敛且最终效果接近”指同一任务上停止步数严格少于Uniform，且最终柔度相对Uniform的差异不超过±1%（包含边界）。这里的接近是数值容差定义，不表示统计等效或拓扑图像相同；与改善>1%、变差>1%三列共同覆盖所有停止更快的已收敛配对。','',
        '“停得更慢但最终改善>5%”指同一任务上停止步数严格多于Uniform，且该方法收敛柔度比Uniform收敛柔度低超过5%；恰好改善5%或停止步数相同均不计入。仅统计双方已收敛的配对任务。','',
        '“停得更慢但最终改善>1%”使用相同的严格步数比较，要求最终柔度比Uniform低超过1%；此列包含“>5%”列的任务，两列不是互斥分类。','',
        '| 方法 | 配对数 | 平均I50% | 平均最终改善% | 中位停止步 | 达共同质量数 | 达标中位步 | 停得更快且最终改善>1% | 更快收敛且最终效果接近（差异≤1%） | 停得更快但最终变差>1% | 停得更慢但最终改善>1% | 停得更慢但最终改善>5% |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["method"]} | {r["paired_converged"]} | {fmt(r["mean_I50"])} | {fmt(r["mean_Iend"])} | {fmtstep(r["median_stop_step"])} | {r["reached_U_end_plus1pct"]} | {fmtstep(r["median_updates_to_common_quality"])} | {r["faster_stop_and_better1pct"]} | {r["faster_stop_and_similar1pct"]} | {r["faster_stop_but_worse1pct"]} | {r["slower_stop_and_better1pct"]} | {r["slower_stop_and_better5pct"]} |')
    lines+=['','解释边界：达到2000步上限仍未收敛的结果不进入以上收敛排名，见REPORT_zh.md和results.csv。最早能认证60步，50步前的密度变化未知。本报告为训练任务上的探索性统计。所有步数都是细网格轨迹OC；COARSE30/80前置粗优化、OC5前置5步、LLM生成和初态FE成本另计，所以本表及Pareto图不是端到端成本前沿。并行墙钟时间不用于公平排名。','',
            '[详细统计](speed_quality_analysis.csv) · [逐任务Pareto](task_pareto.csv) · [主报告](REPORT_zh.md)']
    if any(r['paired_converged'] for r in summary):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        eligible=[r for r in summary if r['paired_converged']==len(cfg['records'])]
        if eligible:
            fig,ax=plt.subplots(figsize=(11,7))
            for r in eligible:
                color='tab:orange' if r['method'] in cfg['llm'] else 'tab:purple' if r['method'].startswith(('COARSE','OC5')) else 'tab:blue'
                ax.scatter(r['median_stop_step'],r['mean_Iend'],color=color,s=25)
                ax.annotate(r['method'],(r['median_stop_step'],r['mean_Iend']),fontsize=7,xytext=(3,3),textcoords='offset points')
            ax.axhline(0,color='gray',lw=.8)
            ax.set(xlabel=f'Median fine OC stopping step (all {len(cfg["records"])} tasks)',ylabel='Mean final improvement vs converged Uniform (%)',title='Speed and quality | preparation costs excluded')
            fig.tight_layout();fig.savefig(OUT/'reports/speed_quality.png',dpi=180);plt.close(fig)
            lines+=['','![速度与质量](speed_quality.png)']
    (OUT/'reports/SPEED_QUALITY_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(settled=len(rows),common_converged=len(common),updated_unix=time.time())),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();globals()['OUT']=args.output.resolve()
    previous=None
    while True:
        try:
            progress=json.loads((OUT/'progress.json').read_text())
            key=(progress['settled'],progress['status'])
            if key!=previous and (OUT/'reports/results.csv').exists():
                analyze();previous=key
            if not args.watch or progress['status'] in ('complete','stopped','paused'):break
            proc=json.loads((OUT/'process.json').read_text())
            try:os.kill(proc['pid'],0)
            except ProcessLookupError:break
        except (FileNotFoundError,json.JSONDecodeError,KeyError,ValueError) as exc:
            print(type(exc).__name__,str(exc),flush=True)
            if not args.watch:raise
        time.sleep(30)


if __name__=='__main__':main()
