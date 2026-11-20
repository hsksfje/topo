#!/usr/bin/env python3
"""Validate the returned cohort and combine disjoint 50+50 result tables only."""
import argparse,csv,json,time
from pathlib import Path
from storage import dump,sha
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(Path(p).read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--local',required=True,type=Path);p.add_argument('--remote',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    local,remote,dest=a.local.resolve(),a.remote.resolve(),a.output.resolve()
    assert dest not in (local,remote),'Merged destination must be separate'
    lc,rc=read(local/'config.json'),read(remote/'config.json');partition=read(ROOT/'provenance/task_partition.json')
    for folder,cfg in ((local,lc),(remote,rc)):
        assert sha(folder/'config.json')==read(folder/'config_identity.json')['sha256'],'Config hash mismatch'
        if not a.allow_partial:
            assert read(folder/'progress.json')['status']=='complete' and (folder/'COMPLETION.json').exists(),'Both cohorts must complete before final merging'
    lids={r['task_id'] for r in lc['records']};rids={r['task_id'] for r in rc['records']}
    assert lids==set(partition['local_tasks']) and rids==set(partition['portable_tasks'])
    assert not lids&rids and len(lids|rids)==100
    for key in ('solver','convergence','max_step','methods'):assert lc[key]==rc[key],key
    verified=False
    if (remote/'RETURN_READY.json').exists():
        marker=read(remote/'RETURN_READY.json');assert sha(remote/'return_manifest.json')==marker['manifest_sha256']
        manifest=read(remote/'return_manifest.json');assert set(manifest['task_ids'])==rids
        for name,digest in manifest['files'].items():
            rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
            assert sha(remote/rel)==digest,'Returned file changed: '+name
        verified=True
    elif not a.allow_partial:raise RuntimeError('Missing RETURN_READY.json; run export_results.py on the completed remote output first')
    rows=[];keys=set()
    for folder,cfg in ((local,lc),(remote,rc)):
        ids={r['task_id'] for r in cfg['records']}
        with (folder/'reports/results.csv').open(encoding='utf-8-sig') as f:source=list(csv.DictReader(f))
        for row in source:
            key=(row['task_id'],row['method']);assert key not in keys and row['task_id'] in ids and row['method'] in cfg['methods']
            keys.add(key);row['source_output']=str(folder);rows.append(row)
    if not a.allow_partial:assert len(rows)==3800
    dest.mkdir(parents=True,exist_ok=True);(dest/'reports').mkdir(exist_ok=True)
    cfg={k:rc[k] for k in ('methods','fixed','llm','extra','solver','convergence','max_step')}
    cfg.update(name='META64_convergence100_merged_reports',records=lc['records']+rc['records'],analysis_only=True)
    dump(dest/'config.json',cfg)
    with (dest/'reports/results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(dest/'merge_audit.json',dict(local=str(local),remote=str(remote),overlap=[],tasks=100,rows=len(rows),expected=3800,remote_return_hashes_verified=verified,partial=a.allow_partial,source_table_hashes={str(folder/'reports/results.csv'):sha(folder/'reports/results.csv') for folder in (local,remote)},unix=time.time()))
    import analyze
    analyze.OUT=dest;analyze.analyze()
    (dest/'README.md').write_text('本目录仅为100任务合并分析，不是续跑检查点。原轨迹保留在merge_audit.json的两个源输出目录。报告见reports/SPEED_QUALITY_zh.md。\n')
    print('MERGED',len(rows),'records;',dest/'reports/SPEED_QUALITY_zh.md',flush=True)
if __name__=='__main__':main()
