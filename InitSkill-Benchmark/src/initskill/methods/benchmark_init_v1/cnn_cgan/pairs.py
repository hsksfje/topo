"""Same-task multi-resolution labels from Train/Valid; never train on CR test."""
import json
from initskill import locking as fcntl
from pathlib import Path
import numpy as np
from ..common import rows,load,solver,dump,sha,trajectory,atomic_savez,fe_backend,MAX_OC_STEPS


def lift(row,fields,high=256):
    f,b,v=fields;old=row['nx']
    if high%old:raise ValueError('Exact lattice refinement required')
    factor=high//old;force=np.zeros((high+1,high+1,2));bc=np.zeros_like(force)
    force[::factor,::factor]=f;bc[::factor,::factor]=b
    new=dict(row,nx=high,ny=high)
    return new,(force,bc,v)


def build_one(job):
    out,subset,row,steps=job;folder=Path(out)/'pairs'/subset;folder.mkdir(parents=True,exist_ok=True)
    # Background preparation and the campaign can join the same work safely.
    with (folder/(row['source_task_id']+'.lock')).open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        from ..resources import high_resolution_slot
        with high_resolution_slot(out,256,256):
            return _build_one(job)


def _build_one(job):
    out,subset,row,steps=job;out=Path(out)
    folder=out/'pairs'/subset;folder.mkdir(parents=True,exist_ok=True)
    p=folder/(row['source_task_id']+'.npz')
    if p.exists() and p.with_suffix('.json').exists():
        if sha(p)!=json.loads(p.with_suffix('.json').read_text())['sha256']:raise ValueError('Changed pair')
        return row['task_key']
    lowp=out/'labels'/subset/(row['source_task_id']+'.npz')
    if not lowp.exists():raise FileNotFoundError('Need common 64-grid teacher first')
    new,fields=lift(row,load(row));m=solver(out,'paired256_'+subset,new,fields)
    runfolder=folder/'high_runs'/row['source_task_id']
    from ..convergence import run_to_convergence
    result=run_to_convergence(m,m.initial(),steps,runfolder,
        identity=dict(task_key=row['task_key'],group_id=row['group_id'],source_split=subset,
                      max_steps=steps,low_label_sha256=sha(lowp),teacher='uniform256-'+fe_backend()+'-common-OC'))
    with np.load(runfolder/'state_final.npz',allow_pickle=False) as a:high=a['x'].reshape((256,256),order='F')
    with np.load(lowp,allow_pickle=False) as a:low=a['x'].reshape((64,64),order='F')
    atomic_savez(p,low=low,high=high)
    dump(p.with_suffix('.json'),dict(task_key=row['task_key'],group_id=row['group_id'],source_split=subset,
        sha256=sha(p),low_label_sha256=sha(lowp),high_teacher=result,
        mapping='same physical nodes injected into exact refined lattice, fixed physical filter radius'))
    return row['task_key']


def build(out,subset,steps=MAX_OC_STEPS,workers=6):
    if subset not in {'train1000','valid100'}:raise ValueError('Pairs only from Train/Valid')
    from concurrent.futures import ProcessPoolExecutor
    jobs=[(str(out),subset,r,steps) for r in rows(subset)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,key in enumerate(pool.map(build_one,jobs)):
            print('paired-label',subset,i+1,key,flush=True)
