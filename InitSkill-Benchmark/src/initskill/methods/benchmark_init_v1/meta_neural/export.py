"""Export native JAX densities; common OC later runs in the single platform env."""
import argparse
from initskill import locking as fcntl
import json
import os
from pathlib import Path
import time
import numpy as np

from ..common import DEFAULT_OUT,rows,dump,sha,atomic_savez
from . import core


def export_one(out,method,row):
    out=Path(out);model_out=out if method=='meta' else out/'early_trajectories'
    frozen=model_out/'models/meta_native/frozen.json';model= json.loads(frozen.read_text())
    folder=out/'native_predictions'/method/row['task_key'];p=folder/'density.npz';sidecar=folder/'metadata.json'
    inputs={k:row['assets'][k]['sha256'] for k in ('forces','normals','target_vf')}
    if p.exists() and sidecar.exists():
        saved=json.loads(sidecar.read_text())
        if (saved['prediction_sha256']!=sha(p) or saved['physical_inputs']!=inputs or
            saved['model_sha256']!=model['model_sha256']):raise ValueError('Native export identity changed')
        return
    started=time.perf_counter();raw,metadata=core.predict(model_out,row)
    folder.mkdir(parents=True,exist_ok=True);atomic_savez(p,raw=raw)
    metadata.update(task_key=row['task_key'],group_id=row['group_id'],physical_inputs=inputs,
        prediction_sha256=sha(p),external_initialization_seconds=time.perf_counter()-started,
        export_environment='isolated author JAX runtime; common OC uses platform Python/SciPy runtime',
        method=method)
    dump(sidecar,metadata)


def export_all(out,method):
    # Native-training and common-OC controllers share these aliases. Join
    # their exports before loading JAX or charging another reference solve.
    folder=Path(out)/'native_predictions'/method;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.export.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        for subset in ('test_id300','test_ood300','test_cr300'):
            for i,row in enumerate(rows(subset)):
                export_one(out,method,row)
                if i%10==0:print('native-export',method,subset,i+1,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--method',choices=['meta','early'],default='meta');a=p.parse_args()
    # Author runtime is lazy: set this before core.backend imports JAX. CUDA
    # may be installed for training; public benchmark prediction stays CPU.
    os.environ['JAX_PLATFORMS']='cpu'
    export_all(a.out,a.method)


if __name__=='__main__':main()
