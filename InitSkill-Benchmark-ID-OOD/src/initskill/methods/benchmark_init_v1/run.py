"""Explicit stages; no API calls or automatic transitions into held-out tests."""
import argparse
import contextlib
from initskill import locking as fcntl
import json
from pathlib import Path
import time

import numpy as np
from .common import DEFAULT_OUT, ROOT, DATA, dump, sha, protocol, rows, solver, calibrate, trajectory


@contextlib.contextmanager
def lock(out):
    Path(out).mkdir(parents=True,exist_ok=True)
    with (Path(out)/'.run.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        yield


def evaluate(out,method,subset,steps,limit=None):
    protocol(out)
    source=rows(subset)
    if limit is not None:source=source[:limit]
    if subset.startswith('test') and limit is not None:
        raise ValueError('No test smoke samples; use train1000 for engineering smoke')
    results=[]
    for row in source:
        folder=Path(out)/'evaluation'/subset/method/row['source_task_id']
        if (folder/'result.json').exists():raise FileExistsError('Evaluation already exists; no silent reuse')
        folder.mkdir(parents=True,exist_ok=True)
        m=solver(out,subset,row);started=time.perf_counter();meta={}
        try:
            if method=='uniform':raw=m.initial()
            elif method=='knn':
                from .knn.core import predict_knn
                raw,meta=predict_knn(out,row)
            elif method in {'ulu','wang'}:
                from .pca_python import predict
                raw,meta=predict(out,row,method)
            elif method=='cnn':
                from .cnn_encoder_decoder.core import predict_cnn
                raw,meta=predict_cnn(out,row)
            elif method=='cgan':
                from .cnn_cgan.core import predict_cgan
                raw,meta=predict_cgan(out,row)
            elif method=='meta':
                from .meta_neural.adapter import predict
                raw,meta=predict(out,row)
            else:raise ValueError(method)
            if raw.size!=row['nx']*row['ny']:
                # Explicit low-resolution design transfer for KNN/PCA only.
                if method not in {'knn','ulu','wang','cnn'}:raise ValueError('Prediction grid mismatch')
                import torch
                a=torch.tensor(raw.reshape((64,64),order='F'))[None,None]
                raw=torch.nn.functional.interpolate(a,size=(row['ny'],row['nx']),mode='bilinear',align_corners=False)[0,0].numpy().ravel(order='F')
                meta['grid_transfer']='64 -> target bilinear design interpolation'
            x=calibrate(raw,m);meta.update(initialization_seconds=time.perf_counter()-started+meta.get('external_initialization_seconds',0.),
                physical_volume=float(m.physical(x).mean()),calibration_max_change=float(np.max(np.abs(x-raw))),
                initialization_FE=meta.get('initialization_FE',0),protocol_sha256=sha(Path(out)/'protocol.json'))
            np.savez_compressed(folder/'initialization.npz',raw=raw,x=x,density=m.physical(x))
            dump(folder/'initialization.json',meta)
            result=trajectory(m,x,steps,folder)
            results.append(dict(task_key=row['task_key'],status='complete',Cend=result['history'][-1]['compliance'],
                FE=result['FE']+meta['initialization_FE']))
        except (ValueError,RuntimeError,FileNotFoundError) as exc:
            failure=dict(task_key=row['task_key'],status='failed',error=str(exc),FE=m.fe_calls,
                         initialization_FE=meta.get('initialization_FE',0))
            dump(folder/'failure.json',failure);results.append(failure)
            # Absent whole-model prerequisites are configuration failures,
            # not 300 different failed task predictions.
            if isinstance(exc,FileNotFoundError):raise
        print('evaluate',method,row['task_key'],results[-1]['status'],flush=True)
    dump(Path(out)/'evaluation'/subset/method/'summary.json',dict(method=method,subset=subset,records=results,
        planned=len(source),failed=sum(r['status']=='failed' for r in results),steps=steps))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['verify','labels','fit-knn','train-pca','train-cnn','train-cgan','train-meta','smoke-meta','evaluate'])
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--subset',default='train1000')
    p.add_argument('--method',choices=['uniform','knn','ulu','wang','cnn','cgan','meta'],default='knn')
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--limit',type=int)
    p.add_argument('--reuse-archive',action='store_true')
    p.add_argument('--epochs',type=int,default=500)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--outer-steps',type=int,default=6000)
    p.add_argument('--inner-steps',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    from .scope import current_scope
    if args.stage=='train-cgan' and not current_scope()['include_cross_resolution']:raise ValueError('cGAN is deferred with cross-resolution')
    if min(args.steps,args.epochs,args.batch_size,args.outer_steps,args.inner_steps)<1:raise ValueError('Positive budgets required')
    with lock(args.out):
        if args.stage=='verify':print(json.dumps(protocol(args.out),indent=2));return
        if args.stage=='labels':
            from .supervised import build_labels
            build_labels(args.out,args.subset,args.reuse_archive,args.steps,args.limit)
        elif args.stage=='fit-knn':
            from .knn.core import train_knn
            protocol(args.out);print(train_knn(args.out))
        elif args.stage=='train-pca':
            from .pca_python import train
            if args.method not in {'ulu','wang'}:raise ValueError('Select --method ulu/wang')
            train(args.out,args.method,args.epochs,args.seed)
        elif args.stage=='train-cnn':
            from .cnn_encoder_decoder.core import train_cnn
            protocol(args.out);train_cnn(args.out,args.epochs,args.batch_size,args.seed)
        elif args.stage=='train-cgan':
            from .cnn_cgan.core import train_cgan
            protocol(args.out);train_cgan(args.out,args.epochs,args.batch_size,args.seed)
        elif args.stage in {'train-meta','smoke-meta'}:
            from .meta_neural.core import train
            smoke=args.stage=='smoke-meta'
            train(args.out,1 if smoke else args.outer_steps,1 if smoke else args.inner_steps,
                  1 if smoke else 5,args.seed,smoke=smoke)
        elif args.stage=='evaluate':evaluate(args.out,args.method,args.subset,args.steps,args.limit)


if __name__=='__main__':main()
