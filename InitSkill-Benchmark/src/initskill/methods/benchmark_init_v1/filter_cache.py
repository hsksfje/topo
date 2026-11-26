"""Read-only shared CSR filter; exact original coefficients, no task outcomes."""
from initskill import locking as fcntl
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import uuid
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from .common import dump,sha


def cached_filter(directory,centers,radius):
    centers=np.ascontiguousarray(centers)
    key=hashlib.sha256(centers.tobytes()+centers.dtype.str.encode()+repr(centers.shape).encode()+float(radius).hex().encode()).hexdigest()
    folder=Path(directory)/key;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'build.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        if not (folder/'ready.json').exists():
            start=time.perf_counter();n=len(centers);tree=cKDTree(centers)
            pairs=tree.sparse_distance_matrix(tree,radius,output_type='coo_matrix')
            H=sparse.coo_matrix((np.maximum(0.,radius-pairs.data),(pairs.row,pairs.col)),shape=(n,n)).tocsr()
            Hs=np.asarray(H.sum(axis=1)).ravel()
            dv=np.asarray(H.T@(np.ones(n)/Hs)).ravel()
            arrays=dict(data=H.data,indices=H.indices,indptr=H.indptr,Hs=Hs,dv=dv)
            for name,value in arrays.items():
                temporary=folder/(name+'.'+uuid.uuid4().hex+'.tmp')
                with temporary.open('wb') as stream:np.save(stream,value,allow_pickle=False)
                fcntl.atomic_replace(temporary,folder/(name+'.npy'))
            dump(folder/'ready.json',dict(shape=[n,n],radius=float(radius),geometry_key=key,
                arrays_sha256={name:sha(folder/(name+'.npy')) for name in arrays},
                build_seconds=time.perf_counter()-start,task_outcomes_used=False))
    return _read(str(folder))


@lru_cache(maxsize=4)
def _read(directory):
    folder=Path(directory);meta=json.loads((folder/'ready.json').read_text());arrays={}
    for name,digest in meta['arrays_sha256'].items():
        p=folder/(name+'.npy')
        if sha(p)!=digest:raise ValueError('Changed deterministic filter cache')
        arrays[name]=np.load(p,mmap_mode='r',allow_pickle=False)
    H=sparse.csr_matrix((arrays['data'],arrays['indices'],arrays['indptr']),shape=tuple(meta['shape']),copy=False)
    if not np.shares_memory(H.data,arrays['data']):raise RuntimeError('Filter CSR unexpectedly copied mmap data')
    return H,arrays['Hs'],arrays['dv']
