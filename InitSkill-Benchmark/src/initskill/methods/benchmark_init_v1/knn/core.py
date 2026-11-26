import json
from pathlib import Path
import numpy as np
from ..common import DATA,dump,sha,features
from ..supervised import arrays

def train_knn(out):
    r,z,x=arrays(out,'train1000');mean=z.mean(0);scale=z.std(0)
    scale[scale<1e-8]=1
    folder=Path(out)/'models/knn';folder.mkdir(parents=True,exist_ok=True)
    p=folder/'model.npz'
    if p.exists():
        frozen=json.loads((folder/'frozen.json').read_text())
        if frozen['model_sha256']!=sha(p) or frozen['manifest_sha256']!=sha(DATA/'manifests/train1000.jsonl'):
            raise ValueError('Frozen KNN identity changed')
        return p
    # Equal total contribution per input block; VF is its own block.
    block=np.ones(z.shape[1]);block[:-1]/=np.sqrt((z.shape[1]-1)/4)
    np.savez_compressed(p,z=(z-mean)/scale*block,x=x,mean=mean,scale=scale,block=block,
                        keys=np.array([a['task_key'] for a in r]))
    dump(folder/'frozen.json',dict(method='1NN-density retrieval adaptation',train_count=len(r),
        model_sha256=sha(p),manifest_sha256=sha(DATA/'manifests/train1000.jsonl'),k=1,
        fitted_on='train1000',extra_reference_FE=0))
    return p

def predict_knn(out,row,k=1):
    p=Path(out)/'models/knn/model.npz';meta=json.loads(p.with_name('frozen.json').read_text())
    if sha(p)!=meta['model_sha256']:raise ValueError('KNN changed')
    with np.load(p,allow_pickle=False) as z:
        distance=np.linalg.norm(z['z']-(features(row)-z['mean'])/z['scale']*z['block'],axis=1)
        ids=np.argsort(distance,kind='stable')[:k]
        if k==1:weights=np.ones(1)
        else:
            weights=1/np.maximum(distance[ids],1e-12);weights/=weights.sum()
        return weights@z['x'][ids],dict(neighbors=z['keys'][ids].tolist(),distances=distance[ids].tolist(),weights=weights.tolist())
