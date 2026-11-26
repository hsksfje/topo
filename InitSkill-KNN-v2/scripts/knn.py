"""Frozen, explicit KNN variants; training inputs alone construct the bank."""
from pathlib import Path
import json
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from initskill.methods.benchmark_init_v1 import common

CONFIG=dict(name='knn_v2',k=3,descriptor_grid=17,gaussian_sigma=1.25,
    load_weight=1.,support_weight=2.,vf_weight=2.,uniform_mix=.1,
    parameter_policy='Fixed engineering variant after v1 test audit; Valid100 is a full validation audit, not hyperparameter search. Not a blind confirmatory benchmark.')


def descriptor(row):
    # Original normalized-coordinate raster with resultant-preserving force
    # splatting. Smooth in physical normalized coordinates, then normalize
    # whole blocks rather than amplify individual rare pixel coordinates.
    z=common.features(row);channels=z[:-1].reshape(4,17,17)
    smooth=np.stack([gaussian_filter(a,CONFIG['gaussian_sigma'],mode='constant') for a in channels])
    load=smooth[:2].ravel();load/=max(np.linalg.norm(load),1e-30)
    support=smooth[2:].copy()
    for a in support:a/=max(np.linalg.norm(a),1e-30)
    return load,support.ravel(),float(z[-1])


def build_bank(out):
    out=Path(out);folder=out/'models/knn';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'bank.npz';meta=folder/'frozen.json'
    if path.exists():
        r=json.loads(meta.read_text())
        if r['sha256']!=common.sha(path):raise ValueError('KNN bank changed')
        return
    rows=common.rows('train1000');designs=[];desc=[];legacy=[]
    for index,row in enumerate(rows,1):
        p=common.ROOT/'data/labels/train1000'/(row['source_task_id']+'.npz');identity=json.loads(p.with_suffix('.json').read_text())
        if identity['task_key']!=row['task_key'] or identity['group_id']!=row['group_id'] or common.sha(p)!=identity['sha256']:raise ValueError('Training label identity changed')
        with np.load(p,allow_pickle=False) as z:designs.append(z['x'].copy())
        desc.append(descriptor(row));legacy.append(common.features(row))
        if index%100==0:print('KNN training bank',index,1000,flush=True)
    z=np.stack(legacy);mean=z.mean(0);scale=z.std(0);scale[scale<1e-8]=1
    block=np.ones(z.shape[1]);block[:-1]/=17
    vf=np.array([d[2] for d in desc]);span=max(float(np.quantile(vf,.95)-np.quantile(vf,.05)),1e-6)
    common.atomic_savez(path,x=np.stack(designs),load=np.stack([d[0] for d in desc]),support=np.stack([d[1] for d in desc]),vf=vf,
        vf_span=np.array(span),keys=np.array([r['task_key'] for r in rows]),legacy=(z-mean)/scale*block,mean=mean,scale=scale,block=block)
    common.dump(meta,dict(sha256=common.sha(path),training_count=1000,training_manifest_sha256=common.sha(common.DATA/'manifests/train1000.jsonl'),config=CONFIG,extra_FE=0))


def distances(bank,load,support,vf):
    plus=np.sum((bank['load']-load)**2,axis=1);minus=np.sum((bank['load']+load)**2,axis=1)
    components=np.stack([np.minimum(plus,minus),np.sum((bank['support']-support)**2,axis=1),((bank['vf']-vf)/float(bank['vf_span']))**2],axis=1)
    return components@np.array([CONFIG['load_weight'],CONFIG['support_weight'],CONFIG['vf_weight']]),components


def prediction(bank,row,method):
    if method=='uniform':return None,dict(initialization_FE=0)
    if method=='knn_legacy':
        d=np.linalg.norm(bank['legacy']-(common.features(row)-bank['mean'])/bank['scale']*bank['block'],axis=1)
        ids=np.argsort(d,kind='stable')[:1];weights=np.ones(1);components=None
    else:
        load,support,vf=descriptor(row);d,components=distances(bank,load,support,vf);ids=np.argsort(d,kind='stable')[:CONFIG['k']]
        if d[ids[0]]<=1e-24:ids=ids[:1];weights=np.ones(1)
        else:weights=1/np.maximum(np.sqrt(d[ids]),1e-12);weights/=weights.sum()
    raw=weights@bank['x'][ids]
    meta=dict(initialization_FE=0,neighbors=bank['keys'][ids].tolist(),distances=d[ids].tolist(),weights=weights.tolist(),neighbor_vf=bank['vf'][ids].tolist(),distance_is_squared=method=='knn_v2')
    if components is not None:meta['distance_components_load_support_vf']=components[ids].tolist()
    return raw,meta


def resize_design(raw,nx,ny):
    if raw.size==nx*ny:return raw.copy()
    from scipy.ndimage import map_coordinates
    yy=(np.arange(ny)+.5)*64/ny-.5;xx=(np.arange(nx)+.5)*64/nx-.5
    coords=np.meshgrid(yy,xx,indexing='ij')
    return map_coordinates(raw.reshape(64,64,order='F'),coords,order=1,mode='nearest',prefilter=False).ravel(order='F')
