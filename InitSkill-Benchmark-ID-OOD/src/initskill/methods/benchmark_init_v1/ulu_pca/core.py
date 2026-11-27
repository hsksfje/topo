"""Independent ulu PCA network training and inference; Python optimizer in optimizer.py."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.linalg import svd
from ..common import dump,sha,features
from ..supervised import arrays
from ..pca_common import Dense
from .optimizer import SCG


def train(out,epochs=1000,seed=42):
    method='ulu'
    torch.set_num_threads(1);torch.manual_seed(seed)
    tr,z,x=arrays(out,'train1000');va,zv,xv=arrays(out,'valid100')
    rank=80;hidden=[80]
    mu=x.mean(0);_,_,vt=svd(x-mu,full_matrices=False);basis=vt[:rank].T
    a=(x-mu)@basis;av=(xv-mu)@basis
    zm=z.mean(0);zs=z.std(0);zs[zs<1e-8]=1
    am=a.mean(0);ast=a.std(0);ast[ast<1e-8]=1
    zt=torch.tensor((z-zm)/zs,dtype=torch.float64);at=torch.tensor((a-am)/ast,dtype=torch.float64)
    zv=torch.tensor((zv-zm)/zs,dtype=torch.float64);av=torch.tensor((av-am)/ast,dtype=torch.float64)
    model=Dense(z.shape[1],rank,hidden).double()
    names=[k for k,_ in model.named_parameters()];shapes=[p.shape for p in model.parameters()];sizes=[p.numel() for p in model.parameters()]
    initial=torch.cat([p.detach().flatten() for p in model.parameters()])
    def parameters(vector):
        return {k:t.reshape(s) for k,t,s in zip(names,vector.split(sizes),shapes)}
    def predict(vector,data):return torch.func.functional_call(model,parameters(vector),(data,))
    # Sum-of-squares residual for LM, MSE for SCG. Scaling is explicit.
    def residual(vector):return (predict(vector,zt)-at).ravel()
    def value_gradient(vector):
        q=torch.tensor(vector,dtype=torch.float64,requires_grad=True)
        loss=(predict(q,zt)-at).square().mean();g=torch.autograd.grad(loss,q)[0]
        return float(loss.detach()),g.detach().numpy()
    optimizer=SCG(value_gradient,initial.numpy())
    folder=Path(out)/'models'/method;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'frozen.json').exists():raise FileExistsError('Frozen PCA model exists')
    np.savez_compressed(folder/'pca.npz',mu=mu,basis=basis,input_mean=zm,input_scale=zs,output_mean=am,output_scale=ast)
    config=dict(method=method,optimizer='SCG' if method=='ulu' else 'LM matrix-free normal equations',rank=rank,
        hidden=hidden,epochs=epochs,seed=seed,train_keys=[r['task_key'] for r in tr],valid_keys=[r['task_key'] for r in va],
        adaptations=['variable-support physical raster input','tanh hidden / linear output explicit assumption',
                     'training-only standardization','Wang rank/width 200 extrapolated to 1000 tasks'],
        MATLAB_execution_available=False,numerical_port_verification='verification/optimizers.json')
    dump(folder/'config.json',config);best=np.inf;stale=0;history=[];started=time.perf_counter()
    for epoch in range(1,epochs+1):
        loss=optimizer.step()
        vector=torch.tensor(optimizer.x)
        with torch.no_grad():val=float((predict(vector,zv)-av).square().mean())
        if val<best:
            best=val;stale=0;torch.save(dict(state_dict=parameters(vector),config=config),folder/'best.pt')
        else:stale+=1
        row=dict(epoch=epoch,validation_MSE=val,seconds=time.perf_counter()-started,
                 training_loss=float(loss))

        history.append(row);dump(folder/'history.json',history)
        torch.save(dict(vector=vector,epoch=epoch),folder/'checkpoint.pt')
        print(method,row,flush=True)
        # Explicit MATLAB legacy training defaults (max_fail=6); the papers
        # do not publish an alternative patience value.
        if stale>=6:break
    baseline=float(av.square().mean())
    dump(folder/'training_health.json',dict(best_validation_coefficient_MSE=best,mean_training_topology_coefficient_MSE=baseline,
        improves_constant_coefficient_baseline=bool(best<baseline),policy='Validation-only diagnostic; original training and model selection unchanged'))
    dump(folder/'frozen.json',dict(model_sha256=sha(folder/'best.pt'),pca_sha256=sha(folder/'pca.npz'),
        config_sha256=sha(folder/'config.json'),best_validation_MSE=best,python_only=True,health_report_sha256=sha(folder/'training_health.json')))

def predict(out,row):
    method='ulu'
    folder=Path(out)/'models'/method;frozen=json.loads((folder/'frozen.json').read_text())
    for name,key in [('best.pt','model_sha256'),('pca.npz','pca_sha256'),('config.json','config_sha256')]:
        if sha(folder/name)!=frozen[key]:raise ValueError('Changed PCA model')
    state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);cfg=state['config']
    with np.load(folder/'pca.npz',allow_pickle=False) as p:
        z=(features(row)-p['input_mean'])/p['input_scale']
        m=Dense(len(z),cfg['rank'],cfg['hidden']).double();m.load_state_dict(state['state_dict']);m.eval()
        with torch.no_grad():a=m(torch.tensor(z,dtype=torch.float64)).numpy()
        raw=p['mu']+p['basis']@(a*p['output_scale']+p['output_mean'])
    return raw,dict(method=method,optimizer=cfg['optimizer'],model_sha256=frozen['model_sha256'])
