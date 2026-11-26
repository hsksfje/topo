"""Yu paper reconstruction; see REPRODUCTION_AUDIT.md for explicit ambiguities."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..common import DATA,dump,sha,rows,nodal
from ..supervised import arrays
from ..training_device import training_device,device_metadata
def conv(a,b,k=3,pad=1,activation=True):
    layers=[nn.Conv2d(a,b,k,padding=pad)]
    if activation:layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class YuEncoderDecoder(nn.Module):
    """Figure 7 channels, VF at bottleneck; text/figure layer count differs."""
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(conv(4,64,2,0),conv(64,64),nn.MaxPool2d(2),
            conv(64,128),conv(128,128),nn.MaxPool2d(2),
            conv(128,256),conv(256,256),nn.MaxPool2d(2),conv(256,512))
        self.decoder=nn.Sequential(conv(513,512),nn.Upsample(scale_factor=2,mode='nearest'),
            conv(512,256),conv(256,256),nn.Upsample(scale_factor=2,mode='nearest'),
            conv(256,128),conv(128,128),nn.Upsample(scale_factor=2,mode='nearest'),
            conv(128,64),conv(64,2),conv(2,1,activation=False),nn.Sigmoid())
        # Match Keras default Glorot rather than PyTorch Kaiming defaults.
        self.apply(glorot)
    def forward(self,z,vf):
        if z.shape[-1]-1!=z.shape[-2]-1 or (z.shape[-1]-1)%8:raise ValueError('Expected nodal (8k+1) square grid')
        h=self.encoder(z)
        v=vf.reshape(-1,1,1,1).expand(-1,1,h.shape[-2],h.shape[-1])
        return self.decoder(torch.cat([h,v],dim=1))

def glorot(layer):
    if isinstance(layer,(nn.Conv2d,nn.Linear)):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:nn.init.zeros_(layer.bias)

def seed_all(seed):
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

def tensors(out,subset):
    records,_,x=arrays(out,subset)
    z,v=zip(*(nodal(r) for r in records))
    y=np.stack([a.reshape((64,64),order='F') for a in x])[:,None]
    return records,torch.tensor(np.stack(z),dtype=torch.float32),torch.tensor(v,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)

def train_cnn(out,epochs=500,batch_size=16,seed=42):
    seed_all(seed);tr,z,v,y=tensors(out,'train1000');va,zv,vv,yv=tensors(out,'valid100')
    folder=Path(out)/'models/cnn';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'frozen.json').exists():raise FileExistsError('Frozen CNN exists')
    device=training_device()
    z,v,y,zv,vv,yv=[a.to(device) for a in (z,v,y,zv,vv,yv)]
    model=YuEncoderDecoder().to(device);opt=torch.optim.Adam(model.parameters(),lr=1e-3,eps=1e-7)
    rng=np.random.default_rng(seed);best=np.inf;stale=0;history=[];start=time.perf_counter()
    config=dict(seed=seed,epochs=epochs,batch_size=batch_size,learning_rate=1e-3,adam_epsilon=1e-7,
        runtime=device_metadata(device),
        input='65x65 nodal Fx,Fy,Bx,By; global force normalization',output='64x64 design x',
        train_keys=[r['task_key'] for r in tr],valid_keys=[r['task_key'] for r in va],
        teacher='audited uniform archive; mixed stopping times explicitly preserved',
        original_grid='33 input -> 32 output',adapted_grid='65 input -> 64 output',
        unspecified='learning rate, epsilon and patience not fully specified in paper; explicit Keras-style defaults',
        criterion='minimum held-out density MAE; patience 30')
    dump(folder/'config.json',config)
    for epoch in range(1,epochs+1):
        model.train();train_loss=0.
        for ids in np.array_split(rng.permutation(len(tr)),int(np.ceil(len(tr)/batch_size))):
            opt.zero_grad();loss=F.l1_loss(model(z[ids],v[ids]),y[ids]);loss.backward();opt.step()
            train_loss+=float(loss.detach())*len(ids)
        model.eval()
        with torch.no_grad():
            val=sum(float(F.l1_loss(model(zv[i:i+batch_size],vv[i:i+batch_size]),yv[i:i+batch_size]))*len(zv[i:i+batch_size]) for i in range(0,len(va),batch_size))/len(va)
        if val<best:
            best=val;stale=0
            torch.save(dict(state_dict=model.state_dict(),config=config,epoch=epoch),folder/'best.pt')
        else:stale+=1
        history.append(dict(epoch=epoch,train_MAE=train_loss/len(tr),valid_MAE=val,seconds=time.perf_counter()-start))
        dump(folder/'history.json',history)
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
                        rng_state_json=json.dumps(rng.bit_generator.state)),folder/'checkpoint.pt')
        print('cnn',history[-1],flush=True)
        if stale>=30:break
    dump(folder/'frozen.json',dict(model_sha256=sha(folder/'best.pt'),config_sha256=sha(folder/'config.json'),
        best_validation_MAE=best,adaptation=True,seed=seed))

def load_cnn(out):
    folder=Path(out)/'models/cnn';f=json.loads((folder/'frozen.json').read_text())
    if sha(folder/'best.pt')!=f['model_sha256'] or sha(folder/'config.json')!=f['config_sha256']:raise ValueError('Changed CNN')
    model=YuEncoderDecoder();model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True)['state_dict']);model.eval()
    return model,f

def predict_cnn(out,row):
    model,f=load_cnn(out)
    # Fully convolutional resolution transfer: explicit adaptation, not forced
    # raster downsampling which could delete CR supports.
    z,v=nodal(row)
    with torch.no_grad():pred=model(torch.tensor(z,dtype=torch.float32)[None],torch.tensor([v],dtype=torch.float32))[0,0].numpy()
    return pred.ravel(order='F'),dict(model_sha256=f['model_sha256'],field_kind='design_x',
                                    grid_transfer='same fully convolutional ED, target nodal grid')
