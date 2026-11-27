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

from ..cnn_encoder_decoder.core import conv,glorot,seed_all,load_cnn
class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Figure 9: conv+BN, ReLU, conv+BN; not the differing cited repo order.
        self.block=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64,momentum=.2,eps=1e-3),nn.ReLU(),
                                 nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64,momentum=.2,eps=1e-3))
    def forward(self,z):return z+self.block(z)

class YuGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=conv(1,64,5,2)
        self.blocks=nn.Sequential(*(ResidualBlock() for _ in range(8)))
        self.post=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64,momentum=.2,eps=1e-3))
        self.up=nn.Sequential(nn.Upsample(scale_factor=2,mode='nearest'),conv(64,256),
            nn.Upsample(scale_factor=2,mode='nearest'),conv(256,256),
            conv(256,1,5,2,False),nn.Sigmoid())
        self.apply(glorot)
    def forward(self,z):
        h=self.first(z);return self.up(h+self.post(self.blocks(h)))

class YuDiscriminator(nn.Module):
    def __init__(self,high_size=256):
        super().__init__();layers=[];a=2
        for i,b in enumerate([64,64,128,128,256,256,512,512]):
            layers+=[nn.Conv2d(a,b,3,stride=1 if i%2==0 else 2,padding=1)]
            if i:layers.append(nn.BatchNorm2d(b,momentum=.2,eps=1e-3))
            layers.append(nn.LeakyReLU(.2))
            a=b
        self.cnn=nn.Sequential(*layers)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(512*(high_size//16)**2,1024),nn.LeakyReLU(.2),nn.Linear(1024,1))
        self.apply(glorot)
    def forward(self,low,high):
        low=F.interpolate(low,size=high.shape[-2:],mode='nearest')
        return self.head(self.cnn(torch.cat([low,high],dim=1)))

def pair_provenance(out,subset):
    if subset not in {'train1000','valid100'}:raise ValueError('Training pairs must be Train/Valid')
    records=rows(subset);identities=[]
    for r in records:
        p=Path(out)/'pairs'/subset/(r['source_task_id']+'.npz')
        meta=json.loads(p.with_suffix('.json').read_text())
        low=Path(out)/'labels'/subset/(r['source_task_id']+'.npz')
        if (meta['task_key']!=r['task_key'] or meta['group_id']!=r['group_id'] or
            meta['source_split']!=subset or sha(p)!=meta['sha256'] or
            sha(low)!=meta['low_label_sha256']):raise ValueError('Changed pair provenance')
        identities.append(dict(task_key=r['task_key'],group_id=r['group_id'],
            pair_sha256=meta['sha256'],low_teacher_sha256=meta['low_label_sha256']))
    return dict(manifest_sha256=sha(DATA/'manifests'/(subset+'.jsonl')),records=identities)


def paired(out,subset):
    """Every pair must belong to this exact manifest, never the CR test pool."""
    records=rows(subset);low=[];high=[]
    for r in records:
        p=Path(out)/'pairs'/subset/(r['source_task_id']+'.npz')
        if not p.exists():raise FileNotFoundError('Missing same-task 64/256 teacher pair: '+str(p))
        meta=json.loads(p.with_suffix('.json').read_text())
        if (meta['task_key']!=r['task_key'] or meta['group_id']!=r['group_id'] or
            meta['sha256']!=sha(p) or meta['source_split']!=subset):raise ValueError('Pair identity/leakage')
        with np.load(p,allow_pickle=False) as a:
            if a['low'].shape!=(64,64) or a['high'].shape!=(256,256):raise ValueError('Pair shape')
            low.append(a['low'].copy());high.append(a['high'].copy())
    return torch.tensor(np.stack(low)[:,None],dtype=torch.float32),torch.tensor(np.stack(high)[:,None],dtype=torch.float32)

def train_cgan(out,epochs=20,batch_size=16,seed=42):
    seed_all(seed);low,high=paired(out,'train1000');vl,vh=paired(out,'valid100')
    # Missing pairs fail before expensive network allocation.
    folder=Path(out)/'models/cgan';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'frozen.json').exists():raise FileExistsError('Frozen GAN exists')
    device=training_device()
    g,d=YuGenerator().to(device),YuDiscriminator(256).to(device)
    go=torch.optim.Adam(g.parameters(),lr=2e-4,betas=(.5,.999),eps=1e-7)
    do=torch.optim.Adam(d.parameters(),lr=2e-4,betas=(.5,.999),eps=1e-7)
    rng=np.random.default_rng(seed);history=[];best=np.inf
    train_data=pair_provenance(out,'train1000');valid_data=pair_provenance(out,'valid100')
    config=dict(epochs=epochs,batch_size=batch_size,seed=seed,lambda_MAE=.001,
        train_keys=[r['task_key'] for r in train_data['records']],
        valid_keys=[r['task_key'] for r in valid_data['records']],
        training_data=train_data,validation_data=valid_data,
        runtime=device_metadata(device),
        optimizer='Adam lr=2e-4 beta1=.5 from cited Keras-GAN; not specified by TO author',
        generator_condition_training='optimized low-resolution teacher',
        discriminator_condition_training='same low-resolution teacher; author prose also mentions predicted low input; unresolved',
        loss='BCE(real)+BCE(fake) for D; non-saturating BCE(fake,real)+.001*MAE for G',
        loss_adaptation='non-saturating G surrogate for paper minimax equation',
        grid='64 -> 256 instead of original 32 -> 128')
    dump(folder/'config.json',config)
    for epoch in range(1,epochs+1):
        g.train();d.train()
        for ids in np.array_split(rng.permutation(len(low)),int(np.ceil(len(low)/batch_size))):
            a,b=low[ids].to(device),high[ids].to(device);fake=g(a)
            do.zero_grad();real=d(a,b);imag=d(a,fake.detach())
            dl=F.binary_cross_entropy_with_logits(real,torch.ones_like(real))+F.binary_cross_entropy_with_logits(imag,torch.zeros_like(imag))
            dl.backward();do.step()
            # D statistics are fixed during the G update; no unintended BN updates.
            d.eval();d.requires_grad_(False);go.zero_grad()
            score=d(a,fake);gl=F.binary_cross_entropy_with_logits(score,torch.ones_like(score))+.001*F.l1_loss(fake,b)
            gl.backward();go.step();d.requires_grad_(True);d.train()
        g.eval()
        with torch.no_grad():val=sum(float(F.l1_loss(g(vl[i:i+batch_size].to(device)),vh[i:i+batch_size].to(device)))*len(vl[i:i+batch_size]) for i in range(0,len(vl),batch_size))/len(vl)
        if val<best:
            best=val;torch.save(dict(state_dict=g.state_dict(),config=config),folder/'best.pt')
        history.append(dict(epoch=epoch,valid_MAE=val));dump(folder/'history.json',history)
        torch.save(dict(generator=g.state_dict(),discriminator=d.state_dict(),generator_optimizer=go.state_dict(),
                        discriminator_optimizer=do.state_dict(),epoch=epoch),folder/'checkpoint.pt')
        print('cgan',history[-1],flush=True)
    dump(folder/'frozen.json',dict(model_sha256=sha(folder/'best.pt'),config_sha256=sha(folder/'config.json'),best_validation_MAE=best,adaptation=True))

def predict_cgan(out,row):
    from ..common import features
    model,cf=load_cnn(out)
    # Conservative force splat and nearest support masks on normalized grid.
    raster=features(row,size=64)[:-1].reshape(4,65,65)
    vf=float(row['volume_fraction'])
    with torch.no_grad():low=model(torch.tensor(raster,dtype=torch.float32)[None],torch.tensor([vf],dtype=torch.float32))
    folder=Path(out)/'models/cgan';f=json.loads((folder/'frozen.json').read_text())
    if sha(folder/'best.pt')!=f['model_sha256']:raise ValueError('Changed cGAN')
    g=YuGenerator();g.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True)['state_dict']);g.eval()
    with torch.no_grad():
        high=g(low)
        if row['nx']!=256:high=F.interpolate(high,size=(row['ny'],row['nx']),mode='area')
    return high[0,0].numpy().ravel(order='F'),dict(model_sha256=f['model_sha256'],CNN_sha256=cf['model_sha256'],
        field_kind='design_x',chain='coarse CNN(64) -> cGAN(256) -> target cell-area averaging if target64',
        CR_condition_mapping='force-conservative bilinear splat, supports nearest grid nodes; explicit coarse raster approximation')
