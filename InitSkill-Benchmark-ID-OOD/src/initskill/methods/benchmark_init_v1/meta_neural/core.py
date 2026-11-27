"""Use immutable upstream JAX model, filters, FEM, loss and Reptile directly.

Only manifest iteration, failure handling, accounting, and density export are
local. No DALI, official pretrained checkpoints, or final topology labels.
"""
import importlib.util
import json
from pathlib import Path
import sys
import time
import os
from initskill import locking as fcntl

import numpy as np
from ..common import ROOT, DATA, dump, sha, rows, load

UPSTREAM=ROOT/'references/init_benchmark_v1/upstream/metatopia'


class TaskSampler:
    """DALI-style shuffled epochs over the restricted manifest, without repeats."""
    def __init__(self,count,batch_size,seed):
        if count%batch_size:raise ValueError('Require complete manifest-sized meta-batches')
        self.count=count;self.batch_size=batch_size;self.rng=np.random.default_rng(seed)
        self.permutation=self.rng.permutation(count);self.cursor=0;self.epoch=0
    def next(self):
        if self.cursor==self.count:
            self.permutation=self.rng.permutation(self.count);self.cursor=0;self.epoch+=1
        indices=self.permutation[self.cursor:self.cursor+self.batch_size]
        self.cursor+=self.batch_size
        return indices
    def state(self):
        return dict(permutation=self.permutation.tolist(),cursor=self.cursor,epoch=self.epoch,
                    rng_state=self.rng.bit_generator.state)
    def restore(self,state):
        permutation=np.asarray(state['permutation'],dtype=int);cursor=state['cursor']
        if (sorted(permutation.tolist())!=list(range(self.count)) or not 0<=cursor<=self.count or
                cursor%self.batch_size or state['epoch']<0):raise ValueError('Invalid native sampler checkpoint')
        self.permutation=permutation;self.cursor=cursor;self.epoch=state['epoch']
        self.rng.bit_generator.state=state['rng_state']


def backend():
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/cache/matplotlib'))
    sys.path.insert(0,str(UPSTREAM))
    if os.name=='nt':
        dlls=Path(sys.prefix)/'Library/bin'
        if dlls.is_dir():os.add_dll_directory(str(dlls))
    import jax
    jax.config.update('jax_enable_x64',True)
    import equinox as eqx
    import optax
    import metatopia as mtp
    spec=importlib.util.spec_from_file_location('init_benchmark_upstream_common',UPSTREAM/'experiments/meta_neural_to/common.py')
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    return jax,eqx,optax,mtp,common


class Ledger:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.count=sum(1 for _ in self.path.open()) if self.path.exists() else 0
        self.failures=0
    def attach(self):
        import metatopia.solver.fem_cvxopt as module
        original=module.displace_cvxopt
        def audited(*args,**kwargs):
            self.count+=1
            with self.path.open('a') as stream:stream.write(json.dumps(dict(attempt=self.count))+'\n')
            try:return original(*args,**kwargs)
            except Exception:
                self.failures+=1
                raise
        module.displace_cvxopt=audited
        self.original=original;self.module=module
    def detach(self):self.module.displace_cvxopt=self.original


def sample(row):
    """Compute reference from uniform physical density, in author coordinates."""
    from ..scope import require_grid
    require_grid(row['nx'],row['ny'])
    import metatopia.solver.fem_cvxopt as fem
    f,b,v=load(row);nx,ny=row['nx'],row['ny']
    ke=fem.get_stiffness_matrix(1.,.3,64.,64.,nx,ny)
    fixed=np.flatnonzero(b.ravel());free=np.setdiff1d(np.arange(f.size),fixed)
    rho=np.full((ny,nx),v)
    u=fem.displace_cvxopt(rho,ke,f.ravel(),free,fixed,penalty=3,E_min=1e-9,E_0=1.)
    ce=fem.compliance(rho,u,ke)
    ref=float(((1e-9+(1-1e-9)*rho**3)*ce).sum())
    log=np.log(np.maximum(ce,1e-30));span=float(log.max()-log.min())
    energy=np.zeros_like(log) if span<1e-30 else 2*(log-log.min())/span-1
    return dict(forces=f,normals=b,target_vf=np.asarray(v),reference_compliance=np.asarray(ref),
        strain_energy=energy,id=np.asarray(int(row['source_task_id'].split('_')[-1])))


def setup(shape,seed=42):
    jax,eqx,optax,mtp,common=backend()
    ny,nx=shape
    model=mtp.models.ResidualSiren(3,256,1,outermost_linear=True,
        num_resnet_blocks=2,num_layers_per_block=2,omega_0=60.,key=jax.random.key(seed))
    solve=mtp.solver.setup_solver_cvxopt(young=1.,young_min=1e-9,poisson=.3,
        Lx=64.,Ly=64.,Nx=nx,Ny=ny,penalty=3.)
    coords=mtp.utils.get_grid_coords(shape)
    parameterisation=common.get_design_parameterisation(shape,1/32)
    loss=common.get_loss_fn(solve,coords,parameterisation,True)
    learner=mtp.learners.Reptile(model,optax.adam(1e-4),optax.adam(1e-6),loss,loss)
    return jax,eqx,optax,mtp,common,model,learner,loss,coords,parameterisation


def train(out,outer_steps=6000,inner_steps=10,batch_size=5,seed=42,validation_every=200,smoke=False):
    folder=Path(out)/('verification/native_meta_smoke' if smoke else 'models/meta_native')
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'training.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        return _train(out,outer_steps,inner_steps,batch_size,seed,validation_every,smoke)


def _train(out,outer_steps=6000,inner_steps=10,batch_size=5,seed=42,validation_every=200,smoke=False):
    jax,eqx,optax,mtp,common,model,learner,loss,coords,param=setup((64,64),seed)
    folder=Path(out)/('verification/native_meta_smoke' if smoke else 'models/meta_native')
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'learned.eqx').exists():
        frozen=json.loads((folder/'frozen.json').read_text());cfg=json.loads((folder/'config.json').read_text())
        if (sha(folder/'learned.eqx')!=frozen['model_sha256'] or sha(folder/'config.json')!=frozen['config_sha256'] or
            cfg['outer_steps']!=outer_steps or cfg['inner_steps']!=inner_steps or cfg['batch_size']!=batch_size or
            cfg['seed']!=seed or cfg['smoke_only']!=smoke):
            raise ValueError('Existing native model identity mismatch')
        return folder
    ledger=Ledger(folder/'FE_attempts.jsonl');ledger.attach()
    try:
        tr=rows('train1000');va=rows('valid100')
        if smoke:tr=tr[:batch_size];va=va[:1]
        config=dict(seed=seed,outer_steps=outer_steps,inner_steps=inner_steps,batch_size=batch_size,
            runtime=dict(jax_version=str(jax.__version__),devices=[str(d) for d in jax.devices()],
                         backend=jax.default_backend(),enable_x64=bool(jax.config.jax_enable_x64)),
            train_keys=[r['task_key'] for r in tr],valid_keys=[r['task_key'] for r in va],
            source_identity=sha(ROOT/'references/init_benchmark_v1/SOURCES.json'),
            native_relative_radius=1/32,platform_relative_radius=2.5/64,
            deliberate_difference='native filters preserved; common OC evaluates exported rho as candidate x',
            train_manifest_sha256=sha(DATA/'manifests/train1000.jsonl'),
            validation_manifest_sha256=sha(DATA/'manifests/valid100.jsonl'),
            optimized_topology_labels_used=False,smoke_only=smoke,
            validation_execution='eqx.filter_jit of unchanged official adapt_and_evaluate')
        from . import checkpointing
        restored=checkpointing.load(folder,model,learner.outer_opt.init,config,eqx)
        params,static=eqx.partition(model,eqx.is_array);outer_state=learner.outer_opt.init(params)
        sampler=TaskSampler(len(tr),batch_size,seed);history=[];best=np.inf;first_step=1;seconds_offset=0.
        if restored is not None:
            model,outer_state,saved=restored;params,static=eqx.partition(model,eqx.is_array)
            sampler.restore(saved['sampler']);history=saved['history'];first_step=saved['step']+1
            seconds_offset=saved['seconds'];best=saved.get('best_validation_loss')
            best=np.inf if best is None else best
            config['resumed_from_step']=saved['step']
            dump(folder/'resume_event.json',dict(step=saved['step'],checkpoint_FE=saved['FE'],
                attempted_FE_before_resume=ledger.count,uncommitted_attempts_preserved=ledger.count-saved['FE'],
                source=saved['recovery_source'],parameters_and_outer_Adam_and_sampler_restored=True,
                old_uncommitted_wall_time='not measured by committed loop clock; retained as unknown overhead'))
            print('native-resume',saved['step'],'attempted-FE',ledger.count,flush=True)
        # Rebuilt uniform references are explicitly charged to the same ledger.
        reference_started=time.perf_counter()
        train_data=[sample(r) for r in tr];valid_data=[sample(r) for r in va]
        config['reference_preparation_seconds_this_invocation']=time.perf_counter()-reference_started
        dump(folder/'config.json',config);started=time.perf_counter()
        update=eqx.filter_jit(learner.update)
        validate=eqx.filter_jit(learner.adapt_and_evaluate)
        # Upstream Reptile reinitializes the inner optimizer per task/episode.
        for step in range(first_step,outer_steps+1):
            indices=sampler.next()
            batch={k:jax.numpy.asarray(np.stack([train_data[i][k] for i in indices])) for k in train_data[0]}
            params,static,outer_state,train_loss,metrics=update(params,static,outer_state,batch,batch,inner_steps)
            train_value=float(train_loss)
            if not np.isfinite(train_value) or ledger.failures:raise RuntimeError('Upstream numerical failure; zero-loss fallback rejected')
            row=dict(step=step,train_loss=train_value,FE=ledger.count,
                     task_indices=indices.tolist(),task_epoch=sampler.epoch)
            if step%validation_every==0 or step==outer_steps:
                losses=[]
                for item in valid_data:
                    one={k:jax.numpy.asarray(np.stack([v])) for k,v in item.items()}
                    (_,_,_,_,val_loss,_)=validate(params,static,one,one,inner_steps)
                    losses.append(float(jax.numpy.mean(val_loss)))
                score=float(np.mean(losses));row['validation_loss']=score
                if not np.isfinite(score) or ledger.failures:raise RuntimeError('Validation FE failure')
                if score<best:
                    best=score;eqx.tree_serialise_leaves(folder/'best.eqx',eqx.combine(params,static))
            history.append(row)
            checkpointing.save(folder,eqx.combine(params,static),outer_state,
                dict(step=step,FE=ledger.count,sampler=sampler.state(),history=history,
                     best_validation_loss=float(best) if np.isfinite(best) else None,
                     seconds=seconds_offset+time.perf_counter()-started),config,eqx)
            print('native-meta',row,flush=True)
        target=folder/'learned.eqx';target.write_bytes((folder/'best.eqx').read_bytes())
        dump(folder/'frozen.json',dict(model_sha256=sha(target),config_sha256=sha(folder/'config.json'),
            FE=ledger.count,best_validation_loss=best,smoke_only=smoke,seed=seed,
            upstream_commit=next(r['commit'] for r in json.loads((ROOT/'references/init_benchmark_v1/SOURCES.json').read_text())['sources'] if r['name']=='metatopia')))
        return folder
    finally:ledger.detach()


def predict(out,row,warmup=0):
    folder=Path(out)/'models/meta_native';frozen=json.loads((folder/'frozen.json').read_text())
    if frozen['smoke_only'] or sha(folder/'learned.eqx')!=frozen['model_sha256']:raise ValueError('Not a frozen formal model')
    jax,eqx,optax,mtp,common,model,learner,loss,coords,param=setup((row['ny'],row['nx']),frozen['seed'])
    model=eqx.tree_deserialise_leaves(folder/'learned.eqx',model)
    ledger=Ledger(Path(out)/'costs/meta_native'/row['task_key']/'FE_attempts.jsonl');before=ledger.count;ledger.attach()
    try:
        s={k:jax.numpy.asarray(v) for k,v in sample(row).items()}
        p,static=eqx.partition(model,eqx.is_array)
        state=optax.adam(1e-4).init(p)
        for _ in range(warmup):
            (_,aux),grad=eqx.filter_value_and_grad(loss,has_aux=True)(p,static,s)
            updates,state=optax.adam(1e-4).update(grad,state);p=eqx.apply_updates(p,updates)
        q=jax.numpy.concatenate([coords,s['strain_energy'].reshape(-1,1)],axis=1)
        rho=np.asarray(param(q,s,p,static))
        if ledger.failures or not np.isfinite(rho).all():raise RuntimeError('Meta prediction failed')
        # Author rho[y,x] with bottom row first -> platform F vector top row first.
        return rho[::-1,:].ravel(order='F').copy(),dict(field_kind='physical_rho',
            handoff='explicit rho-as-x; common platform filter follows',warmup_steps=warmup,
            initialization_FE=ledger.count-before,native_mean_density=float(rho.mean()),
            native_relative_radius=1/32,model_sha256=frozen['model_sha256'],
            native_prediction_backend=jax.default_backend())
    finally:ledger.detach()
