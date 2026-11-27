from dataclasses import asdict, replace
import hashlib
import json
import os
from initskill import locking as fcntl
import uuid
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.special import expit, logit

from initskill.physics import SIMP, SolverConfig, mesh

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT/'data/META64_uniform_1000_100_300_v3'
DEFAULT_OUT = Path(os.environ.get('INITSKILL_RESULT_DIR',str(ROOT/'results_id_ood')))
MAX_OC_STEPS = 500


class BenchmarkSIMP(SIMP):
    """Feasible-side OC update used by the current portable platform protocol."""
    def __init__(self,directory,config=None):
        if fe_backend()=='cudss':
            from .filter_cache import cached_filter
            cache=Path(directory).parents[2]/'filter_cache'
            super().__init__(directory,config,filter_factory=lambda centers,radius:cached_filter(cache,centers,radius))
        else:super().__init__(directory,config)
        if fe_backend()=='cudss' and len(self.cells)>64*64:
            from .gpu_filter import SerialRowFilter
            self._gpu_density_filter=SerialRowFilter(self.H)
            self.H=self._gpu_density_filter
    def update(self,x,ev):
        c=self.config;dc=np.minimum(ev['gradient'],0.)
        lo,hi,feasible=0.,1e9,None
        for _ in range(250):
            mid=.5*(lo+hi)
            new=np.clip(x*np.sqrt(-dc/self.dv/mid),np.maximum(c.xmin,x-c.move),np.minimum(1.,x+c.move))
            volume=(float(self.dv@new)/new.size if fe_backend()=='cudss' else self.physical(new).mean())
            if volume>self.target:lo=mid
            else:hi,feasible=mid,new
            if (hi-lo)/(hi+lo)<=1e-8 and feasible is not None:return feasible
        raise RuntimeError('Feasible-side OC bisection failed')

    def evaluate(self,x):
        # Original float64 assembly/solve, with equilibrium and energy-work
        # guards. Reuse LU as the Decimal refinement preconditioner if needed.
        x=np.asarray(x,dtype=float);c=self.config
        if x.shape!=(len(self.cells),) or not np.isfinite(x).all() or np.any(x<c.xmin-1e-12) or np.any(x>1+1e-12):
            raise ValueError('Invalid design density')
        rho=np.clip(self.physical(x),c.xmin,1.)
        w=c.emin+(c.emax-c.emin)*rho**c.penal
        if fe_backend()=='cudss':
            if not hasattr(self,'_stiffness_assembly'):
                from .assembly import AssemblyPlan
                self._stiffness_assembly=AssemblyPlan(self)
            K=self._stiffness_assembly((w[:,None,None]*self.ke).ravel())
        else:
            K=sparse.coo_matrix(((w[:,None,None]*self.ke).ravel(),(self.rows,self.cols)),
                               shape=(len(self.force),len(self.force))).tocsc()
        self.fe_calls+=1
        if fe_backend()=='cudss':
            if not hasattr(self,'_gpu_factors'):
                from .gpu_sparse import CuDSSFactory
                self._gpu_factors=CuDSSFactory()
            factor=self._gpu_factors(K[self.free][:,self.free])
        else:factor=splu(K[self.free][:,self.free])
        u=np.zeros_like(self.force);u[self.free]=factor.solve(self.force[self.free])
        ue=u[self.edof];energy=np.einsum('ei,ij,ej->e',ue,self.ke,ue)
        compliance=float(w@energy)
        residual=float(np.linalg.norm((K@u-self.force)[self.free])/np.linalg.norm(self.force[self.free]))
        work=float(self.force[self.free]@u[self.free])
        balance=abs(compliance-work)/max(abs(work),1e-30)
        nonnegative_energy=bool(np.isfinite(energy).all() and energy.min()>=-1e-12*max(1.,float(np.max(np.abs(energy)))))
        if not (np.isfinite(compliance) and compliance>0 and np.isfinite(work) and work>0 and residual<c.residual_tolerance and balance<=1e-6 and nonnegative_energy):
            from .precision_fem import refined_evaluation
            result=refined_evaluation(self,x,rho,factor,u,residual)
            if fe_backend()=='cudss':result['linear_solver']='cudss-float64'
            result['linear_solver']='cudss-float64' if fe_backend()=='cudss' else 'superlu-float64'
            return result
        physical_gradient=-c.penal*(c.emax-c.emin)*rho**(c.penal-1)*energy
        gradient=np.asarray(self.H.T@(physical_gradient/self.Hs)).ravel()
        result=dict(compliance=compliance,volume=float(rho.mean()),grayness=float(np.mean(4*rho*(1-rho))),
                    residual=residual,density=rho,gradient=gradient,energy=energy,u=u,external_work=work,energy_balance_relative=balance)
        result['linear_solver']='cudss-float64' if fe_backend()=='cudss' else 'superlu-float64'
        return result

    def __del__(self):
        try:
            if hasattr(self,'_gpu_factors'):self._gpu_factors.close()
            if hasattr(self,'_gpu_density_filter'):self._gpu_density_filter.close()
        except Exception:pass


def fe_backend():
    name=os.environ.get('INITSKILL_FE_SOLVER','superlu')
    if name not in {'superlu','cudss'}:raise ValueError('Unknown explicit FE backend: '+name)
    return name


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    fcntl.atomic_replace(temp,path)


def rows(subset):
    allowed={'train1000':1000,'valid100':100,
             'test_id300':300,'test_ood300':300,'test_cr300':300}
    if subset not in allowed: raise ValueError('Not an authorized benchmark manifest')
    r=[json.loads(s) for s in (DATA/'manifests'/f'{subset}.jsonl').read_text().splitlines()]
    if len(r)!=allowed[subset] or len({x['task_key'] for x in r})!=len(r):
        raise ValueError('Manifest count or task identity mismatch')
    return r


def load(row):
    arrays={}
    for k in ('forces','normals','target_vf'):
        asset=row['assets'][k];path=DATA/asset['path']
        if sha(path)!=asset['sha256']: raise ValueError('Changed physical input: '+str(path))
        arrays[k]=np.load(path,allow_pickle=False)
    f,b,v=arrays['forces'],arrays['normals'],float(arrays['target_vf'].item())
    if f.shape!=b.shape or f.shape!=(row['nx']+1,row['ny']+1,2): raise ValueError('Input shape')
    if not np.isfinite(f).all() or not np.isfinite(b).all() or abs(v-row['volume_fraction'])>1e-12:
        raise ValueError('Invalid physical input')
    return f,b,v


def task_dir(out,subset,row):
    return Path(out)/'inputs'/subset/row['source_task_id']


def prepare_task(out,subset,row,fields=None):
    """Build immutable input assets once, with atomic writes and a shared lease."""
    from .scope import require_grid
    require_grid(row['nx'],row['ny'])
    f,b,v=load(row) if fields is None else fields
    folder=task_dir(out,subset,row);folder.mkdir(parents=True,exist_ok=True)
    signature=dict(task_key=row['task_key'],group_id=row['group_id'],nx=row['nx'],ny=row['ny'],
        force_bytes_sha256=hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest(),
        support_bytes_sha256=hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest(),volume=v)
    with (folder/'.prepare.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        ready=folder/'prepared.json'
        if ready.exists():
            cached=json.loads(ready.read_text())
            if cached['source_signature']!=signature:raise ValueError('Changed prepared physical task')
            if any(not (folder/name).exists() or sha(folder/name)!=digest for name,digest in cached['assets'].items()):
                raise ValueError('Changed prepared input cache')
            return folder
        result=_prepare_task(out,subset,row,(f,b,v))
        names=['source_boundary.npz','boundary_conditions.npz','domain.npz','task.json','input_audit.json']
        dump(ready,dict(source_signature=signature,assets={name:sha(folder/name) for name in names}))
        return result


def atomic_savez(path,**arrays):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
    fcntl.atomic_replace(temporary,path)


def _prepare_task(out,subset,row,fields=None):
    """Preserve source loads; explicitly audit their free/fixed decomposition."""
    from .scope import require_grid
    require_grid(row['nx'],row['ny'])
    f,b,v=load(row) if fields is None else fields
    folder=task_dir(out,subset,row);folder.mkdir(parents=True,exist_ok=True)
    force=f[:,::-1,:].reshape(-1).copy()
    fixed=np.flatnonzero(b[:,::-1,:].reshape(-1)).astype(np.int64)
    effective=force.copy();effective[fixed]=0.
    audit=dict(task_key=row['task_key'],group_id=row['group_id'],raw_force_sha256=row['assets']['forces']['sha256'],
        constrained_load_dofs=np.flatnonzero(force[fixed]).size,
        policy='homogeneous Dirichlet: preserve raw force, solve free RHS; constrained loads affect reactions only',
        mapping='author[x,y,dof] -> local[x,ny-y,dof]; vector components unchanged',
        domain_physical_xy=[64.,64.],element_size_xy=[64/row['nx'],64/row['ny']])
    atomic_savez(folder/'source_boundary.npz',force=force,fixed_dofs=fixed)
    atomic_savez(folder/'boundary_conditions.npz',force=effective,fixed_dofs=fixed)
    nodes,cells,centers=mesh(row['nx'],row['ny'])
    atomic_savez(folder/'domain.npz',nodes=nodes,cells=cells,centers=centers)
    dump(folder/'task.json',dict(schema='initskill-isotropic-task-v1',task_id=row['task_key'],
        elements_xy=[row['nx'],row['ny']],volume_fraction=v,group_id=row['group_id']))
    dump(folder/'input_audit.json',audit)
    return folder


def solver(out,subset,row,fields=None):
    from .scope import require_grid
    require_grid(row['nx'],row['ny'])
    p=prepare_task(out,subset,row,fields)
    config=replace(SolverConfig(),rmin=2.5*row['ny']/64)
    # Unit-square Q4 stiffness is invariant under uniform scaling in 2D with
    # constant thickness. Geometry identity is retained; physical radius scales.
    return BenchmarkSIMP(p,config)


def features(row,size=16):
    """Deterministic physical raster. Never reads source reference fields."""
    f,b,v=load(row);f=f.copy();f[b!=0]=0
    norm=np.linalg.norm(f)
    if norm<=0:raise ValueError('No free load')
    f/=norm
    channels=np.zeros((4,size+1,size+1))
    # Bilinear splat preserves resultant and moments of forces on coarse nodes.
    for i,j in np.argwhere(np.any(f!=0,axis=2)):
        u=i/row['nx']*size;w=(1-j/row['ny'])*size
        a=min(int(u),size-1);c=min(int(w),size-1);du=u-a;dw=w-c
        for di,dj,weight in ((0,0,(1-du)*(1-dw)),(1,0,du*(1-dw)),(0,1,(1-du)*dw),(1,1,du*dw)):
            channels[:2,c+dj,a+di]+=weight*f[i,j]
    for i,j in np.argwhere(np.any(b!=0,axis=2)):
        a=int(round(i/row['nx']*size));c=int(round((1-j/row['ny'])*size))
        channels[2:,c,a]=np.maximum(channels[2:,c,a],b[i,j]!=0)
    return np.r_[channels.ravel(),v]


def nodal(row):
    f,b,v=load(row);f=f.copy();f[b!=0]=0
    scale=np.linalg.norm(f)
    if scale<=0:raise ValueError('No free load')
    # N,C,H,W channels; physical y direction unchanged, raster row reversed.
    z=np.concatenate([f/scale,(b!=0).astype(float)],axis=2)
    return z[:,::-1,:].transpose(2,1,0).copy(),v


def calibrate(raw,m):
    raw=np.asarray(raw,dtype=float).reshape(-1)
    if raw.shape!=(m.nx*m.ny,) or not np.isfinite(raw).all(): raise ValueError('Invalid prediction')
    z=logit(np.clip(raw,1e-8,1-1e-8));lo=-float(z.max())-50;hi=-float(z.min())+50
    for _ in range(120):
        x=m.config.xmin+(1-m.config.xmin)*expit(z+(lo+hi)/2)
        if m.physical(x).mean()>m.target:hi=(lo+hi)/2
        else:lo=(lo+hi)/2
    if abs(m.physical(x).mean()-m.target)>1e-10:raise ValueError('Volume projection failed')
    return x


def trajectory(m,x,steps,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    history=[];start=time.perf_counter();before=m.fe_calls
    for step in range(steps+1):
        ev=m.evaluate(x)
        history.append(dict(step=step,compliance=ev['compliance'],volume=ev['volume'],residual=ev['residual']))
        if step in {0,10,30,50,100,steps}:
            np.savez_compressed(folder/f'state_{step:04d}.npz',x=x,density=ev['density'])
        if step<steps:
            x=m.update(x,ev)
    result=dict(status='complete',steps=steps,history=history,FE=m.fe_calls-before,
                seconds=time.perf_counter()-start,solver=asdict(m.config),
                adaptation='portable platform feasible-side OC multiplier tolerance 1e-8')
    dump(folder/'result.json',result)
    return result


def verify_sources():
    root=ROOT/'references/init_benchmark_v1';report=json.loads((root/'SOURCES.json').read_text())
    for row in report['sources']:
        if row['kind']=='paper' and not (root/row['path']).exists():continue
        if row['status']!='downloaded': raise ValueError('Missing source '+row['name'])
        assets=row['files'] if row['kind']=='repository' else [row]
        for asset in assets:
            if sha(root/asset['path'])!=asset['sha256']:raise ValueError('Changed source '+asset['path'])
    return report


def source_fingerprints():
    sources=Path(__file__).parent
    values={p.relative_to(sources).as_posix():sha(p) for p in sorted(sources.rglob('*.py'))}
    values['@platform_physics.py']=sha(ROOT/'src/initskill/physics.py')
    values['@portable_locking.py']=sha(ROOT/'src/initskill/locking.py')
    return values


def protocol(out):
    verify_sources()
    from .scope import current_scope
    scope=current_scope()
    subsets=['train1000','valid100']+scope['test_subsets']
    table={k:rows(k) for k in subsets}
    for i,a in enumerate(subsets):
        for b in subsets[i+1:]:
            if (a.startswith('train') or a.startswith('valid')) and {r['group_id'] for r in table[a]}&{r['group_id'] for r in table[b]}:
                raise ValueError('Physical group leakage')
    sources=Path(__file__).parent
    value=dict(schema='init-benchmark-id-ood-v1',execution_scope=scope,train='train1000',validation='valid100',
        test=subsets[2:],manifest_sha256={k:sha(DATA/'manifests'/f'{k}.jsonl') for k in subsets},
        task_counts={k:len(v) for k,v in table.items()},sampling_sha256=sha(DATA/'SAMPLING.json'),
        source_sha256=source_fingerprints(),
        solver=asdict(SolverConfig()),filter='fixed physical length: 2.5 cells at 64, 10 at 256',
        oc_max_steps=MAX_OC_STEPS,nonconvergence_policy='stop at500 without convergence; keep separate from numerical failure',
        candidates_per_task=1,source_identity=sha(ROOT/'references/init_benchmark_v1/SOURCES.json'),
        oc_update='portable platform feasible-side multiplier bisection tolerance 1e-8',
        supervised_label='uniform-start local solver; archive reuse independently audited',
        test_used_for_selection=False)
    if fe_backend()=='superlu':value.update(linear_solver_backend='superlu-float64',oc_volume_check='mean(Hx/Hs)')
    if fe_backend()=='cudss':
        value.update(linear_solver_backend='cudss-float64',
            oc_volume_check='dot(H.T*(1/Hs),x)/n; exact linear-filter volume identity',
            linear_solver_dependency_manifest_sha256=sha(ROOT/'.runtime/solver_gpu_install_report.json'))
    dest=Path(out)/'protocol.json'
    if dest.exists():
        if json.loads(dest.read_text())!=value:raise ValueError('Frozen protocol changed; use new output')
        # Preserve the execution's original bytes/SHA when reading Windows
        # results on Linux; do not normalize a committed protocol's newlines.
        return value
    dump(dest,value);return value
