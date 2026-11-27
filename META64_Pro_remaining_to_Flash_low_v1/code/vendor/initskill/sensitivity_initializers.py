"""Local controlled initialization baselines, not native paper reproductions."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, label
from .io import digest

BETAS=(.1,.25,.5,1.)
CONFIGS={}
for count in (4,9,16):
    for beta in BETAS:
        CONFIGS[f'holes{count}_b{int(beta*100):03d}']=dict(kind='regular',count=count,beta=beta)
for seed in range(3):
    for beta in (.25,1.):
        CONFIGS[f'random9_s{seed}_b{int(beta*100):03d}']=dict(kind='random',count=9,replicate=seed,beta=beta)
for gamma in (.5,1.):
    for beta in (.25,1.):
        CONFIGS[f'power_g{int(gamma*10)}_b{int(beta*100):03d}']=dict(kind='power',gamma=gamma,beta=beta)
for beta in (.25,1.):
    CONFIGS[f'guided9_b{int(beta*100):03d}']=dict(kind='guided',count=9,beta=beta)
METHODS=tuple(CONFIGS)


def scaled_volume(model,weights):
    """Positive energy weights, fixed floor .05*Vf, scalar physical-volume solve."""
    weights=np.asarray(weights,dtype=float)
    if weights.shape!=(len(model.cells),) or not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError('Expected positive finite weights')
    floor=max(model.config.xmin,.05*model.target)
    lo,hi=0.,1.
    def mapped(scale):return np.clip(scale*weights,floor,1.)
    while model.physical(mapped(hi)).mean()<model.target:hi*=2
    for _ in range(120):
        mid=(lo+hi)/2;x=mapped(mid);v=model.physical(x).mean()
        if abs(v-model.target)<1e-11:return x
        if v>model.target:hi=mid
        else:lo=mid
    raise ValueError('Energy volume calibration did not converge')


def hole_field(model,items):
    fields=[np.linalg.norm(model.centers-[cx,cy],axis=1)-radius for cx,cy,radius in items]
    return np.clip(np.min(fields,axis=0)/(.04*model.ny),-3.,3.)


def initial_diagnostics(model,x):
    rho=model.physical(x).reshape(model.ny,model.nx,order='F')
    regions,n=label(rho<.5)
    boundary=set(np.r_[regions[0,:],regions[-1,:],regions[:,0],regions[:,-1]])
    return dict(min_design=float(x.min()),max_design=float(x.max()),
        design_std=float(x.std()),grayness=float(np.mean(4*rho*(1-rho))),
        threshold_void_components=int(n),threshold_internal_void_components=int(sum(i not in boundary for i in range(1,n+1))),
        diagnostic_threshold=.5,thresholded_fe_performed=False)


def generate_sensitivity(model,method):
    from .initializers import calibrate
    cfg=CONFIGS[method];kind=cfg['kind']
    meta=dict(method=method,extra_fe=0,fallback=False,configuration=cfg,source='Local literature-informed sensitivity baseline; see SENSITIVITY_PROTOCOL_zh.md')
    if kind in ('regular','random'):
        n=cfg['count'];pitch=min(model.nx,model.ny)/np.sqrt(n)
        radius=.25*pitch
        if kind=='regular':
            side=int(np.sqrt(n))
            items=[(float(cx),float(cy),float(radius)) for cx in (np.arange(side)+.5)*model.nx/side for cy in (np.arange(side)+.5)*model.ny/side]
        else:
            # Seed depends on task and replicate, never on beta or invocation order.
            seed=int(digest([20261110,model.task['task_id'],'random_holes',cfg['replicate']])[:8],16)
            rng=np.random.default_rng(seed);items=[]
            for _ in range(20000):
                r=radius*rng.uniform(.75,1.25)
                cx=rng.uniform(r,model.nx-r);cy=rng.uniform(r,model.ny-r)
                if all(np.hypot(cx-x,cy-y)>1.05*(r+rr) for x,y,rr in items):items.append((float(cx),float(cy),float(r)))
                if len(items)==n:break
            if len(items)!=n:raise ValueError('Random non-overlap hole placement exhausted')
            meta['geometry_seed']=seed
        template=calibrate(model,hole_field(model,items),1.)
        meta.update(nominal_holes=items,field_clip=[-3,3],transition=.04*model.ny)
    else:
        reference=model.evaluate(model.initial());meta['extra_fe']=1
        # Cell rows are top-to-bottom; smoothing is orientation invariant.
        energy=reference['energy'].reshape(model.ny,model.nx,order='F')
        energy=gaussian_filter(energy,sigma=model.config.rmin,mode='nearest').ravel(order='F')
        energy=np.maximum(energy,1e-30)
        meta.update(reference_compliance=reference['compliance'],reference_fe=1,energy_smoothing_sigma=model.config.rmin)
        if kind=='power':
            weights=(energy/energy.mean())**cfg['gamma']
            template=scaled_volume(model,weights)
            meta.update(energy_gamma=cfg['gamma'],template_density_floor=max(model.config.xmin,.05*model.target))
        elif kind=='guided':
            radius=.08*model.ny;items=[]
            nodes=np.unique(np.r_[model.fixed//2,np.flatnonzero(model.force)//2])
            protected_distance=cKDTree(model.nodes[nodes]).query(model.centers)[0]
            pos=model.centers
            valid=(pos[:,0]>=radius)&(pos[:,0]<=model.nx-radius)&(pos[:,1]>=radius)&(pos[:,1]<=model.ny-radius)&(protected_distance>=radius+model.config.rmin)
            for j in np.argsort(energy,kind='stable'):
                if not valid[j]:continue
                cx,cy=pos[j]
                if all(np.hypot(cx-x,cy-y)>=2.2*radius for x,y,_ in items):items.append((float(cx),float(cy),float(radius)))
                if len(items)==cfg['count']:break
            if len(items)!=cfg['count']:raise ValueError('Insufficient distinct low-energy hole locations')
            template=calibrate(model,hole_field(model,items),1.)
            meta.update(nominal_holes=items,initial_only_load_support_exclusion=radius+model.config.rmin,
                field_clip=[-3,3],transition=.04*model.ny)
        else:raise ValueError('Unknown sensitivity kind')
    beta=cfg['beta'];x=(1-beta)*model.initial()+beta*template
    if abs(model.physical(x).mean()-model.target)>2e-11:raise ValueError('Blend failed physical-volume conservation')
    meta.update(template_diagnostics=initial_diagnostics(model,template),initial_diagnostics=initial_diagnostics(model,x))
    return x,meta
