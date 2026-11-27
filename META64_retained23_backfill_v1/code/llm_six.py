"""Frozen six LLM initialization protocols; only required definitions extracted."""
import json,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import logit
from initskill.io import dump,digest,save_npz
from initskill.physics import SIMP as BaseSIMP,SolverConfig
from initskill.initializers import calibrate
from code_runner import execute,FUNCTIONS
ARMS=['A1','A2','A3','B1','B2','B3']

LABELS={'U':'Uniform','A1':'无FE·直接矩阵','A2':'无FE·构想与代码','A3':'无FE·第二LLM修正','B1':'有FE·直接矩阵','B2':'有FE·构想与代码','B3':'有FE·第二LLM修正'}

SHAPE=(10,20)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

class SIMP(BaseSIMP):
    def update(self,x,ev):
        c=self.config;dc=np.minimum(ev['gradient'],0.)
        lo,hi,feasible=0.,1e9,None
        for _ in range(250):
            mid=.5*(lo+hi)
            new=np.clip(x*np.sqrt(-dc/self.dv/mid),np.maximum(c.xmin,x-c.move),np.minimum(1.,x+c.move))
            if self.physical(new).mean()>self.target:lo=mid
            else:hi,feasible=mid,new
            if (hi-lo)/(hi+lo)<=1e-8 and feasible is not None:return feasible
        raise RuntimeError('Feasible-side OC bisection failed')

def schema(properties):
    return dict(type='json_schema',json_schema=dict(name='initialization',strict=True,
        schema=dict(type='object',properties=properties,required=list(properties),additionalProperties=False)))

GRID=dict(type='array',minItems=10,maxItems=10,items=dict(type='array',minItems=20,maxItems=20,items=dict(type='number',minimum=0,maximum=1)))

TEXT=dict(type='string')

def grid(value):
    a=np.asarray(value,dtype=float)
    if a.shape!=SHAPE or not np.isfinite(a).all() or np.any(a<0) or np.any(a>1):raise ValueError('Expected 10x20 density matrix in [0,1]')
    return a

def decode(model,a):
    a=np.asarray(a,dtype=float)
    if a.shape not in (SHAPE,(model.ny,model.nx)):raise ValueError('Density shape mismatch')
    interpolation=RegularGridInterpolator((np.linspace(0,1,a.shape[0]),np.linspace(0,1,a.shape[1])),logit(np.clip(a,.01,.99)))
    field=interpolation(np.c_[1-model.centers[:,1]/model.ny,model.centers[:,0]/model.nx])
    return calibrate(model,field)

def observation(model,ev):
    u=np.asarray(ev['u']).reshape(-1,2);disp=u[model.cells].mean(axis=1)
    scale=max(float(np.linalg.norm(u,axis=1).max()),1e-30)
    c=model.config;rho=ev['density']
    contribution=(c.emin+(c.emax-c.emin)*rho**c.penal)*ev['energy']
    fields=[contribution/max(float(contribution.mean()),1e-30),disp[:,0]/scale,disp[:,1]/scale]
    ix=np.minimum((model.centers[:,0]/model.nx*8).astype(int),7)
    # Same top-to-bottom orientation as generated matrices.
    iy=np.minimum(((1-model.centers[:,1]/model.ny)*8).astype(int),7)
    maps=[[[round(float(v[(iy==y)&(ix==x)].mean()),4) for x in range(8)] for y in range(8)] for v in fields]
    return dict(source='One uniform-density step-zero FE solve',rows='top to bottom',columns='left to right',
                full_element_relative_strain_energy=np.round(fields[0].reshape((model.ny,model.nx),order='F'),4).tolist(),
                full_element_grid_shape=[model.ny,model.nx],
                relative_compliance_contribution=maps[0],normalized_ux=maps[1],normalized_uy=maps[2],
                definition='Full ny x nx per-element strain energy divided by its global mean, rounded to4decimals. Element strain energy=0.5*(Emin+(Emax-Emin)*rho^p)*u_e^T K_unit u_e. Normalized compliance contribution has identical values. Additional block-average energy and signed displacement maps are8x8. No optimized fields or candidate scores.')

def messages(task,mode,obs=None,previous=None,concept=None):
    payload=dict(task=task,objective='Design ONE initial density structure for minimum compliance AFTER exactly 50 fixed SIMP/OC updates. Do not change supports, loads, materials or volume fraction.',
        solver=dict(material='isotropic plane stress',**asdict(SolverConfig())),
        convention='All matrices: row0=TOP, last row=BOTTOM; col0=LEFT. x right, y up. Coordinates u=x/W,v=y/H.',
        decoding='Densities clipped to [.01,.99] before logits; bilinear logit interpolation at fine element centers; global sigmoid shift to enforce exact filtered physical volume. No geometry/path repair. All elements designable.',
        constraints='Construct a meaningful NONUNIFORM structural initialization, rather than a constant neutral field. Avoid unintended islands, missing support/load paths, excessive binarity, thin disconnected members. Reason using ALL explicit loaded-node coordinates and support coordinates. No performance feedback or final topology is available.')
    if obs is not None:payload['initial_FE_observation']=obs
    if mode=='matrix':
        instruction='Output an actual 10-row x20-column numeric density matrix in [0,1], not geometry parameters. Return {values,reason}. No code.'
        contract=schema(dict(values=GRID,reason=TEXT))
    elif mode=='review':
        payload['proposed_matrix']=previous
        instruction='You are an independent structural initialization REVIEWER, not the original designer. Inspect the proposed matrix against the task requirements. Correct implausible load paths, unnecessary islands, fragile connections and excessive voiding if justified. You may keep it unchanged. Return {values,issues,changes} with a complete corrected 10x20 numeric density matrix in [0,1]. No code, no solver calls. Do not claim FE-verified feasibility.'
        contract=schema(dict(values=GRID,issues=TEXT,changes=TEXT))
    elif mode=='concept':
        instruction='Propose ONE concrete structural layout concept with normalized member/path/hole positions, widths, density levels and how it connects actual load nodes to supports. It will be implemented as Python in a separate call. Return {concept}, with concise implementation-ready details.'
        contract=schema(dict(concept=TEXT))
    else:
        payload['layout_concept']=concept
        instruction=('Implement the supplied layout concept as actual vectorized Python source. Return {code,reason}. '
            'Define exactly def initialize(nx, ny, volfrac): returning a numpy array shape (ny,nx) in [0,1], row0 TOP. '
            'Use import numpy as np if needed; np is preprovided. Only assignments, tuple unpacking, expressions and return in the function; no loops, conditionals, comprehensions, helpers, IO, arbitrary attributes, array mutation, methods or builtin calls. '
            'Only np functions allowed: '+','.join(FUNCTIONS)+'. np.pi allowed. Use np.linspace(0,1,nx), np.linspace(1,0,ny), np.meshgrid to construct x,y. '
            'All geometry constants must reflect this task and the concept. Use np.minimum/maximum instead of Python min/max; use np.where for piecewise expressions. No markdown fences. No solver implementation.')
        contract=schema(dict(code=TEXT,reason=TEXT))
    return [dict(role='system',content=instruction),dict(role='user',content=json.dumps(payload,ensure_ascii=False,separators=(',',':')))],contract

def call(client,task_id,name,task,mode,obs=None,previous=None,concept=None):
    msg,contract=messages(task,mode,obs,previous,concept)
    result=client.call(task_id+'_'+name,msg,contract,6144 if mode in ('matrix','review') else 4096)
    if not isinstance(result,dict):raise ValueError('Model returned no valid object')
    return result

def generate_pair(out,client,record,task,prefix,obs=None,suffixes=('1','2','3')):
    root=out/'tasks'/record['task_id'];status={}
    for arm,mode in ((prefix+'1','matrix'),(prefix+'2','code'),(prefix+'3','review')):
        if arm[-1] not in suffixes:continue
        folder=root/arm;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'generation.json').exists():
            status[arm]=read(folder/'generation.json');continue
        try:
            if mode=='matrix':
                result=call(client,record['task_id'],arm,task,mode,obs)
                values=grid(result['values']);dump(folder/'proposal.json',result)
            elif mode=='code':
                concept=call(client,record['task_id'],arm+'_concept',task,'concept',obs)
                dump(folder/'concept.json',concept)
                result=call(client,record['task_id'],arm+'_code',task,'code',obs,concept=concept['concept'])
                dump(folder/'proposal.json',result)
                values=execute(result['code'],task['elements_xy'][0],task['elements_xy'][1],task['volume_fraction'],folder)
            else:
                parent=root/(prefix+'1')
                if not (parent/'raw_matrix.npz').exists():raise ValueError('Original direct matrix invalid; reviewer not called')
                with np.load(parent/'raw_matrix.npz',allow_pickle=False) as a:previous=a['values'].copy()
                result=call(client,record['task_id'],arm,task,mode,obs,previous=previous.tolist())
                values=grid(result['values']);dump(folder/'proposal.json',result)
                dump(folder/'review_delta.json',dict(parent=prefix+'1',changed_cells=int(np.count_nonzero(values!=previous)),mean_absolute_change=float(np.abs(values-previous).mean())))
            save_npz(folder/'raw_matrix.npz',values=values)
            status[arm]=dict(status='valid',shape=list(values.shape),matrix_sha=digest(values.tolist()))
        except RuntimeError:
            # Unknown API outcomes halt the campaign; do not silently continue.
            raise
        except Exception as exc:
            status[arm]=dict(status='invalid',error_type=type(exc).__name__,error=str(exc))
        dump(folder/'generation.json',status[arm])
    return status

def evaluate(out,record,arm):
    folder=out/'tasks'/record['task_id']/arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return read(folder/'result.json')
    if arm!='U' and read(folder/'generation.json')['status']!='valid':
        result=dict(status='invalid',C0=None,C50=None,error=read(folder/'generation.json'))
        dump(folder/'result.json',result);return result
    started=time.monotonic();model=SIMP(out/'inputs'/record['task_id']);history=[]
    cost=read(folder/'cost.json') if (folder/'cost.json').exists() else dict(attempted_fe=0,pending=False)
    def fe(x):
        cost['attempted_fe']+=1;cost['pending']=True;dump(folder/'cost.json',cost)
        ev=model.evaluate(x);cost['pending']=False;dump(folder/'cost.json',cost)
        return ev
    try:
        if (folder/'checkpoint.npz').exists():
            with np.load(folder/'checkpoint.npz',allow_pickle=False) as a:
                x=a['x'].copy();history=json.loads(str(a['history_json']))
            ev=fe(x)
        else:
            if arm=='U':x=model.initial()
            else:
                with np.load(folder/'raw_matrix.npz',allow_pickle=False) as a:x=decode(model,a['values'])
            ev=fe(x)
            history=[dict(step=0,**{k:float(ev[k]) for k in ('compliance','volume','residual','grayness')})]
            save_npz(folder/'state_0000.npz',x=x,density=ev['density'],u=ev['u'],energy=ev['energy'])
        for step in range(history[-1]['step']+1,51):
            if (out/'PAUSE_REQUESTED').exists():raise InterruptedError('Pause requested')
            x=model.update(x,ev);ev=fe(x)
            if abs(ev['volume']-model.target)>1e-8:raise ValueError('Volume constraint failed')
            history.append(dict(step=step,**{k:float(ev[k]) for k in ('compliance','volume','residual','grayness')}))
            if step%10==0:
                # A single atomic checkpoint also includes history for consistent recovery.
                save_npz(folder/'checkpoint.npz',x=x,history_json=json.dumps(history))
                dump(folder/'history.json',history)
        save_npz(folder/'state_0050.npz',x=x,density=ev['density'],u=ev['u'])
        dump(folder/'history.json',history)
        result=dict(status='complete',C0=history[0]['compliance'],C50=history[50]['compliance'],
                    volume0=history[0]['volume'],volume50=history[50]['volume'],max_residual=max(v['residual'] for v in history),
                    attempted_fe=cost['attempted_fe'],seconds=time.monotonic()-started)
    except InterruptedError:raise
    except Exception as exc:result=dict(status='failed',C0=history[0]['compliance'] if history else None,C50=None,error=str(exc))
    dump(folder/'result.json',result)
    return result

def explicit_task(task):
    task=dict(task)
    nx,ny=task['elements_xy']
    def position(node):
        return dict(u=(node//(ny+1))/nx,v=(ny-node%(ny+1))/ny)
    loads={}
    for dof,value in task['forces_sparse']:
        node=int(dof)//2
        loads.setdefault(node,dict(**position(node),fx=0.,fy=0.))['fx' if int(dof)%2==0 else 'fy']+=float(value)
    supports={}
    for dof in task['fixed_dofs']:
        node=int(dof)//2
        supports.setdefault(node,dict(**position(node),ux_fixed=False,uy_fixed=False))['ux_fixed' if int(dof)%2==0 else 'uy_fixed']=True
    task['explicit_loaded_nodes']=list(loads.values())
    task['explicit_support_nodes']=list(supports.values())
    task['dof_definition']='forces_sparse and fixed_dofs use DOF indices, NOT node indices. dof=2*node+axis; axis0=horizontal,axis1=vertical. node=x_index*(ny+1)+row_from_top. Prefer the explicit normalized coordinate lists supplied here.'
    return task
