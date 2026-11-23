"""Frozen L1/L2/L3 only. No transfer, cross-task rules, POD or retrieval."""
import json,time,threading
from pathlib import Path
import numpy as np
from scipy.special import logit
from initskill.io import dump,save_npz
from initskill.initializers import calibrate,decode as geometry_decode
import llm_six as base
read=base.read
NUMERIC=threading.BoundedSemaphore(1)
def scalar(lo, hi): return dict(type='number', minimum=lo, maximum=hi)

def obj(p): return dict(type='object', properties=p, required=list(p), additionalProperties=False)

def arr(item, low, high): return dict(type='array',items=item,minItems=low,maxItems=high)

PATCH = obj(dict(u=scalar(0,1),v=scalar(0,1),radius=scalar(.06,.35),amplitude=scalar(-.8,.8)))

EDIT = obj(dict(strength=scalar(.7,1.3),patches=arr(PATCH,0,4)))

WARP = obj(dict(donor=dict(type='integer',minimum=0,maximum=2),dx=scalar(-.12,.12),dy=scalar(-.12,.12),strength=scalar(.35,1),patches=arr(PATCH,0,4)))

CAPSULE = obj(dict(cx=scalar(0,1),cy=scalar(0,1),length=scalar(.1,3),width=scalar(.04,.35),angle_degrees=scalar(-180,180)))

VECTOR = arr(scalar(-2,2),8,8)

RULE = obj(dict(volume_threshold=scalar(.2,.6),low_base=dict(type='string',enum=['PG2','H4_B035']),high_base=dict(type='string',enum=['PG2','H4_B035']),strength_intercept=scalar(.7,1.3),strength_volume_slope=scalar(-1,1),load_patch_amplitude=scalar(-.8,.8),load_patch_radius=scalar(.06,.35)))

SCHEMAS = dict(L1=obj(dict(PG=EDIT,H4=EDIT,reason=base.TEXT)),L2=obj(dict(items=arr(CAPSULE,2,12),alpha=scalar(.2,1),reason=base.TEXT)),L3=obj(dict(edit=EDIT,reason=base.TEXT)),L4=obj(dict(warp=WARP,reason=base.TEXT)),L6=obj(dict(rule=RULE,reason=base.TEXT)),L7=obj(dict(coefficients=VECTOR,reason=base.TEXT)))

def validate(value, spec):
    t=spec['type']
    if t=='object':
        if not isinstance(value,dict) or set(value)!=set(spec['properties']): raise ValueError('Object contract')
        for k,s in spec['properties'].items(): validate(value[k],s)
    elif t=='array':
        if not isinstance(value,list) or not spec['minItems']<=len(value)<=spec['maxItems']: raise ValueError('Array contract')
        for v in value: validate(v,spec['items'])
    elif t in ('number','integer'):
        if type(value) not in (int,float) or not np.isfinite(value) or not spec['minimum']<=value<=spec['maximum']: raise ValueError('Numeric range')
        if t=='integer' and type(value)!=int: raise ValueError('Integer required')
    elif not isinstance(value,str): raise ValueError('String required')
    if 'enum' in spec and value not in spec['enum']: raise ValueError('Enum')

def vector(path,key='x'):
    with np.load(path,allow_pickle=False) as a: return a[key].copy()

def field(x): return logit(np.clip(x,.02,.98))

def coarse(values,ny=64,nx=64):
    a=np.asarray(values).reshape((ny,nx),order='F')
    if (ny,nx)!=(64,64):raise ValueError('Expected 64x64')
    return np.round(a.reshape(8,8,8,8).mean(axis=(1,3)),3).tolist()

def edit(model,x,p,transfer_strength=False):
    spec=obj(dict(strength=scalar(.35,1),patches=arr(PATCH,0,4))) if transfer_strength else EDIT
    validate(p,spec)
    f=field(x)*p['strength'];u=model.centers[:,0]/model.nx;v=model.centers[:,1]/model.ny
    for q in p['patches']:
        f=f+q['amplitude']*np.exp(-((u-q['u'])**2+(v-q['v'])**2)/(2*q['radius']**2))
    return calibrate(model,f)

def ask(out,client,job,kind,payload):
    msgs=[dict(role='system',content='You design a NONUNIFORM initial material density for isotropic SIMP/OC compliance minimization AFTER exactly 50 updates. Return only the schema JSON. The host enforces exact filtered volume. Coordinates u=x/W and v=y/H, y up. Thumbnail row0 top. No fabricated FE claims. Propose physically plausible load-to-support connections.'),dict(role='user',content=json.dumps(payload,ensure_ascii=False,separators=(',',':')))]
    schema=SCHEMAS[kind];contract=dict(type='json_schema',json_schema=dict(name='initializer',strict=True,schema=schema))
    f=out/'proposals'/f'{job}.json'
    if f.exists():return read(f)
    result=client.call(job,msgs,contract)
    try:validate(result,schema)
    except Exception as e:
        dump(out/'invalid_proposals'/f'{job}.json',dict(value=result,error=str(e)));raise ValueError('Invalid '+job+': '+str(e))
    dump(f,result);return result

def save_start(out,tid,arm,x,meta):
    p=out/'tasks'/tid/arm
    if (p/'start.npz').exists():
        if not np.array_equal(vector(p/'start.npz'),x):raise ValueError('Changed initializer')
        return
    save_npz(p/'start.npz',x=x);dump(p/'initializer.json',meta)

def fe(model,x):
    with NUMERIC:return model.evaluate(x)

def trajectory(out,r,arm):
    p=out/'tasks'/r['task_id']/arm
    if (p/'result.json').exists():return read(p/'result.json')
    if not (p/'start.npz').exists():return None
    m=base.SIMP(out/'inputs'/r['task_id']);start=time.monotonic();history=[]
    # Commit each FE cost before evaluation, so interrupted work remains visible.
    cost=read(p/'cost.json') if (p/'cost.json').exists() else dict(attempted_fe=0)
    def solve(x):
        cost['attempted_fe']+=1;cost['pending']=True;dump(p/'cost.json',cost);v=fe(m,x);cost['pending']=False;dump(p/'cost.json',cost);return v
    try:
        if (p/'checkpoint.npz').exists():
            with np.load(p/'checkpoint.npz',allow_pickle=False) as a:x=a['x'].copy();history=json.loads(str(a['history_json']))
            ev=solve(x)
        else:
            x=vector(p/'start.npz');ev=solve(x);history=[dict(step=0,**{k:float(ev[k]) for k in ('compliance','volume','residual','grayness')})];save_npz(p/'state_0000.npz',x=x,density=ev['density'],u=ev['u'])
        for step in range(history[-1]['step']+1,51):
            if (out/'PAUSE_REQUESTED').exists():raise InterruptedError('Paused')
            x=m.update(x,ev);ev=solve(x)
            if abs(ev['volume']-m.target)>1e-8:raise ValueError('Volume constraint')
            history.append(dict(step=step,**{k:float(ev[k]) for k in ('compliance','volume','residual','grayness')}))
            if step in (10,30,50):save_npz(p/f'state_{step:04d}.npz',x=x,density=ev['density'],u=ev['u']);save_npz(p/'checkpoint.npz',x=x,history_json=json.dumps(history));dump(p/'history.json',history)
        result=dict(status='complete',C0=history[0]['compliance'],C10=history[10]['compliance'],C50=history[50]['compliance'],updates=50,attempted_fe=cost['attempted_fe'],seconds=time.monotonic()-start,max_volume_error=max(abs(h['volume']-m.target) for h in history),max_residual=max(h['residual'] for h in history))
    except InterruptedError:raise
    except Exception as e:result=dict(status='failed',error=type(e).__name__+': '+str(e),C0=history[0]['compliance'] if history else None,C50=None)
    dump(p/'result.json',result);return result
