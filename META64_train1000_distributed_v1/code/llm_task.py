"""Exact previously used per-task LLM protocol (fresh proposals, no donors)."""
import time
from pathlib import Path
import numpy as np
from initskill.io import dump,save_npz
import llm_six as base
import extensions as ext
read=base.read
LLM=tuple(base.ARMS)+('L1_PG','L1_H4','L2_GEOM','L3_FE')
def llm_job(out,cfg,client,r):
    tid=r['task_id'];root=out/'tasks'/tid;root.mkdir(parents=True,exist_ok=True)
    task=base.explicit_task(read(out/r['path']/'physical_input.json'));task.pop('source_root',None)
    base.generate_pair(out,client,r,task,'A')
    if not (root/'observation.json').exists():
        m=base.SIMP(out/r['path']);dump(root/'observation_cost.json',dict(attempted_fe=1,pending=True))
        ev=m.evaluate(m.initial());dump(root/'observation.json',base.observation(m,ev))
        save_npz(root/'observation_raw.npz',density=ev['density'],energy=ev['energy'],u=ev['u'])
        dump(root/'observation_cost.json',dict(attempted_fe=1,pending=False))
    base.generate_pair(out,client,r,task,'B',read(root/'observation.json'))
    for arm in base.ARMS:base.evaluate(out,r,arm)
    m=base.SIMP(out/r['path'])
    bases={a:ext.vector(out/'corpus'/tid/name/'state_0000.npz') for a,name in [('PG','PG2'),('H4','H4_B035')]}
    for kind,payload in [
        ('L1',dict(task=task,base_density={k:ext.coarse(v) for k,v in bases.items()},instruction='Propose ONE bounded logit-field edit for EACH base PG2 and H4_B035. Multiply existing logit field by strength, add up to4 circular Gaussian patches amplitude*exp(-distance_normalized^2/(2*radius^2)); exact target volume is recalibrated. Keep edits small and improve load paths.')),
        ('L2',dict(task=task,instruction='Propose 2..12 solid capsule members. Centers cx,cy in uW,vH; length and full width measured in H units, angles physical degrees counterclockwise from x. Host takes max of (width*H/2-distance_to_segment)/(.04H), scales by alpha and sigmoid-calibrates filtered volume. Avoid isolated rods.'))]:
        try:
            result=ext.ask(out,client,tid+'_'+kind,kind,payload)
            if kind=='L1':
                for key,arm in [('PG','L1_PG'),('H4','L1_H4')]:ext.save_start(out,tid,arm,ext.edit(m,bases[key],result[key]),dict(proposal=tid+'_L1',base=key,edit=result[key]))
            else:ext.save_start(out,tid,'L2_GEOM',ext.geometry_decode(m,dict(representation='components',parameters=dict(items=result['items'],alpha=result['alpha'],transition=.04))),result)
        except ValueError as e:dump(root/(kind+'_invalid.json'),dict(error=str(e)))
    parent=root/'L1_PG/start.npz'
    if parent.exists():
        x=ext.vector(parent);p=root/'feedback.json'
        if not p.exists():
            dump(root/'feedback_cost.json',dict(attempted_fe=1,pending=True));ev=m.evaluate(x);ob=base.observation(m,ev);ob.pop('full_element_relative_strain_energy',None)
            ob.update(source='Candidate L1_PG own step-zero FE; no OC update',C0=float(ev['compliance']),volume=float(ev['volume']),residual=float(ev['residual']),density=ext.coarse(ev['density']))
            dump(p,ob);dump(root/'feedback_cost.json',dict(attempted_fe=1,pending=False))
        try:
            result=ext.ask(out,client,tid+'_L3','L3',dict(task=task,parent_initializer=read(parent.parent/'initializer.json'),candidate_FE=read(p),instruction='Modify THIS candidate using its own FE feedback, not uniform FE. Return a bounded logit edit relative to the actual parent start. Inspect energy/deflection maps to decide where to redistribute density. Objective is C50; C0 alone is not a reliable proxy. Same exact-volume calibration. No OC steps were used.'))
            ext.save_start(out,tid,'L3_FE',ext.edit(m,x,result['edit']),result)
        except ValueError as e:dump(root/'L3_invalid.json',dict(error=str(e)))
    for arm in LLM[6:]:
        result=ext.trajectory(out,r,arm)
        if result is None:dump(root/arm/'result.json',dict(status='invalid',C0=None,C50=None,error='No valid start; no fallback'))
    dump(root/'complete.json',dict(task_id=tid,finished_unix=time.time()))
    return tid
