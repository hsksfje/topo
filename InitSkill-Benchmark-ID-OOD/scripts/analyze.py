"""Independent final analyzer. Refuses incomplete coverage; never trains models.

It reads only settled evaluation results and frozen model identities. Method
implementations and the fixed execution protocol remain unchanged.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/'src'))
from initskill.methods.benchmark_init_v1.scope import current_scope
METHODS=tuple(current_scope()['methods'])
SUBSETS=tuple(current_scope()['test_subsets'])
DATA=ROOT/'data/META64_uniform_1000_100_300_v3/manifests'


def read(path):return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def ledger_count(path):
    path=Path(path)
    if not path.exists():return 0
    with path.open() as stream:return sum(1 for _ in stream)


def failure_detail(base,record):
    """Classify observed failure evidence without inventing a compliance."""
    base=Path(base);error=record['error'] or ''
    detail=dict(subset=record['subset'],method=record['method'],task_key=record['task_key'],
        error=error,trajectory_FE=record['trajectory_FE'],initialization_FE=record['initialization_FE'])
    ledger=base/'FE_attempts.jsonl'
    last=None
    if ledger.exists():
        with ledger.open() as stream:
            for line in stream:
                if line.strip():last=json.loads(line)
    step=last.get('step') if last else None
    detail['last_attempted_step']=step
    detail['failure_phase']=('initial_FE' if step==0 else 'OC_trajectory' if step is not None else 'prediction_or_setup')
    detail['kind']=('FE_acceptance_rejection' if error=='FE equilibrium or positive-compliance check failed'
        else 'requires_individual_review')
    init_path=base/'initialization.json';failure_path=base/'failure.json'
    init=read(init_path) if init_path.exists() else {}
    failure=read(failure_path)
    trace=failure.get('traceback','')
    detail['inline_refinement_entered_evidenced']=bool('precision_fem.py' in trace and 'in refined_evaluation' in trace)
    detail['inline_refinement_evidence_note']=('Failure traceback records entering refined_evaluation; correction count and terminal residual were not retained, so they remain unknown.'
        if detail['inline_refinement_entered_evidenced'] else 'Not established from the preserved failure traceback; this does not prove no inline refinement was attempted.')
    detail['known_initialization_seconds']=init.get('initialization_seconds')
    detail['known_failed_trajectory_seconds']=failure.get('seconds')
    detail['failure_timing_note']='Missing failed-trajectory elapsed time is unknown, not zero; the historical exception wrapper may retain FE attempts but overwrite the inner timing field.'
    for name in ('identity.json','initialization.json','initialization.npz','failure.json'):
        p=base/name
        if p.exists():detail[name+'_sha256']=digest(p)
    replay=ROOT/'research_results/benchmark_init_v1/verification/initial_failure_replay'/record['subset']/record['method']/base.name/'diagnosis.json'
    if replay.exists():
        observed=read(replay)
        if (observed['historical_initialization_sha256']==detail.get('initialization.npz_sha256') and
            observed['historical_failure_sha256']==detail.get('failure.json_sha256') and
            observed['task_key']==record['task_key']):
            detail['separate_initial_replay_diagnostic']=dict(path=str(replay.relative_to(ROOT)),
                sha256=digest(replay),status=observed['replay_status'],
                terminal_residual=observed.get('terminal_refinement_residual'),
                correction_solves_completed=observed.get('correction_solves_completed'),
                scope='New one-FE current-backend diagnostic; original outcome/timing/residual gaps retained, not formal recovery.')
    p=base/'initialization.npz'
    if p.exists():
        with np.load(p,allow_pickle=False) as arrays:
            for name in ('x','density'):
                if name not in arrays:continue
                values=arrays[name]
                finite=bool(np.isfinite(values).all())
                detail[name+'_finite']=finite
                if finite:detail[name+'_summary']=dict(min=float(values.min()),max=float(values.max()),
                    mean=float(values.mean()),fraction_below_1e_minus6=float(np.mean(values<1e-6)))
    return detail


def verify_trajectory_physics(base,result,row,method,subset,cap):
    """Independently check every recorded state and the actual final arrays."""
    identity=result['identity'];history=result['history']
    if identity['method']!=method or identity['subset']!=subset:
        raise ValueError('Method/split identity mismatch: '+str(base))
    if result['stop_step']!=history[-1]['step'] or not 200<=result['stop_step']<=cap:
        raise ValueError('Final step/budget mismatch: '+str(base))
    if result['status']=='max_steps' and result['stop_step']!=cap:
        raise ValueError('Premature budget termination: '+str(base))
    if result['converged']!=(result['status']=='converged'):
        raise ValueError('Inconsistent convergence flag: '+str(base))
    target=row['volume_fraction']
    for record in history:
        C=record['compliance'];volume=record['volume'];residual=record['residual']
        if not (np.isfinite(C) and C>0 and np.isfinite(volume) and abs(volume-target)<=1e-8
                and np.isfinite(residual) and 0<=residual<1e-6):
            raise ValueError('Recorded FE/volume acceptance failed: '+str(base)+' step'+str(record['step']))
        dx=record['max_dx']
        if record['step'] and (dx is None or not np.isfinite(dx) or dx<0):
            raise ValueError('Missing or invalid optimizer change: '+str(base))
    if result['status']=='converged':
        for i in range(len(history)-10,len(history)):
            recent=np.asarray([r['compliance'] for r in history[i-10:i+1]])
            if not (len(recent)==11 and history[i]['max_dx']<=.01
                    and abs(recent[-1]-recent[-2])/max(abs(recent[-2]),1e-30)<=1e-4
                    and np.ptp(recent)/max(abs(recent[-1]),1e-30)<=5e-4):
                raise ValueError('Claimed convergence lacks10 stable steps: '+str(base))
    for step in (50,100,200):
        if result['budget_points'][str(step)]!=history[step]['compliance']:
            raise ValueError('Fixed-budget point differs from trajectory: '+str(base))
    if result['final_compliance']!=history[-1]['compliance']:
        raise ValueError('Final compliance differs from trajectory: '+str(base))
    path=base/'state_final.npz'
    with np.load(path,allow_pickle=False) as arrays:
        x=arrays['x'];rho=arrays['density'];shape=(row['nx']*row['ny'],)
        if x.shape!=shape or rho.shape!=shape or not (np.isfinite(x).all() and np.isfinite(rho).all()):
            raise ValueError('Invalid final density arrays: '+str(base))
        if np.min(x)<1e-6-1e-12 or np.max(x)>1+1e-12 or np.min(rho)<1e-6-1e-12 or np.max(rho)>1+1e-12:
            raise ValueError('Final design/material bounds failed: '+str(base))
        if abs(float(rho.mean())-history[-1]['volume'])>1e-12:
            raise ValueError('Final density volume differs from trajectory: '+str(base))
    return digest(path)


def verify_pair(out,subset,row):
    path=out/'pairs'/subset/(row['source_task_id']+'.npz');meta=read(path.with_suffix('.json'))
    low=out/'labels'/subset/(row['source_task_id']+'.npz')
    if (meta['task_key']!=row['task_key'] or meta['group_id']!=row['group_id'] or
        meta['source_split']!=subset or meta['sha256']!=digest(path) or meta['low_label_sha256']!=digest(low)):
        raise ValueError('Paired teacher identity drift')
    base=path.parent/'high_runs'/row['source_task_id'];result=read(base/'result.json')
    expected=dict(task_key=row['task_key'],group_id=row['group_id'],source_split=subset,
        max_steps=read(out/'protocol.json')['oc_max_steps'],low_label_sha256=digest(low),teacher='uniform256-'+read(out/'protocol.json')['linear_solver_backend'].split('-')[0]+'-common-OC')
    if result['identity']!=expected or meta['high_teacher']!=result:
        raise ValueError('Paired teacher trajectory identity drift')
    if [r['step'] for r in result['history']]!=list(range(result['stop_step']+1)):
        raise ValueError('Paired teacher trajectory incomplete')
    backend=read(out/'protocol.json')['linear_solver_backend']
    if any(r.get('linear_solver')!=backend for r in result['history']):
        raise ValueError('Mixed paired teacher solver backend')
    if ledger_count(base/'FE_attempts.jsonl')!=result['FE']:raise ValueError('Paired teacher FE ledger drift')
    # The numerical checker also serves paired teachers after their original
    # (different-schema) identity has been checked in full immediately above.
    checked=dict(result,identity=dict(expected,method='paired_teacher',subset=subset))
    state_sha=verify_trajectory_physics(base,checked,dict(row,nx=256,ny=256),'paired_teacher',subset,read(out/'protocol.json')['oc_max_steps'])
    with np.load(path,allow_pickle=False) as pair,np.load(low,allow_pickle=False) as a,np.load(base/'state_final.npz',allow_pickle=False) as b:
        if pair['low'].shape!=(64,64) or pair['high'].shape!=(256,256):raise ValueError('Pair resolution drift')
        if pair['low'].ravel(order='F').tobytes()!=a['x'].tobytes():raise ValueError('Pair differs from committed low teacher')
        if pair['high'].ravel(order='F').tobytes()!=b['x'].tobytes():raise ValueError('Pair differs from independently optimized high state')
    return dict(task_key=row['task_key'],group_id=row['group_id'],pair_sha256=meta['sha256'],
        low_teacher_sha256=meta['low_label_sha256'],high_final_state_sha256=state_sha,
        status=result['status'],stop_step=result['stop_step'],FE=result['FE'])


def rows(subset):return [json.loads(x) for x in (DATA/(subset+'.jsonl')).read_text().splitlines()]


def bootstrap(values,seed=20261126,repeats=2000):
    values=np.asarray(values,dtype=float)
    if not len(values):return None
    rng=np.random.default_rng(seed);means=[]
    for _ in range(repeats):means.append(float(np.mean(rng.choice(values,len(values),replace=True))))
    return [float(x) for x in np.quantile(means,[.025,.975])]


def model_audit(out):
    training={x['task_key'] for x in rows('train1000')};validation={x['task_key'] for x in rows('valid100')}
    details=[]
    for method,folder in [('knn',out/'models/knn'),('ulu',out/'models/ulu'),('wang',out/'models/wang'),
                          ('cnn',out/'models/cnn'),('meta',out/'models/meta_native'),
                          ('early',out/'early_trajectories/models/meta_native'),('cgan',out/'models/cgan')]:
        if method not in METHODS:continue
        f=read(folder/'frozen.json')
        if f.get('smoke_only'):raise ValueError('Smoke model in formal benchmark: '+method)
        model=folder/('model.npz' if method=='knn' else 'learned.eqx' if method in {'meta','early'} else 'best.pt')
        if digest(model)!=f['model_sha256']:raise ValueError('Changed model: '+method)
        if method=='knn':
            if f['train_count']!=1000 or f['manifest_sha256']!=digest(DATA/'train1000.jsonl'):raise ValueError('KNN split identity')
        elif method!='cgan':
            cfg=read(folder/'config.json')
            if (set(cfg['train_keys'])!=training or set(cfg['valid_keys'])!=validation or
                len(cfg['train_keys'])!=1000 or len(cfg['valid_keys'])!=100):raise ValueError('Split drift in '+method)
            if digest(folder/'config.json')!=f['config_sha256']:raise ValueError('Config drift in '+method)
            if method in {'ulu','wang','cnn'}:
                history=read(folder/'history.json')
                cap_epochs=500 if method=='cnn' else 1000
                patience=30 if method=='cnn' else 6
                key='valid_MAE' if method=='cnn' else 'validation_MSE'
                if cfg['epochs']!=cap_epochs or not history or [h['epoch'] for h in history]!=list(range(1,len(history)+1)) or len(history)>cap_epochs:
                    raise ValueError('Supervised training epochs incomplete: '+method)
                values=[h[key] for h in history]
                if not np.isfinite(values).all():raise ValueError('Nonfinite supervised validation history')
                selected=int(np.argmin(values))+1
                if len(history)<cap_epochs and len(history)-selected<patience:raise ValueError('Supervised stage ended before its declared stopping rule: '+method)
                if method=='cnn' and cfg['batch_size']!=16:raise ValueError('CNN batch size drift')
                if method in {'ulu','wang'} and cfg['rank']!=(80 if method=='ulu' else 200):raise ValueError('PCA rank drift')
            if method in {'meta','early'}:
                if cfg['outer_steps']!=6000 or cfg['batch_size']!=5 or cfg['inner_steps']!=(10 if method=='meta' else 5):
                    raise ValueError('Incomplete or changed native training budget: '+method)
                history=read(folder/'history.json')
                if [r['step'] for r in history]!=list(range(1,6001)):raise ValueError('Native training updates incomplete')
                if [r['step'] for r in history if 'validation_loss' in r]!=list(range(200,6001,200)):
                    raise ValueError('Native validation schedule incomplete')
                for epoch in range(30):
                    block=history[epoch*200:(epoch+1)*200]
                    indices=[i for row in block for i in row['task_indices']]
                    if sorted(indices)!=list(range(1000)) or any(row['task_epoch']!=epoch or len(row['task_indices'])!=5 for row in block):
                        raise ValueError('Native full-manifest sampling drift')
                pointer=read(folder/'resume_commit.json');state_folder=folder/pointer['slot']
                if pointer['step']!=6000:raise ValueError('Native final state not committed')
                for name,expected_sha in pointer['files'].items():
                    if digest(state_folder/name)!=expected_sha:raise ValueError('Native checkpoint integrity failed')
                state=read(state_folder/'state.json')
                attempted=sum(1 for _ in (folder/'FE_attempts.jsonl').open())
                if state['FE']!=attempted or f['FE']!=attempted:raise ValueError('Native attempted FE ledger drift')
                expected_validation_FE=(cfg['inner_steps']+2)*100
                if state['FE']-history[-1]['FE']!=expected_validation_FE:raise ValueError('Native last validation FE incomplete')
                if state['sampler']['cursor']!=1000 or state['sampler']['epoch']!=29:raise ValueError('Native sampler final state drift')
        else:
            if digest(folder/'config.json')!=f['config_sha256']:raise ValueError('Config drift in cgan')
            cfg=read(folder/'config.json')
            if len(cfg['train_keys'])!=1000 or set(cfg['train_keys'])!=training or len(cfg['valid_keys'])!=100 or set(cfg['valid_keys'])!=validation:
                raise ValueError('cGAN training/validation manifest drift')
            if cfg['epochs']!=20 or cfg['batch_size']!=16:raise ValueError('cGAN declared training budget drift')
            if [r['epoch'] for r in read(folder/'history.json')]!=list(range(1,21)):
                raise ValueError('cGAN full20 epochs incomplete')
            for subset,allowed in [('train1000',training),('valid100',validation)]:
                provenance=cfg['training_data' if subset=='train1000' else 'validation_data']
                if provenance['manifest_sha256']!=digest(DATA/(subset+'.jsonl')):raise ValueError('cGAN training provenance manifest drift')
                saved={r['task_key']:r for r in provenance['records']}
                if len(provenance['records'])!=len(allowed) or set(saved)!=allowed:raise ValueError('cGAN pair provenance coverage')
                pairmeta=[]
                for source in rows(subset):
                    item=verify_pair(out,subset,source)
                    if any(saved[source['task_key']][key]!=item[key] for key in ('task_key','group_id','pair_sha256','low_teacher_sha256')):
                        raise ValueError('cGAN trained pair differs from current teacher')
                    pairmeta.append(item['task_key'])
                if set(pairmeta)!=allowed:raise ValueError('cGAN pair coverage')
        health=None
        if 'health_report_sha256' in f:
            if digest(folder/'training_health.json')!=f['health_report_sha256']:raise ValueError('Changed training health report: '+method)
            health=read(folder/'training_health.json')
        details.append(dict(method=method,model=str(model.relative_to(out)),sha256=digest(model),frozen=f,training_health=health))
    return details


def offline_cost_audit(out):
    inherited=[];new_labels=[];pairs=[]
    bundled_path=ROOT/'data/historical_teacher_costs.json'
    bundled={r['task_key']:r for r in read(bundled_path)['records']} if bundled_path.exists() else {}
    for row in rows('train1000'):
        meta=out/'labels/train1000'/(row['source_task_id']+'.json')
        if not meta.exists():continue
        label=read(meta);original=label.get('origin',{}).get('original_record',{})
        phases=[]
        for key in ('oc50_dir','convergence_dir'):
            folder=original.get(key)
            cost=ROOT/folder/'cost.json' if folder else None
            if cost and cost.exists():
                item=read(cost);n=item.get('attempted_FE',item.get('attempted_fe'))
                phases.append(dict(phase=key,FE=n,cost_sha256=digest(cost),path=str(cost.relative_to(ROOT))))
            else:phases.append(dict(phase=key,FE=None))
        if row['task_key'] in bundled:phases=bundled[row['task_key']]['phases']
        inherited.append(dict(task_key=row['task_key'],phases=phases,
            audited_label_FE=label['FE'],teacher_status=label['origin'].get('status')))
    for subset in ('train1000','valid100'):
        for row in rows(subset):
            p=out/'labels'/subset/(row['source_task_id']+'.json')
            if p.exists() and subset=='valid100':
                m=read(p);new_labels.append(dict(task_key=row['task_key'],FE=m.get('FE'),seconds=m.get('seconds')))
            p=out/'pairs'/subset/(row['source_task_id']+'.json')
            if p.exists():
                m=read(p);pairs.append(dict(task_key=row['task_key'],subset=subset,
                    high_resolution_FE=m['high_teacher']['FE'],high_teacher_status=m['high_teacher']['status']))
    train_cost=[]
    for method,folder in [('knn',out/'models/knn'),('ulu',out/'models/ulu'),('wang',out/'models/wang'),
                          ('cnn',out/'models/cnn'),('meta',out/'models/meta_native'),
                          ('early',out/'early_trajectories/models/meta_native'),('cgan',out/'models/cgan')]:
        if method not in METHODS:continue
        f=folder/'frozen.json'
        if not f.exists() and not (folder/'config.json').exists():continue
        meta=read(f) if f.exists() else {}
        attempted=folder/'FE_attempts.jsonl'
        native_FE=(ledger_count(attempted) if attempted.exists() else meta.get('FE'))
        cost=dict(method=method,training_status='frozen' if f.exists() else 'not_frozen',
            training_physical_FE=native_FE if method in {'meta','early'} else 0)
        h=folder/'history.json'
        if h.exists():
            history=read(h)
            cost['recorded_last_training_seconds']=history[-1].get('seconds') if history else None
            cost['actual_last_epoch_or_outer_step']=(history[-1].get('epoch',history[-1].get('step')) if history else None)
        cfg=folder/'config.json'
        if cfg.exists():
            config=read(cfg)
            cost['runtime']=config.get('runtime')
            cost['runtime_note']='null means runtime was not embedded in this historical model config; see stage command and environment artifacts'
            cost['declared_max_epochs']=config.get('epochs')
            cost['declared_outer_steps']=config.get('outer_steps')
            cost['declared_inner_steps']=config.get('inner_steps')
            cost['train_manifest_task_count']=len(config['train_keys']) if 'train_keys' in config else None
            cost['validation_manifest_task_count']=len(config['valid_keys']) if 'valid_keys' in config else None
            cost['reference_preparation_seconds_this_invocation']=config.get('reference_preparation_seconds_this_invocation')
            cost['artifact_elapsed_seconds']=f.stat().st_mtime-cfg.stat().st_mtime if f.exists() else None
            cost['artifact_elapsed_note']='filesystem elapsed interval, not measured CPU/GPU time'
        progress=folder/'progress.json'
        if progress.exists():
            cost['recorded_training_loop_seconds']=read(progress).get('seconds')
        resume=folder/'resume_event.json'
        if resume.exists():cost['native_resume_overhead']=read(resume)
        stage=out/'campaign'/(method+'_train.complete.json')
        if stage.exists():
            cost['training_stage_command']=read(stage).get('command')
        cost['timing_scope_note']='Training-loop and artifact intervals may exclude PCA factorization, initial reference FE, imports and model construction; they are not total cold-start compute time.'
        train_cost.append(cost)
    known=[p['FE'] for row in inherited for p in row['phases'] if p['FE'] is not None]
    unknown=sum(p['FE'] is None for row in inherited for p in row['phases'])
    teacher_attempts=[]
    for p in (out/'pairs').glob('*/high_runs/*/FE_attempts.jsonl'):
        teacher_attempts.append(dict(path=str(p.relative_to(out)),attempted_FE=ledger_count(p),
            result_committed=(p.parent/'result.json').exists(),failure_recorded=(p.parent/'failure.json').exists()))
    original=ROOT
    engineering=[]
    for p in (ROOT/'verification').rglob('FE_attempts.jsonl'):
        engineering.append(dict(path=str(p.relative_to(original)),attempted_FE=ledger_count(p)))
    retirement=out/'CPU_DIAGNOSTIC_RETIREMENT.json'
    timing_snapshot=out/'audit/native_reference_preparation_timing_20261126/summary.json'
    reference_timing=None
    if timing_snapshot.exists():
        reference_timing=read(timing_snapshot)
        phases=reference_timing['invocations']
        assert len({r['resumed_from_step'] for r in phases})==len(phases)
        for phase in phases:
            preserved=timing_snapshot.parent/phase['preserved_config']
            assert digest(preserved)==phase['preserved_config_sha256']
            cfg=read(preserved)
            assert cfg.get('resumed_from_step',0)==phase['resumed_from_step']
            assert cfg.get('reference_preparation_seconds_this_invocation')==phase['known_reference_preparation_seconds']
        assert sum(r['known_reference_preparation_seconds'] for r in phases if r['known_reference_preparation_seconds'] is not None)==reference_timing['known_measured_reference_preparation_seconds']
        assert sum(r['known_reference_preparation_seconds'] is None for r in phases)==reference_timing['unknown_reference_preparation_phases']
        reference_timing=dict(reference_timing,source=str(timing_snapshot.relative_to(out)),source_sha256=digest(timing_snapshot))
    return dict(shared_training_library=dict(tasks=len(inherited),known_historical_teacher_FE=sum(known),
        unknown_cost_phases=unknown,new_teacher_audit_FE=sum(x['audited_label_FE'] for x in inherited),
        note='Shared teacher construction is counted once globally; it is a prerequisite for each standalone supervised method, not zero training cost'),
        validation_teacher_tasks=len(new_labels),validation_teacher_FE=sum(x['FE'] for x in new_labels if x['FE'] is not None),
        high_resolution_teacher_tasks=len(pairs),high_resolution_teacher_FE=sum(x['high_resolution_FE'] for x in pairs),
        model_training=train_cost,native_reference_timing_snapshot=reference_timing,
        inherited_teacher_records=inherited,validation_teacher_records=new_labels,
        high_resolution_teacher_records=pairs,high_resolution_teacher_attempt_records=teacher_attempts,
        high_resolution_teacher_attempted_FE_including_inflight_and_failures=sum(r['attempted_FE'] for r in teacher_attempts),
        retired_CPU_diagnostics=read(retirement) if retirement.exists() else None,
        engineering_verification_FE_ledger_records=engineering,
        engineering_verification_known_FE=sum(r['attempted_FE'] for r in engineering),
        engineering_cost_note='Known explicit FE ledgers only; diagnostic scripts without a ledger have unmeasured additional cost. These verification ledgers are separate from formal model/evaluation and label ledgers; do not sum overlapping archived checkpoint copies.')


def dataset_roles_audit():
    """Prove the report compares held-out tests, not the training library."""
    names=('train1000','valid100')+SUBSETS
    source={name:rows(name) for name in names}
    for name,records in source.items():
        assert len(records)==(1000 if name=='train1000' else 100 if name=='valid100' else 300)
        assert len({r['task_key'] for r in records})==len(records)
    overlap=[]
    for fitted in ('train1000','valid100'):
        for test in SUBSETS:
            counts={key:len({r[key] for r in source[fitted]} & {r[key] for r in source[test]})
                    for key in ('task_key','group_id')}
            assert all(value==0 for value in counts.values()),(fitted,test,counts)
            overlap.append(dict(fitting_split=fitted,test_split=test,overlap=counts))
    return dict(passed=True,performance_comparison_subsets=list(SUBSETS),
        roles=dict(train1000='Fit models/retrieval library; no training performance comparison',
                   valid100='Validation/model selection; no validation performance comparison',
                   test_id300='Held-out ID performance comparison',
                   test_ood300='Held-out OOD performance comparison',
                   test_cr300='Held-out cross-resolution performance comparison'),
        manifests={name:dict(tasks=len(records),sha256=digest(DATA/(name+'.jsonl')),
                            sample_task_keys=[r['task_key'] for r in records[:2]])
                   for name,records in source.items()},fitting_test_overlap_checks=overlap,
        denominator_note='The300 denominator in each performance row is that test manifest size. Train1000 is a separate fitting library. Teacher generation is training preparation/cost, not a train-set performance comparison.')


def analyze(out,allow_partial=False):
    out=Path(out);protocol=read(out/'protocol.json');cap=protocol['oc_max_steps']
    if protocol.get('execution_scope')!=current_scope():raise ValueError('Analysis scope differs from executed protocol; use benchmark.py analyze with the matching cross-resolution flag')
    if cap!=500:raise ValueError('Requested OC cap is500')
    data_roles=dataset_roles_audit()
    for name,expected in protocol['manifest_sha256'].items():
        if digest(DATA/(name+'.jsonl'))!=expected:raise ValueError('Selected manifest changed: '+name)
    code=ROOT/'src/initskill/methods/benchmark_init_v1'
    extra={'@platform_physics.py':ROOT/'src/initskill/physics.py','@portable_locking.py':ROOT/'src/initskill/locking.py'}
    for name,expected in protocol['source_sha256'].items():
        path=extra[name] if name in extra else code/name
        if digest(path)!=expected:raise ValueError('Executed numerical source changed: '+name)
    # Versioned implementation corrections retain exact source/protocol
    # snapshots. Normative task/solver/budget settings must remain identical.
    normative=lambda p:{k:v for k,v in p.items() if k!='source_sha256'}
    accepted_protocols={digest(out/'protocol.json')}
    for snapshot in (out/'audit').glob('*/protocol.*.json'):
        previous=read(snapshot)
        if normative(previous)!=normative(protocol):raise ValueError('Normative protocol drift: '+str(snapshot))
        accepted_protocols.add(digest(snapshot))
    audit=[];task_metrics=[];coverage=[];failures=[]
    for subset in SUBSETS:
        source=rows(subset)
        if len(source)!=300:raise ValueError('Expected complete sampled300-task test manifest')
        for method in METHODS:
            settled=0;failed=0;metrics=[];missing=[]
            for row in source:
                base=out/'full_evaluation'/subset/method/row['source_task_id']
                p=base/'result.json';f=base/'failure.json'
                if p.exists() and f.exists():raise ValueError('Ambiguous result and failure: '+str(base))
                if not p.exists() and not f.exists():missing.append(row['task_key']);continue
                settled+=1
                if f.exists():
                    failed+=1;record=dict(subset=subset,method=method,task_key=row['task_key'],group_id=row['group_id'],status='failed',error=read(f).get('error'))
                    identity_path=base/'identity.json'
                    if identity_path.exists():
                        failed_identity=read(identity_path)
                        if (failed_identity['task_key']!=row['task_key'] or failed_identity['group_id']!=row['group_id'] or
                            failed_identity['method']!=method or failed_identity['subset']!=subset or failed_identity['max_steps']!=cap or
                            failed_identity['protocol_sha256'] not in accepted_protocols):raise ValueError('Failed execution identity drift')
                        record['protocol_sha256']=failed_identity['protocol_sha256']
                    ledger=base/'FE_attempts.jsonl'
                    count=ledger_count(ledger)
                    init=read(base/'initialization.json') if (base/'initialization.json').exists() else {}
                    # Failed native prediction may not have written a handoff,
                    # but its attempted reference solves still have a ledger.
                    model_out=out if method!='early' else out/'early_trajectories'
                    external=model_out/'costs/meta_native'/row['task_key']/'FE_attempts.jsonl'
                    external_count=ledger_count(external) if method in {'meta','early'} else 0
                    record.update(trajectory_FE=count,initialization_FE=max(external_count,init.get('initialization_FE',0)),
                                  online_FE=count+max(external_count,init.get('initialization_FE',0)))
                    detail=failure_detail(base,record)
                    record['inline_refinement_failed_evidenced']=detail['inline_refinement_entered_evidenced']
                    failures.append(detail)
                else:
                    result=read(p);identity=result['identity']
                    if identity['task_key']!=row['task_key'] or identity['group_id']!=row['group_id']:raise ValueError('Evaluation task mismatch')
                    if identity['protocol_sha256'] not in accepted_protocols:raise ValueError('Unknown execution protocol identity')
                    if identity['max_steps']!=cap:raise ValueError('Different full optimization budget')
                    history=result['history'];steps=[r['step'] for r in history]
                    if any(r.get('linear_solver')!=protocol['linear_solver_backend'] for r in history):
                        raise ValueError('Mixed or unrecorded formal GPU FE backend: '+str(base))
                    if steps!=list(range(steps[-1]+1)):raise ValueError('Non-contiguous optimization trajectory')
                    if not all(k in result['budget_points'] for k in ('50','100','200')):raise ValueError('Missing common fixed-budget result')
                    if result['status'] not in {'converged','max_steps'}:raise ValueError('Unknown final state')
                    final_state_sha=verify_trajectory_physics(base,result,row,method,subset,cap)
                    init=read(base/'initialization.json')
                    ledger=ledger_count(base/'FE_attempts.jsonl')
                    if result['FE']!=ledger:raise ValueError('FE ledger mismatch')
                    external_count=0
                    if method in {'meta','early'}:
                        model_out=out if method=='meta' else out/'early_trajectories'
                        external_count=ledger_count(model_out/'costs/meta_native'/row['task_key']/'FE_attempts.jsonl')
                    initialization_FE=max(external_count,init.get('initialization_FE',0))
                    record=dict(subset=subset,method=method,task_key=row['task_key'],group_id=row['group_id'],status=result['status'],
                        stop_step=result['stop_step'],final_compliance=result['final_compliance'],
                        trajectory_FE=ledger,initialization_FE=initialization_FE,
                        initialization_FE_recorded_handoff=init.get('initialization_FE',0),
                        initialization_retry_FE=max(0,external_count-init.get('initialization_FE',0)),
                        online_FE=ledger+initialization_FE,online_seconds=result['seconds']+init['initialization_seconds'],
                        final_state_sha256=final_state_sha,result_sha256=digest(p),
                        protocol_sha256=identity['protocol_sha256'],
                        inline_refinement_accepted_FE=sum(bool(h.get('precision')) for h in history),
                        inline_refinement_accepted_correction_solves=sum(h.get('refinement_steps',0) for h in history),
                        C50=result['budget_points']['50'],C100=result['budget_points']['100'],C200=result['budget_points']['200'])
                ref=out/'full_evaluation'/subset/'uniform'/row['source_task_id']/'result.json'
                record['target_available']=ref.exists()
                retry=base/'numerical_retry.json'
                record['numerical_recovery_status']=read(retry).get('status') if retry.exists() else 'not_required'
                if ref.exists():
                    uniform=read(ref);target=1.01*uniform['final_compliance']
                    uhits=[r['step'] for r in uniform['history'] if r['compliance']<=target]
                    record.update(target=target,reference_status=uniform['status'],uniform_steps_to_target=min(uhits) if uhits else None)
                    hits=[r['step'] for r in result['history'] if r['compliance']<=target] if p.exists() else []
                    record.update(reached=bool(hits),steps_to_target=min(hits) if hits else None,capped_steps=min(hits) if hits else cap)
                    if p.exists():
                        record['final_ratio']=result['final_compliance']/uniform['final_compliance']
                        for n in (50,100,200):record[f'C{n}_ratio']=record[f'C{n}']/uniform['budget_points'][str(n)]
                task_metrics.append(record);metrics.append(record)
            coverage.append(dict(subset=subset,method=method,planned=300,settled=settled,failed=failed,missing=len(missing)))
            if missing and not allow_partial:raise ValueError(f'Full run unfinished: {subset}/{method}: {settled}/300')
            eligible=[r for r in metrics if r['target_available']]
            good=[r for r in eligible if r['status']!='failed'];hits=[r for r in eligible if r.get('reached')]
            capped=[r['capped_steps'] for r in eligible]
            speed_delta=[r['uniform_steps_to_target']-r['capped_steps'] for r in eligible if r['uniform_steps_to_target'] is not None]
            ratios=[r['final_ratio'] for r in good]
            audit.append(dict(subset=subset,method=method,planned=300,settled=settled,failed=failed,
                target_available=len(eligible),reached=len(hits),not_reached=len(eligible)-len(hits),
                capped_mean_steps=float(np.mean(capped)) if capped else None,capped_steps_CI95=bootstrap(capped),
                paired_step_saving_vs_uniform=float(np.mean(speed_delta)) if speed_delta else None,step_saving_CI95=bootstrap(speed_delta),
                mean_final_ratio=float(np.mean(ratios)) if ratios else None,final_ratio_CI95=bootstrap(ratios),
                lower_final=sum(r['final_ratio']<1-1e-8 for r in good),tied_final=sum(abs(r['final_ratio']-1)<=1e-8 for r in good),
                higher_final=sum(r['final_ratio']>1+1e-8 for r in good),converged=sum(r['status']=='converged' for r in metrics),
                max_steps=sum(r['status']=='max_steps' for r in metrics),
                attempted_online_FE_including_failures=sum(r.get('online_FE',0) for r in metrics),
                numerically_recovered=sum(r['numerical_recovery_status']=='recovered' for r in metrics),
                failed_after_refinement=sum(r['numerical_recovery_status']=='failed_after_refinement' for r in metrics),
                numerical_retry_accounting_scope='Legacy numerically_recovered/failed_after_refinement fields count only separate numerical_retry.json procedures, not inline FE refinement. Failed inline correction counts/residuals are unknown unless retained.',
                inline_refinement_evidenced_failed_tasks=sum(r.get('inline_refinement_failed_evidenced',False) for r in metrics),
                inline_refinement_accepted_FE=sum(r.get('inline_refinement_accepted_FE',0) for r in metrics),
                inline_refinement_accepted_correction_solves=sum(r.get('inline_refinement_accepted_correction_solves',0) for r in metrics),
                means={key:float(np.mean([r[key] for r in good])) if good else None for key in ('C50_ratio','C100_ratio','C200_ratio','online_FE','online_seconds')}))
    dest=out/'reports';dest.mkdir(exist_ok=True)
    dump(dest/'coverage_audit.json',coverage)
    dump(dest/'failure_audit.json',dict(records=failures,failed_task_count=len(failures),
        known_attempted_FE=sum(r['trajectory_FE']+r['initialization_FE'] for r in failures),
        note='Failure classification records observed phase and state only; it does not establish the root cause of an FE rejection. Failed tasks retain no invented compliance.'))
    dump(dest/'two_goal_metrics.json',audit)
    dump(dest/'dataset_roles_audit.json',data_roles)
    offline=offline_cost_audit(out);dump(dest/'offline_cost_audit.json',offline)
    with (dest/'paired_task_metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted({k for r in task_metrics for k in r}));writer.writeheader();writer.writerows(task_metrics)
    fmt=lambda x:'—' if x is None else f'{x:.4f}'
    lines=['# 初始化 Benchmark：全量双目标报告','',
        '状态：'+('探索性中间输出，完整性门槛未要求通过。' if allow_partial else '全量任务覆盖已核对；失败项保留。'),'',
        'Train1000仅用于模型训练或建立检索库，Valid100用于验证与模型选择；两张效果表只比较独立的ID、OOD、CR测试集，不比较训练集或验证集。每个测试集各有固定均匀抽取的300任务，与训练/验证清单的任务键和group_id均无重叠。[数据用途与清单核验](dataset_roles_audit.json)。',
        '', '“测试已结束/测试总数”的分母是该测试集的300例，已结束包括成功和失败；Train1000是单独的训练库。训练标签与高分辨率教师配对属于训练准备，单独报告成本，不是训练集效果对比。优化上限500步，固定预算50/100/200步。',
        '', '方法代号meta/early在本表分别表示Meta零步密度→公共OC、独立k=5训练的零步密度→公共OC。训练内层步数与测试时适应步数须区分；本平台交接没有测试时神经参数热身。',
        '','## 表1：达到相同柔度的速度','',
        '目标为同任务Uniform终态柔度的1.01倍。未达到目标或执行失败按500步封顶计入均值；该值是明确的失败惩罚统计，不冒称只在成功样本上的均值或生存分析RMST。参考失败时目标不可用，单独列数。',
        '','| 测试集 | 方法 | 测试已结束/测试总数 | 达到/可定义目标 | 封顶平均步数 | 比Uniform节省步数 | 失败 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in audit:lines.append(f'| {r["subset"]} | {r["method"]} | {r["settled"]}/{r["planned"]} | {r["reached"]}/{r["target_available"]} | {fmt(r["capped_mean_steps"])} | {fmt(r["paired_step_saving_vs_uniform"])} | {r["failed"]} |')
    lines+=['','## 表2：同预算与最终柔度','',
        '比值为每任务相对Uniform的柔度比再取均值，小于1较好。失败没有虚构柔度，必须连同成功数和第一张表解释。终态包括收敛和到上限两种状态；到上限不称已充分收敛。',
        '','| 测试集 | 方法 | C50比 | C100比 | C200比 | 终态比 | 更低/持平/更高 | 收敛/未收敛(500步上限) |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in audit:lines.append(f'| {r["subset"]} | {r["method"]} | {fmt(r["means"]["C50_ratio"])} | {fmt(r["means"]["C100_ratio"])} | {fmt(r["means"]["C200_ratio"])} | {fmt(r["mean_final_ratio"])} | {r["lower_final"]}/{r["tied_final"]}/{r["higher_final"]} | {r["converged"]}/{r["max_steps"]} |')
    lines+=['','## 验证与解释','',
        '配对任务bootstrap的95%区间、累计在线FE和时间见 `two_goal_metrics.json`；逐任务完整数据见 `paired_task_metrics.csv`。每类测试单独重采样，不将不同种子/分辨率当作额外独立样本。',
        '','实现差异、未知作者细节、不同物理观测与多分辨率标签资源见 [复现审计](../../docs/METHODS.md)。这些差异没有因为最终表现而隐藏或更换。']
    lines+=['','Meta/Early原生输出是已经过滤的物理密度，交接时按rho-as-x进入共同体积校准和平台滤波；原始预测与交接场均保存。此表评估学习密度初始化迁移到共同OC后的表现，未评估原论文神经参数空间中在线Adam适应的速度，也不能据此宣称复现了原生Meta在线优化的性能排名。参考物理场的实际FE与导出时间按初始化账本记录。']
    lines+=['','计时范围是已测初始化加OC轨迹，未包含公共任务目录构建与滤波矩阵装配；FE尝试账本包含失败和恢复重算。不能把该计时口径直接称为含全部准备的冷启动总时间。']
    lines+=['','中断发生在两次检查点之间时，尚未保存的那部分耗时可能缺失；恢复后的seconds不能代表包含全部已丢弃工作时间的总墙钟成本。对应FE尝试仍按逐次账本保留，缺失耗时不填零。']
    reference_timing=offline.get('native_reference_timing_snapshot')
    if reference_timing is not None:
        lines+=['',f'元训练参考场准备另有保存配置可追溯：截至恢复点{reference_timing["observed_latest_resume_step"]}的调用，已知测得耗时合计{reference_timing["known_measured_reference_preparation_seconds"]:.6f}秒，另有{reference_timing["unknown_reference_preparation_phases"]}个阶段未记录。该值是此快照的已知下限，后续调用与未记录阶段不能推断为零；其中{reference_timing["reference_FE_through_these_invocations"]}次参考FE已包含在模型账本，不能再次相加。[各次调用与配置SHA](../audit/native_reference_preparation_timing_20261126/summary.json)。']
    lines+=['','原生预测若有中断重试，成功交接记录之外的参考FE尝试也从任务账本计入在线总FE；缺失的先前失败导出耗时仍为未知，不能用最后一次成功导出时间代替全部尝试耗时。']
    lines+=['','历史KNN全量测试曾暴露了高条件数场的float64残差失败。公共求解器随后加入相同Q4方程、原始1e-6阈值下的扩展精度迭代精化；所有方法均适用，已接受的float64路径保持不变。原失败记录归档，重算与精化账本保留，恢复数量见 `two_goal_metrics.json`。该过程属于测试期间数值调试；未用测试柔度选择模型或超参数，但本轮不能表述为一次性全盲确认实验。']
    lines+=['','精化统计区分单次FE评估内的数值修正与另行启动的版本化重试。JSON旧字段numerically_recovered/failed_after_refinement只计numerical_retry.json，不代表所有内联精化次数；新增inline_refinement字段分别计已接受FE/修正求解次数和由失败回溯证实进入精化的任务。失败未保存的修正次数及终末残差仍为未知，不能从独立重试为0推断未经过精化。']
    shared=offline['shared_training_library']
    lines+=['',f'共同训练标签库：已核实历史教师FE {shared["known_historical_teacher_FE"]}，另有 {shared["unknown_cost_phases"]} 个历史阶段账本未取得；本次标签核验FE {shared["new_teacher_audit_FE"]}。共享库不在总资源中重复相加，但每个独立监督方法都依赖它。',
            f'验证教师FE {offline["validation_teacher_FE"]}；已完成高分辨率配对教师FE {offline["high_resolution_teacher_FE"]}；包含进行中与失败的高分辨率教师FE尝试 {offline["high_resolution_teacher_attempted_FE_including_inflight_and_failures"]}（此总数已包含前述已完成部分，不能重复相加）。网络训练FE、耗时来源及逐任务账本见 `offline_cost_audit.json`，缺失成本不填成零。',
            f'失败任务的最后尝试步数、初始场范围、证据文件SHA与阶段分类见 `failure_audit.json`。失败轨迹未保留的耗时仍为未知，不填零；在线FE总数包含失败尝试，成功样本的平均耗时不能代表包含全部失败成本的总时间。有明确FE账本的独立工程核验共 {offline["engineering_verification_known_FE"]} 次，未记账的诊断成本仍为未知额外开销。']
    if offline['retired_CPU_diagnostics']:
        retired=offline['retired_CPU_diagnostics']
        lines+=['',f'已退役CPU诊断另计 {retired["attempted_OC_FE"]} 次OC FE尝试，旧结果不并入本次固定后端比较。其已完成与失败数、原生训练恢复、运行库/并发I/O修复及所有保留的失败证据见审计。']
    common_runtime=('统一Python/NumPy/SciPy装配及OC、cuDSS float64直接因子；全部方法相同，逐步记录后端，采用等价线性滤波体积点积检查'
                    if protocol.get('linear_solver_backend')=='cudss-float64' else '同一CPU Python/NumPy/SciPy环境')
    lines+=['','训练设备、库版本、实际轮数/外层步数及阶段命令也保存在 `offline_cost_audit.json`。公共OC环境：'+common_runtime+'。原生Meta预测显式以CPU导出。安装完成、设备验证、正式训练完成是不同状态；训练配置未记录的设备信息不推测填充。']
    if (out/'audit/factor_lifetime_and_assembly_20261126/CHANGE.json').exists():
        lines+=['','运行中修复了每次FE动态建类保留矩阵的生命周期问题，并接入固定网格的等值装配：材料、方程、残差阈值及OC规则未改变，978个原型轨迹状态逐字节核对矩阵，生产入口再次通过两个64完整收敛轨迹和256的20步诊断。初始额外严格1e-10轨迹门槛曾未通过，原实现重复运行同样出现约5e-10量级差异；全部观察和既有CPU/GPU准则保留于审计。',
            '','早期完成结果保留原装配实现的计时，后续结果使用更快的装配。逐任务协议SHA、迁移检查点及源快照可追溯；迭代步数/柔度是本比较的主要目标，当前原始耗时不能作为受控、统一装配版本的模型速度排名。失败任务已恢复其失败当时的执行身份，未把旧失败错误标成新版本执行。']
    if (out/'audit/exact_gpu_filter_20261126/CHANGE.json').exists():
        lines+=['','高分辨率随后接入保持原CSR逐行求和顺序、禁用FMA收缩的float64 CUDA滤波；CPU和64路径不变。两个完整500步预算训练任务共2834次FE、5668次实际正反向乘积均与CPU逐字节一致，分别500步到上限和832步收敛；与既有同任务教师的路径柔度相对差最高约3.14e-10。实际生产入口另核验23次FE及6项精度/恢复检查，模型、材料、OC和停止条件不变。中断检查点、源码和协议迁移见 `audit/exact_gpu_filter_20261126/`。这属于执行路径加速，不能把跨装配/滤波版本和并发负载的原始计时作为受控模型速度排名。',
            '','冻结Ulu/Wang/CNN的完整ID/OOD可由同一评测函数提前执行，方法互斥锁与主控制器去重；各方法完整CR仍由主队列运行。评测顺序与并发占用影响墙钟耗时，所有数据清单、单任务预算和统计分母不变。']
    if (out/'audit/pinned_filter_cache_20261126/CHANGE.json').exists():
        lines+=['','随后释放高分辨率CSR上传后未使用的锁页缓存，核及算术方法不变：局部诊断约400MiB缓存释放后正反向结果逐字节相同，实际生产入口另通过23次训练FE和6项精度/续跑检查。一次并发观测见三个高分辨率槽均在用，RSS加共享内存约6.0GB；这不是长期内存上限保证或方法计时排名。该次提前评测重启误用shell覆盖写，旧stdout日志丢失，逐任务结果、状态、FE账本及原执行身份保留；缺口明确保留而不补造日志。详见 `audit/pinned_filter_cache_20261126/`。']
    replay_summary=ROOT/'research_results/benchmark_init_v1/verification/initial_failure_replay/summary.json'
    if replay_summary.exists():
        diagnostic=read(replay_summary)
        lines+=['',f'另对当时已保存的{diagnostic["selected_initial_failures"]}个step0 FE拒绝，各用同一初态/输入及当前等值后端重放一次，新增{diagnostic["FE_added"]}次诊断FE；结果为{diagnostic["statuses"]}。此重放只记录原生产函数的残差/精化过程，不改变模型、参数、容差或正式失败成绩；新残差与耗时不能补充为历史原始执行实测值。[完整诊断](../../benchmark_init_v1/verification/initial_failure_replay/summary.json)与failure_audit中的逐例SHA链接可追溯。']
        if replay_summary.with_name('figure_metadata.json').exists():
            lines+=['','![初始FE失败重放的残差诊断](../../benchmark_init_v1/verification/initial_failure_replay/residual_diagnostics.png)',
                '','图中横轴为FE内部精化修正次数；19例的171个检查点均未达到原1e-6验收阈值。右图使用每例全部检查点中的最低残差，并非仅比较最后一步。[PDF](../../benchmark_init_v1/verification/initial_failure_replay/residual_diagnostics.pdf) · [SVG](../../benchmark_init_v1/verification/initial_failure_replay/residual_diagnostics.svg) · [逐例极值](../../benchmark_init_v1/verification/initial_failure_replay/trace_extrema.csv)。']
    report_name='INTERIM_REPORT_zh.md' if allow_partial else 'FINAL_REPORT_zh.md'
    lines.insert(2,'当前执行范围：'+', '.join(SUBSETS)+'；方法：'+', '.join(METHODS)+'。以RUN_PLAN.json为完整性边界；默认不要求cGAN或跨分辨率结果。')
    (dest/report_name).write_text('\n'.join(lines)+'\n')
    if not allow_partial:
        models=model_audit(out)
        dump(dest/'FINAL_REPORT_READY.json',dict(status='full_coverage_verified',coverage=coverage,models=models,
            protocol_sha256=digest(out/'protocol.json'),analysis_source_sha256=digest(Path(__file__)),
            report_sha256=digest(dest/'FINAL_REPORT_zh.md')))
    return coverage


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'results')
    parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    analyze(args.out,args.allow_partial)
