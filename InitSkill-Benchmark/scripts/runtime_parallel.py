"""CPU allocation and an audited, resource-only Windows compatibility upgrade."""
import hashlib
import json
from pathlib import Path
import tempfile
import time

# The initial published Windows release. All numerical functions stay identical;
# only shared leases and the Windows cleanup branch changed in resources.py.
OLD_RESOURCE_SHA='dc8c9f6a6ed17a04a301d8f68e8795f7343cc44963ed186478dad684f92f4164'


def cpu_allocation(cpus,ram_gb):
    cpus=int(cpus);ram_gb=float(ram_gb)
    if cpus<4 or ram_gb<8:raise ValueError('At least4 CPU slots and8GiB RAM are required for this preset')
    reserved=2 if cpus>=6 else 1
    small=2 if cpus>=6 else 1
    reserve_ram=min(16.,max(8.,ram_gb/4))
    large=min(cpus-reserved-small,max(1,int((ram_gb-reserve_ram)//2)))
    return dict(cpu_budget=cpus,ram_gb=ram_gb,small_workers=small,large_slots=large,
        cr_workers=large,teacher_workers=large,reserved_training_cpus=reserved,
        maximum_optimization_tasks=small+large,large_job_estimate_gib=2.,reserved_ram_gib=reserve_ram,
        note='CR tests and teacher generation share ONE large-job budget. Workers use one numerical thread; installed RAM is a planning estimate, not a measured peak guarantee.')


def settings(profile):
    if profile.get('concurrency'):
        if profile['fe_solver']!='superlu':raise ValueError('This CPU allocation preset requires the SuperLU profile')
        return profile['concurrency']
    return dict(small_workers=2,cr_workers=3,teacher_workers=2,large_slots=None,
        note='Original default:2 small jobs,2 shared baseline large slots; Linux may admit an optional third large job.')


def migrate_resource_protocol(out,common):
    """Preserve finished identities; explicitly transition incomplete OC cases.

    Unknown source/config changes are rejected. Native model/Adam/sampler files,
    all predictions, trajectories, labels, convergence conditions and budgets
    are not changed by this resource-management-only upgrade.
    """
    out=Path(out);path=out/'protocol.json'
    if not path.exists():return
    old_bytes=path.read_bytes();old=json.loads(old_bytes)
    with tempfile.TemporaryDirectory(prefix='initskill-protocol-') as temp:
        new=common.protocol(Path(temp));new_bytes=(Path(temp)/'protocol.json').read_bytes()
    if old==new:return
    normative=lambda p:{k:v for k,v in p.items() if k!='source_sha256'}
    if normative(old)!=normative(new):raise ValueError('Changed numerical/data protocol; use a new results directory')
    delta={k for k in set(old['source_sha256'])|set(new['source_sha256'])
           if old['source_sha256'].get(k)!=new['source_sha256'].get(k)}
    if delta!={'resources.py'} or old['source_sha256'].get('resources.py')!=OLD_RESOURCE_SHA:
        raise ValueError('Unrecognized source change; automatic resource-only migration refused')
    execution=out/'campaign/execution.json'
    if execution.exists():
        import os
        import psutil
        record=json.loads(execution.read_text())
        try:process=psutil.Process(record['pid'])
        except psutil.NoSuchProcess:process=None
        if (process is not None and process.pid!=os.getpid() and
                abs(process.create_time()-record['process_created'])<1e-3):
            raise RuntimeError('Stop the existing benchmark before updating its resource protocol')
    old_sha=hashlib.sha256(old_bytes).hexdigest();new_sha=hashlib.sha256(new_bytes).hexdigest()
    folder=out/'audit'/('resource_parallel_upgrade_'+str(time.time_ns()));folder.mkdir(parents=True)
    (folder/'protocol.before.json').write_bytes(old_bytes);(folder/'protocol.after.json').write_bytes(new_bytes)
    changed=[]
    for identity in (out/'full_evaluation').glob('*/*/*/identity.json'):
        case=identity.parent
        if (case/'result.json').exists() or (case/'failure.json').exists():continue
        record=json.loads(identity.read_text())
        if record.get('protocol_sha256')==new_sha:
            transition=json.loads((case/'resource_execution_transition.json').read_text())
            if transition['after']!=record or transition['before'].get('protocol_sha256')!=old_sha:
                raise ValueError('Unrecognized partial migration: '+str(case))
            changed.append(case.relative_to(out/'full_evaluation').as_posix());continue
        if record.get('protocol_sha256')!=old_sha:raise ValueError('Unknown pending execution identity: '+str(case))
        rel=case.relative_to(out/'full_evaluation');saved=folder/'pending_before'/rel;saved.mkdir(parents=True)
        (saved/'identity.json').write_bytes(identity.read_bytes())
        if (case/'checkpoint.json').exists():
            contents=(case/'checkpoint.json').read_bytes();checkpoint=json.loads(contents)
            (saved/'checkpoint.json').write_bytes(contents)
            state=case/checkpoint['state_file']
            if common.sha(state)!=checkpoint['state_sha256']:raise ValueError('Corrupt pre-upgrade checkpoint')
            target=saved/checkpoint['state_file'];target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(state.read_bytes())
        before=dict(record);record['protocol_sha256']=new_sha
        common.dump(case/'resource_execution_transition.json',dict(before=before,after=record,
            audit=folder.relative_to(out).as_posix(),reason='Resource leases/Windows cleanup only; no numerical implementation or budget change'))
        common.dump(identity,record)
        changed.append(rel.as_posix())
    temporary=path.with_name('protocol.resource_upgrade.tmp');temporary.write_bytes(new_bytes)
    from initskill.locking import atomic_replace
    atomic_replace(temporary,path)
    common.dump(folder/'CHANGE.json',dict(created_unix=time.time(),old_protocol_sha256=old_sha,new_protocol_sha256=new_sha,
        changed_source_files=sorted(delta),completed_and_failed_cases_unchanged=True,pending_cases_transitioned=changed,
        physical_functions_and_models_unchanged=True,numerical_budgets_unchanged=True))
