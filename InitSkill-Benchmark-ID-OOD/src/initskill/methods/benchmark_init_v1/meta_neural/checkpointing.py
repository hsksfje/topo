from initskill import locking as fcntl
"""Coherent model/Adam/sampler commits for exact native training continuation."""
import json
from pathlib import Path
import shutil
import uuid
from ..common import dump,sha

RESUME_KEYS=('seed','outer_steps','inner_steps','batch_size','train_keys','valid_keys',
    'source_identity','native_relative_radius','platform_relative_radius',
    'train_manifest_sha256','validation_manifest_sha256','optimized_topology_labels_used','smoke_only')


def _tree_write(eqx,path,tree):
    temporary=path.with_name(path.stem+'_'+uuid.uuid4().hex+'.eqx')
    eqx.tree_serialise_leaves(temporary,tree);fcntl.atomic_replace(temporary,path)


def save(folder,model,outer_state,state,config,eqx):
    """Write an inactive slot, then atomically commit a hash-checked pointer."""
    folder=Path(folder);slot=folder/'checkpoint_slots'/str(state['step']%2);slot.mkdir(parents=True,exist_ok=True)
    _tree_write(eqx,slot/'model.eqx',model);_tree_write(eqx,slot/'outer_optimizer.eqx',outer_state)
    dump(slot/'state.json',dict(state,config=config))
    names=['model.eqx','outer_optimizer.eqx','state.json']
    if state.get('best_validation_loss') is not None:
        shutil.copy2(folder/'best.eqx',slot/'best.eqx');names.append('best.eqx')
    pointer=dict(step=state['step'],slot=slot.relative_to(folder).as_posix(),files={n:sha(slot/n) for n in names})
    dump(folder/'resume_commit.json',pointer)
    # The commit manifest is authoritative; omit nonportable public symlinks.
    dump(folder/'history.json',state['history'])
    dump(folder/'progress.json',{k:v for k,v in state.items() if k not in {'history','best_validation_loss'}})


def load(folder,model,outer_init,expected,eqx):
    folder=Path(folder);pointer=folder/'resume_commit.json';legacy=False
    if pointer.exists():
        commit=json.loads(pointer.read_text());base=folder/commit['slot']
        for name,digest in commit['files'].items():
            if sha(base/name)!=digest:raise ValueError('Incomplete native checkpoint commit: '+name)
        state=json.loads((base/'state.json').read_text());config=state['config']
        model_path=base/'model.eqx';outer_path=base/'outer_optimizer.eqx'
        if 'best.eqx' in commit['files']:shutil.copy2(base/'best.eqx',folder/'best.eqx')
    else:
        candidates=sorted((folder/'recovery').glob('step_*/MANIFEST.json'))
        if not candidates:
            if (folder/'checkpoint.eqx').exists():raise ValueError('Native checkpoint lacks an audited coherent state')
            return None
        manifest=candidates[-1];base=manifest.parent;audit=json.loads(manifest.read_text());legacy=True
        for name,digest in audit['source_files_sha256'].items():
            if sha(base/name)!=digest:raise ValueError('Changed audited recovery snapshot: '+name)
        state=json.loads((base/'progress.json').read_text());config=json.loads((base/'config.json').read_text())
        state['history']=json.loads((base/'history.json').read_text())
        scores=[x['validation_loss'] for x in state['history'] if 'validation_loss' in x]
        state['best_validation_loss']=min(scores) if scores else None
        if scores:raise ValueError('Legacy recovery with selected validation model requires an audited best.eqx')
        model_path=base/'checkpoint.eqx';outer_path=base/'outer_optimizer.eqx'
    for key in RESUME_KEYS:
        if config.get(key)!=expected.get(key):raise ValueError('Native resume configuration drift: '+key)
    if config.get('runtime',{}).get('backend')!=expected.get('runtime',{}).get('backend'):
        raise ValueError('Native resume backend drift')
    if [x['step'] for x in state['history']]!=list(range(1,state['step']+1)):
        raise ValueError('Non-contiguous native checkpoint history')
    if state['history']:
        last=state['history'][-1]
        # A history row is created after its outer update; validation is then
        # charged before the coherent checkpoint is committed.
        # Upstream adapt evaluates a final inner loss, then evaluates the
        # outer loss as well: k inner updates + final inner + outer = k+2.
        validation_FE=((config['inner_steps']+2)*len(config['valid_keys'])
                       if 'validation_loss' in last else 0)
        if state['FE']-last['FE']!=validation_FE:raise ValueError('Native checkpoint FE phase mismatch')
    model=eqx.tree_deserialise_leaves(model_path,model)
    params,_=eqx.partition(model,eqx.is_array)
    outer=eqx.tree_deserialise_leaves(outer_path,outer_init(params))
    state['recovery_source']=str(base);state['legacy_audited_snapshot']=legacy
    return model,outer,state
