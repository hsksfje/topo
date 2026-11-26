"""Paper B5/k10 update continuation across model/Adam/sampler serialization."""
from pathlib import Path
import sys
import os
os.environ["JAX_PLATFORMS"]="cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from initskill.methods.benchmark_init_v1.meta_neural import core,checkpointing
from initskill.methods.benchmark_init_v1.common import DEFAULT_OUT,rows,dump,sha,DATA


def max_error(jax,a,b):
    return max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)))


def main():
    folder=ROOT/'verification/native_training_resume';folder.mkdir(parents=True,exist_ok=True)
    jax,eqx,optax,mtp,common,model,learner,*_=core.setup((64,64))
    tr=rows('train1000')[:5];ledger=core.Ledger(folder/'FE_attempts.jsonl');ledger.attach()
    try:
        samples=[core.sample(r) for r in tr];sampler=core.TaskSampler(5,5,42)
        config=dict(seed=42,outer_steps=2,inner_steps=10,batch_size=5,train_keys=[r['task_key'] for r in tr],
            valid_keys=[tr[0]['task_key']],source_identity=sha(ROOT/'references/init_benchmark_v1/SOURCES.json'),
            native_relative_radius=1/32,platform_relative_radius=2.5/64,
            train_manifest_sha256=sha(DATA/'manifests/train1000.jsonl'),
            validation_manifest_sha256=sha(DATA/'manifests/valid100.jsonl'),
            optimized_topology_labels_used=False,smoke_only=True,runtime=dict(backend=jax.default_backend()))
        def batch(ids):return {k:jax.numpy.asarray(np.stack([samples[i][k] for i in ids])) for k in samples[0]}
        p,s=eqx.partition(model,eqx.is_array);o=learner.outer_opt.init(p);update=eqx.filter_jit(learner.update)
        b=batch(sampler.next());p,s,o,loss,_=update(p,s,o,b,b,10);float(loss)
        history=[dict(step=1,FE=ledger.count,train_loss=float(loss))]
        checkpointing.save(folder,eqx.combine(p,s),o,dict(step=1,FE=ledger.count,sampler=sampler.state(),
            history=history,best_validation_loss=None,seconds=1.),config,eqx)
        assert '\\' not in __import__('json').loads((folder/'resume_commit.json').read_text())['slot']
        ids=sampler.next();b=batch(ids);expected_p,expected_s,expected_o,expected_loss,_=update(p,s,o,b,b,10)
        float(expected_loss)
        # A different learner and deserialized optimizer emulate a fresh process.
        _,_,_,_,_,template,learner2,*_=core.setup((64,64))
        resumed,restored_o,state=checkpointing.load(folder,template,learner2.outer_opt.init,config,eqx)
        rp,rs=eqx.partition(resumed,eqx.is_array)
        assert max_error(jax,p,rp)==0 and max_error(jax,o,restored_o)==0
        restored_sampler=core.TaskSampler(5,5,42);restored_sampler.restore(state['sampler'])
        restored_ids=restored_sampler.next();assert np.array_equal(ids,restored_ids)
        update2=eqx.filter_jit(learner2.update);b=batch(restored_ids)
        actual_p,actual_s,actual_o,actual_loss,_=update2(rp,rs,restored_o,b,b,10);float(actual_loss)
        pe=max_error(jax,expected_p,actual_p);oe=max_error(jax,expected_o,actual_o)
        assert pe<1e-12 and oe<1e-12 and abs(float(actual_loss)-float(expected_loss))<1e-12
        bad=dict(config,outer_steps=3)
        try:checkpointing.load(folder,template,learner2.outer_opt.init,bad,eqx)
        except ValueError:rejected=True
        else:raise AssertionError('Resume accepted changed training budget')
        dump(folder/'comparison.json',dict(passed=True,training_tasks=5,test_tasks_used=0,
            batch_size=5,inner_steps=10,formal_model_saved=False,serialized_parameter_error=0.,serialized_Adam_error=0.,
            resumed_update_parameter_max_error=pe,resumed_update_Adam_max_error=oe,
            resumed_update_loss_error=abs(float(actual_loss)-float(expected_loss)),
            epoch_boundary_permutation_equal=True,changed_budget_rejected=rejected,FE=ledger.count))
        print('native resume PASS',pe,oe,float(actual_loss),flush=True)
    finally:ledger.detach()


if __name__=='__main__':main()
