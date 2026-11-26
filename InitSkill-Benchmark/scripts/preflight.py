"""Data isolation, numerical ports and runtime checks before a full run."""
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def run_checks():
    import numpy as np
    import scipy
    import torch
    from initskill.methods.benchmark_init_v1 import common,verification
    from scripts.analyze import dataset_roles_audit
    package=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())
    for name,record in package['files'].items():
        assert common.sha(ROOT/name)==record['sha256'],('Changed packaged file',name)
    roles=dataset_roles_audit()
    sampling=json.loads((common.DATA/'SAMPLING.json').read_text())
    specs=[('valid100','valid300',100),('test_id300','test_id1000',300),
           ('test_ood300','test_ood1000',300),('test_cr300','test_cr1000',300)]
    for (name,source,n),seed in zip(specs,np.random.SeedSequence(42).spawn(4)):
        p=common.DATA/'source_manifests'/(source+'.jsonl')
        original=[json.loads(line) for line in p.read_text().splitlines()]
        indices=np.sort(np.random.default_rng(seed).choice(len(original),n,replace=False)).tolist()
        saved=sampling['splits'][name]
        assert indices==saved['selected_indices']
        assert [original[i]['task_key'] for i in indices]==[r['task_key'] for r in common.rows(name)]
        assert common.sha(p)==saved['source_sha256']
        assert common.sha(common.DATA/'manifests'/(name+'.jsonl'))==saved['manifest_sha256']
    verification.main(ROOT/'verification/platform')
    from scripts.verify_oc_resume import run_check
    run_check()
    profile=json.loads((ROOT/'.runtime/profile.json').read_text())
    subprocess.run([profile['jax_python'],str(ROOT/'scripts/verify_native.py')],check=True,
                   env=dict(os.environ,JAX_PLATFORMS='cpu',PYTHONUTF8='1'))
    manifest=dict(passed=True,scope=common.protocol(ROOT/'results')['task_counts'],data_roles=roles,
        platform=platform.platform(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,
        longdouble_mantissa_bits=np.finfo(np.longdouble).nmant+1,
        common_backend=profile['fe_solver'],neural_train_device=profile['torch_device'],
        tests_use_training_tasks_only=True,formal_neural_models_trained_by_preflight=False,retrieval_bank_fitted_from_train1000=True,
        note='Numerical SCG/LM equivalence, filtered FE gradient, CNN/GAN operations and native model/Adam/sampler continuation checked. MATLAB binary parity is not claimed.')
    (ROOT/'verification/PREFLIGHT_PASS.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='data_roles'},ensure_ascii=False,indent=2))


if __name__=='__main__':run_checks()
