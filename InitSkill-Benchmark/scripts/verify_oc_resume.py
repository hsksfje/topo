"""Real training-task OC continuation, corruption rejection and FE accounting."""
from pathlib import Path
import json
import shutil
import tempfile
import numpy as np

from initskill.methods.benchmark_init_v1 import common
from initskill.methods.benchmark_init_v1.convergence import run_to_convergence

ROOT=Path(__file__).resolve().parents[1]


def run_check():
    (ROOT/'verification').mkdir(exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='oc_resume_',dir=ROOT/'verification'))
    row=common.rows('train1000')[0]
    a=common.solver(folder,'train1000',row);b=common.solver(folder,'train1000',row)
    identity=dict(training_task=row['task_key'],engineering_only=True,max_steps=30)
    expected=run_to_convergence(a,a.initial(),30,folder/'continuous',identity)
    class Interrupt:
        def __init__(self,base):self.base=base;self.calls=0
        def __getattr__(self,name):return getattr(self.base,name)
        def evaluate(self,x):
            self.calls+=1
            if self.calls==14:raise RuntimeError('Injected engineering interruption')
            return self.base.evaluate(x)
    try:run_to_convergence(Interrupt(b),b.initial(),30,folder/'resumed',identity)
    except RuntimeError as exc:assert str(exc)=='Injected engineering interruption'
    else:raise AssertionError('Expected interruption')
    pointer=json.loads((folder/'resumed/checkpoint.json').read_text())
    assert pointer['step']==10 and '\\' not in pointer['state_file']
    shutil.copytree(folder/'resumed',folder/'corrupt')
    broken=folder/'corrupt'/pointer['state_file'];data=bytearray(broken.read_bytes());data[-1]^=1;broken.write_bytes(data)
    try:run_to_convergence(b,b.initial(),30,folder/'corrupt',identity)
    except ValueError as exc:assert 'coherent OC checkpoint' in str(exc)
    else:raise AssertionError('Changed checkpoint accepted')
    # This is an intentional training-only engineering failure, not a formal
    # failed test task; retain its evidence under verification/.
    (folder/'resumed/failure.json').rename(folder/'resumed/injected_failure.json')
    actual=run_to_convergence(b,b.initial(),30,folder/'resumed',identity)
    assert actual['history']==expected['history']
    with np.load(folder/'continuous/state_final.npz') as x,np.load(folder/'resumed/state_final.npz') as y:
        assert x['x'].tobytes()==y['x'].tobytes() and x['density'].tobytes()==y['density'].tobytes()
    assert expected['FE']==31 and actual['FE']==35
    # The corrupt-copy ledger duplicates the original interruption; remove
    # only its copied ledger to avoid double-counting the same attempts.
    (folder/'corrupt/FE_attempts.jsonl').unlink()
    report=dict(passed=True,training_task=row['task_key'],test_tasks_used=0,
        final_arrays_byte_equal=True,all_history_records_equal=True,corrupt_state_rejected=True,
        continuous_attempts=31,resume_including_failed_attempt_and_replay=35,total_engineering_attempts=66,
        checkpoint_paths_POSIX=True,artifact_folder=folder.relative_to(ROOT).as_posix())
    common.dump(ROOT/'verification/OC_RESUME_PASS.json',report);print(json.dumps(report));return report


if __name__=='__main__':run_check()
