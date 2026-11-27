from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from initskill.methods.benchmark_init_v1 import common

class DecimalRegressionTest(unittest.TestCase):
    def test_eight_known_failed64_initial_fields(self):
        with np.load(ROOT/'tests/fixtures/windows_initial_failures.npz') as z:fields={k:z[k].copy() for k in z.files}
        rows={r['task_key']:r for s in ['test_id300','test_ood300'] for r in common.rows(s)};records=[]
        with tempfile.TemporaryDirectory() as temp:
            for key,subset,x in zip(fields['keys'],fields['subsets'],fields['x']):
                m=common.solver(temp,str(subset),rows[str(key)]);e=m.evaluate(x)
                self.assertLess(e['residual'],1e-6);self.assertGreater(e['compliance'],0)
                self.assertLess(e.get('energy_balance_relative',0),1e-6)
                records.append(dict(task=str(key),residual=e['residual'],precision=e.get('precision'),compliance=e['compliance']))
        common.dump(ROOT/'verification/decimal64_regression.json',dict(passed=True,FE_attempts=8,scope='Known v1 numerical regression only, no model selection',cases=records))
