"""Numerical regressions include8 actual v1 Windows rejected initial fields."""
from pathlib import Path
import json
import time
import unittest
from scripts.prepare_data import prepare
from initskill.methods.benchmark_init_v1.common import ROOT


def verify():
    prepare()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise RuntimeError('KNN numerical/physical regression failed; do not start')
    print(json.dumps(dict(status='PASS',tests=result.testsRun,decimal_digits=40,scope='Engineering/numerical checks only; full training-validation-test benchmark not run by verify')))
