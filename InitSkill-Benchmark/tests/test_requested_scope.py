"""Cross-check requested sample counts and the producer's shared OC limit."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from initskill.methods.benchmark_init_v1 import campaign,common


class RequestedScopeTest(unittest.TestCase):
    def test_all_test_jobs_share300_tasks_and500_step_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            protocol=common.protocol(temp)
            self.assertEqual(protocol['oc_max_steps'],500)
            self.assertEqual(protocol['task_counts'],dict(train1000=1000,valid100=100,test_id300=300,test_ood300=300,test_cr300=300))
            for subset in campaign.SUBSETS:
                reference=None
                for method in campaign.METHODS:
                    jobs,settled=campaign.evaluation_plan(temp,method,subset)
                    self.assertEqual(len(jobs),300);self.assertEqual(settled,[])
                    self.assertEqual({job[4] for job in jobs},{500})
                    keys=[job[3]['task_key'] for job in jobs]
                    if reference is None:reference=keys
                    self.assertEqual(keys,reference)


if __name__=='__main__':unittest.main()
