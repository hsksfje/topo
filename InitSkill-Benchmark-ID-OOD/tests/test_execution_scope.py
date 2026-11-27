import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from initskill.methods.benchmark_init_v1 import common,campaign
from initskill.methods.benchmark_init_v1.scope import make_scope,require_grid

class ExecutionScopeTest(unittest.TestCase):
    def test_default_plan_has_no_cross_resolution_work(self):
        spec=make_scope()
        self.assertEqual(spec['methods'],['uniform','ulu','wang','cnn','meta','early'])
        self.assertEqual(spec['test_subsets'],['test_id300','test_ood300'])
        self.assertEqual(spec['actors'],['train','test64'])
        self.assertEqual(spec['planned_test_runs'],3600)
        self.assertEqual(spec['high_resolution_teachers'],0)
        self.assertEqual(spec['native_export_tasks_per_method'],600)

    def test_cli_plan_is_safe_without_setup_and_full_is_explicit(self):
        a=json.loads(subprocess.check_output([sys.executable,str(ROOT/'benchmark.py'),'plan'],text=True))
        b=json.loads(subprocess.check_output([sys.executable,str(ROOT/'benchmark.py'),'plan','--enable-cross-resolution'],text=True))
        self.assertFalse(a['include_cross_resolution']);self.assertTrue(b['include_cross_resolution'])
        self.assertEqual(b['planned_test_runs'],6300);self.assertEqual(b['high_resolution_teachers'],1100)
        self.assertNotEqual(a['output'],b['output'])
        p=subprocess.run([sys.executable,str(ROOT/'benchmark.py'),'pairs'],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0);self.assertIn('disabled',p.stdout+p.stderr)

    def test_high_grid_rejected_before_any_input_or_fe_work(self):
        with patch.dict(os.environ,{'INITSKILL_INCLUDE_CR':'0'}):
            with patch.object(common,'prepare_task') as prepare:
                with self.assertRaises(ValueError):common.solver('unused','test_cr300',dict(nx=256,ny=256))
                prepare.assert_not_called()
            from initskill.methods.benchmark_init_v1.meta_neural import core
            with self.assertRaises(ValueError):core.sample(dict(nx=256,ny=256))

    def test_native_export_only_reads_id_and_ood(self):
        from initskill.methods.benchmark_init_v1.meta_neural import export
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'INITSKILL_INCLUDE_CR':'0'}),patch.object(export,'rows',return_value=[{}]) as rows,patch.object(export,'export_one') as run:
            export.export_all(temp,'meta')
            self.assertEqual([r.args[0] for r in rows.call_args_list],['test_id300','test_ood300'])
            self.assertEqual(run.call_count,2)
            with self.assertRaises(ValueError):export.export_all(temp,'meta',['test_cr300'])

    def test_all_default_jobs_share_same300_and500_limit(self):
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'INITSKILL_INCLUDE_CR':'0'}):
            p=common.protocol(temp)
            self.assertEqual(p['task_counts'],dict(train1000=1000,valid100=100,test_id300=300,test_ood300=300))
            self.assertEqual(p['execution_scope']['high_resolution_teachers'],0)
            for subset in ['test_id300','test_ood300']:
                reference=None
                for method in make_scope()['methods']:
                    jobs,settled=campaign.evaluation_plan(temp,method,subset)
                    self.assertEqual(len(jobs),300);self.assertFalse(settled)
                    self.assertEqual({r[4] for r in jobs},{500});self.assertTrue(all(r[3]['nx']==64 for r in jobs))
                    keys=[r[3]['task_key'] for r in jobs]
                    if reference is None:reference=keys
                    self.assertEqual(keys,reference)

    def test_cpu_budget_uses_low_grid_workers_by_default(self):
        from scripts.runtime_parallel import cpu_allocation
        import types
        with patch.dict(os.environ,{'INITSKILL_INCLUDE_CR':'0'}),patch('psutil.virtual_memory',return_value=types.SimpleNamespace(total=64*1024**3)):
            cfg=cpu_allocation(28,64)
        self.assertEqual(cfg['small_workers'],26);self.assertEqual(cfg['large_slots'],0)
