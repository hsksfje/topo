"""Shared large-job limits and native Windows resource cleanup, no FEM loads."""
from pathlib import Path
import multiprocessing as mp
from queue import Empty
import os
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.runtime_parallel import cpu_allocation


def large_job(folder,queue,release):
    os.environ['INITSKILL_HIGH_RES_SLOTS']='2'
    from initskill.methods.benchmark_init_v1.resources import high_resolution_slot
    with high_resolution_slot(folder,256,256):
        queue.put(os.getpid());release.wait(timeout=30)


class ParallelBudgetTest(unittest.TestCase):
    def test_28cpu_64gib_allocation_is_aggregate(self):
        plan=cpu_allocation(28,64)
        self.assertEqual(plan['small_workers'],2)
        self.assertEqual(plan['large_slots'],24)
        self.assertEqual(plan['teacher_workers'],plan['cr_workers'])
        self.assertEqual(plan['maximum_optimization_tasks']+plan['reserved_training_cpus'],28)

    def test_shared_limit_and_windows_cleanup(self):
        context=mp.get_context('spawn');queue=context.Queue();release=context.Event()
        with tempfile.TemporaryDirectory() as directory:
            processes=[context.Process(target=large_job,args=(directory,queue,release)) for _ in range(3)]
            try:
                for process in processes:process.start()
                entered={queue.get(timeout=30),queue.get(timeout=30)}
                self.assertEqual(len(entered),2)
                with self.assertRaises(Empty):queue.get(timeout=1)
                release.set();entered.add(queue.get(timeout=30));self.assertEqual(len(entered),3)
                for process in processes:process.join(timeout=30);self.assertEqual(process.exitcode,0)
            finally:
                release.set()
                for process in processes:
                    if process.is_alive():process.terminate();process.join(timeout=10)
                queue.close()


if __name__=='__main__':unittest.main()
