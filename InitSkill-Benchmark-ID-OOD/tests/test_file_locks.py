"""Real spawn-mode contention, including native Windows msvcrt locking."""
from pathlib import Path
import multiprocessing as mp
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from initskill import locking


def contender(path,queue):
    with open(path,'a') as stream:
        try:locking.flock(stream,locking.LOCK_EX|locking.LOCK_NB)
        except BlockingIOError:queue.put(False)
        else:
            queue.put(True);locking.flock(stream,locking.LOCK_UN)


class LockTest(unittest.TestCase):
    def test_cross_process_exclusion_and_release(self):
        context=mp.get_context('spawn')
        with tempfile.TemporaryDirectory() as temp:
            path=str(Path(temp)/'lease.lock');queue=context.Queue()
            with open(path,'a') as held:
                locking.flock(held,locking.LOCK_EX)
                child=context.Process(target=contender,args=(path,queue));child.start()
                self.assertFalse(queue.get(timeout=20));child.join(timeout=20)
                self.assertEqual(child.exitcode,0);locking.flock(held,locking.LOCK_UN)
            child=context.Process(target=contender,args=(path,queue));child.start()
            self.assertTrue(queue.get(timeout=20));child.join(timeout=20)
            self.assertEqual(child.exitcode,0);queue.close()


if __name__=='__main__':unittest.main()
