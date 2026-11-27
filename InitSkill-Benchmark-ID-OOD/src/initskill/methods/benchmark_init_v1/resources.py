"""Share large-job slots across formal OC pipelines; allow an explicit CPU budget.

This is scheduling only: no task, model, convergence rule or budget changes.
After the factor-lifetime fix, observed workers use about0.9-1.2GiB RSS.
By default the extra slot requires anonymous/shared RSS below70% of the host
limit. An explicit CPU profile registers a fixed shared slot budget.
"""
from contextlib import contextmanager
import ctypes
from initskill import locking as fcntl
import gc
import os
from pathlib import Path
import time

_requested=os.environ.get('INITSKILL_HIGH_RES_SLOTS')
BASE_HIGH_RESOLUTION_SLOTS=int(_requested) if _requested else 2
HIGH_RESOLUTION_SLOTS=int(_requested) if _requested else 3
if not 1<=BASE_HIGH_RESOLUTION_SLOTS<=HIGH_RESOLUTION_SLOTS<=128:
    raise ValueError('Invalid shared high-resolution worker budget')


def extra_slot_allowed():
    try:
        base=Path('/sys/fs/cgroup/memory')
        limit=int((base/'memory.limit_in_bytes').read_text())
        stat=dict(line.split() for line in (base/'memory.stat').read_text().splitlines())
        rss=int(stat['rss'])+int(stat.get('shmem',0))
        return 0<limit<2**60 and rss<.7*limit
    except (OSError,ValueError,KeyError):return False


@contextmanager
def high_resolution_slot(out,nx,ny):
    if nx*ny<=64*64:
        yield
        return
    directory=Path(out)/'campaign/high_resolution_slots';directory.mkdir(parents=True,exist_ok=True)
    streams=[(directory/(str(i)+'.lock')).open('a') for i in range(HIGH_RESOLUTION_SLOTS)]
    held=None
    try:
        while held is None:
            for index,stream in enumerate(streams):
                if index>=BASE_HIGH_RESOLUTION_SLOTS and not extra_slot_allowed():continue
                try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if index>=BASE_HIGH_RESOLUTION_SLOTS and not extra_slot_allowed():
                    fcntl.flock(stream,fcntl.LOCK_UN);continue
                held=stream;break
            if held is None:time.sleep(.1)
        yield
    finally:
        # The caller's task-local solver has left scope before this cleanup.
        # Release Python cycles and unused libc arenas before another large
        # task enters, rather than retaining peak allocations in idle workers.
        try:
            if held is not None:
                try:
                    gc.collect()
                    if os.name!='nt':
                        libc=ctypes.CDLL(None)
                        trim=getattr(libc,'malloc_trim',None)
                        if trim is not None:trim(0)
                finally:fcntl.flock(held,fcntl.LOCK_UN)
        finally:
            for stream in streams:stream.close()
