"""Share two baseline slots and an optional third across formal OC pipelines.

This is scheduling only: no task, model, convergence rule or budget changes.
After the factor-lifetime fix, observed workers use about0.9-1.2GiB RSS.
The extra slot requires anonymous/shared RSS below70% of the host limit.
"""
from contextlib import contextmanager
import ctypes
from initskill import locking as fcntl
import gc
from pathlib import Path
import time

BASE_HIGH_RESOLUTION_SLOTS=2
HIGH_RESOLUTION_SLOTS=3


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
        if held is not None:
            gc.collect()
            libc=ctypes.CDLL(None)
            trim=getattr(libc,'malloc_trim',None)
            if trim is not None:trim(0)
            fcntl.flock(held,fcntl.LOCK_UN)
        for stream in streams:stream.close()
