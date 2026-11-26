"""Cross-platform advisory locks; matches the flock interface used here."""
import errno
import os
import time

if os.name!='nt':
    from fcntl import flock,LOCK_EX,LOCK_NB,LOCK_UN
else:
    import msvcrt
    LOCK_EX,LOCK_NB,LOCK_UN=2,4,8

    def flock(stream,operation):
        # A single byte shared by every participant; no symlink/admin rights.
        if os.fstat(stream.fileno()).st_size==0:
            stream.write('\0');stream.flush()
        stream.seek(0)
        if operation & LOCK_UN:
            msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1);return
        while True:
            try:
                stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1);return
            except OSError as exc:
                if exc.errno not in (errno.EACCES,errno.EAGAIN,errno.EDEADLK):raise
                if operation & LOCK_NB:raise BlockingIOError(exc.errno,str(exc)) from exc
                time.sleep(.05)


def atomic_replace(source,target):
    """Readers/antivirus may briefly hold a Windows destination handle."""
    for attempt in range(101):
        try:os.replace(source,target);return
        except PermissionError:
            if attempt==100:raise
            time.sleep(.05)
