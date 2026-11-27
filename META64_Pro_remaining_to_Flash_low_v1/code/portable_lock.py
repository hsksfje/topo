"""POSIX/Windows interprocess file-lock compatibility for API receipts."""
import os,time
LOCK_EX=2;LOCK_NB=4;LOCK_UN=8
if os.name!='nt':
 from fcntl import flock,LOCK_EX,LOCK_NB,LOCK_UN
else:
 import msvcrt
 def flock(handle,operation):
  fd=handle.fileno() if hasattr(handle,'fileno') else int(handle)
  os.lseek(fd,0,os.SEEK_SET)
  if os.fstat(fd).st_size==0:os.write(fd,b'0');os.lseek(fd,0,os.SEEK_SET)
  if operation&LOCK_UN:msvcrt.locking(fd,msvcrt.LK_UNLCK,1);return
  while True:
   try:msvcrt.locking(fd,msvcrt.LK_NBLCK,1);return
   except OSError:
    if operation&LOCK_NB:raise
    time.sleep(.05)
