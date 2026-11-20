"""Child-code resource limits: POSIX rlimits or Windows Job Object."""
import os
_WINDOWS_JOB=None
def install_limits():
 if os.name!='nt':
  import resource
  resource.setrlimit(resource.RLIMIT_CPU,(8,8));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));resource.setrlimit(resource.RLIMIT_FSIZE,(0,0));return
 import ctypes
 from ctypes import wintypes
 class Basic(ctypes.Structure):
  _fields_=[('PerProcessUserTimeLimit',ctypes.c_longlong),('PerJobUserTimeLimit',ctypes.c_longlong),('LimitFlags',wintypes.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),('MaximumWorkingSetSize',ctypes.c_size_t),('ActiveProcessLimit',wintypes.DWORD),('Affinity',ctypes.c_size_t),('PriorityClass',wintypes.DWORD),('SchedulingClass',wintypes.DWORD)]
 class IO(ctypes.Structure):
  _fields_=[(name,ctypes.c_ulonglong) for name in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
 class Extended(ctypes.Structure):
  _fields_=[('BasicLimitInformation',Basic),('IoInfo',IO),('ProcessMemoryLimit',ctypes.c_size_t),('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]
 kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.CreateJobObjectW.restype=wintypes.HANDLE;kernel.GetCurrentProcess.restype=wintypes.HANDLE
 job=kernel.CreateJobObjectW(None,None)
 if not job:raise ctypes.WinError(ctypes.get_last_error())
 limits=Extended();limits.BasicLimitInformation.PerProcessUserTimeLimit=8*10**7;limits.BasicLimitInformation.LimitFlags=0x2|0x100|0x2000;limits.ProcessMemoryLimit=2*1024**3
 if not kernel.SetInformationJobObject(wintypes.HANDLE(job),9,ctypes.byref(limits),ctypes.sizeof(limits)):raise ctypes.WinError(ctypes.get_last_error())
 if not kernel.AssignProcessToJobObject(wintypes.HANDLE(job),wintypes.HANDLE(kernel.GetCurrentProcess())):raise ctypes.WinError(ctypes.get_last_error())
 global _WINDOWS_JOB
 _WINDOWS_JOB=job
