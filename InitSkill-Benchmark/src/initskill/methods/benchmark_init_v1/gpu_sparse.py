"""Float64 cuDSS factors, lazily loaded in each GPU execution worker.

Same Q4 matrices and RHS as the CPU path. Symbolic analysis is reused only
when CSR structure is identical. See training-only numerical/OC checks.
"""
import ctypes
import logging
from pathlib import Path
import sys
import numpy as np


class CuDSSFactory:
    def __init__(self):
        root=Path(__file__).resolve().parents[4]
        packages=root/'.runtime/solver_gpu_packages'
        if str(packages) not in sys.path:sys.path.insert(0,str(packages))
        # The physics backend needs only the pinned NVIDIA libraries, not
        # PyTorch's large Python/CPU runtime. Neural inference loads its own.
        site=Path(sys.prefix)/'lib/python3.11/site-packages/nvidia'
        for name in [site/'cublas/lib/libcublasLt.so.12',site/'cublas/lib/libcublas.so.12',
                     site/'cuda_runtime/lib/libcudart.so.12']:
            ctypes.CDLL(str(name),mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(packages/'nvidia/cu12/lib/libcudss.so.0'),mode=ctypes.RTLD_GLOBAL)
        import nvmath.sparse.advanced as sa
        import cupy
        if cupy.cuda.runtime.getDeviceCount()<1:raise RuntimeError('Explicit cuDSS backend requires CUDA execution')
        self.sa=sa;self.packages=packages;self.solver=None;self.signature=None
        self.logger=logging.getLogger('initskill_cudss');self.logger.setLevel(logging.ERROR)

    def __call__(self,matrix):
        a=matrix.tocsr();a.sort_indices()
        return _Factor(self,a)

    def close(self):
        if self.solver is not None:self.solver.free();self.solver=None
        self.signature=None


class _Factor:
    """A regular short-lived object; no per-evaluation dynamic class cycle."""
    def __init__(self,owner,a):
        self.owner=owner;self.a=a;self.factored=False

    def solve(self,b):
        owner=self.owner;a=self.a
        b=np.ascontiguousarray(b,dtype=np.float64)
        signature=(a.shape,a.indptr.tobytes(),a.indices.tobytes())
        if owner.signature!=signature:
            owner.close()
            options=dict(sparse_system_type=owner.sa.DirectSolverMatrixType.SPD,
                sparse_system_view=owner.sa.DirectSolverMatrixViewType.FULL,
                multithreading_lib=str(owner.packages/'nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0'),
                blocking=True,logger=owner.logger)
            owner.solver=owner.sa.DirectSolver(a,b,options=options)
            owner.solver.plan();owner.signature=signature
        elif not self.factored:owner.solver.reset_operands(a=a,b=b)
        else:owner.solver.reset_operands(b=b)
        if not self.factored:owner.solver.factorize();self.factored=True
        return np.asarray(owner.solver.solve()).copy()
