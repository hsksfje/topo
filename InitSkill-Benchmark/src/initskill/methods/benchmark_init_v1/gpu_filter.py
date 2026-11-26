"""Float64 serial-row CSR filtering, verified against CPU products bytewise.

Used only for the common GPU backend above64x64. CPU and64 filters remain
unchanged. See full training-trajectory and independent baseline audits.
"""
import numpy as np
from initskill.methods.benchmark_init_v1.gpu_sparse import CuDSSFactory

CUDA_SOURCE=r'''
extern "C" __global__ void serial_csr(
    const int n, const int* rowptr, const int* indices,
    const double* values, const double* x, double* y) {
    const int row=blockDim.x*blockIdx.x+threadIdx.x;
    if (row<n) {
        double sum=0.0;
        for (int j=rowptr[row];j<rowptr[row+1];++j)
            sum=sum+values[j]*x[indices[j]];
        y[row]=sum;
    }
}
'''


class _Transpose:
    def __init__(self,owner):self.owner=owner
    def __matmul__(self,x):return self.owner.apply(x,transpose=True)
    def __getattr__(self,name):return getattr(self.owner.matrix.T,name)


class SerialRowFilter:
    def __init__(self,matrix,verify=False):
        # Initialize the same pinned CUDA libraries without any FE solve.
        loader=CuDSSFactory();loader.close()
        import cupy as cp
        import cupyx.scipy.sparse as sparse_gpu
        if matrix.indices.dtype!=np.dtype('int32') or matrix.indptr.dtype!=np.dtype('int32'):
            raise ValueError('Expected int32 CSR indices')
        self.matrix=matrix;self.cp=cp;self.verify=verify;self.comparisons=0
        self.kernel=cp.RawKernel(CUDA_SOURCE,'serial_csr',options=('--fmad=false',))
        forward=sparse_gpu.csr_matrix(matrix)
        reverse=forward.T.tocsr();reverse.sort_indices()
        self.matrices=(forward,reverse)
        self.buffers=tuple(cp.empty(a.shape[0],dtype=cp.float64) for a in self.matrices)
        # Large host-to-device CSR uploads leave unused pinned-host blocks.
        # All uploads/transpose kernels must finish before releasing caches.
        cp.cuda.get_current_stream().synchronize()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    @property
    def T(self):return _Transpose(self)

    def __getattr__(self,name):return getattr(self.matrix,name)

    def __matmul__(self,x):return self.apply(x)

    def apply(self,x,transpose=False):
        x=np.asarray(x)
        if x.ndim!=1 or x.dtype!=np.dtype('float64'):
            return (self.matrix.T if transpose else self.matrix)@x
        cp=self.cp;index=int(transpose);a=self.matrices[index];n=a.shape[0]
        if len(x)!=a.shape[1]:raise ValueError('Filter vector length mismatch')
        gx=cp.asarray(np.ascontiguousarray(x));y=self.buffers[index]
        self.kernel(((n+127)//128,),(128,),
            (np.int32(n),a.indptr,a.indices,a.data,gx,y))
        result=cp.asnumpy(y)
        if self.verify:
            expected=(self.matrix.T if transpose else self.matrix)@x
            if expected.tobytes()!=result.tobytes():
                raise AssertionError('GPU/CPU filter product differs: transpose='+str(transpose)+
                    ', max_abs='+str(float(np.max(np.abs(expected-result)))))
            self.comparisons+=1
        return result

    def close(self):
        self.matrices=();self.buffers=()
        self.cp.get_default_memory_pool().free_all_blocks()
        self.cp.get_default_pinned_memory_pool().free_all_blocks()
