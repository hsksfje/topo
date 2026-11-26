"""Fixed-geometry assembly preserving the pinned SciPy duplicate summation.

Only the COO routing/sort is reused. The same SciPy float64 duplicate kernel
still accumulates each matrix. Verified by exact matrix bytes and training OC
trajectories; see verification/planned_assembly_trajectories.
"""
import numpy as np
from scipy import sparse
from scipy.sparse import _sparsetools


class AssemblyPlan:
    def __init__(self,m):
        n=len(m.force);length=len(m.rows)
        index=np.int32 if max(n,length)<2**31 else np.int64
        indptr=np.empty(n+1,dtype=index);indices=np.empty(length,dtype=index)
        order=np.empty(length,dtype=np.int64)
        _sparsetools.coo_tocsr(n,n,length,m.cols.astype(index,copy=False),
            m.rows.astype(index,copy=False),np.arange(length,dtype=np.int64),indptr,indices,order)
        shuffled=sparse.csc_matrix((order,indices,indptr),shape=(n,n));shuffled.sort_indices()
        self.sorted_indices=shuffled.indices.copy();self.unmerged_indptr=shuffled.indptr.copy()
        boundaries=np.r_[True,shuffled.indices[1:]!=shuffled.indices[:-1]]
        boundaries[shuffled.indptr[:-1]]=True
        self.starts=np.flatnonzero(boundaries);self.order=shuffled.data
        self.indices=shuffled.indices[self.starts].copy()
        self.indptr=np.searchsorted(self.starts,shuffled.indptr).astype(index)
        self.shape=(n,n)
    def __call__(self,values):
        data=values[self.order];indices=self.sorted_indices.copy();indptr=self.unmerged_indptr.copy()
        _sparsetools.csr_sum_duplicates(self.shape[1],self.shape[0],indptr,indices,data)
        length=int(indptr[-1]);assert length==len(self.indices)
        return sparse.csc_matrix((data[:length],self.indices,self.indptr),shape=self.shape,copy=False)
