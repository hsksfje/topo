"""Verbatim feasible-side OC class extracted from the original frozen solver."""
import numpy as np
from physics import SIMP as BaseSIMP

class SIMP(BaseSIMP):
    def update(self,x,ev):
        c=self.config;dc=np.minimum(ev['gradient'],0.)
        lo,hi,feasible=0.,1e9,None
        for _ in range(250):
            mid=.5*(lo+hi)
            new=np.clip(x*np.sqrt(-dc/self.dv/mid),np.maximum(c.xmin,x-c.move),np.minimum(1.,x+c.move))
            if self.physical(new).mean()>self.target:lo=mid
            else:hi,feasible=mid,new
            if (hi-lo)/(hi+lo)<=1e-8 and feasible is not None:return feasible
        raise RuntimeError('Feasible-side OC bisection failed')
