import numpy as np
import torch

class SCG:
    """Moller scaled conjugate gradients with finite-difference curvature."""
    def __init__(self,value_gradient,x):
        self.fun=value_gradient;self.x=x.copy();self.f,self.g=self.fun(self.x)
        self.d=-self.g;self.beta=5e-7;self.success=True;self.nsuccess=0
    def step(self):
        if self.success:
            self.mu=float(self.d@self.g)
            if self.mu>=0:self.d=-self.g;self.mu=float(self.d@self.g)
            self.kappa=float(self.d@self.d)
            if self.kappa<1e-30:return self.f
            sigma=5e-5/np.sqrt(self.kappa)
            _,gplus=self.fun(self.x+sigma*self.d)
            self.theta=float(self.d@(gplus-self.g))/sigma
        delta=self.theta+self.beta*self.kappa
        if delta<=0:
            delta=self.beta*self.kappa;self.beta-=self.theta/self.kappa
        alpha=-self.mu/delta;proposal=self.x+alpha*self.d
        fnew,gnew=self.fun(proposal)
        ratio=2*(fnew-self.f)/(alpha*self.mu)
        if np.isfinite(ratio) and ratio>=0:
            self.success=True;self.nsuccess+=1;oldg=self.g
            self.x=proposal;self.f=fnew;self.g=gnew
        else:self.success=False
        if not np.isfinite(ratio) or ratio<.25:self.beta=min(4*self.beta,1e100)
        elif ratio>.75:self.beta=max(.5*self.beta,1e-15)
        if self.success:
            if self.nsuccess==len(self.x):self.d=-self.g;self.nsuccess=0
            else:
                gamma=float((oldg-self.g)@self.g)/self.mu
                self.d=gamma*self.d-self.g
        return self.f
