import numpy as np
import torch

def conjugate_gradient(operator,b,tol=1e-8,maxiter=200):
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rr=torch.dot(r,r)
    initial=float(rr.sqrt())
    if initial==0:return x,0,0.
    for iteration in range(1,maxiter+1):
        ap=operator(p);alpha=rr/torch.dot(p,ap).clamp_min(torch.finfo(b.dtype).tiny)
        x=x+alpha*p;r=r-alpha*ap;new=torch.dot(r,r)
        if float(new.sqrt())<=tol*initial:return x,iteration,float(new.sqrt())/initial
        p=r+(new/rr)*p;rr=new
    return x,iteration,float(rr.sqrt())/initial

class LM:
    """Solve (J.T J + mu I) delta = -J.T residual using exact AD products.

    This changes only the linear algebra implementation, not the LM normal
    equations. CG tolerance and residual are retained for every attempt.
    """
    def __init__(self,residual,x):
        self.residual=residual;self.x=x.detach().clone();self.mu=.001;self.history=[]
    def step(self):
        r,vjp=torch.func.vjp(self.residual,self.x)
        gradient=vjp(r)[0];old=float(torch.dot(r,r));accepted=False
        while self.mu<=1e10:
            def operator(v):
                jv=torch.func.jvp(self.residual,(self.x,),(v,))[1]
                return vjp(jv)[0]+self.mu*v
            delta,count,relative=conjugate_gradient(operator,-gradient)
            proposed=self.x+delta;rn=self.residual(proposed);new=float(torch.dot(rn,rn))
            self.history.append(dict(mu=self.mu,cg_iterations=count,cg_relative_residual=relative,old=old,new=new))
            if np.isfinite(new) and new<old:
                self.x=proposed.detach();self.mu=max(self.mu*.1,1e-20);accepted=True;break
            self.mu*=10
        return (new if accepted else old),accepted
