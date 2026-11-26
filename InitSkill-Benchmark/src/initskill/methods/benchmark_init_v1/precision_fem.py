"""Mixed-precision Q4 refinement for otherwise rejected equilibrium solves.

The physical problem, density and tolerance do not change. Existing float64
factors are reused as a preconditioner; displacement/residual and quadrature
energy are accumulated with extended precision. Healthy float64 solves never
enter this path.
"""
import numpy as np
from scipy import sparse


def quadrature(nu):
    ld=np.longdouble;nu=ld(nu)
    D=np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]],dtype=ld)/(1-nu**2)
    Bs=[];ke=np.zeros((8,8),dtype=ld)
    for xi in (-1/np.sqrt(ld(3)),1/np.sqrt(ld(3))):
        for eta in (-1/np.sqrt(ld(3)),1/np.sqrt(ld(3))):
            dn=np.array([[-1+eta,1-eta,1+eta,-1-eta],[-1+xi,-1-xi,1+xi,1-xi]],dtype=ld)/2
            B=np.zeros((3,8),dtype=ld)
            B[0,::2]=dn[0];B[1,1::2]=dn[1];B[2,::2]=dn[1];B[2,1::2]=dn[0]
            Bs.append(B);ke+=B.T@D@B/4
    return np.stack(Bs),D,ke


def refined_evaluation(m,x,rho,factor,u64,raw_residual):
    ld=np.longdouble;c=m.config
    if not hasattr(m,'_extended_quadrature'):m._extended_quadrature=quadrature(c.nu)
    Bs,D,ke=m._extended_quadrature
    rh=rho.astype(ld);w=ld(c.emin)+(ld(c.emax)-ld(c.emin))*rh**c.penal
    K=sparse.coo_matrix(((w[:,None,None]*ke).ravel(),(m.rows,m.cols)),
                       shape=(len(m.force),len(m.force))).tocsc()
    A=K[m.free][:,m.free];force=m.force[m.free].astype(ld)
    norm=np.sqrt(np.sum(force*force));free_u=u64[m.free].astype(ld)
    residual=np.inf;refinements=0
    for iteration in range(9):
        r=force-A@free_u;residual=float(np.sqrt(np.sum(r*r))/norm)
        if residual<c.residual_tolerance*.1:break
        if iteration==8:break
        correction=factor.solve(np.asarray(r,dtype=float))
        free_u+=correction.astype(ld);refinements+=1
    if not np.isfinite(residual) or residual>=c.residual_tolerance:
        raise ValueError('FE equilibrium or positive-compliance check failed')
    u=np.zeros(len(m.force),dtype=ld);u[m.free]=free_u
    ue=u[m.edof]
    # Strains annihilate rigid translation analytically; subtract it before
    # contraction rather than cancelling terms of order displacement squared.
    relative=ue-np.tile(ue[:,:2],(1,4))
    strains=np.einsum('qij,ej->eqi',Bs,relative)
    energy=np.einsum('eqi,ij,eqj->e',strains,D,strains)/4
    compliance=float(w@energy);work=float(force@free_u)
    if not np.isfinite(compliance) or compliance<=0:raise ValueError('FE equilibrium or positive-compliance check failed')
    physical_gradient=-ld(c.penal)*(ld(c.emax)-ld(c.emin))*rh**(c.penal-1)*energy
    gradient=np.asarray(m.H.T@(physical_gradient/m.Hs),dtype=float).ravel()
    if not np.isfinite(gradient).all():raise ValueError('FE equilibrium or positive-compliance check failed')
    return dict(compliance=compliance,volume=float(rho.mean()),grayness=float(np.mean(4*rho*(1-rho))),
        residual=residual,density=rho,gradient=gradient,energy=np.asarray(energy,dtype=float),u=u,
        precision='extended-Q4-iterative-refinement',raw_residual=raw_residual,
        refinement_steps=refinements,external_work=work,energy_balance_relative=abs(compliance-work)/max(abs(work),1e-30))
