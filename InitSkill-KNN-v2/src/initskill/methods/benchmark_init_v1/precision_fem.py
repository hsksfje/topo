"""Platform-independent Decimal Q4 refinement. No longdouble dependency.

Only rejected float64 evaluations enter here. The same physical coefficients,
Dirichlet conditions and1e-6 acceptance threshold are retained. 40 decimal
significant digits are used on Windows and Linux alike. LU is a preconditioner.
"""
from decimal import Decimal, localcontext
import math
import time
import numpy as np

DIGITS=40
D=Decimal.from_float


def quadrature_decimal(nu):
    n=D(float(nu));one=Decimal(1);zero=Decimal(0)
    mat=[[one,n,zero],[n,one,zero],[zero,zero,(one-n)/2]]
    mat=[[a/(one-n*n) for a in row] for row in mat]
    gauss=one/Decimal(3).sqrt();Bs=[];ke=[[zero for _ in range(8)] for _ in range(8)]
    for xi in [-gauss,gauss]:
        for eta in [-gauss,gauss]:
            dx=[(-one+eta)/2,(one-eta)/2,(one+eta)/2,(-one-eta)/2]
            dy=[(-one+xi)/2,(-one-xi)/2,(one+xi)/2,(one-xi)/2]
            B=[[zero for _ in range(8)] for _ in range(3)]
            for j in range(4):B[0][2*j]=dx[j];B[1][2*j+1]=dy[j];B[2][2*j]=dy[j];B[2][2*j+1]=dx[j]
            Bs.append(B)
            for i in range(8):
                for j in range(8):ke[i][j]+=sum(B[a][i]*mat[a][b]*B[b][j] for a in range(3) for b in range(3))/4
    return Bs,mat,ke


def refined_evaluation(m,x,rho,factor,u64,raw_residual):
    start=time.perf_counter();c=m.config
    with localcontext() as ctx:
        ctx.prec=DIGITS
        if not hasattr(m,'_decimal_quadrature'):m._decimal_quadrature=quadrature_decimal(c.nu)
        Bs,mat,ke=m._decimal_quadrature
        emin=D(float(c.emin));span=D(float(c.emax))-emin;p=D(float(c.penal))
        rh=[D(float(v)) for v in rho];weights=[emin+span*ctx.power(v,p) for v in rh]
        # Assemble free rows directly in Decimal, including the coefficient
        # products and duplicate sums. Casting an assembled float64 K would
        # already have lost the residual precision we need to recover.
        lookup=np.full(len(m.force),-1,dtype=np.int64);lookup[m.free]=np.arange(len(m.free))
        A=[{} for _ in m.free]
        for e,edof in enumerate(m.edof):
            ids=lookup[edof];w=weights[e]
            for i,ri in enumerate(ids):
                if ri<0:continue
                row=A[int(ri)]
                for j,cj in enumerate(ids):
                    if cj>=0:
                        col=int(cj);row[col]=row.get(col,Decimal(0))+w*ke[i][j]
        rows=[list(row.items()) for row in A];del A
        force=[D(float(v)) for v in m.force[m.free]]
        un=[D(float(v)) for v in u64[m.free]]
        norm=sum(v*v for v in force).sqrt();refinements=0;residual=math.inf
        for iteration in range(9):
            r=[f-sum(a*un[j] for j,a in row) for f,row in zip(force,rows)]
            residual=float(sum(v*v for v in r).sqrt()/norm)
            if residual<c.residual_tolerance*.1:break
            if iteration==8:break
            correction=factor.solve(np.array([float(v) for v in r],dtype=float))
            if not np.isfinite(correction).all():break
            un=[v+D(float(delta)) for v,delta in zip(un,correction)];refinements+=1
        if not math.isfinite(residual) or residual>=c.residual_tolerance:
            raise ValueError(f'Decimal{DIGITS} FE residual {residual:.8g} exceeds {c.residual_tolerance}; raw={raw_residual:.8g}; corrections={refinements}')
        u=[Decimal(0) for _ in m.force]
        for idx,v in zip(m.free,un):u[int(idx)]=v
        energy=[]
        for edof in m.edof:
            ue=[u[int(i)] for i in edof];base=ue[:2];relative=[v-base[j%2] for j,v in enumerate(ue)]
            en=Decimal(0)
            for B in Bs:
                strain=[sum(v*a for v,a in zip(relative,row)) for row in B]
                en+=sum(strain[i]*mat[i][j]*strain[j] for i in range(3) for j in range(3))/4
            energy.append(en)
        compliance=float(sum(w*e for w,e in zip(weights,energy)));work=float(sum(f*v for f,v in zip(force,un)))
        if not math.isfinite(compliance) or compliance<=0:raise ValueError('Decimal FE positive compliance failed')
        gradient_rho=np.array([float(-p*span*ctx.power(v,p-1)*e) for v,e in zip(rh,energy)])
        gradient=np.asarray(m.H.T@(gradient_rho/m.Hs)).ravel()
        if not np.isfinite(gradient).all():raise ValueError('Decimal FE gradient is not finite')
        balance=abs(compliance-work)/max(abs(work),1e-30)
        if balance>1e-6:raise ValueError(f'Decimal FE energy balance failed: {balance}')
        return dict(compliance=compliance,volume=float(rho.mean()),grayness=float(np.mean(4*rho*(1-rho))),
            residual=residual,density=rho,gradient=gradient,energy=np.array([float(v) for v in energy]),
            u=np.array([float(v) for v in u]),precision='decimal40-Q4-iterative-refinement',
            raw_residual=float(raw_residual),refinement_steps=refinements,external_work=work,
            energy_balance_relative=balance,refinement_seconds=time.perf_counter()-start)
