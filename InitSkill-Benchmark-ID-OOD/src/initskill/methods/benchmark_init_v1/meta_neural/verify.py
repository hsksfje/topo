"""Compare immutable author FEM and native parameterization with platform FEM."""
import copy
from pathlib import Path
import numpy as np
from scipy import sparse
from ..common import DEFAULT_OUT,rows,solver,dump
from .core import setup,sample,Ledger


def main(out=None):
    out=DEFAULT_OUT if out is None else Path(out)
    jax,eqx,optax,mtp,common,model,learner,loss,coords,param=setup((64,64))
    row=rows('train1000')[0]
    ledger=Ledger(out/'verification/native_meta_numeric/FE_attempts.jsonl');ledger.attach()
    try:
        s={k:jax.numpy.asarray(v) for k,v in sample(row).items()}
        m=solver(out,'train1000',row)
        # Evaluate identical physical density in both solvers, disabling only
        # the local *design* filter for this FEM equality check.
        physical=copy.copy(m);n=m.nx*m.ny
        physical.H=sparse.eye(n,format='csr');physical.Hs=np.ones(n);physical.dv=np.ones(n)
        rng=np.random.default_rng(42);rho=rng.uniform(.25,.65,(64,64))
        source_solve=mtp.solver.setup_solver_cvxopt(young=1.,young_min=1e-9,poisson=.3,Lx=64.,Ly=64.,Nx=64,Ny=64,penalty=3.)
        author,gradient=jax.value_and_grad(source_solve)(jax.numpy.asarray(rho),s)
        local=physical.evaluate(rho[::-1,:].ravel(order='F'))
        error=abs(float(author)/local['compliance']-1)
        g=np.asarray(gradient)[::-1,:].ravel(order='F')
        ge=float(np.linalg.norm(g-local['gradient'])/np.linalg.norm(g))
        assert error<1e-7,(float(author),local['compliance'],error)
        assert ge<1e-6,ge
        params,static=eqx.partition(model,eqx.is_array)
        q=jax.numpy.concatenate([coords,s['strain_energy'].reshape(-1,1)],axis=1)
        rho0=np.asarray(param(q,s,params,static))
        native_loss,grads=eqx.filter_value_and_grad(loss,has_aux=True)(params,static,s)
        leaves=jax.tree.leaves(grads)
        assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
        count=sum(np.size(p) for p in jax.tree.leaves(params) if p is not None)
        report=dict(status='PASS',training_task=row['task_key'],compliance_relative_error=error,
            physical_gradient_relative_error=ge,native_volume=float(rho0.mean()),target=float(s['target_vf']),
            parameter_count=count,FE=ledger.count,native_loss=float(native_loss[0]),
            native_filter='author cone + Hoyer (unchanged published tolerance)')
        dump(out/'verification/native_meta_numeric.json',report);print(report,flush=True)
    finally:ledger.detach()


if __name__=='__main__':
    import argparse,os
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    # This independent comparison runs in the author's JAX/NumPy runtime;
    # formal common OC uses the separate, unified execution environment.
    os.environ['INITSKILL_FE_SOLVER']='superlu'
    main(parser.parse_args().out)
