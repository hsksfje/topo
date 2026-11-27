"""Numerical port and platform checks using training tasks only."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
import copy

from .common import DEFAULT_OUT,dump,rows,solver,calibrate,load,features
from .ulu_pca.optimizer import SCG
from .wang_pca.optimizer import LM,conjugate_gradient


def main(out=DEFAULT_OUT):
    torch.set_num_threads(1)
    rng=np.random.default_rng(42)
    a=rng.normal(size=(10,10));h=a.T@a+np.eye(10);b=rng.normal(size=10)
    def quadratic(x):return .5*x@h@x+b@x,h@x+b
    scg=SCG(quadratic,np.zeros(10))
    for _ in range(300):scg.step()
    error=float(np.linalg.norm(scg.x+np.linalg.solve(h,b)))
    assert error<1e-5,error
    a=torch.tensor(rng.normal(size=(20,8)),dtype=torch.float64)
    b=torch.tensor(rng.normal(size=20),dtype=torch.float64)
    def residual(x):return a@x-b
    initial=torch.zeros(8,dtype=torch.float64)
    lm=LM(residual,initial);expected=torch.linalg.solve(a.T@a+.001*torch.eye(8),a.T@b)
    lm.step();lm_error=float(torch.linalg.norm(lm.x-expected))
    assert lm_error<1e-7,lm_error
    for _ in range(10):lm.step()
    lm_final=float(torch.linalg.norm(lm.x-torch.linalg.lstsq(a,b).solution))
    assert lm_final<1e-6,lm_final
    nonlinear=lambda x:torch.stack([torch.sin(x[0])+x[1]**2,x[0]*x[1],x[0]**2-1])
    x=torch.tensor([.3,.4],dtype=torch.float64);r,vjp=torch.func.vjp(nonlinear,x)
    jac=torch.func.jacrev(nonlinear)(x)
    def op(v):return vjp(torch.func.jvp(nonlinear,(x,),(v,))[1])[0]+.01*v
    q,it,rel=conjugate_gradient(op,-vjp(r)[0])
    dense=torch.linalg.solve(jac.T@jac+.01*torch.eye(2,dtype=torch.float64),-jac.T@r)
    nonlinear_error=float(torch.linalg.norm(q-dense));assert nonlinear_error<1e-9
    report=dict(SCG_quadratic_solution_error=error,LM_first_step_vs_explicit_J_error=lm_error,
        LM_linear_least_squares_solution_error=lm_final,LM_nonlinear_AD_vs_dense_step_error=nonlinear_error,
        port_status='Python mathematical equivalence checked; no MATLAB binary available for bitwise cross-language test')
    dump(Path(out)/'verification/optimizers.json',report)
    # FEM finite difference on a real training task, including density filter.
    row=rows('train1000')[0];m=solver(out,'train1000',row)
    ledger=Path(out)/'verification/FE_attempts.jsonl';ledger.parent.mkdir(parents=True,exist_ok=True)
    attempts=sum(1 for _ in ledger.open()) if ledger.exists() else 0
    evaluate=m.evaluate
    def counted(x):
        nonlocal attempts
        attempts+=1
        with ledger.open('a') as stream:stream.write(json.dumps(dict(attempt=attempts,scope='training-only numerical verifier'))+'\n')
        return evaluate(x)
    m.evaluate=counted
    x=calibrate(rng.uniform(.2,.7,m.nx*m.ny),m);ev=m.evaluate(x)
    d=rng.normal(size=len(x));d/=np.linalg.norm(d);eps=1e-5
    fd=(m.evaluate(x+eps*d)['compliance']-m.evaluate(x-eps*d)['compliance'])/(2*eps)
    analytic=float(ev['gradient']@d);gradient_error=abs(fd-analytic)/max(1,abs(fd))
    assert gradient_error<1e-5,gradient_error
    uniform=m.evaluate(m.initial())
    # Reference scalar is read by this independent verifier only.
    from .common import DATA
    ref=float(np.load(DATA/row['assets']['reference_compliance']['path']).item())
    reference_error=abs(uniform['compliance']/ref-1);assert reference_error<1e-5
    from .cnn_encoder_decoder.core import YuEncoderDecoder
    from .scope import current_scope
    ed=YuEncoderDecoder().eval()
    with torch.no_grad():
        image=ed(torch.randn(1,4,33,33),torch.tensor([.3]));assert image.shape==(1,1,32,32)
    if current_scope()['include_cross_resolution']:
        from .cnn_cgan.core import YuGenerator,YuDiscriminator
        g=YuGenerator().eval();disc=YuDiscriminator(32).eval()
        with torch.no_grad():
            fine=g(torch.rand(1,1,8,8));assert fine.shape==(1,1,32,32)
            score=disc(torch.rand(1,1,8,8),fine);assert score.shape==(1,1)
        report.update(cGAN_8_to32_shape=True,conditional_discriminator=True)
    else:report['cGAN_status']='deferred; not imported or executed'
    optimizer=torch.optim.Adam(ed.parameters());optimizer.zero_grad()
    loss=ed(torch.randn(1,4,33,33),torch.tensor([.3])).abs().mean();loss.backward();optimizer.step()
    report.update(filtered_FE_gradient_relative_error=gradient_error,uniform_reference_relative_error=reference_error,
        CNN_input33_output32=True,CNN_backward_step=True,
        CNN_conv_count=sum(isinstance(x,torch.nn.Conv2d) for x in ed.modules()),
        status='PASS',training_task=row['task_key'])
    dump(Path(out)/'verification/python_platform.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    main(parser.parse_args().out)
