"""Untrained initializers; explicit adaptations, not native-paper reproductions."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.linalg import splu
from scipy.ndimage import gaussian_filter,map_coordinates,maximum_filter
from dataclasses import replace
from types import SimpleNamespace
from initskill.io import digest
from initskill.physics import mesh
from initskill.initializers import calibrate
from initskill.literature_initializers import generate_literature,prolong_design
from initskill.sensitivity_initializers import generate_sensitivity
from llm_six import SIMP
METHODS=('PLSM15','MMC_CROSS','GUIDED9_B025','SMOOTH_NOISE05','SMOOTH_NOISE10','STRESS_TRAJ_B025','STRESS_TRAJ_B050','STRESS_RIDGE_B025','COARSE30_MPC','COARSE80_MPC','OC5_PREOPT')

def blend(m,x,b):return (1-b)*m.initial()+b*x

def stress(m,ev):
 u=ev['u'][m.edof];dx=np.array([-.5,.5,.5,-.5]);dy=np.array([-.5,-.5,.5,.5])
 ux=u[:,0::2];uy=u[:,1::2];strain=np.c_[ux@dx,uy@dy,ux@dy+uy@dx];nu=m.config.nu
 D=np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])/(1-nu**2)
 return strain@D.T

def stress_paths(m,ev):
 """Uniform-FE principal lines, midpoint integration, sign continuity.
 Local fixed-spacing adaptation; no claim of exact Wang2022 seeding/width rules.
 """
 sig=stress(m,ev);grids=sig.reshape(m.ny,m.nx,3,order='F')
 def vector(p,axis,previous=None):
  coords=np.array([[m.ny-p[1]-.5],[p[0]-.5]])
  sx,sy,tau=[float(map_coordinates(grids[:,:,k],coords,order=1,mode='nearest')[0]) for k in range(3)]
  theta=.5*np.arctan2(2*tau,sx-sy)+axis*np.pi/2;v=np.array([np.cos(theta),np.sin(theta)])
  if previous is not None and np.dot(v,previous)<0:v=-v
  return v
 selected=[]
 for i in np.unique(np.r_[m.fixed//2,np.flatnonzero(m.force)//2]):
  selected.append(np.clip(m.nodes[i],[.5,.5],[m.nx-.5,m.ny-.5]))
 selected += [np.array([x,y]) for x in np.linspace(2,m.nx-2,5) for y in np.linspace(2,m.ny-2,5)]
 seeds=[]
 for p in selected:
  if all(np.linalg.norm(p-q)>2 for q in seeds):seeds.append(p)
 points=[];segments=0
 for seed in seeds:
  for axis in (0,1):
   for sign in (-1,1):
    pos=seed.copy();v=sign*vector(pos,axis);previous=[pos.copy()]
    for k in range(2*max(m.nx,m.ny)):
     v=vector(pos,axis,v);mid=pos+.25*v;vm=vector(mid,axis,v);new=pos+.5*vm
     if np.any(new<[.5,.5]) or np.any(new>[m.nx-.5,m.ny-.5]):break
     if k>12 and min(np.linalg.norm(new-q) for q in previous[:-8])<.3:break
     points.append(new);previous.append(new.copy());pos=new;v=vm;segments+=1
 if not points:raise ValueError('No principal stress lines')
 points=np.asarray(points);dist=cKDTree(points).query(m.centers)[0]
 # Width responds to this same unoptimized reference energy, no optimizer output.
 e=np.maximum(ev['energy'],1e-30);width=.35+1.2*np.sqrt(np.clip(e/np.percentile(e,90),0,1))
 field=(width-dist)/.6
 return calibrate(m,field),dict(seeds=len(seeds),trace_points=len(points),integration_step=.5,source='Wang et al2022 stress-trajectory principle; local seeding/width adaptation',native_paper_reproduction=False)

class CoarseMPC:
 """Factor2 Q4 mesh with exact physical-point supports via bilinear MPC rows."""
 def __init__(self,fine):
  self.audit_hook=getattr(fine,'coarse_audit_hook',None);factor=2;self.nx=fine.nx//2;self.ny=fine.ny//2;self.target=fine.target;self.config=replace(fine.config,rmin=fine.config.rmin/2)
  self.nodes,self.cells,self.centers=mesh(self.nx,self.ny);self.ke=fine.ke;self.fe_calls=0;self.edof=(2*self.cells[:,:,None]+np.arange(2)).reshape(-1,8);self.rows=np.repeat(self.edof,8,axis=1).ravel();self.cols=np.tile(self.edof,(1,8)).ravel()
  rr=[];cc=[];vv=[]
  for i,p in enumerate(fine.nodes/factor):
   ix=min(int(np.floor(p[0])),self.nx-1);iy=min(int(np.floor(p[1])),self.ny-1);a,b=p[0]-ix,p[1]-iy
   for dx,dy,w in [(0,0,(1-a)*(1-b)),(1,0,a*(1-b)),(0,1,(1-a)*b),(1,1,a*b)]:
    if abs(w)<1e-15:continue
    node=(ix+dx)*(self.ny+1)+(self.ny-(iy+dy))
    for axis in (0,1):rr.append(2*i+axis);cc.append(2*node+axis);vv.append(w)
  P=sparse.coo_matrix((vv,(rr,cc)),shape=(len(fine.force),2*len(self.nodes))).tocsr();self.P=P;self.force=np.asarray(P.T@fine.force).ravel();self.A=P[fine.fixed,:].tocsr()
  # Prescribed fine physical points retain their coordinates and constrained axes;
  # no nearest-node rounding. Constraint dependencies are rejected explicitly.
  if np.linalg.matrix_rank(self.A.toarray())!=self.A.shape[0]:raise ValueError('Dependent coarse MPC constraints; no silent dropping')
  f=fine.force.reshape(-1,2);c=self.force.reshape(-1,2)
  assert np.allclose(f.sum(0),c.sum(0),rtol=1e-10,atol=1e-10)
  mf=np.sum(fine.nodes[:,0]*f[:,1]-fine.nodes[:,1]*f[:,0]);mc=np.sum(2*self.nodes[:,0]*c[:,1]-2*self.nodes[:,1]*c[:,0]);assert np.isclose(mf,mc,rtol=1e-10,atol=1e-10)
  pairs=cKDTree(self.centers).sparse_distance_matrix(cKDTree(self.centers),self.config.rmin,output_type='coo_matrix');self.H=sparse.coo_matrix((np.maximum(0,self.config.rmin-pairs.data),(pairs.row,pairs.col)),shape=(len(self.cells),len(self.cells))).tocsr();self.Hs=np.asarray(self.H.sum(1)).ravel();self.dv=np.asarray(self.H.T@(np.ones(len(self.cells))/self.Hs)).ravel()
 def initial(self):return np.full(len(self.cells),self.target)
 def physical(self,x):return np.asarray(self.H@x).ravel()/self.Hs
 def update(self,x,e):return SIMP.update(self,x,e)
 def evaluate(self,x):
  c=self.config;rho=self.physical(x);w=c.emin+(c.emax-c.emin)*rho**c.penal
  K=sparse.coo_matrix(((w[:,None,None]*self.ke).ravel(),(self.rows,self.cols)),shape=(len(self.force),len(self.force))).tocsc();A=self.A
  system=sparse.bmat([[K,A.T],[A,None]],format='csc');rhs=np.r_[self.force,np.zeros(A.shape[0])];self.fe_calls+=1;
  if self.audit_hook:self.audit_hook('start')
  sol=splu(system).solve(rhs);
  if self.audit_hook:self.audit_hook('end')
  u=sol[:len(self.force)];res=np.linalg.norm(system@sol-rhs)/max(np.linalg.norm(rhs),1e-30);constraint=np.linalg.norm(A@u)/max(np.linalg.norm(u),1e-30)
  energy=np.einsum('ei,ij,ej->e',u[self.edof],self.ke,u[self.edof]);C=float(w@energy)
  if not np.isfinite(C) or C<=0 or res>1e-6 or constraint>1e-8:raise ValueError('Coarse equilibrium/MPC residual failed')
  grad=np.asarray(self.H.T@(-c.penal*(c.emax-c.emin)*rho**(c.penal-1)*energy/self.Hs)).ravel()
  return dict(compliance=C,volume=float(rho.mean()),residual=res,grayness=float(np.mean(4*rho*(1-rho))),gradient=grad,density=rho,energy=energy,u=u,constraint_residual=constraint)


def generate(m,name):
 before=m.fe_calls;meta=dict(method=name,native_paper_reproduction=False,fallback=False)
 if name=='PLSM15':x,z=generate_literature(m,'plsm_holes');meta.update(z)
 elif name=='MMC_CROSS':x,z=generate_literature(m,'mmc_cross');meta.update(z)
 elif name=='GUIDED9_B025':x,z=generate_sensitivity(m,'guided9_b025');meta.update(z)
 elif name.startswith('SMOOTH_NOISE'):
  amp=.05 if name.endswith('05') else .1;seed=int(digest([20261119,m.task['task_id'],'smooth_noise'])[:16],16);rng=np.random.default_rng(seed);noise=gaussian_filter(rng.standard_normal((m.ny,m.nx)),sigma=3,mode='reflect');noise-=noise.mean();noise/=max(noise.std(),1e-30);x=calibrate(m,amp*noise.ravel(order='F'));meta.update(seed=seed,logit_amplitude=amp,sigma_elements=3,source='Local frozen smooth perturbation control')
 elif name.startswith('STRESS_TRAJ'):
  ev=m.evaluate(m.initial());x,z=stress_paths(m,ev);b=.25 if name.endswith('025') else .5;x=blend(m,x,b);meta.update(z,beta=b)
 elif name=='STRESS_RIDGE_B025':
  ev=m.evaluate(m.initial());sig=stress(m,ev);s1=.5*(sig[:,0]+sig[:,1])+np.sqrt(.25*(sig[:,0]-sig[:,1])**2+sig[:,2]**2);s2=.5*(sig[:,0]+sig[:,1])-np.sqrt(.25*(sig[:,0]-sig[:,1])**2+sig[:,2]**2);v=np.maximum(abs(s1),abs(s2)).reshape(m.ny,m.nx,order='F');v=gaussian_filter(v,1);ridge=v>=maximum_filter(v,size=(3,3));ridge&=v>=np.percentile(v,60)
  if not ridge.any():raise ValueError('No stress ridge')
  pts=m.centers[ridge.ravel(order='F')];distance=cKDTree(pts).query(m.centers)[0];x=blend(m,calibrate(m,(1-distance)/.6),.25);meta.update(source='Local stress-magnitude ridge heuristic; NOT degenerate-point/separatrix topology reproduction',ridge_points=len(pts),beta=.25)
 elif name.startswith('COARSE'):
  updates=30 if name.startswith('COARSE30') else 80;coarse=CoarseMPC(m);q=coarse.initial();history=[];t=time.monotonic()
  for k in range(updates+1):
   ev=coarse.evaluate(q);history.append(dict(step=k,C=ev['compliance'],residual=ev['residual'],constraint_residual=ev['constraint_residual']))
   if k<updates:q=coarse.update(q,ev)
  suggested=prolong_design(m,coarse,q,2);from scipy.special import logit
  x=calibrate(m,logit(np.clip(suggested,1e-6,1-1e-6)));meta.update(coarse_FE=coarse.fe_calls,coarse_updates=updates,coarse_history=history,coarse_x=q.tolist(),coarse_nodes_xy=(2*coarse.nodes).tolist(),coarse_force=coarse.force.tolist(),mpc_shape=list(coarse.A.shape),mpc_rows=coarse.A.tocoo().row.tolist(),mpc_cols=coarse.A.tocoo().col.tolist(),mpc_values=coarse.A.tocoo().data.tolist(),initialization_seconds=time.monotonic()-t,source='Local coarse-to-fine adaptation; physical support points imposed by bilinear multipoint constraints; not nodal rounding')
 elif name=='OC5_PREOPT':
  x=m.initial();h=[]
  for k in range(6):
   ev=m.evaluate(x);h.append(dict(step=k,C=ev['compliance']))
   if k<5:x=m.update(x,ev)
  meta.update(preoptimization_updates=5,preoptimization_history=h,source='Budget control: Uniform5+OC50 equivalent to Uniform55 with same solver; not free improvement')
 else:raise ValueError(name)
 meta.update(initialization_fine_FE=m.fe_calls-before)
 if abs(m.physical(x).mean()-m.target)>1e-8 or not np.isfinite(x).all():raise ValueError('Initializer density/volume failed')
 return x,meta
