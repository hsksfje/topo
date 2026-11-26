"""Q4 plane-stress SIMP/OC. Adapted from the local isotropic_simp.py equations.

Same x-major, top-row-first ordering and density filter as EngiBench/top88.
Independent implementation: no CFRC material, RBF, fiber variables or MMA.
All evaluations represent the design AFTER exactly `step` OC updates.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class SolverConfig:
    rmin: float = 2.5
    penal: float = 3.0
    move: float = 0.2
    xmin: float = 1e-6
    emin: float = 1e-9
    emax: float = 1.0
    nu: float = 0.3
    residual_tolerance: float = 1e-6

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Nonfinite solver parameter')
        if not (self.rmin > 0 and self.penal >= 1 and 0 < self.move <= 1
                and 0 <= self.xmin < 1 and 0 < self.emin < self.emax
                and -1 < self.nu < .5 and self.residual_tolerance > 0):
            raise ValueError('Invalid SIMP/OC settings')


def mesh(nx, ny):
    xx, yy = np.meshgrid(np.arange(nx+1), np.arange(ny, -1, -1))
    nodes = np.c_[xx.ravel(order='F'), yy.ravel(order='F')].astype(float)
    nr = np.arange(len(nodes)).reshape(ny+1, nx+1, order='F')
    cells = np.column_stack([a.ravel(order='F') for a in
                            (nr[1:, :-1], nr[1:, 1:], nr[:-1, 1:], nr[:-1, :-1])])
    return nodes, cells, nodes[cells].mean(axis=1)


def stiffness(nu):
    d = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])/(1-nu**2)
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    ke = np.zeros((8, 8))
    for xi in (-1/np.sqrt(3), 1/np.sqrt(3)):
        for eta in (-1/np.sqrt(3), 1/np.sqrt(3)):
            dn = np.array([[-1+eta, 1-eta, 1+eta, -1-eta],
                           [-1+xi, -1-xi, 1+xi, 1-xi]])/4
            jac = dn @ coords
            deriv = np.linalg.solve(jac, dn)
            b = np.zeros((3, 8))
            b[0, 0::2], b[1, 1::2] = deriv[0], deriv[1]
            b[2, 0::2], b[2, 1::2] = deriv[1], deriv[0]
            ke += b.T @ d @ b * np.linalg.det(jac)
    return ke


class SIMP:
    def __init__(self, directory, config=None, *, filter_factory=None):
        self.root = Path(directory)
        self.task = json.loads((self.root/'task.json').read_text())
        if self.task['schema'] != 'initskill-isotropic-task-v1':
            raise ValueError('Use isolated isotropic tasks, not old CFRC tasks')
        self.config = config or SolverConfig()
        self.nx, self.ny = self.task['elements_xy']
        self.target = self.task['volume_fraction']
        if not self.config.xmin < self.target < 1:
            raise ValueError('Target volume outside feasible bounds')
        with np.load(self.root/'domain.npz', allow_pickle=False) as a:
            self.nodes, self.cells, self.centers = [a[k].copy() for k in ('nodes', 'cells', 'centers')]
        with np.load(self.root/'boundary_conditions.npz', allow_pickle=False) as a:
            self.fixed = a['fixed_dofs'].copy()
            self.force = a['force'].copy()
        n = len(self.cells)
        if self.force.shape != (2*len(self.nodes),) or n != self.nx*self.ny:
            raise ValueError('Invalid geometry/force shapes')
        if not np.isfinite(self.force).all() or np.linalg.norm(self.force) == 0:
            raise ValueError('Nonfinite or zero force')
        if self.fixed.dtype.kind not in 'iu' or np.any(self.fixed < 0) or np.any(self.fixed >= len(self.force)):
            raise ValueError('Invalid constrained DOFs')
        if len(np.unique(self.fixed)) != len(self.fixed) or np.any(self.force[self.fixed] != 0):
            raise ValueError('Duplicate supports or force on constrained DOF')
        # The current kernel supports full, unit-square rectangular grids only.
        expected = mesh(self.nx, self.ny)
        if not all(np.array_equal(a, b) for a, b in zip((self.nodes, self.cells, self.centers), expected)):
            raise ValueError('Unsupported geometry: expected audited unit square grid')
        rigid = []
        for dof in self.fixed:
            x, y = self.nodes[dof//2]
            rigid.append([1., 0., -y] if dof % 2 == 0 else [0., 1., x])
        if np.linalg.matrix_rank(np.asarray(rigid)) < 3:
            raise ValueError('Supports leave a planar rigid-body mode')
        self.free = np.setdiff1d(np.arange(len(self.force)), self.fixed)
        self.edof = (2*self.cells[:, :, None]+np.arange(2)).reshape(-1, 8)
        self.rows = np.repeat(self.edof, 8, axis=1).ravel()
        self.cols = np.tile(self.edof, (1, 8)).ravel()
        self.ke = stiffness(self.config.nu)
        if filter_factory is None:
            tree = cKDTree(self.centers)
            pairs = tree.sparse_distance_matrix(tree, self.config.rmin, output_type='coo_matrix')
            self.H = sparse.coo_matrix((np.maximum(0., self.config.rmin-pairs.data),
                                       (pairs.row, pairs.col)), shape=(n, n)).tocsr()
            self.Hs = np.asarray(self.H.sum(axis=1)).ravel()
            self.dv = np.asarray(self.H.T @ (np.ones(n)/self.Hs)).ravel()
        else:
            self.H,self.Hs,self.dv=filter_factory(self.centers,self.config.rmin)
        self.fe_calls = 0

    def physical(self, x):
        return np.asarray(self.H @ x).ravel()/self.Hs

    def initial(self):
        return np.full(len(self.cells), self.target)

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        c = self.config
        if x.shape != (len(self.cells),) or not np.isfinite(x).all() or np.any(x < c.xmin-1e-12) or np.any(x > 1+1e-12):
            raise ValueError('Invalid design density')
        rho = np.clip(self.physical(x), c.xmin, 1.)
        w = c.emin + (c.emax-c.emin)*rho**c.penal
        K = sparse.coo_matrix(((w[:, None, None]*self.ke).ravel(), (self.rows, self.cols)),
                              shape=(len(self.force), len(self.force))).tocsc()
        # Count every attempted factorization, including failed numerical solves.
        self.fe_calls += 1
        u = np.zeros_like(self.force)
        u[self.free] = splu(K[self.free][:, self.free]).solve(self.force[self.free])
        ue = u[self.edof]
        energy = np.einsum('ei,ij,ej->e', ue, self.ke, ue)
        compliance = float(w @ energy)
        residual = float(np.linalg.norm((K@u-self.force)[self.free])/np.linalg.norm(self.force[self.free]))
        if not (np.isfinite(compliance) and compliance > 0 and residual < c.residual_tolerance):
            raise ValueError('FE equilibrium or positive-compliance check failed')
        physical_gradient = -c.penal*(c.emax-c.emin)*rho**(c.penal-1)*energy
        gradient = np.asarray(self.H.T @ (physical_gradient/self.Hs)).ravel()
        return dict(compliance=compliance, volume=float(rho.mean()), grayness=float(np.mean(4*rho*(1-rho))),
                    residual=residual, density=rho, gradient=gradient, energy=energy, u=u)

    def update(self, x, evaluation):
        c = self.config
        dc = np.minimum(evaluation['gradient'], 0.)
        low, high = 0., 1e9
        for _ in range(250):
            mid = .5*(low+high)
            new = np.clip(x*np.sqrt(-dc/self.dv/mid), np.maximum(c.xmin, x-c.move), np.minimum(1., x+c.move))
            if self.physical(new).mean() > self.target:
                low = mid
            else:
                high = mid
            if (high-low)/(high+low) <= 1e-3:
                return new
        raise RuntimeError('OC volume bisection failed')
