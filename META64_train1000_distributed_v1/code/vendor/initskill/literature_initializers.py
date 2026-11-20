"""Literature-derived layouts and coarse-to-fine warm starts.

These are initial-state transfers, not reproductions of the original optimizers.
See docs/LITERATURE_INITIALIZERS_zh.md for primary sources and adaptations.
"""
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import logit

from .io import dump, save_npz
from .physics import SIMP, mesh

METHODS = ('plsm_holes', 'mmc_cross', 'coarse_simp30', 'coarse_simp80')


def plsm_field(model):
    # Wei et al. 2018, publisher MATLAB supplement, initialization lines 4-10.
    hx = np.r_[np.tile([1/6, 5/6], 3), np.tile([0, 1/3, 2/3, 1], 2), .5]
    hy = np.r_[np.repeat([0, .5, 1], 2), np.repeat([.25, .75], 4), .5]
    centers = np.c_[hx*model.nx, hy*model.ny]
    distances = np.linalg.norm(model.centers[:, None, :]-centers[None, :, :], axis=2)
    return np.clip(distances.min(axis=1)-.1*model.ny, -3., 3.)


def mmc_field(model):
    # Zhang et al. 2016: paired crossed components on a regular .5H grid.
    # Independently evaluate their variable-thickness superellipse equation.
    coords = model.centers/model.ny
    fields, components = [], []
    for cx in np.arange(.25, model.nx/model.ny, .5):
        for cy in (.25, .75):
            for st in (.7, -.7):
                ct = np.sqrt(1-st*st)
                delta = coords-[cx, cy]
                along = ct*delta[:, 0]+st*delta[:, 1]
                across = -st*delta[:, 0]+ct*delta[:, 1]
                length, t1, t2, t3 = .38, .04, .06, .04
                width = ((t2+t1-2*t3)/(2*length**2))*along**2+(t2-t1)/(2*length)*along+t3
                # Even exponent p=6; zero width outside the ends is safely void.
                inside = np.abs(along) <= length
                phi = np.full(len(coords), -3.)
                phi[inside] = 1-(along[inside]/length)**6-(across[inside]/np.maximum(width[inside], 1e-12))**6
                fields.append(phi)
                components.append([float(cx),cy,length,t1,t2,t3,st])
    return np.clip(np.max(fields, axis=0), -3., 3.), components


def coarse_inputs(model, factor=2):
    """Coarsen only audited D1 symmetry/roller and D2 clamped-left supports.

    Shape-function nodal load restriction preserves resultant and physical moment.
    The coarse unit grid is a uniform coordinate scaling of the original domain;
    plane stress stiffness for square elements is scale invariant (fixed thickness).
    """
    if model.nx % factor or model.ny % factor:
        raise ValueError('Coarse grid must divide both fine dimensions')
    nx, ny = model.nx//factor, model.ny//factor
    nodes, cells, centers = mesh(nx, ny)
    family = model.task['family']
    if family == 'D1_mbb':
        expected = np.r_[2*np.arange(model.ny+1), 2*(len(model.nodes)-1)+1]
        fixed = np.r_[2*np.arange(ny+1), 2*(len(nodes)-1)+1]
    elif family == 'D2_cantilever':
        expected = np.arange(2*(model.ny+1))
        fixed = np.arange(2*(ny+1))
    else:
        raise ValueError('Unaudited boundary family for coarse transfer')
    if not np.array_equal(np.sort(model.fixed), np.sort(expected)):
        raise ValueError('Support pattern differs from audited family')
    force = np.zeros(2*len(nodes))
    for dof in np.flatnonzero(model.force):
        px, py = model.nodes[dof//2]/factor
        ix, iy = min(int(np.floor(px)), nx-1), min(int(np.floor(py)), ny-1)
        wx, wy = px-ix, py-iy
        for dx, dy, w in [(0,0,(1-wx)*(1-wy)), (1,0,wx*(1-wy)), (0,1,(1-wx)*wy), (1,1,wx*wy)]:
            node = (ix+dx)*(ny+1)+(ny-(iy+dy))
            force[2*node+dof%2] += w*model.force[dof]
    fine = model.force.reshape(-1,2); coarse = force.reshape(-1,2)
    moment_fine = np.sum(model.nodes[:,0]*fine[:,1]-model.nodes[:,1]*fine[:,0])
    moment_coarse = np.sum(factor*nodes[:,0]*coarse[:,1]-factor*nodes[:,1]*coarse[:,0])
    if not np.allclose(fine.sum(axis=0),coarse.sum(axis=0),atol=1e-10,rtol=1e-10) or not np.isclose(moment_fine,moment_coarse,atol=1e-9,rtol=1e-10):
        raise ValueError('Coarse transfer failed force/moment conservation')
    if np.any(force[fixed] != 0):
        raise ValueError('Restricted load reaches a constrained coarse DOF')
    return nodes, cells, centers, force, fixed


def prolong_design(model, coarse, x, factor=2):
    # mesh() stores each column from top to bottom; interpolation y is ascending.
    values = x.reshape(coarse.nx,coarse.ny).T[::-1, :]
    interpolator = RegularGridInterpolator((np.arange(coarse.ny)+.5,np.arange(coarse.nx)+.5), values)
    pos = np.clip(model.centers/factor,[.5,.5],[coarse.nx-.5,coarse.ny-.5])
    return interpolator(pos[:,[1,0]])


def coarse_start(model, updates):
    from .initializers import calibrate
    factor=2
    nodes, cells, centers, force, fixed = coarse_inputs(model, factor)
    cfg = replace(model.config, rmin=model.config.rmin/factor)
    task = dict(model.task, elements_xy=[model.nx//factor, model.ny//factor])
    t0=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='initskill_coarse_') as tmp:
        root=Path(tmp)
        dump(root/'task.json', task)
        save_npz(root/'domain.npz', nodes=nodes, cells=cells, centers=centers)
        save_npz(root/'boundary_conditions.npz', force=force, fixed_dofs=fixed)
        coarse=SIMP(root,cfg)
        x=coarse.initial(); history=[]
        for k in range(updates+1):
            evaluation=coarse.evaluate(x)
            history.append(dict(step=k,compliance=evaluation['compliance'],volume=evaluation['volume']))
            if k<updates:x=coarse.update(x,evaluation)
        # Prolong the design variables, retaining physical coordinate orientation.
        suggested=prolong_design(model,coarse,x,factor)
        # No additional contrast factor: bounded logit + scalar volume shift only.
        result=calibrate(model,logit(np.clip(suggested,1e-6,1-1e-6)),1.)
        metadata=dict(extra_fe=coarse.fe_calls,coarse_fe_calls=coarse.fe_calls,coarse_elements=len(cells),
            fine_elements=len(model.cells),coarse_updates=updates,factor=factor,coarse_solver=asdict(cfg),
            coarse_seconds=time.monotonic()-t0,coarse_history=history,
            coarse_design=x.tolist(),coarse_elements_xy=[coarse.nx,coarse.ny],
            coarse_force_sparse=[[int(i),float(force[i])] for i in np.flatnonzero(force)],
            coarse_fixed_dofs=fixed.tolist(),
            design_interpolation_volume_error_before=float(model.physical(suggested).mean()-model.target),
            calibration_max_design_change=float(np.max(np.abs(result-suggested))),
            source='COMSOL coarse-to-fine topology optimization example; local SIMP implementation',
            adaptation='factor2 square grids; conserved resultant/moment; rmin/factor; bilinear design transfer')
        # Existing runner counter is aggregate solves; metadata disaggregates resolutions.
        model.fe_calls += coarse.fe_calls
        return result, metadata


def generate_literature(model, method):
    from .initializers import calibrate
    metadata=dict(method=method,extra_fe=0,fallback=False,adaptation=True)
    if method=='plsm_holes':
        metadata.update(source='Wei et al. 2018, DOI 10.1007/s00158-018-1904-8, publisher supplement lines4-10',
            adaptation='same 15 hole centers and r=.1H; element-center field; sigmoid/physical-volume calibration; then SIMP/OC')
        return calibrate(model,plsm_field(model),1.),metadata
    if method=='mmc_cross':
        field,components=mmc_field(model)
        metadata.update(source='Zhang et al. 2016, DOI 10.1007/s00158-015-1372-3; MMC188_python geometry initialization',
            components=components,adaptation='crossed component superellipse at element centers; clip[-3,3]; sigmoid/volume calibration; then SIMP/OC')
        return calibrate(model,field,1.),metadata
    if method in ('coarse_simp30','coarse_simp80'):
        x,extra=coarse_start(model,30 if method=='coarse_simp30' else 80)
        metadata.update(extra)
        return x,metadata
    raise ValueError('Unknown literature initialization')
