"""Literature-category adaptations, not native-paper reproductions."""
import hashlib
import numpy as np
from scipy.special import expit, logit

METHODS = ('U', 'HC1_B020', 'RAND_IID_S01', 'RAND_IID_S02', 'RAND_IID_S03')


def calibrate(model, field):
    field = np.asarray(field, dtype=float)
    if field.shape != (len(model.cells),) or not np.isfinite(field).all():
        raise ValueError('Invalid initializer field')
    bound = float(np.max(np.abs(field))) + 50.
    lo, hi = -bound, bound
    for _ in range(120):
        mid = (lo + hi) / 2
        x = model.config.xmin + (1-model.config.xmin)*expit(field + mid)
        volume = model.physical(x).mean()
        if abs(volume-model.target) < 1e-11:
            return x
        if volume > model.target:
            hi = mid
        else:
            lo = mid
    raise ValueError('Physical volume calibration did not converge')


def generate(model, method):
    meta = dict(method=method, extra_FE=0, native_paper_reproduction=False,
                source_reference='Alvarez et al. 2019, DOI:10.3390/ma12223736')
    if method == 'U':
        x = model.initial()
        meta['source_reference'] = 'Sigmund 2001, DOI:10.1007/s001580050176'
    elif method == 'HC1_B020':
        center = np.array([model.nx/2, model.ny/2])
        phi = np.clip((np.linalg.norm(model.centers-center, axis=1)-.25*model.ny)
                      / (.04*model.ny), -3., 3.)
        x = .8*model.initial() + .2*calibrate(model, phi)
        meta.update(center_xy=center.tolist(), radius_H=.25, transition_H=.04,
                    field_clip=[-3, 3], beta=.2,
                    adaptation='Center-hole category; local radius, soft mapping, and Uniform blend')
    elif method in METHODS[2:]:
        replicate = int(method[-2:])
        text = model.task['task_id'] + '|RAND_IID|' + str(replicate)
        seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:16], 16)
        values = np.random.default_rng(seed).uniform(.01, .99, (model.ny, model.nx))
        x = calibrate(model, logit(values).ravel(order='F'))
        meta.update(seed=seed, replicate=replicate, raw_density_range=[.01, .99],
                    row0='TOP', flatten_order='F',
                    adaptation='Fully-random category; local frozen seed and logit volume mapping')
    else:
        raise ValueError('Unknown initializer: '+method)
    if not np.isfinite(x).all() or abs(model.physical(x).mean()-model.target) > 1e-8:
        raise ValueError('Initializer failed volume/finiteness checks')
    return x, meta
