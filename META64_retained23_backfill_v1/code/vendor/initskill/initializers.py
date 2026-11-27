"""Representation decoders. All methods share physical-volume calibration."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import expit, logit


def calibrate(model, field, alpha=1.):
    field = np.asarray(field, dtype=float)
    if field.shape != (len(model.cells),) or not np.isfinite(field).all():
        raise ValueError('Invalid decoded field')
    if not np.isfinite(alpha) or not 0 <= alpha <= 2:
        raise ValueError('alpha must be in [0,2]')
    lo, hi = -float(np.max(np.abs(alpha*field)))-50, float(np.max(np.abs(alpha*field)))+50
    xmin = model.config.xmin
    for _ in range(120):
        mid = (lo+hi)/2
        x = xmin+(1-xmin)*expit(alpha*field+mid)
        volume = model.physical(x).mean()
        if abs(volume-model.target) < 1e-11:
            return x
        if volume > model.target:
            hi = mid
        else:
            lo = mid
    raise ValueError('Physical volume cannot be calibrated')


def grid_shape(model):
    return 6, 12 if model.nx/model.ny <= 2.1 else 18


def decode(model, candidate):
    if not isinstance(candidate, dict) or set(candidate) != {'representation', 'parameters'}:
        raise ValueError('Expected representation and parameters only')
    representation, params = candidate['representation'], candidate['parameters']
    if not isinstance(params, dict):
        raise ValueError('parameters must be an object')
    alpha = params.get('alpha', 1.)
    if representation == 'grid':
        if set(params)-{'values', 'alpha'}:
            raise ValueError('Unexpected grid parameters')
        values = np.asarray(params['values'], dtype=float)
        if values.shape != grid_shape(model) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f'Expected density grid shape {grid_shape(model)}, values in [0,1]')
        interp = RegularGridInterpolator((np.linspace(0, 1, values.shape[0]), np.linspace(0, 1, values.shape[1])),
                                         logit(np.clip(values, .01, .99)), bounds_error=True)
        field = interp(np.c_[1-model.centers[:, 1]/model.ny, model.centers[:, 0]/model.nx])
    elif representation == 'components':
        if set(params)-{'items', 'alpha', 'transition'}:
            raise ValueError('Unexpected component parameters')
        items = params['items']
        if not isinstance(items, list) or not 1 <= len(items) <= 16:
            raise ValueError('Expected 1 to 16 solid capsule components')
        transition = float(params.get('transition', .04))*model.ny
        if not .005*model.ny <= transition <= .2*model.ny:
            raise ValueError('Invalid component transition width')
        fields = []
        for item in items:
            if set(item) != {'cx', 'cy', 'length', 'width', 'angle_degrees'}:
                raise ValueError('Invalid component fields')
            cx, cy, length, width, angle = [float(item[k]) for k in ('cx', 'cy', 'length', 'width', 'angle_degrees')]
            if not all(np.isfinite([cx, cy, length, width, angle])) or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < length <= 4 and 0 < width <= 1 and -180 <= angle <= 180):
                raise ValueError('Invalid component geometry')
            delta = model.centers - [cx*model.nx, cy*model.ny]
            direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
            t = np.clip(delta@direction, -length*model.ny/2, length*model.ny/2)
            distance = np.linalg.norm(delta-t[:, None]*direction, axis=1)
            fields.append((width*model.ny/2-distance)/transition)
        field = np.max(fields, axis=0)
    elif representation == 'holes':
        if set(params)-{'items', 'alpha', 'transition'}:
            raise ValueError('Unexpected hole parameters')
        items = params['items']
        if not isinstance(items, list) or len(items) > 24:
            raise ValueError('Expected at most 24 holes')
        if not items:
            return model.initial()
        transition = float(params.get('transition', .05))*model.ny
        if not .005*model.ny <= transition <= .2*model.ny:
            raise ValueError('Invalid hole transition width')
        fields = []
        for item in items:
            if set(item) != {'cx', 'cy', 'radius'}:
                raise ValueError('Invalid hole fields')
            cx, cy, radius = [float(item[k]) for k in ('cx', 'cy', 'radius')]
            if not all(np.isfinite([cx, cy, radius])) or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < radius <= 1):
                raise ValueError('Invalid circle geometry')
            fields.append((np.linalg.norm(model.centers-[cx*model.nx, cy*model.ny], axis=1)-radius*model.ny)/transition)
        field = np.min(fields, axis=0)
    else:
        raise ValueError('Supported representations: grid/components/holes')
    return calibrate(model, field, alpha)


def generate(model, method, seed, candidate=None):
    metadata = dict(method=method, extra_fe=0, fallback=False)
    if method == 'uniform':
        return model.initial(), metadata
    if method == 'candidate':
        metadata['candidate'] = candidate
        try:
            x = decode(model, candidate)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            metadata.update(fallback=True, fallback_reason=str(exc))
            x = model.initial()
        return x, metadata
    if method == 'random_grid':
        values = np.random.default_rng(seed).uniform(.05, .95, grid_shape(model))
        candidate = dict(representation='grid', parameters=dict(values=values.tolist(), alpha=1.))
        metadata['candidate'] = candidate
        return decode(model, candidate), metadata
    if method == 'regular_holes':
        candidate = dict(representation='holes', parameters=dict(items=[dict(cx=float(x), cy=float(y), radius=.12)
                         for x in np.linspace(.15, .85, 5) for y in (.25, .75)], alpha=1., transition=.05))
        metadata['candidate'] = candidate
        return decode(model, candidate), metadata
    if method == 'energy':
        reference = model.evaluate(model.initial())
        field = np.log(np.maximum(reference['energy'], 1e-30))
        field = (field-np.median(field))/max(float(np.std(field)), 1e-12)
        metadata.update(extra_fe=1, field_mapping='standardized log strain energy; alpha=.8; one uniform FE')
        return calibrate(model, field, .8), metadata
    from .literature_initializers import METHODS, generate_literature
    if method in METHODS:
        return generate_literature(model, method)
    from .sensitivity_initializers import CONFIGS, generate_sensitivity
    if method in CONFIGS:
        return generate_sensitivity(model, method)
    if method in ('llm_holes','random_holes_flex'):
        from .llm_holes import generate_holes
        return generate_holes(model,method,seed,candidate)
    raise ValueError('Unknown initialization method')
