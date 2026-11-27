from typing import Tuple

import jax
import jax.numpy as jnp
import jax.scipy.signal as jsps
import numpy as np
import optimistix as optx


# This wrapper allows for dynamically configuring the filters
def get_hoyer_filter():
    def _hoyer_filter(design, sample):
        return hoyer_volume_enforcing_filter(design, sample["target_vf"])

    return _hoyer_filter


def get_cone_filter(
    domain_shape,
    filter_radius: float,
):
    def _cone_filter(design, sample):
        return cone_filter(design, domain_shape, filter_radius)

    return _cone_filter


def get_xu_volume_preserving_filter():
    def _xu_volume_preserving_filter(design, beta, target_vf=None):
        return xu_volume_preserving_filter(design, beta, target_vf)

    return _xu_volume_preserving_filter


def cone_filter(design: jax.Array, domain_shape: Tuple, r_min: float) -> jax.Array:
    # Reshape to the actual design shape
    design = design.reshape(domain_shape)

    # Prepare the filtering window
    range_ = np.arange(-np.ceil(r_min) + 1, np.ceil(r_min))
    dx, dy = np.meshgrid(range_, range_)
    window = np.maximum(0, r_min - np.sqrt(dx**2 + dy**2))

    # Use FFT-based convolution to avoid cuDNN workspace allocation failures
    # on large grids (e.g. 256x256 with f64) — cuDNN can hit CUDNN_STATUS_ALLOC_FAILED
    # even when memory is technically available.
    weights = jsps.convolve(np.ones(domain_shape), window, mode="same", method="fft")
    filtered_design = (
        jsps.convolve(design, window, mode="same", method="fft") / weights
    )

    return filtered_design.ravel()


def hoyer_volume_enforcing_filter(
    design: jax.Array,
    target_vf: float,
) -> jax.Array:
    def logit(p):
        p = jnp.clip(p, 0, 1)
        return jnp.log(p) - jnp.log(1.0 - p)

    b1 = logit(target_vf) - jnp.max(design)
    b2 = logit(target_vf) - jnp.min(design)
    bounds = jnp.asarray([b1, b2])

    lb, ub = jnp.min(bounds), jnp.max(bounds)

    # Ensure the bounds are wide enough that the volume violation function
    # changes sign. When the design has very low variance (e.g. after cone
    # filtering on certain boundary conditions), the analytical bounds can
    # be too tight for the bisection solver's root-containment check.
    # A margin of 10 in logit-space corresponds to sigmoid values near
    # 0.0000454 and 0.99995, guaranteeing a sign change for any target_vf
    # in a practical range.
    min_half_width = 10.0
    center = (lb + ub) / 2.0
    half_width = jnp.maximum((ub - lb) / 2.0, min_half_width)
    lb = center - half_width
    ub = center + half_width

    def transform_design(design, b):
        # Sigmoidal transform
        transformed = jax.nn.sigmoid(design + b)
        return transformed

    def volume_violation_fn(y, args):
        loss = transform_design(args, y).mean() - target_vf
        return loss

    # NOTE: these tolerances are on the logit shift ``b``, not on the resulting
    # volume fraction. Combined with the widened bounds above they admit a
    # volume fraction error of order 1e-4. Tightening them to 1e-9 reduces that
    # error to ~1e-10 for roughly 20 extra bisection steps (max_steps is 750),
    # but changes results relative to those reported in the paper, so the
    # published setting is kept here deliberately.
    bisection = optx.Bisection(atol=1e-3, rtol=1e-3)

    b = optx.root_find(
        fn=volume_violation_fn,
        solver=bisection,
        y0=(ub+lb)/2.0,
        args=design,
        options=dict(lower=lb, upper=ub),
        max_steps=750,
        throw=False,
    )

    return transform_design(design, b.value)


def xu_projection(design: jax.Array, beta: float, eta: float):
    """Density projection used to obtain sharp black and white designs.
    Also Needed for volume preserving heavyside filter from Xu et al.

    Parameters
    ----------
    design : jax.Array
        Design array.
    beta : float
        Steepness of the heavy side function. Larger beta gives a sharper
        transition [more black and white designs]. Has to be greater than
        zero.
    eta : float
        Factor for preserving the volume fraction. Found using a bisection
        method. Should be between 0 and 1.

    Returns
    -------
    jax.Array
        Filtered design.

    """
    design = (jnp.tanh(beta * eta) + jnp.tanh(beta * (design - eta))) / (
        jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta))
    )
    return design


def xu_volume_preserving_filter(
    design: jax.Array,
    beta: float,
    target_vf: float = None,
) -> jax.Array:
    """Volume preserving filter from Xu et al.

    See: https://doi.org/10.1007/s00158-009-0452-7

    The filter is applied to the design array in-place and projects the
    input design to a design that meets the target volume fraction.

    Parameters
    ----------
    design : jax.Array
        Input design.
    beta : float
        Steepness of the heavy side function. Larger beta gives a sharper
        transition [more black and white designs]. Has to be greater than
        zero.
    target_vf : float
        Target volume fraction to enforce on the design.

    Returns
    -------
    jax.Array
        Filtered design.
    """

    lb, ub = 0, 1

    # if target_vf is not provided, preserve the current volume fraction
    if target_vf is None:
        target_vf = design.mean()

    def transform_design(design, eta):
        # Sigmoidal transform
        transformed = xu_projection(design, beta, eta)
        return transformed

    def volume_violation_fn(y, args):
        loss = transform_design(args, y).mean() - target_vf
        return loss

    bisection = optx.Bisection(atol=1e-5, rtol=1e-5)

    sol = optx.root_find(
        fn=volume_violation_fn,
        solver=bisection,
        y0=0.5,
        args=design,
        options=dict(lower=lb, upper=ub),
        max_steps=128,
        throw=False,
    )

    return transform_design(design, sol.value)


def _threshold_design(rho: jax.Array, target_vf: float) -> jax.Array:
    k = jnp.floor(np.size(rho) * target_vf).astype(int)

    rho_f = rho.flatten()
    _, indices = jax.lax.top_k(rho_f, k)
    rho_t = jnp.zeros_like(rho_f)
    rho_t = rho_t.at[indices].set(1.0)
    rho_t = rho_t.reshape(rho.shape)
    return rho_t


def threshold_design(rho, problem):
    def _threshold_fn_host(rho, target_vf):
        rho = _threshold_design(rho, target_vf)

        return rho.astype(rho.dtype)

    result_shape_dtypes = jax.ShapeDtypeStruct(
        jnp.broadcast_shapes(rho.shape), rho.dtype
    )

    return jax.pure_callback(
        _threshold_fn_host,
        result_shape_dtypes,
        rho,
        problem["target_vf"],
        vmap_method="sequential",
    )
