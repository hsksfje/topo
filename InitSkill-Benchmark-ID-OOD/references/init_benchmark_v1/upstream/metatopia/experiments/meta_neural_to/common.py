"""
Common functions used for running topology optimisation experiments.
"""

import os
from functools import partial
from typing import Any, Callable, Literal, Tuple

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree as jt
import metatopia.utils as mtpu
import optax
from jaxtyping import PyTree
from metatopia.filters import (
    cone_filter,
    hoyer_volume_enforcing_filter,
    threshold_design,
)
from metatopia.utils import (
    tree_extend,
)


jax.config.update("jax_enable_x64", True)


###############################################################################
# Design parameterisation and loss functions
###############################################################################
def get_design_parameterisation(domain_shape, relative_r_min):
    """
    Returns the mapping from decision variables to the design space.
    """
    # Check that r_min is betweeen 0 and 1
    if not 0.0 < relative_r_min < 1.0:
        raise ValueError("r_min must be between 0 and 1.")

    r_min = relative_r_min * domain_shape[0]
    density_filter = partial(cone_filter, domain_shape=domain_shape, r_min=r_min)

    def design_parameterisation(input, sample, params, static):
        model = eqx.combine(params, static)
        rho = eqx.filter_vmap(model)(input)
        rho *= 10.0  # Boost the signal prior to filtering
        rho = density_filter(rho)  # Volume preserving
        rho = hoyer_volume_enforcing_filter(rho, sample["target_vf"])
        return rho.reshape(domain_shape)

    return design_parameterisation


def get_loss_fn(
    solver: Any,
    coords: jax.Array,
    design_parameterisation: Callable,
    condition_on_strain_energy: bool,
    kind: Literal["thresholded", "standard"] = "standard",
) -> Tuple[Callable, Callable]:
    ####################################################################

    def loss_fn(
        params: PyTree,
        static: PyTree,
        sample: PyTree,
    ):
        if condition_on_strain_energy:
            e = sample["strain_energy"].reshape(-1, 1)
            input = jnp.concatenate([coords, e], axis=1)
        else:
            input = coords

        rho = design_parameterisation(input, sample, params, static)

        if kind == "thresholded":
            rho = threshold_design(rho, sample)

        # Volume scaling to account for minor volume deviations
        compliance = solver(rho, sample)
        loss = compliance / sample["reference_compliance"]
        volume = jnp.mean(rho)

        return loss, {
            "compliance": compliance,
            "volume": volume,
            "design": rho,
        }

    return loss_fn


###############################################################################
# Meta-testing
###############################################################################
class AdaptationResults(eqx.Module):
    params: PyTree

    inner_losses: jax.Array
    outer_losses: jax.Array
    inner_metrics: PyTree
    outer_metrics: PyTree

    losses_bw: jax.Array
    steps: jax.Array
    id: jax.Array


class AdaptationSummary(eqx.Module):
    outer_losses: jax.Array
    outer_metrics: PyTree
    losses_bw: jax.Array
    steps: jax.Array
    id: jax.Array


def adapt_until_convergence(
    params: PyTree,
    static: PyTree,
    batch: Any,
    loss_fn: Callable,
    thresholded_loss_fn: Callable,
    task_opt: optax.GradientTransformation,
    *,
    steps: int,
    min_steps: int,
    tol: float,
) -> Tuple[PyTree, jax.Array, PyTree]:
    def _adapt_until_convergence(
        params: PyTree,
        static: PyTree,
        sample: Any,
        min_steps: int,
        max_steps: int,
    ):
        """
        Single task adaptation until convergence
        """

        def step_fn(params, opt_state, step):
            """Single inner loop step"""

            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, static, sample
            )

            updates, opt_state = task_opt.update(grads, opt_state)
            params = eqx.apply_updates(params, updates)

            return params, opt_state, step + 1, loss, metrics

        def terminate_fn(loss, loss_prev, current_step, min_steps):
            """
            Termination condition. For now, just checks if the loss
            has stabilised.
            """
            # Terminate if the loss has stabilised, or reached sufficient steps
            scale = tol * (1 + jnp.abs(loss))
            stagnated = jnp.abs(loss - loss_prev) / scale < 1.0
            enough_steps = current_step > min_steps

            terminate = jnp.logical_and(stagnated, enough_steps)

            return terminate

        def scan_fn(carry, _):
            params, opt_state, step, loss_prev, metrics, terminate = carry

            def no_op(params, opt_state, step):
                return params, opt_state, step, loss_prev, metrics

            # Only update if we haven't terminated yet
            params, opt_state, step, loss, metrics = jax.lax.cond(
                terminate, no_op, step_fn, params, opt_state, step
            )

            # Check if we should terminate
            terminate = terminate_fn(loss, loss_prev, step, min_steps)

            return (params, opt_state, step, loss, metrics, terminate), (
                loss,
                metrics,
            )

        opt_state = task_opt.init(params)

        init_metrics = jt.map(
            lambda x: jnp.zeros_like(x),
            eqx.filter_eval_shape(loss_fn, params, static, sample)[-1],
        )

        init_carry = (params, opt_state, 0, 0.0, init_metrics, False)

        (final_params, _, final_step, _, _, _), (inner_losses, metrics) = eqxi.scan(
            scan_fn,
            init_carry,
            jnp.arange(max_steps),
            kind="lax",
        )

        final_loss, (final_metrics) = loss_fn(
            final_params,
            static,
            sample,
        )

        # Loss for a thresholded design
        final_loss_bw, _ = thresholded_loss_fn(
            final_params,
            static,
            sample,
        )

        # Calculate the final loss and metrics.
        inner_losses = tree_extend(inner_losses, final_loss)
        inner_metrics = tree_extend(metrics, final_metrics)

        return AdaptationResults(
            params=final_params,
            inner_losses=inner_losses,
            outer_losses=final_loss,
            inner_metrics=inner_metrics,
            outer_metrics=final_metrics,
            losses_bw=final_loss_bw,
            steps=final_step,
            id=sample["id"],
        )

    adaptation_results = eqx.filter_vmap(
        _adapt_until_convergence, in_axes=(None, None, 0, None, None)
    )(params, static, batch, min_steps, steps)

    return adaptation_results


def load_learned_model(
    generic_model: PyTree,
    path: str,
):
    if not os.path.isdir(path):
        raise FileNotFoundError("The specified path does " f"not exist: {path}")

    meta_learned_model = mtpu.load_tree(generic_model, path, "learned_model")

    return meta_learned_model
