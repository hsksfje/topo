import time
from functools import partial
from pathlib import Path

import equinox as eqx
import equinox.internal as eqxi
import hydra
import jax
import jax.numpy as jnp
import jax.tree as jt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import metatopia as mtp
import metatopia.utils as mtpu
import numpy as np
from hydra.utils import instantiate
from metatopia.filters import cone_filter, hoyer_volume_enforcing_filter
from metatopia.utils import tree_extend
from omegaconf import DictConfig
from visualize_loss import add_loss_landscape_visualization


jax.config.update("jax_enable_x64", True)


@hydra.main(version_base=None, config_path="conf", config_name="conf")
def main(cfg: DictConfig) -> None:
    # Topology optimization setup
    domain_shape = (cfg.top_opt.Ny, cfg.top_opt.Nx)
    coords = mtpu.get_grid_coords(domain_shape)
    solver = mtp.solver.setup_solver_cvxopt(**cfg.top_opt)

    # Pre-processing
    solver_se = mtp.solver.setup_solver_cvxopt_elementwise(**cfg.top_opt)

    # Data sample
    sample = instantiate(cfg.problem)

    reference_rho = np.ones(domain_shape) * sample["target_vf"]
    reference_compliance, reference_se = solver_se(reference_rho, sample)

    # Duplicate code, will be removed in the future
    def process_strain_energy(ce):
        ce = jnp.log(ce)
        ce = 2.0 * (ce - jnp.min(ce)) / (jnp.max(ce) - jnp.min(ce)) - 1.0
        return ce

    reference_se = process_strain_energy(reference_se)

    sample = eqx.tree_at(
        lambda x: x["reference_compliance"], sample, reference_compliance
    )

    sample = eqx.tree_at(lambda x: x["strain_energy"], sample, reference_se)

    # Reparameterization model
    init_model = instantiate(cfg.model)
    model = init_model

    density_filter = partial(
        cone_filter, domain_shape=domain_shape, r_min=cfg.top_opt.filter_radius
    )
    hoyer_filter = partial(hoyer_volume_enforcing_filter, target_vf=sample["target_vf"])

    if cfg.model.in_features == 3:
        e = sample["strain_energy"].reshape(-1, 1)
        input = jnp.concatenate([coords, e], axis=1)
    else:
        input = coords

    def design_parameterisation(input, sample, params, static):
        model = eqx.combine(params, static)
        rho = jax.vmap(model)(input)
        rho *= 10.0  # Boost the signal prior to filtering
        rho = density_filter(rho)  # Volume preserving
        rho = hoyer_filter(rho)  # Enforce volume

        return rho.reshape(domain_shape)

    def loss_fn(params):
        rho = design_parameterisation(input, sample, params, static)
        compliance = solver(rho, sample) / sample["reference_compliance"]
        loss = compliance
        return loss, {"rho": rho}

    def make_step(params, opt_state):
        prev_opt_state = opt_state
        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            params
        )

        def value_fn(params):
            return loss

        updates, new_opt_state = opt.update(
            grads, prev_opt_state, params, grad=grads, value=loss, value_fn=value_fn
        )

        # Type correction for optimizer state (need in LBFGS)
        new_opt_state = jt.map(
            lambda x, y: x.astype(y.dtype), new_opt_state, prev_opt_state
        )

        params = eqx.apply_updates(params, updates)

        return params, new_opt_state, loss, metrics

    def scan_fn(carry, _):
        params, opt_state = carry
        params, opt_state, loss, metrics = make_step(params, opt_state)
        return (params, opt_state), (loss, metrics)

    ####################################################################
    partial_filter_1 = instantiate(cfg.filter_1)
    filter_1, _ = partial_filter_1(model=model)
    params, static = eqx.partition(model, filter_1)

    t0 = time.time()
    opt = instantiate(cfg.opt_1)
    opt_state = opt.init(params)

    n_steps_1 = cfg.n_steps_1
    n_steps_2 = cfg.n_steps_2
    n_steps = n_steps_1 + n_steps_2 if cfg.two_stage else n_steps_1

    mtp.logger.info(f"Running optimization (stage 1) for {n_steps_1} steps.")
    t0 = time.time()

    # First optimizer
    (params, opt_state), (losses_1, metrics_1) = eqxi.scan(
        scan_fn,
        (params, opt_state),
        None,
        length=n_steps_1,  # For the final loss
        kind="lax",
    )

    # Second optimizer (optional)
    if cfg.two_stage:
        mtp.logger.info(f"Running optimization (stage 2) for {n_steps_2} steps.")
        opt = instantiate(cfg.opt_2)
        opt_state = opt.init(params)

        partial_filter_2 = instantiate(cfg.filter_2)
        filter_2, _ = partial_filter_2(model=model)

        model = eqx.combine(params, static)
        params, static = eqx.partition(model, filter_2)

        (params, opt_state), (losses_2, metrics_2) = eqxi.scan(
            scan_fn,
            (params, opt_state),
            None,
            length=n_steps_2,  # For the final loss
            kind="lax",
        )

        losses = tree_extend(losses_1, losses_2, expand_dims=False)
        metrics = tree_extend(metrics_1, metrics_2, expand_dims=False)

    else:
        losses = losses_1
        metrics = metrics_1

    rhos = metrics["rho"]
    t1 = time.time()

    mtp.logger.info(
        f"Optimization finished in {t1-t0:.3f} seconds. "
        f"Final loss: {losses[-1]:.3f}. "
        "Creating animation."
    )

    ###################################################################
    # Create animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_settings = {
        "vmin": 0.0,
        "vmax": 1.0,
        "cmap": "Grays",
    }

    # Set static axes for loss plot

    ax2.set_xlim(0, n_steps)
    ax2.set_ylim(jnp.min(losses), jnp.max(losses))
    (line,) = ax2.plot([], [], "b-", linewidth=2.0)

    def animate(i):
        # Structure plot
        ax1.clear()
        im = ax1.imshow(
            rhos[i].reshape(domain_shape),
            **plot_settings,
        )

        ax1.set_title(f"Design at iteration {i}")

        # Update loss plot
        xdata = range(i + 1)
        ydata = losses[: i + 1]
        line.set_data(xdata, ydata)
        ax2.set_title(f"Compliance: {losses[i]:.4f}")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Compliance")
        ax2.grid(True)
        ax2.set_ylim(0, 1.0)

        return [im, line]

    anim = animation.FuncAnimation(fig, animate, frames=n_steps, interval=50, blit=True)

    save_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    save_name = "optimization_trajectory.gif"

    mtp.logger.info(f"Saving results to {save_path}")

    anim.save(save_path / save_name, writer="pillow")
    plt.close()

    # Save the final topology as well

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(
        rhos[-1],
        **plot_settings,
    )
    ax.set_title("Final design")
    plt.savefig(save_path / "final_design.png")

    def simplified_loss_fn(params):
        loss, _ = loss_fn(params)
        return loss, {}

    final_model = eqx.combine(params, static)

    loss_visualization = False

    if loss_visualization:
        add_loss_landscape_visualization(final_model, simplified_loss_fn, save_path)

    mtp.logger.info("Done.")


if __name__ == "__main__":
    main()
