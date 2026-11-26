"""
1D Loss Landscape Visualization using jax.tree.map.

This version uses jax.tree.map for parameter interpolation instead of
flattening/unflattening, and loads both meta and standard models from
their respective saved checkpoints with their own configurations.
"""

import argparse
from os.path import join
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import matplotlib as mpl
import matplotlib.pyplot as plt
import metatopia as mtp
import numpy as np
import optax
import seaborn as sns
from common import get_design_parameterisation, get_loss_fn, load_learned_model
from hydra.utils import instantiate
from metatopia.dataset import get_next
from metatopia.dataset.to import get_iterator_fn
from metatopia.learners import get_filters
from metatopia.solver import setup_solver_cvxopt
from metatopia.utils import get_grid_coords
from omegaconf import OmegaConf


jax.config.update("jax_enable_x64", True)

# Matplotlib styling to match plot_utils.py
plt.set_loglevel("ERROR")
MM = 0.0393701  # millimeter in inches

# Apply matplotlib style settings
mpl.rcdefaults()
mpl.rcParams["font.size"] = 7
mpl.rcParams["axes.labelsize"] = 7
mpl.rcParams["xtick.labelsize"] = 7
mpl.rcParams["ytick.labelsize"] = 7
mpl.rcParams["legend.fontsize"] = 7
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"  # Embed fonts as text for editable SVGs
mpl.rcParams["axes.linewidth"] = 0.5
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["lines.linewidth"] = 1.0
mpl.rcParams["lines.markersize"] = 3

# Seaborn color palette
palette = sns.color_palette("Dark2")
sns.set_palette(palette)

DPI = 300
FIG_WIDTH_MM = 60
FIG_WIDTH = FIG_WIDTH_MM * MM  # 60mm in inches

# UPDATE THESE PATHS
META_MODEL_PATH = "/home/igork/projects/saved_runs/NewExperiments/24-01_10-34"
STD_MODEL_PATH = "/home/igork/projects/saved_runs/NewExperiments/07-03_23-29/14"
DATASET_NAME = "test_id"

WORST_ID = 30437
BEST_ID = 30736


# ---------------------------------------------------------------------------
# Task & Optimization
# ---------------------------------------------------------------------------
def load_task(dataset_name, task_id, seed=0):
    current_file = Path(__file__).resolve()
    root_path = current_file.parents[2]
    data_path = root_path / "data" / "to_new" / dataset_name
    iterator_fn = get_iterator_fn(path=data_path, seed=seed)
    iterator = iterator_fn(batch_size=1)
    for _ in range(len(iterator)):
        sample = get_next(iterator)
        if int(sample["id"].squeeze()) == task_id:
            return jt.map(lambda x: x.squeeze(0), sample)
    raise ValueError(f"Task ID {task_id} not found.")


def get_final_params(params, static, sample, loss_fn, steps, lr=1e-4):
    """
    Runs optimization using eqx.filter_jit to handle static parts correctly.
    """
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @eqx.filter_jit
    def step_fn(params, opt_state):
        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, static, sample
        )
        updates, new_opt_state = opt.update(grads, opt_state)
        return eqx.apply_updates(params, updates), new_opt_state, loss

    losses = []
    for i in range(steps):
        params, opt_state, loss = step_fn(params, opt_state)
        losses.append(float(loss))
        if (i + 1) % 50 == 0 or i == 0:
            mtp.logger.info(f"  Step {i + 1}/{steps}, Loss: {loss:.4f}")

    return params, losses


# ---------------------------------------------------------------------------
# 1D Landscape Along Optimization Direction
# ---------------------------------------------------------------------------
def compute_1d_landscape(
    init_params,
    final_params,
    static,
    sample,
    loss_fn,
    alpha_min=0.0,
    alpha_max=1.0,
    n_points=1_000,
):
    """
    Evaluate loss along the line:
        theta(alpha) = theta_init + alpha * (theta_final - theta_init)
    Uses jax.tree.map for direct pytree interpolation without flattening.
    """
    # Compute direction in pytree space
    direction = jt.map(lambda f, i: f - i, final_params, init_params)

    @eqx.filter_jit
    def eval_alpha(alpha):
        # Interpolate params at this alpha using tree_map
        perturbed_params = jt.map(lambda i, d: i + alpha * d, init_params, direction)
        loss, _ = loss_fn(perturbed_params, static, sample)
        return loss

    alphas = np.linspace(alpha_min, alpha_max, n_points)
    mtp.logger.info(f"Scanning 1D landscape ({n_points} points)...")

    losses = []
    for i, a in enumerate(alphas):
        val = float(eval_alpha(jnp.array(a)))
        losses.append(val)

    return alphas, np.array(losses)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_1d_comparison(
    alphas_meta,
    losses_meta,
    loss_init_meta,
    loss_final_meta,
    alphas_std,
    losses_std,
    loss_init_std,
    loss_final_std,
    design_meta,
    design_std,
    opt_losses_meta,
    opt_losses_std,
    save_path,
):
    save_dir = save_path.parent
    task_id = save_path.stem.split("task")[-1]

    # Get colors from seaborn palette
    colors = sns.color_palette()

    # Subplot 1: Meta design
    fig1, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_WIDTH), layout="constrained")
    ax1.imshow(design_meta, cmap="gray_r", origin="upper", vmin=0, vmax=1)
    ax1.set_title("Meta-neural TO Final Design")
    ax1.axis("off")
    fig1.savefig(
        save_dir / f"design_meta_task{task_id}.svg",
        format="svg",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig1)

    # Subplot 2: Standard design
    fig2, ax2 = plt.subplots(figsize=(FIG_WIDTH, FIG_WIDTH), layout="constrained")
    ax2.imshow(design_std, cmap="gray_r", origin="upper", vmin=0, vmax=1)
    ax2.set_title("Neural TO Final Design")
    ax2.axis("off")
    fig2.savefig(
        save_dir / f"design_std_task{task_id}.svg",
        format="svg",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig2)

    # Subplot 3: Loss landscape
    fig3, ax3 = plt.subplots(
        figsize=(FIG_WIDTH, FIG_WIDTH),
        subplot_kw=dict(box_aspect=1),
        layout="constrained",
    )

    # 1. Meta-Learned Trajectory
    ax3.plot(
        alphas_meta,
        losses_meta,
        color=colors[0],  # Dark2 palette color 0 (green)
        linewidth=1.5,
        label="Meta-neural TO",
    )

    # 2. Standard Trajectory
    ax3.plot(
        alphas_std,
        losses_std,
        color=colors[1],  # Dark2 palette color 1 (orange)
        linewidth=1.5,
        label="Neural TO",
    )

    # 4. Labels
    ax3.set_xlabel(r"Interpolation $\beta$ (0=Init, 1=Final)")
    ax3.set_ylabel("Loss")
    ax3.set_title("1D Loss Landscape")
    ax3.legend(loc="upper right", frameon=False)
    ax3.grid(True, alpha=0.3, linewidth=0.5)

    # 5. Y-Limits and Aspect
    ax3.set_ylim(0, 1)
    ax3.set_xlim(0, 1)

    fig3.savefig(
        save_dir / f"landscape_1d_task{task_id}.svg",
        format="svg",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig3)

    # Subplot 4: Loss vs. Iteration
    fig4, ax4 = plt.subplots(
        figsize=(FIG_WIDTH, FIG_WIDTH),
        subplot_kw=dict(box_aspect=1),
        layout="constrained",
    )

    # Plot optimization trajectories
    iterations_meta = np.arange(1, len(opt_losses_meta) + 1)
    iterations_std = np.arange(1, len(opt_losses_std) + 1)

    ax4.plot(
        iterations_meta,
        opt_losses_meta,
        color=colors[0],  # Dark2 palette color 0 (green)
        linewidth=1.5,
        label="Meta-neural TO",
    )
    ax4.plot(
        iterations_std,
        opt_losses_std,
        color=colors[1],  # Dark2 palette color 1 (orange)
        linewidth=1.5,
        label="Neural TO",
    )

    # Reference line
    ax4.axhline(
        y=1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5, label="Loss=1.0"
    )

    # Labels
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Loss")
    ax4.set_title("Loss vs. Iteration")
    ax4.legend(loc="upper right", frameon=False)
    ax4.grid(True, alpha=0.3, linewidth=0.5)
    ax4.set_ylim(0, 1)
    ax4.set_xlim(1, len(opt_losses_meta))

    fig4.savefig(
        save_dir / f"loss_vs_iteration_task{task_id}.svg",
        format="svg",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig4)

    mtp.logger.info(
        f"Saved plots to {save_dir}:\n"
        f"  - design_meta_task{task_id}.svg\n"
        f"  - design_std_task{task_id}.svg\n"
        f"  - landscape_1d_task{task_id}.svg\n"
        f"  - loss_vs_iteration_task{task_id}.svg"
    )


# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=WORST_ID)
    parser.add_argument("--alpha_min", type=float, default=0.0)
    parser.add_argument("--alpha_max", type=float, default=1.0)
    parser.add_argument("--n_points", type=int, default=200)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument(
        "--std_seed",
        type=int,
        default=1,
        help="Random seed for standard model initialization",
    )
    parser.add_argument("--save_path", type=str, default=".")
    args = parser.parse_args()

    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load task once (used for both models)
    sample = load_task(DATASET_NAME, args.task_id, seed=args.seed)

    # Load dataset configuration
    current_file = Path(__file__).resolve()
    root_path = current_file.parents[2]
    data_path = root_path / "data" / "to_new"
    dataset_cfg = OmegaConf.load(data_path / f"{DATASET_NAME}.yaml")

    # 1. Setup Meta Model with its own config
    mtp.logger.info("Setting up meta-learned model...")
    cfg_meta = OmegaConf.load(join(META_MODEL_PATH, ".hydra/config.yaml"))

    # Topology optimization setup for meta model
    domain_shape = (dataset_cfg.Ny, dataset_cfg.Nx)
    coords = get_grid_coords(domain_shape)
    solver_cvxopt = setup_solver_cvxopt(**dataset_cfg)

    base_model_meta_partial = instantiate(cfg_meta.base_model)
    model_key_meta = instantiate(cfg_meta.model_key)
    base_model_meta = base_model_meta_partial(key=model_key_meta)

    design_param_meta = get_design_parameterisation(
        domain_shape, cfg_meta.top_opt.relative_r_min
    )

    # Check if model uses strain energy conditioning
    condition_on_strain_energy_meta = (
        cfg_meta.base_model.in_features == 3
        if hasattr(cfg_meta.base_model, "in_features")
        else False
    )

    loss_fn_meta = get_loss_fn(
        solver_cvxopt,
        coords,
        design_param_meta,
        condition_on_strain_energy=condition_on_strain_energy_meta,
        kind="standard",
    )

    inner_filter_meta, _ = get_filters(name=cfg_meta.filter_name, model=base_model_meta)

    # Load learned meta model from checkpoint
    meta_model = load_learned_model(base_model_meta, META_MODEL_PATH)
    meta_init, meta_static = eqx.partition(meta_model, inner_filter_meta)

    # Sanity check: Evaluate loss at uniform design with target volume fraction
    mtp.logger.info("Evaluating loss at uniform design...")
    target_vf = float(sample["target_vf"])
    uniform_design = jnp.ones(domain_shape) * target_vf

    # Compute compliance with uniform design using solver
    uniform_compliance = solver_cvxopt(uniform_design, sample)
    reference_compliance = sample["reference_compliance"]
    uniform_loss = uniform_compliance / reference_compliance

    mtp.logger.info(
        f"Uniform design (vf={target_vf:.3f}):\n"
        f"  Uniform compliance: {float(uniform_compliance):.6e}\n"
        f"  Reference compliance: {float(reference_compliance):.6e}\n"
        f"  Normalized loss: {float(uniform_loss):.6f} "
        f"(expected ~1.0 if properly normalized)"
    )

    # 2. Setup Standard Model with its own config (random initialization)
    mtp.logger.info(
        f"Setting up Standard Model with random initialization "
        f"(seed={args.std_seed})..."
    )
    cfg_std = OmegaConf.load(join(STD_MODEL_PATH, ".hydra/config.yaml"))

    # Create randomly initialized model with specified seed
    model_key_std = jr.key(args.std_seed)
    std_model_partial = instantiate(cfg_std.base_model)
    std_model = std_model_partial(key=model_key_std)

    design_param_std = get_design_parameterisation(
        domain_shape, cfg_std.top_opt.relative_r_min
    )

    # Check if model uses strain energy conditioning
    condition_on_strain_energy_std = (
        cfg_std.base_model.in_features == 3
        if hasattr(cfg_std.base_model, "in_features")
        else False
    )

    loss_fn_std = get_loss_fn(
        solver_cvxopt,
        coords,
        design_param_std,
        condition_on_strain_energy=condition_on_strain_energy_std,
        kind="standard",
    )

    inner_filter_std, _ = get_filters(name=cfg_std.filter_name, model=std_model)

    # Partition the randomly initialized model
    std_init, std_static = eqx.partition(std_model, inner_filter_std)

    # 3. Optimize Meta Model
    mtp.logger.info("Optimizing meta-learned model...")
    meta_final, opt_losses_meta = get_final_params(
        meta_init, meta_static, sample, loss_fn_meta, args.steps
    )

    # 4. Optimize Standard Model
    mtp.logger.info("Optimizing standard model...")
    std_final, opt_losses_std = get_final_params(
        std_init, std_static, sample, loss_fn_std, args.steps
    )

    # 5. Compute 1D Landscapes with model-specific loss functions
    mtp.logger.info("Computing meta loss landscape...")
    alphas_meta, losses_meta = compute_1d_landscape(
        meta_init,
        meta_final,
        meta_static,
        sample,
        loss_fn_meta,
        args.alpha_min,
        args.alpha_max,
        args.n_points,
    )

    mtp.logger.info("Computing standard landscape...")
    alphas_std, losses_std = compute_1d_landscape(
        std_init,
        std_final,
        std_static,
        sample,
        loss_fn_std,
        args.alpha_min,
        args.alpha_max,
        args.n_points,
    )

    # 6. Extract specific points using eqx.filter_jit
    @eqx.filter_jit
    def eval_loss_meta(p, s):
        return loss_fn_meta(p, s, sample)[0]

    @eqx.filter_jit
    def eval_loss_std(p, s):
        return loss_fn_std(p, s, sample)[0]

    l_meta_init = float(eval_loss_meta(meta_init, meta_static))
    l_meta_final = float(eval_loss_meta(meta_final, meta_static))
    l_std_init = float(eval_loss_std(std_init, std_static))
    l_std_final = float(eval_loss_std(std_final, std_static))

    improvement = 100.0 * (l_std_final - l_meta_final) / l_std_final
    mtp.logger.info(
        f"Final losses — Meta: {l_meta_final:.6f}, Std: {l_std_final:.6f}\n"
        f"  Meta-neural TO improvement over Standard TO: {improvement:.2f}%"
    )

    # 7. Extract final designs using model-specific components
    mtp.logger.info("Extracting final designs...")

    # Meta model design
    if condition_on_strain_energy_meta:
        e = sample["strain_energy"].reshape(-1, 1)
        model_input_meta = jnp.concatenate([coords, e], axis=1)
    else:
        model_input_meta = coords
    design_meta = design_param_meta(model_input_meta, sample, meta_final, meta_static)

    # Standard model design
    if condition_on_strain_energy_std:
        e = sample["strain_energy"].reshape(-1, 1)
        model_input_std = jnp.concatenate([coords, e], axis=1)
    else:
        model_input_std = coords
    design_std = design_param_std(model_input_std, sample, std_final, std_static)

    # 8. Plot comparison
    plot_1d_comparison(
        alphas_meta,
        losses_meta,
        l_meta_init,
        l_meta_final,
        alphas_std,
        losses_std,
        l_std_init,
        l_std_final,
        design_meta,
        design_std,
        opt_losses_meta,
        opt_losses_std,
        save_dir / f"landscape_task{args.task_id}",
    )


if __name__ == "__main__":
    main()
