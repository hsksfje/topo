"""
Optimize 3 random tasks with the same base model and save designs to SVG.

Uses a randomly initialized base model to optimize 3 different tasks
(with volume > 0.3) for 200 steps each, saving the final designs as SVG files.
"""

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import metatopia as mtp
import numpy as np
import optax
from common import get_design_parameterisation, get_loss_fn, load_learned_model
from hydra.utils import instantiate
from metatopia.dataset import get_next
from metatopia.dataset.to import get_iterator_fn
from metatopia.learners import get_filters
from metatopia.solver import setup_solver_cvxopt
from metatopia.utils import get_grid_coords
from omegaconf import OmegaConf


jax.config.update("jax_enable_x64", True)

# Matplotlib styling
plt.set_loglevel("ERROR")
MM = 0.0393701  # millimeter in inches

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

DPI = 300
FIG_WIDTH_MM = 60
FIG_WIDTH = FIG_WIDTH_MM * MM  # 60mm in inches

# UPDATE THESE PATHS
META_MODEL_PATH = "/home/igork/projects/saved_runs/NewExperiments/07-03_23-29/15"
DATASET_NAME = "test_cr"


def load_all_tasks(dataset_name, seed=0):
    """Load all tasks from the dataset."""
    current_file = Path(__file__).resolve()
    root_path = current_file.parents[2]
    data_path = root_path / "data" / "to_new" / dataset_name
    iterator_fn = get_iterator_fn(path=data_path, seed=seed)
    iterator = iterator_fn(batch_size=1)

    tasks = []
    for _ in range(len(iterator)):
        sample = get_next(iterator)
        tasks.append(jax.tree.map(lambda x: x.squeeze(0), sample))

    return tasks


def select_random_tasks(tasks, n_tasks=3, min_volume=0.3, seed=42):
    """Select n random tasks with volume > min_volume."""
    # Filter tasks by volume
    valid_tasks = []
    for task in tasks:
        vf = float(task["target_vf"])
        if vf > min_volume:
            valid_tasks.append(task)

    mtp.logger.info(
        f"Found {len(valid_tasks)}/{len(tasks)} tasks with volume > {min_volume}"
    )

    # Randomly select n_tasks
    rng = np.random.RandomState(seed)
    indices = rng.choice(
        len(valid_tasks), size=min(n_tasks, len(valid_tasks)), replace=False
    )
    selected = [valid_tasks[i] for i in indices]

    # Log selected tasks
    for i, task in enumerate(selected):
        task_id = int(task["id"])
        vf = float(task["target_vf"])
        mtp.logger.info(f"Task {i+1}: ID={task_id}, volume={vf:.3f}")

    return selected


def optimize_task(params, static, sample, loss_fn, steps=200, lr=1e-4):
    """Run optimization for a single task."""
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


def save_design_svg(design, task_id, save_dir):
    """Save design as SVG file."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_WIDTH), layout="constrained")
    ax.imshow(design, cmap="gray_r", origin="upper", vmin=0, vmax=1)
    ax.set_title(f"Task {task_id}")
    ax.axis("off")

    save_path = save_dir / f"design_task_{task_id}.svg"
    fig.savefig(save_path, format="svg", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    mtp.logger.info(f"Saved design to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_tasks", type=int, default=10, help="Number of tasks to optimize"
    )
    parser.add_argument(
        "--min_volume", type=float, default=0.2, help="Minimum volume fraction"
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Optimization steps per task"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for task selection"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./results_random_tasks",
        help="Directory to save results",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset configuration
    current_file = Path(__file__).resolve()
    root_path = current_file.parents[2]
    data_path = root_path / "data" / "to_new"
    dataset_cfg = OmegaConf.load(data_path / f"{DATASET_NAME}.yaml")

    # Load all tasks and select random ones
    mtp.logger.info(f"Loading tasks from {DATASET_NAME}...")
    all_tasks = load_all_tasks(DATASET_NAME)
    selected_tasks = select_random_tasks(
        all_tasks, args.n_tasks, args.min_volume, args.seed
    )

    # Setup meta-learned model
    mtp.logger.info("Loading meta-learned model...")
    cfg = OmegaConf.load(f"{META_MODEL_PATH}/.hydra/config.yaml")

    domain_shape = (dataset_cfg.Ny, dataset_cfg.Nx)
    coords = get_grid_coords(domain_shape)
    solver_cvxopt = setup_solver_cvxopt(**dataset_cfg)

    # Load meta-learned model from checkpoint
    model_key = instantiate(cfg.model_key)
    base_model_partial = instantiate(cfg.base_model)
    base_model = base_model_partial(key=model_key)
    model = load_learned_model(base_model, META_MODEL_PATH)
    mtp.logger.info("Meta-learned model loaded successfully")

    design_param = get_design_parameterisation(
        domain_shape, cfg.top_opt.relative_r_min
    )

    # Check if model uses strain energy conditioning
    condition_on_strain_energy = (
        cfg.base_model.in_features == 3
        if hasattr(cfg.base_model, "in_features")
        else False
    )

    loss_fn = get_loss_fn(
        solver_cvxopt,
        coords,
        design_param,
        condition_on_strain_energy=condition_on_strain_energy,
        kind="standard",
    )

    inner_filter, _ = get_filters(name=cfg.filter_name, model=model)
    init_params, static = eqx.partition(model, inner_filter)

    # Optimize each task
    for i, task in enumerate(selected_tasks):
        task_id = int(task["id"])
        vf = float(task["target_vf"])

        mtp.logger.info(f"\n{'='*60}")
        mtp.logger.info(
            f"Task {i+1}/{len(selected_tasks)}: ID={task_id}, volume={vf:.3f}"
        )
        mtp.logger.info(f"{'='*60}")

        # Optimize (start from same initial parameters for all tasks)
        final_params, losses = optimize_task(
            init_params, static, task, loss_fn, args.steps
        )

        # Extract final design
        if condition_on_strain_energy:
            e = task["strain_energy"].reshape(-1, 1)
            model_input = jnp.concatenate([coords, e], axis=1)
        else:
            model_input = coords

        design = design_param(model_input, task, final_params, static)

        # Save design
        save_design_svg(design, task_id, save_dir)

        # Log final loss
        mtp.logger.info(f"Final loss: {losses[-1]:.4f}")

    mtp.logger.info(f"\nAll designs saved to {save_dir}")


if __name__ == "__main__":
    main()
