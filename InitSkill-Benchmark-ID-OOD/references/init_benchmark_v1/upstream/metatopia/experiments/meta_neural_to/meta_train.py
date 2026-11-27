import time
from functools import partial
from pathlib import Path

import equinox as eqx
import hydra
import jax
import metatopia as mtp
from common import get_design_parameterisation, get_loss_fn
from hydra.utils import instantiate
from jaxtyping import Array, Float
from metatopia.dataset.to import get_iterator_fn, to_single_task_iterator_fn
from metatopia.learners import AbstractMetaLearner, get_filters, meta_learn
from metatopia.utils import (
    get_grid_coords,
    load_tree,
    plot_meta_losses,
    serialise_array,
    serialise_tree,
)
from nvidia.dali.plugin.jax import DALIGenericIterator
from omegaconf import DictConfig, OmegaConf


jax.config.update("jax_enable_x64", True)


###############################################################################
# Traning loop
###############################################################################
def run_meta_training(
    meta_learner: AbstractMetaLearner,
    meta_train_iterator_fn: DALIGenericIterator,
    *,
    batch_size: int,
    inner_steps: int,
    outer_steps: int,
    log_every_n: int | None = None,
):
    # Placeholder for the initial hyperparameters
    meta_learned_model, meta_losses, metrics = meta_learn(
        meta_learner=meta_learner,
        train_iterator_fn=meta_train_iterator_fn,
        batch_size=batch_size,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        log_every_n=log_every_n,
    )

    return meta_learner, meta_learned_model, metrics, meta_losses


###############################################################################
# Model serialisation and loading
###############################################################################
def serialise(
    meta_learner: AbstractMetaLearner,
    base_model: eqx.Module,
    meta_learned_model: eqx.Module,
    metrics: Float[Array, "meta_step batch"],
    meta_losses: Float[Array, "meta_step 1"],
    save_path: str,
):
    mtp.logger.info(f"Saving trained models in {save_path}.")

    for tree, name in [
        (base_model, "base_model"),
        (meta_learned_model, "learned_model"),
        (meta_learner, "meta_learner"),
        (metrics, "train_metrics"),
    ]:
        serialise_tree(tree, name, save_path)

    serialise_array(meta_losses, "meta_losses", save_path)


###############################################################################
# Main
###############################################################################
@hydra.main(version_base=None, config_path="conf", config_name="meta_train")
def main(cfg: DictConfig) -> None:
    # Load the dataset configuration
    current_file = Path(__file__).resolve()
    root_path = current_file.parents[2]
    data_path = root_path / "data" / "to"
    dataset_cfg = OmegaConf.load(data_path / f"{cfg.dataset.name}.yaml")

    # Topology optimization setup
    domain_shape = (dataset_cfg.Ny, dataset_cfg.Nx)
    coords = get_grid_coords(domain_shape)

    solver = mtp.solver.setup_solver_cvxopt(**dataset_cfg)

    # Reparameterization setup
    base_model_partial = instantiate(cfg.base_model)
    model_key = instantiate(cfg.model_key)
    base_model = base_model_partial(key=model_key)

    # (Optional) Load a training checkpoint
    if cfg.resume_from_checkpoint:
        base_model = load_tree(base_model, cfg.resume_from_checkpoint, "learned_model")

    design_parameterisation = get_design_parameterisation(
        domain_shape, cfg.top_opt.relative_r_min
    )

    # Loss functions for continuous designs
    loss_fn = get_loss_fn(
        solver,
        coords,
        design_parameterisation,
        condition_on_strain_energy=(cfg.base_model.in_features == 3),
        kind="standard",
    )

    inner_filter, outer_filter = get_filters(name=cfg.filter_name, model=base_model)

    # Instantiate the meta-learner
    meta_learner_partial = instantiate(cfg.meta_learner)
    meta_learner = meta_learner_partial(
        base_model=base_model,
        outer_loss_fn=loss_fn,
        inner_loss_fn=loss_fn,
        inner_filter=inner_filter,
        outer_filter=outer_filter,
    )
    mtp.logger.info(
        f"Training {meta_learner.name} for "
        f"{cfg.meta_train.outer_steps} outer steps, "
        f"{cfg.meta_train.inner_steps} inner steps, "
        f"{cfg.meta_train.batch_size} task(s) per batch."
    )

    t0 = time.time()

    if cfg.single_task:
        meta_train_iterator_fn = partial(
            to_single_task_iterator_fn,
            domain_shape=domain_shape,
            target_vf=0.3,
            name="mbb_beam",
        )

        mtp.logger.warning("Single task training is enabled.")

    else:
        # Get the meta-training iterator function
        meta_train_iterator_fn = get_iterator_fn(
            path=data_path / cfg.dataset.name,
            seed=cfg.dataset.seed,
        )

    meta_learner, meta_learned_model, metrics, meta_losses = run_meta_training(
        meta_learner,
        meta_train_iterator_fn,
        **cfg.meta_train,
    )

    t1 = time.time()
    elapsed = t1 - t0
    mtp.logger.info(f"Training completed in {elapsed:.3f} seconds.")

    # Calculate the average loss over the first and last 10% of the meta steps
    # to get a sense of the convergence
    first_10_mean = meta_losses[: int(0.1 * len(meta_losses))].mean()
    last_10_mean = meta_losses[int(0.9 * len(meta_losses)) :].mean()

    mtp.logger.info(
        f"Average loss over the first 10% of meta steps: {first_10_mean:.5f}"
    )

    mtp.logger.info(f"Average loss over the last 10% of meta steps: {last_10_mean:.5f}")

    save_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Save the model
    serialise(
        meta_learner=meta_learner,
        base_model=base_model,
        meta_learned_model=meta_learned_model,
        metrics=metrics,
        meta_losses=meta_losses,
        save_path=save_path,
    )

    # Plot the meta losses
    plot_meta_losses(
        meta_losses,
        save_path=save_path,
        save_name="meta_losses",
    )


if __name__ == "__main__":
    main()
