import time
from functools import partial
from typing import Callable, Tuple

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import metatopia as mtp
import metatopia.learners as mtpl
from hydra.utils import instantiate
from metatopia.dataset.sine import sin_task_iterator
from metatopia.utils import plot_meta_losses
from omegaconf import DictConfig
from plot import plot_sinusoids


jax.config.update("jax_enable_x64", True)


###############################################################################
# Data iterators
###############################################################################
def get_meta_train_iterators(
    seed: int, num_points: int
) -> Tuple[Callable, Callable]:
    # [NOTE] Using the same seed for both train and validation iterators
    # but different splits
    train_iterator_fn = partial(
        sin_task_iterator, num_points=num_points, seed=seed, split="train"
    )
    validation_iterator_fn = partial(
        sin_task_iterator,
        num_points=num_points,
        seed=seed,
        split="val",
    )
    return train_iterator_fn, validation_iterator_fn


def get_meta_test_iterator(seed: int, num_points: int) -> Callable:
    test_iterator_fn = partial(
        sin_task_iterator, num_points=num_points, seed=seed, split="val"
    )
    return test_iterator_fn


###############################################################################
# Loss functions
# ###############################################################################
def mse_loss_fn(params, static, sample):
    model = eqx.combine(params, static)
    pred = jax.vmap(model)(sample["x"])
    return jnp.mean((sample["y"] - pred) ** 2)


def inner_loss_fn(params, static, sample):
    loss = mse_loss_fn(params, static, sample)
    return loss, None


def outer_loss_fn(params, static, sample):
    loss = mse_loss_fn(params, static, sample)
    return loss, None


###############################################################################
# Traning
###############################################################################


def run_meta_training(
    meta_learner,
    batch_size,
    outer_steps,
    inner_steps,
    log_every_n,
    num_points,
    seed,
):
    train_iterator_fn, validation_iterator_fn = get_meta_train_iterators(
        seed, num_points
    )

    meta_learned_model, meta_losses, metrics = mtpl.meta_learn(
        meta_learner=meta_learner,
        train_iterator_fn=train_iterator_fn,
        validation_iterator_fn=validation_iterator_fn,
        batch_size=batch_size,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        log_every_n=log_every_n,
    )

    return meta_learner, meta_learned_model, meta_losses, metrics


###############################################################################
# Validation
###############################################################################
def run_meta_testing(
    meta_learner,
    base_model,
    meta_learned_model,
    plot_results: bool = False,
    *,
    meta_test_batch,
    inner_steps,
):
    mtp.logger.info(f"Running meta-testing with {inner_steps} steps.")

    params_mtl, static_mtl = eqx.partition(
        meta_learned_model, meta_learner.inner_filter
    )
    params_std, static_std = eqx.partition(
        base_model, meta_learner.inner_filter
    )

    adapted_params_mtl, mtl_losses, _ = meta_learner.adapt(
        params_mtl, static_mtl, meta_test_batch, inner_steps
    )

    adapted_params_std, std_losses, _ = meta_learner.adapt(
        params_std, static_std, meta_test_batch, inner_steps
    )

    @partial(jax.vmap, in_axes=(0, None, None))
    def apply_model(params, static, X):
        model = eqx.combine(params, static)
        return jax.vmap(model)(X)

    #  We measure the performance across 100 points
    X = jnp.linspace(-5, 5, 100).reshape(-1, 1)

    Ys0 = apply_model(adapted_params_std, static_std, X)
    Ys = apply_model(adapted_params_mtl, static_mtl, X)

    def get_gt_sinusoid(amp, phase, X):
        return amp * jnp.sin(X + phase)

    gt = jax.vmap(get_gt_sinusoid, in_axes=(0, 0, None))(
        meta_test_batch["amp"], meta_test_batch["phase"], X
    )

    mean_loss_mtl = jnp.mean((Ys - gt) ** 2)
    mean_loss_std = jnp.mean((Ys0 - gt) ** 2)

    mtp.logger.info(f"{'Test loss (standard):':<35}{mean_loss_std:.4f}")
    mtp.logger.info(f"{'Test loss (meta):':<35}{mean_loss_mtl:.4f}")

    if plot_results:
        plot_sinusoids(
            X,
            Ys,
            Ys0,
            meta_test_batch,
            n_plots=3,
            algo_name="MAML",
            n_steps=inner_steps,
        )
    return None


###############################################################################
# Main script
###############################################################################
@hydra.main(version_base=None, config_path="conf", config_name="conf")
def main(cfg: DictConfig) -> None:
    base_model = instantiate(cfg.base_model)
    inner_filter, outer_filter = mtp.learners.get_filters(
        name=cfg.filters, model=base_model
    )

    meta_learner_partial = instantiate(cfg.meta_learner)
    meta_learner = meta_learner_partial(
        base_model=base_model,
        outer_loss_fn=outer_loss_fn,
        inner_loss_fn=inner_loss_fn,
        inner_filter=inner_filter,
        outer_filter=outer_filter,
    )

    mtp.logger.info("Running meta-training.")
    t0 = time.time()
    meta_learner, meta_learned_model, meta_losses, _ = run_meta_training(
        meta_learner, **cfg.meta_train
    )
    t1 = time.time()
    elapsed = t1 - t0

    mtp.logger.info(f"Training completed in {elapsed:.3f} seconds.")

    t0 = time.time()

    meta_test_iterator = get_meta_test_iterator(
        cfg.meta_test.seed, cfg.meta_test.num_points
    )(cfg.meta_test.batch_size)

    meta_test_batch = mtp.dataset.get_next(meta_test_iterator)

    run_meta_testing(
        meta_learner,
        base_model,
        meta_learned_model,
        plot_results=True,
        inner_steps=cfg.meta_test.inner_steps,
        meta_test_batch=meta_test_batch,
    )

    t1 = time.time()
    elapsed = t1 - t0
    mtp.logger.info(f"Testing completed in {elapsed:.3f} seconds.")

    plot_meta_losses(
        meta_losses, window_size=max(cfg.meta_train.outer_steps // 100, 1)
    )


if __name__ == "__main__":
    main()
