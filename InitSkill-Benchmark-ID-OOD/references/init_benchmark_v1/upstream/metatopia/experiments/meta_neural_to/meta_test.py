import logging
import os
import time
from functools import partial
from os.path import join
from pathlib import Path


# Suppress massive JAX/XLA error tracebacks — failed batches are caught and
# logged as one-line warnings by our own code.
logging.getLogger("jax._src.callback").setLevel(logging.CRITICAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import equinox as eqx
import hydra
import jax
import jax.random as jr
import jax.tree as jt
import metatopia as mtp
import numpy as np
import pandas as pd
from common import (
    adapt_until_convergence,
    get_design_parameterisation,
    get_loss_fn,
    load_learned_model,
)
from hydra.utils import instantiate
from metatopia.dataset import get_next
from metatopia.dataset.to import get_iterator_fn
from metatopia.learners import get_filters

# import metatopia.learners as mtpl
from metatopia.solver import setup_solver_cvxopt
from metatopia.utils import get_grid_coords

# import optax
# from jaxtyping import PyTree
from omegaconf import DictConfig, OmegaConf


jax.config.update("jax_enable_x64", True)


def save_batch_data(save_dir, adaptation_results, append=False):
    """Save a single batch of results.

    If append=False (or metrics.csv does not exist), writes a fresh file with
    header. Otherwise appends without header so resumed runs do not clobber
    earlier batches.
    """
    save_dir = Path(save_dir)

    # Create directories
    designs_dir = save_dir / "designs"
    designs_dir.mkdir(parents=True, exist_ok=True)
    losses_dir = save_dir / "losses"
    losses_dir.mkdir(parents=True, exist_ok=True)

    # Create metrics DataFrame for this batch
    metrics_df = pd.DataFrame(
        {
            "id": adaptation_results.id,
            "loss": adaptation_results.outer_losses,
            "bw_loss": adaptation_results.losses_bw,
            "steps": adaptation_results.steps,
            "compliance": adaptation_results.outer_metrics["compliance"],
            "volume": adaptation_results.outer_metrics["volume"],
        }
    )

    # Save or append metrics
    metrics_file = save_dir / "metrics.csv"

    if append and metrics_file.exists():
        metrics_df.to_csv(metrics_file, mode="a", header=False, index=False)
    else:
        metrics_df.to_csv(metrics_file, index=False)

    for i, (design, id) in enumerate(
        zip(adaptation_results.outer_metrics["design"], adaptation_results.id)
    ):
        # Save design in float16 for space efficiency
        np.save(designs_dir / f"{id}.npy", design.astype(np.float16))

        # Save inner losses
        np.save(
            losses_dir / f"{id}.npy",
            adaptation_results.inner_losses[i][: adaptation_results.steps[i] + 1],
        )


###############################################################################
# Testing loop
###############################################################################
def run_meta_testing(
    model,
    meta_test_iterator_fn,
    inner_filter,
    inner_opt,
    loss_fn,
    thresholded_loss_fn,
    *,
    sample_size,
    batch_size,
    steps,
    min_steps,
    tol,
    save_path,
    dataset_name,
    model_name,
):
    """Tests a single model at a time."""
    meta_test_iterator = meta_test_iterator_fn(batch_size=batch_size)

    save_dir = Path(save_path) / dataset_name / model_name / f"{steps}_steps"
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = save_dir / "metrics.csv"

    sample_size = sample_size or len(meta_test_iterator) * batch_size
    params, static = eqx.partition(model, inner_filter)

    # Resume: dedupe against IDs already present in metrics.csv. Iterator order
    # is deterministic for a given seed, so re-fetched batches will overlap with
    # what's already done; we filter those out per batch.
    processed_ids = set()
    if metrics_file.exists():
        try:
            existing_df = pd.read_csv(metrics_file)
            processed_ids = set(int(i) for i in existing_df["id"].tolist())
            mtp.logger.info(
                f"Resuming {save_dir}: {len(processed_ids)} samples already done."
            )
        except (pd.errors.EmptyDataError, KeyError):
            processed_ids = set()

    samples_processed = len(processed_ids)
    if samples_processed >= sample_size:
        mtp.logger.info(
            f"Target sample_size={sample_size} already reached "
            f"({samples_processed} done); skipping."
        )
        return save_dir

    total_batches = (sample_size + batch_size - 1) // batch_size  # ceiling division
    current_batch = 0

    while samples_processed < sample_size:
        batch = get_next(meta_test_iterator)
        current_batch += 1

        # Drop samples whose IDs are already in metrics.csv.
        batch_ids = np.asarray(batch["id"]).astype(int).reshape(-1).tolist()
        keep_mask = np.array([i not in processed_ids for i in batch_ids])

        # Cap to remaining quota.
        remaining = sample_size - samples_processed
        if keep_mask.sum() > remaining:
            cumsum = np.cumsum(keep_mask)
            keep_mask = keep_mask & (cumsum <= remaining)

        if not keep_mask.any():
            continue  # entire batch already processed

        if not keep_mask.all():
            batch = jt.map(lambda x, m=keep_mask: x[m], batch)

        try:
            adaptation_results = adapt_until_convergence(
                params,
                static,
                batch,
                loss_fn,
                thresholded_loss_fn,
                inner_opt,
                steps=steps,
                min_steps=min_steps,
                tol=tol,
            )
        except Exception as e:
            skipped_ids = np.asarray(batch["id"]).astype(int).reshape(-1).tolist()
            mtp.logger.warning(
                f"Batch {current_batch} failed ({type(e).__name__}), "
                f"skipping IDs {skipped_ids}."
            )
            processed_ids.update(skipped_ids)
            samples_processed = len(processed_ids)
            continue

        # Append (or create) — save_batch_data handles both cases.
        save_batch_data(save_dir, adaptation_results, append=True)

        new_ids = np.asarray(adaptation_results.id).astype(int).reshape(-1).tolist()
        processed_ids.update(new_ids)
        samples_processed = len(processed_ids)

        # Log every batch
        progress_percentage = (samples_processed / sample_size) * 100

        mtp.logger.info(
            f"Processed batch {current_batch}/{total_batches} "
            f"({progress_percentage:.1f}% complete, "
            f"{samples_processed}/{sample_size} samples)."
        )

    # Log final statistics from the saved CSV
    metrics_df = pd.read_csv(save_dir / "metrics.csv")
    mtp.logger.info(f"Average loss: {metrics_df['loss'].mean():.3f}")
    mtp.logger.info(f"Average steps: {metrics_df['steps'].mean():.1f}")

    return save_dir


###############################################################################
# Main
###############################################################################
@hydra.main(version_base=None, config_path="conf", config_name="meta_test")
def main(cfg_test: DictConfig) -> None:
    # Load the training config
    cfg_train = OmegaConf.load(join(cfg_test.model_path, ".hydra/config.yaml"))

    # Load the dataset configuration
    current_file = Path(__file__).resolve()
    root_path = current_file.parents[2]
    data_path = root_path / "data" / "to"
    test_path = data_path / cfg_test.dataset.name
    dataset_cfg = OmegaConf.load(data_path / f"{cfg_test.dataset.name}.yaml")

    # Topology optimization setup
    domain_shape = (dataset_cfg.Ny, dataset_cfg.Nx)
    coords = get_grid_coords(domain_shape)
    solver = setup_solver_cvxopt(**dataset_cfg)

    base_model_partial = instantiate(cfg_train.base_model)
    model_key = instantiate(cfg_train.model_key)

    # This is used to instantiate the learned model
    base_model = base_model_partial(key=model_key)

    design_parameterisation = get_design_parameterisation(
        domain_shape, cfg_test.top_opt.relative_r_min
    )

    _get_loss_fn = partial(
        get_loss_fn,
        solver,
        coords,
        design_parameterisation,
        condition_on_strain_energy=True,  # temporary
    )
    loss_fn = _get_loss_fn(kind="standard")
    thresholded_loss_fn = _get_loss_fn(kind="thresholded")

    # At test time we always adapt all parameters, regardless of the filter
    # used during meta-training (e.g. FiLM-only).
    inner_filter, outer_filter = get_filters(name="default", model=base_model)

    # Load the meta_learner
    meta_learner_partial = instantiate(cfg_train.meta_learner)
    meta_learner = meta_learner_partial(
        base_model=base_model,
        outer_loss_fn=loss_fn,
        inner_loss_fn=loss_fn,
        inner_filter=inner_filter,
        outer_filter=outer_filter,
    )

    mtp.logger.info(f"Loading model weights from {cfg_test.model_path}")
    learned_model = load_learned_model(base_model, cfg_test.model_path)

    current_file = Path(__file__).resolve()

    meta_test_iterator_fn = get_iterator_fn(
        path=current_file.parent / test_path,
        seed=cfg_test.dataset.seed,
    )

    mtp.logger.info(
        f"Testing the meta-learned model with {cfg_test.meta_test.steps} "
        "adaptation steps."
    )

    t0 = time.time()
    meta_save_dir = run_meta_testing(
        learned_model,
        meta_test_iterator_fn,
        inner_filter,
        meta_learner.inner_opt,
        loss_fn,
        thresholded_loss_fn,
        save_path=cfg_test.results_root,
        dataset_name=cfg_test.dataset.name,
        model_name="meta",
        **cfg_test.meta_test,
    )

    mtp.logger.info(f"Meta-learned model results saved to {meta_save_dir}")

    # Test standard models if requested
    if cfg_test.std_model.test:
        mtp.logger.info("Testing standard models...")
        # Load the training config for standard NRTO
        cfg_std = OmegaConf.load(join(cfg_test.std_model.path, ".hydra/config.yaml"))
        model_key_std = instantiate(cfg_std.model_key)
        std_model_partial = instantiate(cfg_std.base_model)

        # Generate random keys for multiple initializations
        n_inits = cfg_test.std_model.n_inits
        keys = jr.split(model_key_std, n_inits)
        base_models = [std_model_partial(key=key) for key in keys]

        for i, base_model in enumerate(base_models):
            mtp.logger.info(
                f"Testing randomly initialized base model {i + 1}/{n_inits}."
            )
            inner_filter_std, _ = get_filters(name="default", model=base_model)
            inner_opt_std = instantiate(cfg_std.meta_learner.inner_opt)

            std_save_dir = run_meta_testing(
                base_model,
                meta_test_iterator_fn,
                inner_filter_std,
                inner_opt_std,
                loss_fn,
                thresholded_loss_fn,
                save_path=cfg_test.results_root,
                dataset_name=cfg_test.dataset.name,
                model_name=f"std_{i}",
                **cfg_test.meta_test,
            )

            mtp.logger.info(f"Standard model {i + 1} results saved to {std_save_dir}")

    t1 = time.time()
    elapsed = t1 - t0
    mtp.logger.info(f"Testing completed in {elapsed:.3f} seconds.")


if __name__ == "__main__":
    main()
