from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import metatopia as mtp
import nlopt
import numpy as np
import pandas as pd
from metatopia.dataset.to import get_iterator_fn
from metatopia.filters import cone_filter, threshold_design
from omegaconf import OmegaConf


jax.config.update("jax_enable_x64", True)

########################################################################
use_learned_init = False
########################################################################


def save_batch_data(
    save_dir,
    problem,
    rho,
    compliance,
    loss,
    loss_t,
    losses,
    steps,
    append=False,
):
    """Save results in a consistent format with meta-testing script."""
    save_dir = Path(save_dir)

    # Create directories
    designs_dir = save_dir / "designs"
    designs_dir.mkdir(parents=True, exist_ok=True)
    losses_dir = save_dir / "losses"
    losses_dir.mkdir(parents=True, exist_ok=True)

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(
        {
            "id": [problem["id"].item()],
            "loss": [loss],
            "bw_loss": [loss_t],
            "steps": [steps],
            "compliance": [compliance],
            "volume": [rho.mean()],
        }
    )

    # Save or append metrics
    metrics_file = save_dir / "metrics.csv"

    if append and metrics_file.exists():
        metrics_df.to_csv(metrics_file, mode="a", header=False, index=False)
    else:
        metrics_df.to_csv(metrics_file, index=False)

    id = problem["id"].item()

    # Save design in float16 for space efficiency
    np.save(designs_dir / f"{id}.npy", rho.astype(np.float16))

    # Save inner losses
    np.save(
        losses_dir / f"{id}.npy",
        losses,
    )


@hydra.main(version_base=None, config_path="conf", config_name="meta_test")
def main(cfg):
    # Directory setup
    current_file = Path(__file__).resolve()
    root_path = current_file.parents[2]
    data_path = root_path / "data" / "to"
    test_path = data_path / cfg.dataset.name
    dataset_cfg = OmegaConf.load(data_path / f"{cfg.dataset.name}.yaml")

    # Setup the solver
    domain_shape = (dataset_cfg.Ny, dataset_cfg.Nx)

    relative_r_min = cfg.top_opt.relative_r_min
    r_min = relative_r_min * domain_shape[0]
    density_filter = partial(cone_filter, domain_shape=domain_shape, r_min=r_min)
    dataset_name = cfg.dataset.name

    model_name = "mma"

    steps = cfg.meta_test.steps
    save_dir = Path(cfg.model_path) / dataset_name / model_name / f"{steps}_steps"
    save_dir.mkdir(parents=True, exist_ok=True)

    def create_optimization_functions(problem):
        def design_parameterisation(x):
            rho = x.reshape(domain_shape)
            rho = density_filter(rho)
            return rho.reshape(domain_shape)

        @jax.jit
        def objective(x):
            rho = design_parameterisation(x)
            compliance = solver(rho, problem)
            return compliance

        @jax.jit
        def constraint(x):
            rho = design_parameterisation(x)
            return jnp.mean(rho) - problem["target_vf"].item()

        return design_parameterisation, objective, constraint

    def wrap_jax_func(func, compliances=None):
        def wrapper(x, grad):
            if grad.size > 0:
                value, _grad = jax.value_and_grad(func)(x)
                grad[:] = np.array(_grad)
            else:
                value = func(x)

            if compliances is not None:
                compliances.append(value)

            return np.array(value) * 1.0

        return wrapper

    test_iterator_fn = get_iterator_fn(
        path=current_file.parent / test_path,
        seed=cfg.dataset.seed,
    )

    iterator = test_iterator_fn(batch_size=1)
    solver = mtp.solver.setup_solver_cvxopt(**dataset_cfg)

    # Resume support: skip IDs already present in metrics.csv.
    metrics_file = save_dir / "metrics.csv"
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

    sample_size = cfg.meta_test.sample_size or len(iterator)
    if len(processed_ids) >= sample_size:
        mtp.logger.info(
            f"Target sample_size={sample_size} already reached "
            f"({len(processed_ids)} done); skipping."
        )
        return

    mtp.logger.info(f"Running MMA on test problems (target {sample_size}).")

    # Get initial design
    def get_initial_design(problem):
        x0 = (
            np.ones(shape=(dataset_cfg.Nx * dataset_cfg.Ny,))
            * problem["target_vf"].item()
        )

        return x0.astype(np.float64)

    for i, problem in enumerate(iterator):
        if len(processed_ids) >= sample_size:
            break

        problem_id = int(np.asarray(problem["id"]).reshape(-1)[0])
        if problem_id in processed_ids:
            continue

        if (i + 1) % 100 == 0:
            mtp.logger.info(f"Solving problem {i + 1}/{len(iterator)}")

        try:
            design_parameterisation, objective, constraint = (
                create_optimization_functions(problem)
            )

            # Pick a single problem to remove the batch dimension
            problem = jax.tree.map(lambda x: x[0], problem)

            compliances = []
            objective = wrap_jax_func(objective, compliances)
            constraint = wrap_jax_func(constraint)

            # Default initial guess
            x0 = get_initial_design(problem)

            opt = nlopt.opt(nlopt.LD_MMA, x0.size)
            opt.set_lower_bounds(0.0)
            opt.set_upper_bounds(1.0)

            tol = cfg.meta_test.tol
            opt.set_ftol_abs(tol)
            opt.set_ftol_rel(tol)

            opt.set_min_objective(objective)
            opt.add_inequality_constraint(constraint, 1e-8)
            max_iterations = cfg.meta_test.steps
            opt.set_maxeval(max_iterations + 1)

            # Optimise the design
            x = opt.optimize(x0)

            # Thresholding
            rho = design_parameterisation(x)
            rho_t = threshold_design(rho, problem)

            def design_compliance_and_loss(rho, problem):
                # Scale the compliance accordingly
                compliance = solver(rho, problem)
                loss = compliance / problem["reference_compliance"]
                return compliance, loss

            compliance, loss = design_compliance_and_loss(rho, problem)
            compliance_t, loss_t = design_compliance_and_loss(rho_t, problem)
            losses = np.array(compliances) / problem["reference_compliance"]

            save_batch_data(
                save_dir,
                problem,
                rho,
                compliance,
                loss,
                loss_t,
                losses,
                steps=opt.get_numevals() - 1,
                append=True,
            )
        except Exception as e:
            mtp.logger.warning(
                f"Problem {problem_id} failed ({type(e).__name__}), skipping."
            )

        processed_ids.add(problem_id)


if __name__ == "__main__":
    main()
