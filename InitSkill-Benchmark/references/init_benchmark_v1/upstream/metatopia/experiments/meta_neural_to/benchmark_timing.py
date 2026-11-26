"""
Wall-clock timing benchmark: Standard TO (MMA via nlopt) vs Neural TO (SIREN + Adam).

Compares per-iteration cost across two mesh resolutions (64x64, 256x256) and
three validation problems (MBB beam, cantilever, tensile).
"""

import csv
import time
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import metatopia as mtp
import nlopt
import numpy as np
import optax
from metatopia.dataset.to.validation_problems import (
    cantilever_beam,
    mbb_beam,
    tensile,
)
from metatopia.filters import cone_filter, hoyer_volume_enforcing_filter
from metatopia.models import Siren
from metatopia.utils import get_grid_coords


jax.config.update("jax_enable_x64", True)

# ── Configuration ────────────────────────────────────────────────────────────
MESH_CONFIGS = {
    "64x64": dict(Nx=64, Ny=64, Lx=64, Ly=64),
    "256x256": dict(Nx=256, Ny=256, Lx=64, Ly=64),
}

SOLVER_PARAMS = dict(young=1.0, young_min=1e-9, poisson=0.3, penalty=3.0)

PROBLEMS = {
    "MBB Beam": mbb_beam,
    "Cantilever": cantilever_beam,
    "Tensile": tensile,
}

# SIREN hyperparameters (from conf/base_model/siren.yaml)
SIREN_CFG = dict(
    in_features=2,
    out_features=1,
    hidden_features=256,
    hidden_layers=4,
    outermost_linear=True,
    omega_0=30.0,
)

RELATIVE_R_MIN = 0.03125  # from conf/top_opt/default.yaml
TARGET_VF = 0.5
N_WARMUP = 10
N_TIMED = 200
LR = 1e-2


# ── Helpers ──────────────────────────────────────────────────────────────────
def make_problem(problem_fn, Nx, Ny):
    """Create a problem dict with proper array types for the solver."""
    problem = problem_fn(Nx, Ny, TARGET_VF)
    # Convert to jax arrays
    return jax.tree.map(lambda x: jnp.array(x) if x is not None else x, problem)


# ── Neural TO benchmark ─────────────────────────────────────────────────────
def benchmark_neural_to(mesh_name, mesh_cfg, problem_name, problem_fn):
    Nx, Ny = mesh_cfg["Nx"], mesh_cfg["Ny"]
    domain_shape = (Ny, Nx)

    solver = mtp.solver.setup_solver_cvxopt(**mesh_cfg, **SOLVER_PARAMS)
    coords = get_grid_coords(domain_shape)

    r_min = RELATIVE_R_MIN * Ny
    density_filter = partial(cone_filter, domain_shape=domain_shape, r_min=r_min)

    problem = make_problem(problem_fn, Nx, Ny)

    # Compute reference compliance
    reference_rho = jnp.ones(domain_shape) * TARGET_VF
    reference_compliance = solver(reference_rho, problem)
    problem["reference_compliance"] = reference_compliance

    # Init SIREN
    key = jr.key(42)
    model = Siren(**SIREN_CFG, key=key)
    params, static = eqx.partition(model, eqx.is_array)

    # Optimizer
    opt = optax.adam(LR)
    opt_state = opt.init(params)

    def design_parameterisation(input, sample, params, static):
        model = eqx.combine(params, static)
        rho = jax.vmap(model)(input)
        rho *= 10.0
        rho = density_filter(rho)
        rho = hoyer_volume_enforcing_filter(rho, sample["target_vf"])
        return rho.reshape(domain_shape)

    def loss_fn(params):
        rho = design_parameterisation(coords, problem, params, static)
        compliance = solver(rho, problem) / problem["reference_compliance"]
        return compliance

    @eqx.filter_jit
    def make_step(params, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = eqx.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Warmup
    mtp.logger.info(
        f"  Neural TO | {mesh_name} | {problem_name} | warmup ({N_WARMUP} steps)..."
    )
    for _ in range(N_WARMUP):
        params, opt_state, _ = make_step(params, opt_state)
    # Block until warmup is done
    jax.block_until_ready(params)

    # Timed iterations
    mtp.logger.info(
        f"  Neural TO | {mesh_name} | {problem_name} | timing ({N_TIMED} steps)..."
    )
    iter_times = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        params, opt_state, loss = make_step(params, opt_state)
        jax.block_until_ready(loss)
        t1 = time.perf_counter()
        iter_times.append((t1 - t0) * 1000)  # ms

    mean_ms = np.mean(iter_times)
    std_ms = np.std(iter_times)
    mtp.logger.info(
        f"  Neural TO | {mesh_name} | {problem_name} | "
        f"{mean_ms:.2f} ± {std_ms:.2f} ms/iter"
    )
    return mean_ms, std_ms, iter_times


# ── MMA benchmark ───────────────────────────────────────────────────────────
def benchmark_mma(mesh_name, mesh_cfg, problem_name, problem_fn):
    Nx, Ny = mesh_cfg["Nx"], mesh_cfg["Ny"]
    domain_shape = (Ny, Nx)

    solver = mtp.solver.setup_solver_cvxopt(**mesh_cfg, **SOLVER_PARAMS)

    r_min = RELATIVE_R_MIN * Ny
    density_filter = partial(cone_filter, domain_shape=domain_shape, r_min=r_min)

    problem = make_problem(problem_fn, Nx, Ny)

    # Compute reference compliance
    reference_rho = jnp.ones(domain_shape) * TARGET_VF
    reference_compliance = solver(reference_rho, problem)
    problem["reference_compliance"] = reference_compliance

    def design_parameterisation(x):
        rho = x.reshape(domain_shape)
        rho = density_filter(rho)
        return rho.reshape(domain_shape)

    @jax.jit
    def objective(x):
        rho = design_parameterisation(x)
        return solver(rho, problem)

    @jax.jit
    def constraint(x):
        rho = design_parameterisation(x)
        return jnp.mean(rho) - TARGET_VF

    def wrap_jax_func(func):
        def wrapper(x, grad):
            if grad.size > 0:
                value, _grad = jax.value_and_grad(func)(x)
                grad[:] = np.array(_grad)
            else:
                value = func(x)
            return np.array(value) * 1.0

        return wrapper

    def run_mma(n_iters):
        """Run MMA for n_iters iterations and return elapsed time in seconds."""
        x0 = np.ones(Nx * Ny) * TARGET_VF

        opt = nlopt.opt(nlopt.LD_MMA, x0.size)
        opt.set_lower_bounds(0.0)
        opt.set_upper_bounds(1.0)
        opt.set_ftol_abs(0.0)
        opt.set_ftol_rel(0.0)
        opt.set_min_objective(wrap_jax_func(objective))
        opt.add_inequality_constraint(wrap_jax_func(constraint), 1e-8)
        opt.set_maxeval(n_iters + 1)

        t0 = time.perf_counter()
        opt.optimize(x0)
        t1 = time.perf_counter()

        return t1 - t0

    # Warmup (JIT + caches)
    mtp.logger.info(
        f"  MMA       | {mesh_name} | {problem_name} | warmup ({N_WARMUP} steps)..."
    )
    run_mma(N_WARMUP)

    # Timed run
    mtp.logger.info(
        f"  MMA       | {mesh_name} | {problem_name} | timing ({N_TIMED} steps)..."
    )
    total_s = run_mma(N_TIMED)
    mean_ms = (total_s / N_TIMED) * 1000
    # No per-iteration std for MMA (we time the full block)
    std_ms = float("nan")

    mtp.logger.info(
        f"  MMA       | {mesh_name} | {problem_name} | "
        f"{mean_ms:.2f} ms/iter (total {total_s:.2f}s)"
    )
    return mean_ms, std_ms, total_s


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    mtp.logger.info("=" * 70)
    mtp.logger.info("Timing Benchmark: Neural TO vs MMA")
    mtp.logger.info(f"Warmup: {N_WARMUP} | Timed: {N_TIMED}")
    mtp.logger.info("=" * 70)

    results = []

    for mesh_name, mesh_cfg in MESH_CONFIGS.items():
        for problem_name, problem_fn in PROBLEMS.items():
            # Neural TO
            mean, std, iter_times = benchmark_neural_to(
                mesh_name, mesh_cfg, problem_name, problem_fn
            )
            results.append(
                {
                    "method": "Neural TO",
                    "mesh": mesh_name,
                    "task": problem_name,
                    "mean_ms": mean,
                    "std_ms": std,
                    "iter_times": iter_times,
                }
            )

            # MMA
            mean, std, total_s = benchmark_mma(
                mesh_name, mesh_cfg, problem_name, problem_fn
            )
            results.append(
                {
                    "method": "MMA",
                    "mesh": mesh_name,
                    "task": problem_name,
                    "mean_ms": mean,
                    "std_ms": std,
                    "iter_times": None,
                }
            )

    # ── Print summary table ──────────────────────────────────────────────
    print()
    print("=" * 75)
    header = (
        f"| {'Method':<12}| {'Mesh':<10}| {'Task':<18}| "
        f"{'Mean (ms)':>10} | {'Std (ms)':>10} |"
    )
    separator = (
        "|"
        + "-" * 13
        + "|"
        + "-" * 11
        + "|"
        + "-" * 19
        + "|"
        + "-" * 12
        + "|"
        + "-" * 12
        + "|"
    )
    print(header)
    print(separator)
    for r in results:
        std_str = (
            f"{r['std_ms']:>10.2f}" if not np.isnan(r["std_ms"]) else f"{'N/A':>10}"
        )
        print(
            f"| {r['method']:<12}| {r['mesh']:<10}| {r['task']:<18}| "
            f"{r['mean_ms']:>10.2f} | {std_str} |"
        )
    print("=" * 75)

    # ── Save CSV ─────────────────────────────────────────────────────────
    output_dir = Path(__file__).resolve().parent / "benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    summary_file = output_dir / "timing_summary.csv"
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "mesh", "task", "mean_ms", "std_ms"])
        for r in results:
            writer.writerow(
                [r["method"], r["mesh"], r["task"], r["mean_ms"], r["std_ms"]]
            )
    mtp.logger.info(f"Summary saved to {summary_file}")

    # Per-iteration CSV for Neural TO
    iter_file = output_dir / "neural_to_iter_times.csv"
    with open(iter_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mesh", "task", "iteration", "time_ms"])
        for r in results:
            if r["iter_times"] is not None:
                for i, t in enumerate(r["iter_times"]):
                    writer.writerow([r["mesh"], r["task"], i, t])
    mtp.logger.info(f"Per-iteration times saved to {iter_file}")


if __name__ == "__main__":
    main()
