import time
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import metatopia as mtp
import numpy as np
import numpy.random as npr
from omegaconf import DictConfig, OmegaConf
from utils import random_problem


jax.config.update("jax_enable_x64", True)


def process_strain_energy(ce):
    ce = np.log(ce)
    ce = 2.0 * (ce - np.min(ce)) / (np.max(ce) - np.min(ce)) - 1.0
    return ce


def process_files(directory, remove_npz=False):
    """
    Processes all .npz files in the specified directory.

    Args:
    directory (str): Path to the directory containing .npz files.
    remove_npz (bool): Whether to remove .npz files after processing.
    """
    # Convert directory to Path object
    directory = Path(directory)

    # Get a list of all .npz files in the directory
    npz_files = list(directory.glob("*.npz"))
    processed_ids = set()

    # Process each .npz file
    for npz_file in npz_files:
        # Load the .npz file
        data = np.load(npz_file)

        # Get base filename without extension
        base_filename = npz_file.stem

        # Create problem id by splitting the filename on _
        id = int(base_filename.split("_")[1])

        # Iterate through each key in the .npz file
        for key in data.keys():
            if key == "id":
                continue
            npy_filename = f"{base_filename}_{key}.npy"
            save_path = npz_file.parent / npy_filename

            # Save the data under the current key to a .npy file
            np.save(save_path, data[key])

        if id in processed_ids:
            continue

        # Save problem id to a separate .npy file
        save_path = npz_file.parent / f"{base_filename}_id.npy"
        np.save(save_path, id)
        processed_ids.add(id)

        # Optionally remove the .npz file
        if remove_npz:
            npz_file.unlink()


@hydra.main(version_base=None, config_path="conf", config_name="conf")
def main(cfg: DictConfig):
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    data_path = project_root / "data" / cfg.data_subdir / cfg.name
    data_path.mkdir(parents=True, exist_ok=True)

    solver = mtp.solver.setup_solver_cvxopt_elementwise(
        young=cfg.young,
        young_min=cfg.young_min,
        poisson=cfg.poisson,
        Lx=cfg.Lx,
        Ly=cfg.Ly,
        Nx=cfg.Nx,
        Ny=cfg.Ny,
        penalty=cfg.penalty,
    )

    domain_shape = (cfg.Ny, cfg.Nx)

    mtp.logger.info(f"Generating dataset with {cfg.n_problems} tasks.")

    # Set a random seed for reproducibility
    npr.seed(cfg.seed)

    j = 0
    skipped_problems = 0

    t0 = time.time()

    while j < cfg.n_problems:
        random_vf = npr.uniform(low=cfg.vf_min, high=cfg.vf_max)

        problem = random_problem(
            Nx=cfg.Nx,
            Ny=cfg.Ny,
            target_vf=random_vf,
            p_edge=cfg.p_edge,
            distributed=cfg.distributed,
        )

        ref_design = jnp.ones(domain_shape) * problem["target_vf"]
        init_compliance, init_ce = solver(ref_design, problem)

        if (
            init_compliance > 0.0
            and not (init_ce <= 0.0).any()
            and not np.any(np.isnan(init_ce))
            and np.ptp(init_ce) > 0.0
        ):
            problem["reference_compliance"] = init_compliance
            problem["strain_energy"] = process_strain_energy(init_ce)
            data = problem

            np.savez(f"{data_path}/problem_{j:06d}.npz", **data)
            j += 1

            if j % 100 == 0:
                mtp.logger.info(f"Generated {j} problems.")
        else:
            mtp.logger.warning("Skipping problem.")
            skipped_problems += 1

    mtp.logger.info(f"Dataset generation completed. Dataset saved to {data_path}")

    t1 = time.time()

    mtp.logger.info(f"Time per problem: {(t1 - t0)/cfg.n_problems:.2f} s")
    mtp.logger.info(f"Skipped {skipped_problems} problems.")

    process_files(data_path, remove_npz=True)

    # [TODO] Save the config to the same folder
    # with the dataset for reproducibility
    yaml_config = OmegaConf.to_yaml(cfg)
    with open(f"{data_path}/conf.yaml", "w") as f:
        f.write(yaml_config)


if __name__ == "__main__":
    main()
