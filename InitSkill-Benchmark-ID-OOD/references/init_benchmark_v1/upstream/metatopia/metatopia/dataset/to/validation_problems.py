from typing import Callable, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ..common import AbstractTaskIterator


def mbb_beam(width, height, target_vf):
    normals = np.zeros((width + 1, height + 1, 2))
    normals[-1, -1, 1] = 1
    normals[0, :, 0] = 1
    forces = np.zeros((width + 1, height + 1, 2))
    forces[0, 0, 1] = -1

    return {
        "normals": normals,
        "forces": forces,
        "target_vf": target_vf,
        "id": 0,
        "reference_compliance": 1.0,
        "strain_energy": None,
    }


def cantilever_beam(width, height, target_vf, force_y=1):
    normals = np.zeros((width + 1, height + 1, 2))
    normals[0, :, :] = 1

    forces = np.zeros((width + 1, height + 1, 2))
    forces[-1, round((1 - force_y) * height), 1] = -1

    return {
        "normals": normals,
        "forces": forces,
        "target_vf": target_vf,
        "id": 0,
        "reference_compliance": 1.0,
        "strain_energy": None,
    }


def tensile(width, height, target_vf):
    normals = np.zeros((width + 1, height + 1, 2))
    normals[0, :, :] = 1

    forces = np.zeros((width + 1, height + 1, 2))
    forces[-1, round(0.5 * height), 0] = -1

    return {
        "normals": normals,
        "forces": forces,
        "target_vf": target_vf,
        "id": 0,
        "reference_compliance": 1.0,
        "strain_energy": None,
    }


class TOSingleTaskIterator(AbstractTaskIterator):
    batch_size: int
    key: jr.PRNGKey
    generator_fn: Callable
    split: str

    def __init__(
        self,
        domain_shape: tuple[int, int],
        name: Literal["mbb_beam", "cantilever_beam", "tensile"] = "mbb_beam",
    ):
        self.batch_size = 1
        self.key = jr.key(0)  # Dummy key

        Ny, Nx = domain_shape

        problem_dict = {
            "mbb_beam": mbb_beam,
            "cantilever_beam": cantilever_beam,
            "tensile": tensile,
        }

        def _generator_fn(batch_size, key):
            # Only vary the volume fraction
            # target_vf = jr.uniform(key, (batch_size,), minval=0.1, maxval=0.5).ravel()

            target_vf = jnp.full((batch_size,), 0.5) # Dummy value - single vf

            def get_problem(target_vf):
                return problem_dict[name](Nx, Ny, target_vf)

            return jax.vmap(get_problem)(target_vf)

        self.generator_fn = _generator_fn


def to_single_task_iterator_fn(domain_shape, name, batch_size):
    return TOSingleTaskIterator(domain_shape, name)
