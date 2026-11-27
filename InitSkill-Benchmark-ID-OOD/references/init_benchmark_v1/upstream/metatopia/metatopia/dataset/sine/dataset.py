"""
Dataset utilities
"""

# Modules
# =============================================================================
from functools import partial
from typing import Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ..common import AbstractTaskIterator


# Authorship and Credits
# =============================================================================
__author__ = "Igor Kuszczak (igor.kuszczak.18@ucl.ac.uk)"
__credits__ = ["Igor Kuszczak"]
__status__ = "WIP"


# =============================================================================
def generate_sinusoids(
    batch_size: int, num_points: int, key: jr.PRNGKey
) -> Tuple[jax.Array, jax.Array]:
    """
    Return data generated from sinusoids with a random amplitude and phase
    y = A * sin(x + phase)
    """
    k1, k2, k3 = jr.split(key, 3)

    amp = jr.uniform(key=k1, shape=(batch_size, 1, 1), minval=0.1, maxval=5.0)
    phase = jr.uniform(key=k2, shape=(batch_size, 1, 1), minval=0.0, maxval=jnp.pi)
    x = jr.uniform(key=k3, shape=(batch_size, num_points, 1), minval=-5.0, maxval=5.0)

    y = amp * jnp.sin(x + phase)

    return x, y, amp, phase


@eqx.filter_jit
def generate_sin_tasks(
    batch_size: int,
    num_points: int,
    key: jax.random.PRNGKey,
    split: str = "train",
) -> Dict[str, Tuple[jax.Array, jax.Array]]:
    """Generates random batches of sinusoidal tasks. The output is a
    dictionary, similar to the return type in NVIDIA DALI iterators."""
    x, y, amp, phase = generate_sinusoids(batch_size, 2 * num_points, key=key)
    x_train, x_val = jnp.split(x, 2, axis=1)
    y_train, y_val = jnp.split(y, 2, axis=1)

    if split == "train":
        return {"x": x_train, "y": y_train, "amp": amp, "phase": phase}
    elif split == "val":
        return {"x": x_val, "y": y_val, "amp": amp, "phase": phase}


class SinTaskIterator(AbstractTaskIterator):
    batch_size: int
    key: jr.PRNGKey
    generator_fn: Callable
    split: str

    def __init__(
        self,
        batch_size: int,
        num_points: int,
        seed: int,
        split: str = "train",
    ):
        self.batch_size = batch_size
        self.key = jr.PRNGKey(seed)
        self.generator_fn = partial(
            generate_sin_tasks, num_points=num_points, split=split
        )


def sin_task_iterator(
    batch_size: int, num_points: int, seed=int, split: str = "train"
) -> SinTaskIterator:
    return SinTaskIterator(
        batch_size=batch_size, num_points=num_points, seed=seed, split=split
    )
