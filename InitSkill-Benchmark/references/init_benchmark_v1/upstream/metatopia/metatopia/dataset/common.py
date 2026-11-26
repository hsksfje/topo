from typing import Callable

import jax.random as jr
from equinox import AbstractVar

from .._custom_types import TaskBatch


class AbstractTaskIterator:
    """
    Base class for iterators that generate infinite streams of random
    task batches based on a generator function. This class provides an iterator
    interface consistent with NVIDIA's DALI iterators.

    Parameters
    ----------
    generator_fn : Callable[[int, jr.PRNGKey], TaskBatch]
        Function that generates random batches of tasks. Should accept batch_size
        and key parameters and return a PyTree with array leaves. All arrays
        should have a leading dimension equal to batch_size.
    batch_size : int
        Number of tasks to generate in each batch.
    key : jr.PRNGKey
        JAX random key for task generation.

    Notes
    -----
    This iterator provides an infinite stream of randomly generated tasks, making
    it useful for training scenarios where you need continuous random task
    generation. Since tasks are generated on-the-fly rather than drawn from a
    fixed dataset, there is no concept of dataset exhaustion or reset.
    """

    generator_fn: AbstractVar[Callable[[int, jr.PRNGKey], TaskBatch]]
    batch_size: AbstractVar[int]
    key: AbstractVar[jr.PRNGKey]

    def init_key(self, seed: int) -> jr.PRNGKey:
        """Initialize the random key with a given seed."""
        return jr.key(seed)

    def __iter__(self) -> "AbstractTaskIterator":
        """Make the class iterable."""
        return self

    def __next__(self) -> TaskBatch:
        """Generate the next batch of random tasks."""
        self.key, subkey = jr.split(self.key)
        return self.generator_fn(self.batch_size, key=subkey)


def get_next(
    iterator: AbstractTaskIterator,
) -> TaskBatch:
    """
    Get the next batch from an iterator.

    Parameters
    ----------
    iterator :
        Either an instance of AbstractTaskIterator or NVIDIA DALI JAX iterator

    Returns
    -------
    TaskBatch
        Next batch of tasks
    """
    try:
        return next(iterator)
    except StopIteration:
        iterator.reset()
        return next(iterator)
