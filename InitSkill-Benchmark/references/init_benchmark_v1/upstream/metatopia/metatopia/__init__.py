from .__version__ import __version__
from ._src.logger import logger


try:
    import jax

    _backend = jax.default_backend()
    logger.info(f"Running version {__version__} on the {_backend.upper()}.")
    del _backend
except ImportError:
    logger.error("JAX not found. Please install JAX to use Metatopia.")

# Import the core modules
from . import (
    dataset,
    filters,
    learners,
    models,
    solver,
    utils,
)


__all__ = [
    "dataset",
    "filters",
    "learners",
    "models",
    "solver",
    "utils",
]
