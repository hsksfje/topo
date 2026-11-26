from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def process_strain_energy(ce):
    ce = np.log(ce)
    ce = 2.0 * (ce - np.min(ce)) / (np.max(ce) - np.min(ce)) - 1.0
    return ce


def get_grid_coords(shape: Tuple[int, int]) -> jax.Array:
    """
    Create a 2D grid of coordinates normalized between -1 and 1.

    Parameters
    ----------
    shape : Tuple[int, int]
        Desired grid shape as (Ny, Nx), where Ny is the number of points
        in the y-direction and Nx is the number of points in the x-direction.

    Returns
    -------
    jax.Array
        Flattened array of coordinates with shape (Ny*Nx, 2), where each row
        contains (x, y) coordinates normalized between -1 and 1.

    Notes
    -----
    The output coordinates are arranged in row-major order, with the x-coordinate
    varying fastest. The resulting grid is uniformly spaced in both dimensions.
    """
    Ny, Nx = shape

    x = jnp.linspace(-1, 1, Nx)
    y = jnp.linspace(-1, 1, Ny)

    coords = jnp.meshgrid(x, y)
    coords = jnp.stack(coords, axis=-1)
    coords = coords.reshape(-1, 2)

    return coords


def serialise_tree(tree, name, output_dir):
    """
    Serialize a PyTree to disk using equinox serialization.

    Parameters
    ----------
    tree : PyTree
        Any valid JAX PyTree structure to be serialized.
    name : str
        Base name for the output file (without extension).
    output_dir : str
        Directory path where the serialized file will be saved.
        The file will be saved as '{output_dir}/{name}.eqx'.

    """
    eqx.tree_serialise_leaves(f"{output_dir}/{name}.eqx", tree)


def serialise_array(array, array_name, output_dir):
    """
    Save a JAX array to disk in NumPy's binary format.

    Parameters
    ----------
    array : jax.Array
        The array to be saved.
    array_name : str
        Base name for the output file (without extension).
    output_dir : str
        Directory path where the array will be saved.
        The file will be saved as '{output_dir}/{array_name}.npy'.

    Notes
    -----
    The output directory must exist and be writable. The array is saved
    using NumPy's binary format (.npy).
    """
    with open(f"{output_dir}/{array_name}.npy", "wb") as f:
        jnp.save(f, array)


def load_tree(tree, load_dir, name):
    """
    Load a serialized PyTree from disk using equinox deserialization.

    Parameters
    ----------
    tree : PyTree
        Template PyTree with the same structure as the serialized tree.
        Used to determine the structure for deserialization.
    load_dir : str
        Directory path containing the serialized file.
    name : str
        Base name of the file to load (without extension).
        The file will be loaded from '{load_dir}/{name}.eqx'.

    Returns
    -------
    PyTree
        Deserialized PyTree with the same structure as the input template tree.

    Notes
    -----
    The template tree is required because equinox needs to know the expected
    PyTree structure for proper deserialization.
    """

    model = eqx.tree_deserialise_leaves(f"{load_dir}/{name}.eqx", tree)
    return model
