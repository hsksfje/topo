import operator as op

import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import PyTree


def tree_extend(
    base: PyTree, extension: PyTree, axis: int = 0, expand_dims=True
) -> PyTree:
    """
    Concatenates an extension PyTree to a base PyTree along specified axis.

    Parameters
    ----------
    base : PyTree
        Base tree to extend
    extension : PyTree
        Tree to append to base
    axis : int, optional
        Axis along which to concatenate, by default 0
    expand_dims : bool, optional
        Whether to expand dims of extension before concatenating, by default True

    Returns
    -------
    PyTree
        Concatenated PyTree
    """
    if expand_dims:

        def _tree_extend(base, extension):
            return jnp.concatenate(
                [base, jnp.expand_dims(extension, axis)], axis=axis
            )

    else:

        def _tree_extend(base, extension):
            return jnp.concatenate([base, extension], axis=axis)

    return jt.map(_tree_extend, base, extension)


def tree_sum(tree):
    """
    Computes sum of all leaves in a PyTree.

    Parameters
    ----------
    tree : PyTree
        Input tree to sum

    Returns
    -------
    jax.Array
        Sum of all leaves
    """
    sums = jt.map(jnp.sum, tree)
    return jt.reduce(op.add, sums)


def tree_add_scalar_mul(tree_x, scalar, tree_y):
    """
    Computes tree_x + scalar * tree_y element-wise.

    Parameters
    ----------
    tree_x : PyTree
        First tree
    scalar : float
        Scalar multiplier
    tree_y : PyTree
        Second tree

    Returns
    -------
    PyTree
        Result of tree_x + scalar * tree_y
    """
    return jt.map(lambda x, y: x + scalar * y, tree_x, tree_y)


def tree_l2_norm(tree, squared=True, mean=False):
    """
    Computes L2 norm of PyTree leaves.

    Parameters
    ----------
    tree : PyTree
        Input tree
    squared : bool, optional
        Whether to return squared norm, by default True
    mean : bool, optional
        Whether to average across leaves, by default False

    Returns
    -------
    jax.Array
        L2 norm of tree
    """

    squared_tree = jt.map(jnp.square, tree)
    norm = tree_mean(squared_tree) if mean else tree_sum(squared_tree)

    if not squared:
        return jnp.sqrt(norm)
    else:
        return norm


def tree_mean(tree, axis=0):
    """
    Computes mean of PyTree leaves along specified axis.

    Parameters
    ----------
    tree : PyTree
        Input tree
    axis : int, optional
        Axis along which to compute mean, by default 0

    Returns
    -------
    PyTree
        Tree with means along specified axis
    """
    return jt.map(lambda g: g.mean(axis=axis), tree)


def tree_stack(trees):
    """
    Stacks multiple PyTrees along a new first axis.

    Parameters
    ----------
    trees : Sequence[PyTree]
        Sequence of trees to stack

    Returns
    -------
    PyTree
        Stacked trees
    """
    return jt.map(lambda *v: jnp.stack(v), *trees)
