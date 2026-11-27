from typing import Any, Callable, Tuple

import equinox as eqx
import equinox.internal as eqxi
import jax.numpy as jnp
import jax.tree as jt
import optax
from jaxtyping import PyTree

from .base_learner import (
    AbstractMetaLearner,
    AxisSpec,
    OuterLosses,
    OuterMetrics,
)


class Reptile(AbstractMetaLearner):
    """
    This class implements the Reptile algorithm [1], which performs
    meta-learning through a simplified first-order approach that treats
    the meta-update as a step towards the parameters adapted on each
    task.

    Notes
    -----
    Unlike MAML, Reptile does not require computing second-order
    derivatives and uses a simpler update rule: move the initialization
    towards the parameters obtained after adapting to each task. This
    makes it more memory-efficient but potentially less expressive than
    MAML.

    Reptile always adapts and meta-learns all array parameters, ignoring
    any provided filter values. This is part of the original algorithm
    design and differs from MAML's flexible parameter filtering.

    References
    ----------
    [1] Nichol, A., Achiam, J., & Schulman, J. (2018). On First-Order
    Meta-Learning Algorithms.
    https://arxiv.org/abs/1803.02999
    """

    base_model: eqx.Module
    inner_opt: optax.GradientTransformation
    outer_opt: optax.GradientTransformation
    inner_loss_fn: Callable
    outer_loss_fn: Callable
    inner_filter: PyTree[AxisSpec]
    outer_filter: PyTree[AxisSpec]
    name: str

    def __init__(
        self,
        base_model: eqx.Module,
        inner_opt: optax.GradientTransformation,
        outer_opt: optax.GradientTransformation,
        inner_loss_fn: Callable,
        outer_loss_fn: Callable,
        inner_filter: PyTree[AxisSpec] = eqx.is_array,
        outer_filter: PyTree[AxisSpec] = eqx.is_array,
        name: str = "Reptile",
    ):
        """
        Initialize the Reptile meta-learner.

        Parameters
        ----------
        base_model : eqx.Module
            Model used to solve the base task. Should include parameters and
            hyperparameters that will be optimized in inner and outer
            loops.
        inner_opt : optax.GradientTransformation
            Optimizer used for task adaptation (inner loop).
        outer_opt : optax.GradientTransformation
            Optimizer used for meta-optimization (outer loop).
        inner_loss_fn : Callable[[PyTree, PyTree, PyTree], Tuple[jax.Array, PyTree]]
            Loss function for task adaptation. Takes (params, static, batch)
            and returns a tuple of (scalar_loss, metrics), where metrics is
            a PyTree of jax.Arrays/floats.
        outer_loss_fn : Callable[[PyTree, PyTree, PyTree], Tuple[jax.Array, PyTree]]
            Loss function for meta-optimization. Takes (params, static,
            batch) and returns a tuple of (scalar_loss, metrics), where
            metrics is a PyTree of jax.Arrays/floats.
        inner_filter : PyTree[AxisSpec], optional
            Ignored by Reptile. All array parameters are adapted, by default
            eqx.is_array.
        outer_filter : PyTree[AxisSpec], optional
            Ignored by Reptile. All array parameters are meta-learned, by
            default eqx.is_array.

        """

        self.base_model = base_model
        self.inner_opt = eqxi.closure_to_pytree(inner_opt)
        self.outer_opt = eqxi.closure_to_pytree(outer_opt)
        self.inner_loss_fn = inner_loss_fn
        self.outer_loss_fn = outer_loss_fn

        if inner_filter != eqx.is_array or outer_filter != eqx.is_array:
            import warnings

            warnings.warn(
                "Reptile uses eqx.is_array for both inner and outer filters "
                "regardless of the provided filter values. The provided filters "
                "will be ignored.",
                UserWarning,
            )

        self.inner_filter = eqx.is_array
        self.outer_filter = eqx.is_array
        self.name = name

    def meta_loss_value_and_grad(
        self, params, static, train_batch, inner_steps
    ) -> Tuple[Tuple[OuterLosses, OuterMetrics], PyTree]:
        """
        Compute the meta-loss and meta-gradients for a batch of tasks.

        Parameters
        ----------
        params : PyTree
            Current learnable parameters of the model.
        static : PyTree
            Static parameters that remain fixed during meta-learning.
        train_batch : PyTree
            Batch of training data used for task adaptation. Contains
            jax.Arrays/floats. First dimension is the batch dimension.
        inner_steps : int
            Number of adaptation steps to perform for each task.

        Returns
        -------
        Tuple[Tuple[OuterLosses, OuterMetrics], PyTree]
            - Tuple of (meta_loss, metrics):
                - meta_loss: Scalar loss averaged over the batch of tasks
                - metrics: PyTree of evaluation metrics
            - meta_grad: PyTree of gradients for meta-update

        """
        (
            adapted_params,
            _,
            _,
            _,
            meta_losses,
            outer_metrics,
        ) = self.adapt_and_evaluate(
            params, static, train_batch, train_batch, inner_steps
        )

        adapted_params = jt.map(lambda x: x.mean(axis=0), adapted_params)
        meta_grad = jt.map(jnp.subtract, params, adapted_params)
        meta_loss = jnp.mean(meta_losses)

        return (meta_loss, outer_metrics), meta_grad

    def update(
        self,
        outer_params: PyTree,
        outer_static: PyTree,
        outer_opt_state: optax.OptState,
        train_batch: Any,
        validation_batch: Any,
        inner_steps: int,
    ):
        """
        Perform a meta-update step on a batch of tasks.

        Parameters
        ----------
        outer_params : PyTree
            Current learnable parameters of the model.
        outer_static : PyTree
            Static parameters that remain fixed during meta-learning.
        outer_opt_state : optax.OptState
            Current state of the meta-optimizer.
        train_batch : PyTree
            Batch of training data used for task adaptation. Contains
            jax.Arrays/floats. First dimension is the batch dimension.
        validation_batch : PyTree
            Ignored in Reptile as it uses the training batch for both
            adaptation and meta-update. Contains jax.Arrays/floats.
        inner_steps : int
            Number of adaptation steps to perform for each task.

        Returns
        -------
        Tuple[PyTree, PyTree, optax.OptState, OuterLosses, OuterMetrics]
            - Updated model parameters
            - Updated static parameters
            - Updated meta-optimizer state
            - Meta-loss computed during the update
            - Meta-metrics computed during the update

        Notes
        -----
        Reptile does not use a separate validation batch for computing
        meta-gradients. Instead, it uses the training batch for both
        adaptation and evaluation.
        """

        (meta_loss, metrics), meta_grad = self.meta_loss_value_and_grad(
            outer_params,
            outer_static,
            train_batch,
            inner_steps,
        )

        updates, outer_opt_state = self.outer_opt.update(
            meta_grad, outer_opt_state
        )
        outer_params = eqx.apply_updates(outer_params, updates)
        return outer_params, outer_static, outer_opt_state, meta_loss, metrics
