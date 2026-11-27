from typing import Any, Callable, Union

import equinox as eqx
import equinox.internal as eqxi
import jax.numpy as jnp
import optax
from jaxtyping import PyTree

from .base_learner import AbstractMetaLearner


AxisSpec = Union[bool, Callable[[Any], bool]]


class MAML(AbstractMetaLearner):
    """
    Implementation of Model-Agnostic Meta-Learning (MAML) [1], which
    learns an initialization for a base model that can be quickly
    adapted to new tasks with few gradient steps.

    Notes
    -----
    MAML's memory usage can be reduced by enabling gradient
    checkpointing (use_grad_checkpointing=True), which recomputes
    intermediate values during the backward pass instead of storing
    them. This comes at the cost of increased computation time.

    When using Adam as the inner optimizer, set eps_root=1e-8 to enable
    differentiation through the inner loop.

    The inner_filter and outer_filter parameters can be used to
    implement variants of MAML that only adapt or meta-learn specific
    parts of the model, e.g. ANIL, BOIL or CAVIA. By default, all
    learnable parameters are both adapted and meta-learned.

    References
    ----------
    [1] Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic
    Meta-Learning for Fast Adaptation of Deep Networks.
    https://arxiv.org/abs/1703.03400
    """

    base_model: eqx.Module
    inner_opt: optax.GradientTransformation
    outer_opt: optax.GradientTransformation
    inner_loss_fn: Callable
    outer_loss_fn: Callable
    inner_filter: PyTree[AxisSpec]
    outer_filter: PyTree[AxisSpec]

    first_order: bool
    use_grad_checkpointing: bool
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
        *,
        first_order: bool = False,
        use_grad_checkpointing: bool = True,
        name: str = "MAML",
    ):
        """
        Initialize a MAML meta-learner.

        Parameters
        ----------
        base_model : eqx.Module
            Model used to solve the base task. Should include parameters and
            hyperparameters that will be optimized in inner and outer loops.
        inner_opt : optax.GradientTransformation
            Optimizer used for inner (task) level learning.
        outer_opt : optax.GradientTransformation
            Optimizer used for meta-optimization (outer loop).
        inner_loss_fn : Callable[[PyTree, PyTree, PyTree], Tuple[jax.Array, PyTree]]
            Loss function for task adaptation. Takes (params, static, batch) and
            returns a tuple of (scalar_loss, metrics).
        outer_loss_fn : Callable[[PyTree, PyTree, PyTree], Tuple[jax.Array, PyTree]]
            Loss function for meta-optimization. Takes (params, static, batch) and
            returns a tuple of (scalar_loss, metrics).
        inner_filter : PyTree[AxisSpec], optional
            PyTree matching the structure of base_model that determines which
            parameters are adapted in the inner loop, by default eqx.is_array.
        outer_filter : PyTree[AxisSpec], optional
            PyTree matching the structure of base_model that determines which
            parameters are updated in the outer loop, by default eqx.is_array.
        first_order : bool, optional
            If True, uses first-order approximation by ignoring second-order
            derivatives, by default False.
        use_grad_checkpointing : bool, optional
            If True, uses gradient checkpointing to reduce memory usage during
            meta-training, by default True.

        """

        self.base_model = base_model
        self.inner_opt = eqxi.closure_to_pytree(inner_opt)
        self.outer_opt = eqxi.closure_to_pytree(outer_opt)
        self.inner_loss_fn = inner_loss_fn
        self.outer_loss_fn = outer_loss_fn
        self.inner_filter = inner_filter
        self.outer_filter = outer_filter

        # MAML-specific settings
        self.first_order = first_order
        self.use_grad_checkpointing = use_grad_checkpointing
        self.name = name

    def meta_loss(
        self,
        outer_params,
        outer_static,
        train_batch,
        validation_batch,
        inner_steps,
    ):
        """
        Compute the meta-loss for a batch of tasks.

        Parameters
        ----------
        outer_params : PyTree
            Current learnable meta-parameters of the model.
        outer_static : PyTree
            Static meta-parameters that remain fixed during
            meta-learning.
        train_batch : PyTree
            Batch of training data used for task adaptation. Contains
            jax.Arrays/floats.
        validation_batch : PyTree
            Batch of validation data used for computing meta-gradients.
            Contains jax.Arrays/floats.
        inner_steps : int
            Number of adaptation steps to perform for each task.

        Returns
        -------
        Tuple[jax.Array, PyTree]
            - meta_loss: Scalar meta-loss averaged over the batch of
              tasks
            - metrics: PyTree of evaluation metrics computed on
              validation data

        Notes
        -----
        This method: 1. Reconstructs the full model from outer_params
        and outer_static 2. Partitions it according to inner_filter for
        adaptation 3. Performs task adaptation and evaluation 4. Returns
        the mean meta-loss and metrics
        """
        model = eqx.combine(outer_params, outer_static)
        inner_params, inner_static = eqx.partition(model, self.inner_filter)

        _, _, _, _, meta_losses, metrics = self.adapt_and_evaluate(
            inner_params,
            inner_static,
            train_batch,
            validation_batch,
            inner_steps,
            self.first_order,
            self.use_grad_checkpointing,
        )

        meta_loss = jnp.mean(meta_losses, axis=0)

        return meta_loss, metrics

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
            Current learnable meta-parameters of the model.
        outer_static : PyTree
            Static meta-parameters that remain fixed during meta-learning.
        outer_opt_state : optax.OptState
            Current state of the meta-optimizer.
        train_batch : PyTree
            Batch of training data used for task adaptation. Contains
            jax.Arrays/floats.
        validation_batch : PyTree
            Batch of validation data used for computing meta-gradients.
            Contains jax.Arrays/floats.
        inner_steps : int
            Number of adaptation steps to perform for each task.

        Returns
        -------
        Tuple[PyTree, PyTree, optax.OptState, jax.Array, PyTree]
            - Updated meta-parameters
            - Updated static parameters
            - Updated meta-optimizer state
            - Meta-loss computed during the update
            - Meta-metrics computed during the update

        Notes
        -----
        This method computes meta-gradients through the entire adaptation
        process (unless first_order=True) and applies them using the
        meta-optimizer to update the meta-parameters.
        """

        (meta_loss, metrics), meta_grad = eqx.filter_value_and_grad(
            self.meta_loss, has_aux=True
        )(
            outer_params,
            outer_static,
            train_batch,
            validation_batch,
            inner_steps,
        )

        updates, outer_opt_state = self.outer_opt.update(meta_grad, outer_opt_state)
        outer_params = eqx.apply_updates(outer_params, updates)

        return outer_params, outer_static, outer_opt_state, meta_loss, metrics
