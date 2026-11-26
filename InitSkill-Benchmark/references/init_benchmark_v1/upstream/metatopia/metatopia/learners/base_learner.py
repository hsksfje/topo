"""
Module description
"""

# Modules
# =============================================================================
from abc import abstractmethod
from typing import Any, Callable, Tuple

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import optax
from equinox import AbstractVar
from jaxtyping import PyTree

import metatopia as mtp

# Custom types
from metatopia._custom_types import (
    AxisSpec,
    InnerLosses,
    InnerMetrics,
    OuterLosses,
    OuterMetrics,
)
from metatopia.utils import tree_extend, tree_stack


# Authorship and Credits
# =============================================================================
__author__ = "Igor Kuszczak (igor.kuszczak.18@ucl.ac.uk)"
__credits__ = ["Igor Kuszczak"]
__status__ = "Stable"


# =============================================================================
class AbstractMetaLearner(eqx.Module, strict=True):
    """
    Abstract base class for gradient-based meta-learning algorithms.
    This class defines the interface for meta-learning algorithms that
    optimize a base model's parameters via bi-level optimization.

    Parameters
    ----------
    base_model : eqx.Module
        Model used to solve the base task. Should include parameters and
        hyperparameters that will be optimized in inner and outer
        loops.
    inner_opt : optax.GradientTransformation
        Optimizer used for inner (task) level learning.
    outer_opt : optax.GradientTransformation
        Optimizer used for outer (meta) level learning.
    inner_loss_fn : Callable[[PyTree, PyTree, PyTree], Tuple[jax.Array, PyTree]]
        Loss function used in the inner loop (adaptation). Takes
        (params, static, batch) and returns a tuple of (scalar_loss,
        metrics), where metrics is a PyTree of jax.Arrays/floats.
    outer_loss_fn : Callable[[PyTree, PyTree, PyTree], Tuple[jax.Array, PyTree]]
        Loss function used in the outer loop. Takes (params, static,
        batch) and returns a tuple of (scalar_loss, metrics), where
        metrics is a PyTree of jax.Arrays/floats.
    inner_filter : PyTree[AxisSpec]
        PyTree matching the structure of base_model. Determines which
        parameters are adapted in the inner loop.
    outer_filter : PyTree[AxisSpec]
        PyTree matching the structure of base_model. Determines which
        parameters are updated in the outer loop.
    name : str
        Identifier for the meta-learning algorithm.

    Notes
    -----
    Different combinations of inner_filter and outer_filter enable
    implementation of common meta-learning variants. For example, ANIL
    can be implemented by only adapting the last layer (head) of the
    base_model in the inner loop, while updating all learnable
    parameters in the outer loop.
    """

    base_model: AbstractVar[eqx.Module]
    inner_opt: AbstractVar[optax.GradientTransformation]
    outer_opt: AbstractVar[optax.GradientTransformation]
    inner_loss_fn: AbstractVar[Callable]
    outer_loss_fn: AbstractVar[Callable]
    inner_filter: AbstractVar[PyTree[AxisSpec]]
    outer_filter: AbstractVar[PyTree[AxisSpec]]
    name: str

    def adapt(
        self,
        inner_params: PyTree,
        inner_static: PyTree,
        train_batch: Any,
        inner_steps: int,
        first_order: bool = False,
        use_grad_checkpointing: bool = False,
    ) -> Tuple[PyTree, PyTree, InnerLosses, InnerMetrics]:
        """
        Adapt the base learner to a batch of tasks.

        Parameters
        ----------
        inner_params : PyTree
            Initial learnable parameters of the base learner that will
            be adapted during task optimization.
        inner_static : PyTree
            Static (frozen) parameters of the base learner that remain
            fixed during adaptation.
        train_batch : PyTree
            Batch of training data used for task adaptation. First
            dimension is the batch dimension.
        inner_steps : int
            Number of adaptation steps to perform for each task.
        first_order : bool, optional
            If True, uses first-order approximation by stopping
            gradients through the adaptation process, by default False.
        use_grad_checkpointing : bool, optional
            If True, uses gradient checkpointing to reduce memory usage
            at the cost of increased computation time, by default False.

        Returns
        -------
        Tuple[PyTree, PyTree, InnerLosses, InnerMetrics]
            - Adapted task-specific parameters
            - Static parameters unchanged during adaptation
            - Array of adaptation losses
            - PyTree of metrics computed during adaptation

        Notes
        -----
        This method performs task adaptation by running on the provided
        batch of tasks. The adaptation is vectorized across the batch
        dimension for efficiency.
        """

        def _adapt(
            inner_params: PyTree,
            inner_static: PyTree,
            sample: Any,
            inner_steps: int,
        ):
            """
            Single task adaptation.
            """

            def scan_fn(carry, _):
                inner_params, opt_state = carry
                (loss, metrics), grads = eqx.filter_value_and_grad(
                    self.inner_loss_fn, has_aux=True
                )(inner_params, inner_static, sample)

                if first_order:
                    grads = jax.lax.stop_gradient(grads)

                updates, opt_state = self.inner_opt.update(grads, opt_state)
                inner_params = eqx.apply_updates(inner_params, updates)

                return (inner_params, opt_state), (
                    loss,
                    metrics,
                )

            opt_state = self.inner_opt.init(inner_params)

            (inner_params, opt_state), (losses, metrics) = eqxi.scan(
                scan_fn,
                (inner_params, opt_state),
                None,
                length=inner_steps,
                kind=("checkpoint" if use_grad_checkpointing else "lax"),
            )

            final_loss, final_metrics = self.inner_loss_fn(
                inner_params, inner_static, sample
            )

            losses = tree_extend(losses, final_loss)
            metrics = tree_extend(metrics, final_metrics)

            return inner_params, inner_static, losses, metrics

        # vmap across the batch dimension
        adapted_params, _, losses, metrics = eqx.filter_vmap(
            _adapt, in_axes=(None, None, 0, None)
        )(inner_params, inner_static, train_batch, inner_steps)

        return adapted_params, losses, metrics

    def adapt_and_evaluate(
        self,
        inner_params,
        inner_static,
        train_batch,
        validation_batch,
        inner_steps,
        first_order=False,
        use_grad_checkpointing=False,
    ) -> Tuple[PyTree, PyTree, InnerLosses, InnerMetrics, OuterLosses, OuterMetrics]:
        """
        Performs adaptation on a batch of training tasks and evaluates
        the adapted model on the validation batch.

        Parameters
        ----------
        inner_params : PyTree
            Initial learnable parameters of the base learner that will
            be adapted during task optimization.
        inner_static : PyTree
            Static (frozen) parameters of the base learner that remain
            fixed during adaptation.
        train_batch : PyTree
            Batch of training data used for task adaptation. Contains
            jax.Arrays/floats. First dimension is the batch dimension.
        validation_batch : PyTree
            Batch of validation data used for meta-evaluation. Contains
            jax.Arrays/floats. First dimension is the batch dimension.
        inner_steps : int
            Number of adaptation steps to perform for each task.
        first_order : bool, optional
            If True, uses first-order approximation by stopping
            gradients through the adaptation process, by default False.
        use_grad_checkpointing : bool, optional
            If True, uses gradient checkpointing to reduce memory usage
            at the cost of increased computation time, by default False.

        Returns
        -------
        Tuple[PyTree, PyTree, InnerLosses, InnerMetrics, OuterLosses, OuterMetrics]
            - Adapted task-specific parameters
            - Static parameters unchanged during adaptation
            - Array of losses during task adaptation
            - PyTree of metrics computed during adaptation
            - Array of losses computed on validation data
            - PyTree of metrics computed on validation data

        Notes
        -----
        This method combines task adaptation with evaluation on
        validation data, making it useful for computing meta-gradients
        in the outer loop. The validation loss and metrics can be used
        to assess the model's few-shot learning performance.
        """

        inner_params, task_losses, task_metrics = self.adapt(
            inner_params,
            inner_static,
            train_batch,
            inner_steps,
            first_order=first_order,
            use_grad_checkpointing=use_grad_checkpointing,
        )

        outer_losses, outer_metrics = eqx.filter_vmap(
            self.outer_loss_fn, in_axes=(0, None, 0)
        )(inner_params, inner_static, validation_batch)

        return (
            inner_params,
            inner_static,
            task_losses,
            task_metrics,
            outer_losses,
            outer_metrics,
        )

    @abstractmethod
    def update(
        self,
        outer_params: PyTree,
        outer_static: PyTree,
        outer_opt_state: optax.OptState,
        train_batch: Any,
        validation_batch: Any,
        inner_steps: int,
    ) -> Tuple[PyTree, PyTree, optax.OptState, OuterLosses, OuterMetrics]:
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
            jax.Arrays/floats. First dimension is the batch dimension.
        validation_batch : PyTree
            Batch of validation data used for computing meta-gradients.
            Contains jax.Arrays/floats. First dimension is the batch dimension.
        inner_steps : int
            Number of adaptation steps to perform for each task.

        Returns
        -------
        Tuple[PyTree, PyTree, optax.OptState, OuterLosses, OuterMetrics]
            - Updated meta-parameters
            - Updated static parameters
            - Updated meta-optimizer state
            - Meta-loss computed during the update
            - Meta-metrics computed during the update

        Notes
        -----
        This abstract method should be implemented by concrete meta-learning
        algorithms to define how meta-parameters are updated using gradients
        computed through the task adaptation process.
        """
        pass


def meta_learn(
    meta_learner: AbstractMetaLearner,
    train_iterator_fn: Callable,
    validation_iterator_fn: Callable | None = None,
    batch_size: int = 1,
    outer_steps: int = 1,
    inner_steps: int = 1,
    log_every_n: int = None,
) -> Tuple[eqx.Module, OuterLosses, OuterMetrics]:
    """
    Performs meta-learning using the provided train and validation iterators.

    Parameters
    ----------
    meta_learner : AbstractMetaLearner
        AbstractMetaLearner instance that implements the inner
        (adaptation) and outer (meta-learning) update logic.
    train_iterator_fn : Callable[[int], Iterator[PyTree]]
        Function that takes a batch_size parameter and returns an
        iterator over training tasks. Each iteration yields a PyTree
        containing batched
        training data (jax.Arrays/floats).
    validation_iterator_fn : Callable[[int], Iterator[PyTree]] | None
        Function that takes a batch_size parameter and returns an
        iterator over validation tasks. Each iteration yields a PyTree
        containing batched validation data (jax.Arrays/floats). If None,
        the training batch will be used for both adaptation and
        meta-update. Default: None
    batch_size : int, optional
        Number of tasks to process in parallel during meta-training.
        Default: 1
    outer_steps : int, optional
        Total number of outer loop steps to perform. Default: 1
    inner_steps : int, optional
        Number of adaptation steps to perform for each task. Default: 1
    log_every_n : int | None, optional
        Frequency of logging meta-learning progress. If None, will log
        approximately 10 times during training. Default: None

    Returns
    -------
    Tuple[eqx.Module, OuterLosses, OuterMetrics]
    - meta_learned_model: The meta-trained model
    - outer_losses: Array containing meta-losses recorded during training
    - outer_metrics: PyTree of metrics with containing outer metrics
    recorded during training

    Notes
    -----
    Memory usage scales with batch_size and inner_steps. If you
    encounter OOM errors, consider: - Reducing batch_size - Using
    gradient checkpointing if supported by the meta-learner - Reducing
    inner_steps

    Progress logging prints the current step and loss value. The
    frequency defaults to approximately 10 logs over the course of
    training, but can be customized using log_every_n.
    """

    outer_params, outer_static = eqx.partition(
        meta_learner.base_model, meta_learner.outer_filter
    )

    outer_opt_state = meta_learner.outer_opt.init(outer_params)

    update = eqx.filter_jit(meta_learner.update)

    outer_losses = []
    outer_metrics_list = []

    train_iterator = train_iterator_fn(batch_size=batch_size)
    validation_iterator = (
        validation_iterator_fn(batch_size=batch_size)
        if validation_iterator_fn
        else None
    )

    # Default value
    if not log_every_n:
        log_every_n = max(1, outer_steps // 10)

    for step in range(outer_steps):
        train_batch = mtp.dataset.get_next(train_iterator)

        validation_batch = (
            mtp.dataset.get_next(validation_iterator)
            if validation_iterator
            else train_batch
        )

        (
            outer_params,
            outer_static,
            outer_opt_state,
            meta_loss,
            outer_metrics,
        ) = update(
            outer_params,
            outer_static,
            outer_opt_state,
            train_batch,
            validation_batch,
            inner_steps,
        )

        if step == 0 or step % log_every_n == 0 or step == outer_steps - 1:
            mtp.logger.info(f"Step: {step:<5} Loss: {meta_loss:.5f}")

        outer_losses.append(jnp.array(meta_loss))
        outer_metrics_list.append(outer_metrics)

    meta_learned_model = eqx.combine(outer_params, outer_static)

    outer_metrics = tree_stack(outer_metrics_list) if outer_metrics_list else None
    outer_losses = tree_stack(outer_losses) if outer_losses else None

    return meta_learned_model, outer_losses, outer_metrics
