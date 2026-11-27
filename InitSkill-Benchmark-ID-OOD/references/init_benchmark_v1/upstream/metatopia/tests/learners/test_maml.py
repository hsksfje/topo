# tests/learners/test_maml.py

import equinox as eqx
import jax
import jax.tree as jt
import pytest
from jax import config
from jax.test_util import check_grads
from metatopia.learners.maml import MAML


config.update("jax_enable_x64", True)


def create_maml(
    base_model,
    optimizers,
    loss_fns,
    *,
    first_order=False,
    use_grad_checkpointing=False,
    custom_filters=None,
):
    """Helper function to create MAML instances with different configurations."""
    inner_opt, outer_opt = optimizers
    inner_loss_fn, outer_loss_fn = loss_fns

    inner_filter = outer_filter = eqx.is_array
    if custom_filters:
        inner_filter, outer_filter = custom_filters

    return MAML(
        base_model=base_model,
        inner_opt=inner_opt,
        outer_opt=outer_opt,
        inner_loss_fn=inner_loss_fn,
        outer_loss_fn=outer_loss_fn,
        inner_filter=inner_filter,
        outer_filter=outer_filter,
        first_order=first_order,
        use_grad_checkpointing=use_grad_checkpointing,
    )


@pytest.mark.parametrize("inner_steps", [0, 1, 3])
@pytest.mark.parametrize("iterators", [1, 4], indirect=True)
@pytest.mark.parametrize("first_order", [True, False])
@pytest.mark.parametrize("use_grad_checkpointing", [True, False])
def test_maml_adapt(
    base_model,
    optimizers,
    loss_fns,
    inner_steps,
    iterators,
    first_order,
    use_grad_checkpointing,
):
    maml = create_maml(
        base_model,
        optimizers,
        loss_fns,
        first_order=first_order,
        use_grad_checkpointing=use_grad_checkpointing,
    )
    train_iter, _ = iterators

    train_batch = next(train_iter)
    batch_size = train_batch["x"].shape[0]

    inner_params, inner_static = eqx.partition(maml.base_model, maml.inner_filter)

    adapted_params, losses, metrics = maml.adapt(
        inner_params, inner_static, train_batch, inner_steps=inner_steps
    )

    # Check if adapted_params have the same structure as inner_params
    assert jt.structure(inner_params) == jt.structure(
        adapted_params
    ), "Adapted params have incorrect structure"

    # Verify parameter updates occurred
    assert not eqx.tree_equal(inner_params, adapted_params)

    # Check the shape of losses and metrics is correct
    def check_element_shape(x):
        assert x.shape[:2] == (
            batch_size,
            inner_steps + 1,
        ), "Incorrect shape after adaptation"

    # # Check the shape of adapted_params
    jt.map(check_element_shape, losses)
    jt.map(check_element_shape, metrics)


@pytest.mark.parametrize("inner_steps", [3])
@pytest.mark.parametrize("iterators", [4], indirect=True)
@pytest.mark.parametrize("first_order", [True, False])
@pytest.mark.parametrize("use_grad_checkpointing", [True, False])
def test_maml_meta_loss(
    base_model,
    optimizers,
    loss_fns,
    inner_steps,
    iterators,
    first_order,
    use_grad_checkpointing,
):
    maml = create_maml(
        base_model,
        optimizers,
        loss_fns,
        first_order=first_order,
        use_grad_checkpointing=use_grad_checkpointing,
    )
    train_iter, val_iter = iterators

    train_batch = next(train_iter)
    val_batch = next(val_iter)
    batch_size = train_batch["x"].shape[0]

    outer_params, outer_static = eqx.partition(maml.base_model, maml.outer_filter)

    meta_loss, metrics = maml.meta_loss(
        outer_params,
        outer_static,
        train_batch,
        val_batch,
        inner_steps=inner_steps,
    )

    # Check meta_loss is scalar
    assert isinstance(meta_loss, jax.Array)
    assert meta_loss.ndim == 0, "Meta loss should be a scalar"

    # Check metrics have correct batch dimension
    def check_element_shape(x):
        assert x.shape[0] == batch_size, "Incorrect batch dimension in metrics"

    jt.map(check_element_shape, metrics)


@pytest.mark.parametrize("inner_steps", [3])
@pytest.mark.parametrize("iterators", [4], indirect=True)
@pytest.mark.parametrize("first_order", [True, False])
@pytest.mark.parametrize("use_grad_checkpointing", [True, False])
def test_maml_update(
    base_model,
    optimizers,
    loss_fns,
    inner_steps,
    iterators,
    first_order,
    use_grad_checkpointing,
):
    maml = create_maml(
        base_model,
        optimizers,
        loss_fns,
        first_order=first_order,
        use_grad_checkpointing=use_grad_checkpointing,
    )
    train_iter, val_iter = iterators

    train_batch = next(train_iter)
    val_batch = next(val_iter)
    batch_size = train_batch["x"].shape[0]

    outer_params, outer_static = eqx.partition(maml.base_model, maml.outer_filter)
    opt_state = maml.outer_opt.init(outer_params)

    new_params, new_static, new_opt_state, loss, metrics = maml.update(
        outer_params,
        outer_static,
        opt_state,
        train_batch,
        val_batch,
        inner_steps=inner_steps,
    )

    # Check structure preservation
    assert jt.structure(outer_params) == jt.structure(new_params)
    assert jt.structure(outer_static) == jt.structure(new_static)

    # Verify parameter updates occurred
    assert not eqx.tree_equal(outer_params, new_params)

    # Check loss is scalar
    assert isinstance(loss, jax.Array)
    assert loss.ndim == 0, "Loss should be a scalar"

    # Check metrics have correct batch dimension
    def check_element_shape(x):
        assert x.shape[0] == batch_size, "Incorrect batch dimension in metrics"

    jt.map(check_element_shape, metrics)


@pytest.mark.parametrize("inner_steps", [1, 3])
@pytest.mark.parametrize("iterators", [1, 4], indirect=True)
def test_maml_gradients(
    base_model,
    optimizers,
    loss_fns,
    inner_steps,
    iterators,
    first_order=False,
    use_grad_checkpointing=False,
):
    maml = create_maml(
        base_model,
        optimizers,
        loss_fns,
        first_order=first_order,
        use_grad_checkpointing=use_grad_checkpointing,
    )

    train_iter, val_iter = iterators
    train_batch = next(train_iter)
    val_batch = next(val_iter)

    # Initialize parameters
    outer_params, outer_static = eqx.partition(base_model, maml.outer_filter)

    def meta_loss_fn(params):
        """Wrapper for meta_loss that returns just the loss value."""
        loss, _ = maml.meta_loss(
            params,
            outer_static,
            train_batch,
            val_batch,
            inner_steps,
        )
        return loss

    # Check gradients of meta_loss w.r.t. outer_params
    # [NOTE] We use a very high tolerance here because the gradients are noisy
    check_grads(
        meta_loss_fn,
        [outer_params],
        order=1,
        modes=["rev"],
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("inner_steps", [3])
@pytest.mark.parametrize("iterators", [4], indirect=True)
def test_filtered_maml(base_model, optimizers, loss_fns, inner_steps, iterators):
    """Test MAML with custom filters."""
    from metatopia.learners import get_filters

    custom_filters = get_filters("ANIL", base_model)
    inner_filter, outer_filter = custom_filters
    maml = create_maml(
        base_model,
        optimizers,
        loss_fns,
        custom_filters=custom_filters,
    )
    train_iter, val_iter = iterators
    train_batch = next(train_iter)
    val_batch = next(val_iter)

    outer_params, outer_static = eqx.partition(base_model, outer_filter)
    inner_params, inner_static = eqx.partition(base_model, inner_filter)

    outer_opt_state = maml.outer_opt.init(outer_params)

    # Perform adaptation
    adapted_params, losses, metrics = maml.adapt(
        inner_params, inner_static, train_batch, inner_steps=inner_steps
    )

    # Perform one meta-update
    new_outer_params, new_static, new_opt_state, loss, metrics = maml.update(
        outer_params,
        outer_static,
        outer_opt_state,
        train_batch,
        val_batch,
        inner_steps=3,
    )

    assert not eqx.tree_equal(outer_params, new_outer_params)
    assert eqx.tree_equal(outer_static, new_static)
