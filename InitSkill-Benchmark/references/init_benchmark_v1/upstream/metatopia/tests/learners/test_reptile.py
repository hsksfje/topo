import warnings

import equinox as eqx
import jax
import jax.tree as jt
import pytest
from jax import config
from metatopia.learners.reptile import Reptile


config.update("jax_enable_x64", True)


def create_reptile(base_model, optimizers, loss_fns):
    """Helper function to create Reptile instances with different configurations."""
    inner_opt, outer_opt = optimizers
    inner_loss_fn, outer_loss_fn = loss_fns

    return Reptile(
        base_model=base_model,
        inner_opt=inner_opt,
        outer_opt=outer_opt,
        inner_loss_fn=inner_loss_fn,
        outer_loss_fn=outer_loss_fn,
    )


@pytest.mark.parametrize("inner_steps", [0, 1, 3])
@pytest.mark.parametrize("iterators", [1, 4], indirect=True)
def test_reptile_adapt(
    base_model,
    optimizers,
    loss_fns,
    inner_steps,
    iterators,
):
    reptile = create_reptile(
        base_model,
        optimizers,
        loss_fns,
    )
    train_iter, _ = iterators

    train_batch = next(train_iter)
    batch_size = train_batch["x"].shape[0]

    inner_params, inner_static = eqx.partition(reptile.base_model, reptile.inner_filter)

    adapted_params, losses, metrics = reptile.adapt(
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

    jt.map(check_element_shape, losses)
    jt.map(check_element_shape, metrics)


@pytest.mark.parametrize("inner_steps", [3])
@pytest.mark.parametrize("iterators", [4], indirect=True)
def test_reptile_meta_loss(
    base_model,
    optimizers,
    loss_fns,
    inner_steps,
    iterators,
):
    reptile = create_reptile(
        base_model,
        optimizers,
        loss_fns,
    )
    train_iter, _ = iterators

    train_batch = next(train_iter)
    batch_size = train_batch["x"].shape[0]

    outer_params, outer_static = eqx.partition(reptile.base_model, reptile.outer_filter)

    (meta_loss, metrics), meta_grad = reptile.meta_loss_value_and_grad(
        outer_params,
        outer_static,
        train_batch,
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
def test_reptile_update(
    base_model,
    optimizers,
    loss_fns,
    inner_steps,
    iterators,
):
    reptile = create_reptile(
        base_model,
        optimizers,
        loss_fns,
    )
    train_iter, val_iter = iterators

    train_batch = next(train_iter)
    val_batch = next(val_iter)
    batch_size = train_batch["x"].shape[0]

    outer_params, outer_static = eqx.partition(reptile.base_model, reptile.outer_filter)
    opt_state = reptile.outer_opt.init(outer_params)

    new_params, new_static, new_opt_state, loss, metrics = reptile.update(
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


def test_reptile_filter_warning(base_model, optimizers, loss_fns):
    """Test that Reptile warns when custom filters are provided."""
    inner_opt, outer_opt = optimizers
    inner_loss_fn, outer_loss_fn = loss_fns

    # Create a custom filter that's different from eqx.is_array
    custom_filter = lambda x: False

    # Test warning when custom inner_filter is provided
    with pytest.warns(
        UserWarning,
        match="Reptile uses eqx.is_array for both inner and outer filters",
    ):
        reptile = Reptile(
            base_model=base_model,
            inner_opt=inner_opt,
            outer_opt=outer_opt,
            inner_loss_fn=inner_loss_fn,
            outer_loss_fn=outer_loss_fn,
            inner_filter=custom_filter,
        )
        assert reptile.inner_filter == eqx.is_array
        assert reptile.outer_filter == eqx.is_array

    # Test warning when custom outer_filter is provided
    with pytest.warns(
        UserWarning,
        match="Reptile uses eqx.is_array for both inner and outer filters",
    ):
        reptile = Reptile(
            base_model=base_model,
            inner_opt=inner_opt,
            outer_opt=outer_opt,
            inner_loss_fn=inner_loss_fn,
            outer_loss_fn=outer_loss_fn,
            outer_filter=custom_filter,
        )
        assert reptile.inner_filter == eqx.is_array
        assert reptile.outer_filter == eqx.is_array

    # Test warning when both custom filters are provided
    with pytest.warns(
        UserWarning,
        match="Reptile uses eqx.is_array for both inner and outer filters",
    ):
        reptile = Reptile(
            base_model=base_model,
            inner_opt=inner_opt,
            outer_opt=outer_opt,
            inner_loss_fn=inner_loss_fn,
            outer_loss_fn=outer_loss_fn,
            inner_filter=custom_filter,
            outer_filter=custom_filter,
        )
        assert reptile.inner_filter == eqx.is_array
        assert reptile.outer_filter == eqx.is_array

    # Test no warning when default filters are used
    with warnings.catch_warnings():
        reptile = Reptile(
            base_model=base_model,
            inner_opt=inner_opt,
            outer_opt=outer_opt,
            inner_loss_fn=inner_loss_fn,
            outer_loss_fn=outer_loss_fn,
        )
        warnings.simplefilter("error")
        assert reptile.inner_filter == eqx.is_array
        assert reptile.outer_filter == eqx.is_array
