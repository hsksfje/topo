# tests/learners/conftest.py

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from metatopia.dataset.sine import sin_task_iterator


@pytest.fixture
def base_model():
    """Create a simple MLP model for testing."""
    return eqx.nn.MLP(in_size=1, out_size=1, width_size=40, depth=2, key=jr.key(0))


@pytest.fixture
def key():
    return jr.key(0)


@pytest.fixture
def base_siren_params():
    return {
        "in_features": 2,
        "hidden_features": 32,
        "hidden_layers": 3,
        "out_features": 1,
        "omega_0": 30.0,
    }


@pytest.fixture
def optimizers():
    """Create standard inner/outer optimizers for testing."""
    inner_opt = optax.sgd(0.01)
    outer_opt = optax.sgd(0.001)
    return inner_opt, outer_opt


@pytest.fixture
def loss_fns():
    """Create standard inner/outer loss functions for testing."""

    def mse_loss_fn(params, static, batch):
        model = eqx.combine(params, static)
        pred = jax.vmap(model)(batch["x"])
        loss = jnp.mean((batch["y"] - pred) ** 2)
        return loss, {"mse": loss}

    return mse_loss_fn, mse_loss_fn


@pytest.fixture
def iterators(batch_size=4):
    """Create train/val iterators for testing."""
    train_iter = sin_task_iterator(
        batch_size=batch_size, num_points=10, seed=42, split="train"
    )
    val_iter = sin_task_iterator(
        batch_size=batch_size, num_points=10, seed=42, split="val"
    )
    return train_iter, val_iter
