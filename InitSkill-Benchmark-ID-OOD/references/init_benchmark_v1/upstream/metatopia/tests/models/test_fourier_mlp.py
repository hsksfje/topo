import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from metatopia.models import FourierFeatures, FourierMLP


def test_fourier_features_initialization():
    key = jr.key(0)
    in_features = 2
    fourier_features = 256
    scale = 10.0

    features = FourierFeatures(
        in_features=in_features,
        fourier_features=fourier_features,
        scale=scale,
        key=key,
    )

    # Test frequency matrix dimensions
    assert features.frequencies.shape == (
        fourier_features,
        in_features,
    ), "Incorrect frequency matrix shape"

    # Test scale parameter
    assert features.scale == scale, "Incorrect scale parameter"

    # Test output shape
    batch_size = 10
    x = jr.uniform(key, (batch_size, in_features))
    output = features(x)
    assert output.shape == (
        batch_size,
        2 * fourier_features,
    ), "Incorrect output shape"


def test_fourier_features_output_range():
    key = jr.key(0)
    in_features = 2
    fourier_features = 256
    scale = 10.0

    features = FourierFeatures(
        in_features=in_features,
        fourier_features=fourier_features,
        scale=scale,
        key=key,
    )

    # Test that outputs are in [-1, 1] (sine/cosine range)
    batch_size = 100
    x = jr.uniform(key, (batch_size, in_features))
    output = features(x)
    assert jnp.all(output >= -1.0)
    assert jnp.all(output <= 1.0)


def test_fourier_mlp_initialization():
    key = jr.key(0)
    in_features = 2
    fourier_features = 256
    hidden_features = 32
    hidden_layers = 3
    out_features = 1
    fourier_scale = 10.0

    model = FourierMLP(
        in_features=in_features,
        fourier_features=fourier_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        fourier_scale=fourier_scale,
        key=key,
    )

    # Test Fourier feature layer
    assert model.fourier_layer.frequencies.shape == (
        fourier_features,
        in_features,
    ), "Incorrect Fourier feature frequencies shape"

    # Test MLP structure
    assert isinstance(model.mlp, eqx.nn.MLP), "MLP not initialized correctly"
    assert (
        model.mlp.layers[0].in_features == 2 * fourier_features
    ), "Incorrect input size"
    assert model.mlp.layers[-1].out_features == out_features, "Incorrect output size"

    # Test name attribute
    assert model.name == "FourierMLP"


def test_fourier_mlp_forward():
    key = jr.key(0)
    in_features = 2
    fourier_features = 256
    hidden_features = 32
    hidden_layers = 3
    out_features = 1
    fourier_scale = 10.0
    batch_size = 10

    model = FourierMLP(
        in_features=in_features,
        fourier_features=fourier_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        fourier_scale=fourier_scale,
        key=key,
    )

    # Test forward pass dimensions
    x = jr.uniform(key, (batch_size, in_features))
    output = vmap(model)(x)
    assert output.shape == (batch_size, out_features), "Incorrect output shape"


def test_fourier_mlp_gradient():
    key = jr.key(0)
    in_features = 2
    fourier_features = 256
    hidden_features = 32
    hidden_layers = 3
    out_features = 1
    batch_size = 10

    model = FourierMLP(
        in_features=in_features,
        fourier_features=fourier_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        key=key,
    )

    # Test input
    x = jr.uniform(key, (batch_size, in_features))

    from jax.test_util import check_grads

    # Check gradients up to 2nd order
    check_grads(vmap(model), (x,), order=1, modes=["fwd", "rev"])
