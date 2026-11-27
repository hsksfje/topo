"""Hoyer's CNN model.

Copied from CRAX.

Classes
-------
LearnableOffset
    Add an offset to the input.
HoyerCNN
    Convolutional neural network model.

Functions
---------
global_normalization
    Global normalization function.
get_up_sampling_2d
    Get the 2D upsampling function.
update_model
    Update the model parameters with a new set of weights.
variance_scaling_init
    Variance scaling initialization.
orthogonal_init
    Orthogonal initialization.
"""

# =============================================================================
# Standard
from functools import partial
from typing import Callable

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


#                                                          Authorship & Credits
# =============================================================================
__author__ = "Gawel"
__credits__ = ["Igor, Surya"]
__status__ = "Stable"


# =============================================================================
#
# =============================================================================
def global_normalization(inputs, epsilon=1e-6):
    """Global normalization function.

    Recenters and normalizes the input tensor using the mean and variance
    of its values.

    Parameters
    ----------
    inputs : jnp.ndarray
        Input tensor.
    epsilon : float, optional
        Small value to prevent division by zero, by default 1e-6

    Returns
    -------
    jnp.ndarray
        Normalized tensor.
    """
    mean = jnp.mean(inputs)
    variance = jnp.var(inputs)
    net = inputs - mean
    net = net * jax.lax.rsqrt(variance + epsilon)
    return net


def upsample(x, factor=2, method="bilinear"):
    C, H, W = x.shape
    new_shape = (C, H * factor, W * factor)
    return jax.image.resize(x, new_shape, method=method)


class LearnableOffset(eqx.Module):
    """Adds a trainable offset to the input.

    The trainable parameters have the same shape as the input tensor and
    is scaled and added to the input tensor.

    Attributes
    ----------
    bias : jnp.ndarray
        LearnableOffset tensor with trainable parameters.
    scale : float
        Scaling factor.
    """

    bias: jax.Array
    scale: float

    def __init__(self, input_shape, scale=1):
        self.scale = scale
        self.bias = jnp.zeros(input_shape)

    def __call__(self, x):
        return x + self.scale * self.bias


def initialize_layer_weights(layer: eqx.Module, init_fn: Callable, key: jr.PRNGKey):
    get_weights = lambda x: x.weight
    weights = get_weights(layer)
    new_weights = init_fn(key, weights.shape, weights.dtype)
    return eqx.tree_at(get_weights, layer, new_weights)


class HoyerCNN(eqx.Module):
    """Convolutional neural network model.

    This is a decoder-like model that takes a latent vector as input and
    produces an image-like output, as used by Hoyer et al. (2019).
    https:\\arxiv.org/abs/1909.04240

    Attributes
    ----------
    layers : list
        List of neural network layers.
    latent_vector : jnp.ndarray
        Latent vector.
    fc1 : eqx.nn.Linear
        Fully connected layer.
    dense_shape : tuple
        Shape of the dense layer.
    domain_shape : tuple
        Shape of the expected output.
    """

    layers: list
    latent_vector: jax.Array
    name: str

    def __init__(
        self,
        key,
        domain_shape,
        latent_size=128,
        dense_channels=32,
        resizes=(1, 2, 2, 2, 1),
        conv_filters=(128, 64, 32, 16, 1),  # Like U-Net
        offset_scale=10.0,
        kernel_size=5,
        latent_scale=1.0,
        activation=jax.nn.relu,
        normalization=global_normalization,
        output_sigmoid=False,
        upsampling_method="bilinear",
        name="CNN",
    ):
        """Constructs the neural network graph.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        domain_shape : tuple, optional
            Shape of the expected output.
            This is needed to determine the sizes of
            many of the layers in the network.
            Should be (Ny, Nx).
        latent_size : int, optional
            Size of the latent vector (starting), by default 128
        dense_channels : int, optional
            Number of channels in the dense layer, by default 32
        resizes : tuple, optional
            Resize factors for hidden layer, by default (1, 2, 2, 2, 1)
        conv_filters : tuple, optional
            Number of filters for each convolutional layer,
            by default (128, 64, 32, 16, 1)
            Same size as resizes.
        offset_scale : int, optional
            Scaling factor for the offset, by default 10
        kernel_size : int, optional
            Kernel size for the convolutional layers, by default 5x5
        latent_scale : float, optional
            Scaling factor for the latent vector, by default 1.0
        dense_init_scale : float, optional
            Scaling factor for the dense layer initialization, by default 1.0
        activation : Callable, optional
            Activation function, by default jax.nn.tanh
        normalization : Callable, optional
            Normalization function, by default global_normalization
        output_sigmoid : bool, optional
            Whether to apply a sigmoid activation to the output,
            by default False
        """
        self.name = name

        total_resize = int(np.prod(np.array(resizes)))
        h = domain_shape[0] // total_resize
        w = domain_shape[1] // total_resize

        filters = h * w * dense_channels

        latent_key, linear_key, conv_key = jr.split(key, 3)

        # linear_init_scale = 1.0 * np.sqrt(max(filters / latent_size, 1))

        # Build initializers functions for different layers
        latent_init_fn = jax.nn.initializers.normal(latent_scale)

        # Initializations from the original paper
        # linear_init_fn = jax.nn.initializers.orthogonal(linear_init_scale)
        # conv_init_fn = jax.nn.initializers.variance_scaling(
        #     scale=1.0, mode="fan_in", distribution="truncated_normal"
        # )

        # Initialize the latent vector
        self.latent_vector = latent_init_fn(latent_key, (latent_size,))

        layers = []

        linear_layer = eqx.nn.Linear(
            in_features=latent_size, out_features=filters, key=linear_key
        )

        # linear_layer = initialize_layer_weights(
        #     linear_layer, linear_init_fn, linear_key
        # )

        layers.append(linear_layer)

        # Reshape the output of the dense layer to (C, H, W) format
        def reshape_fn(x):
            return x.reshape(
                (
                    dense_channels,
                    h,
                    w,
                )
            )

        layers.append(reshape_fn)

        current_h = h
        current_w = w
        in_channels = dense_channels

        for resize, filters in zip(resizes, conv_filters):
            # Activation function
            layers.append(activation)

            # Upsampling
            upsample_fn = partial(upsample, factor=resize, method=upsampling_method)
            layers.append(upsample_fn)

            # Normalization
            if normalization is not None:
                layers.append(normalization)

            # Convolutional layer
            conv_key, _ = jr.split(conv_key, 2)
            conv = eqx.nn.Conv2d(
                in_channels, filters, kernel_size, key=conv_key, padding="SAME"
            )
            # conv = initialize_layer_weights(conv, conv_init_fn, conv_key)

            layers.append(conv)

            current_h = current_h * resize
            current_w = current_w * resize

            if offset_scale is not None:
                layers.append(
                    LearnableOffset((filters, current_h, current_w), scale=offset_scale)
                )

            in_channels = filters  # For the next iteration

        if output_sigmoid:
            layers.append(jax.nn.sigmoid)

        # Save the layers
        self.layers = layers

    def __call__(self, z=None):
        x = self.latent_vector

        for layer in self.layers:
            x = layer(x)
        return x.squeeze(axis=0)


###############################################################################


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    Ny, Nx = 64, 64
    domain_shape = (Ny, Nx)
    model = HoyerCNN(key, domain_shape=domain_shape)
    y = model()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")
    im = ax.imshow(y.squeeze())
    fig.colorbar(im, ax=ax)
    plt.show()
