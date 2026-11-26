import math
from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


###############################################################################
# Custom initialization utils
###############################################################################
def siren_weight_init(
    weight: jax.Array,
    omega_0: float,
    is_first: bool,
    *,
    key: jr.PRNGKey,
):
    """
    Custom initialization for Siren weights following the original publication.

    Parameters
    ----------
    weight : jax.Array
        Weight matrix to be initialized, shape (f_out, f_in).
    omega_0 : float
        Frequency hyperparameter for the sine activations.
    is_first : bool
        Whether this is the first layer of the network.
    key : jr.PRNGKey
        Random key for initialization.

    Returns
    -------
    jax.Array
        Initialized weight matrix with same shape as input.

    """

    f_out, f_in = weight.shape

    if is_first:
        lim = 1.0 / f_in
    else:
        lim = math.sqrt(6.0 / f_in) / omega_0

    return jr.uniform(key, (f_out, f_in), weight.dtype, minval=-lim, maxval=lim)


def init_siren_weights(
    linear: eqx.nn.Linear, omega_0: float, is_first: bool, *, key: jr.PRNGKey
):
    """
    Initialize weights of a linear layer for use in a SIREN network.

    Parameters
    ----------
    linear : eqx.nn.Linear
        Linear layer whose weights will be initialized.
    omega_0 : float
        Frequency multiplier for the sine activation.
    is_first : bool
        Whether this is the first layer of the network.
    key : jr.PRNGKey
        Random key for initialization.

    Returns
    -------
    eqx.nn.Linear
        Linear layer with initialized weights.

    Notes
    -----
    This is a helper function that applies siren_weight_init to an equinox
    Linear layer, preserving the layer's structure and only updating its weights.
    """

    get_weights = lambda x: x.weight
    weights = get_weights(linear)
    new_weight = siren_weight_init(weights, omega_0, is_first, key=key)

    return eqx.tree_at(get_weights, linear, new_weight)


###############################################################################
# Standard sine layer
###############################################################################
class SineLayer(eqx.Module):
    """
    Linear layer followed by a sine activation with frequency modulation
    omega_0. Basic building block of SIREN networks [sitzmann2020]_.

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    bias : bool, optional
        Whether to include bias in linear layer, by default True.
    is_first : bool, optional
        Whether this is the first layer in network, affects
        initialization, by default False.
    omega_0 : float, optional
        Frequency multiplier for sine activation, by default 30.0.
    key : jr.PRNGKey
        Random key for initialization.

    References
    ----------
    .. [sitzmann2020] Sitzmann et al. (2020) "Implicit Neural Representations
           with Periodic Activation Functions"
    """

    omega_0: float
    is_first: bool
    in_features: int
    linear: eqx.nn.Linear

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0: float = 30.0,
        *,
        key,
    ):
        self.omega_0 = omega_0

        self.is_first = is_first
        self.in_features = in_features

        linear = eqx.nn.Linear(in_features, out_features, bias, key=key)
        self.linear = init_siren_weights(linear, omega_0, is_first, key=key)

    def __call__(self, x):
        return jnp.sin(self.omega_0 * self.linear(x))


###############################################################################
# Sine layer with FilM modulation
###############################################################################
class FiLM(eqx.Module):
    """
    Feature-wise Linear Modulation layer proposed in [perez2017]_. Performs
    affine transformation on input features.

    References
    ----------
    .. [perez2017] Perez et al. (2017) "FiLM: Visual Reasoning with a General
           Conditioning Layer" https://arxiv.org/abs/1709.07871
    """

    scale: jax.Array | float
    shift: jax.Array | float

    def __init__(self, scale, shift):
        """
        Initialize the FiLM layer.

        Parameters
        ----------
        scale : jax.Array | float
            Multiplicative scaling factor.
        shift : jax.Array | float
            Additive shift factor.
        """
        self.scale = scale
        self.shift = shift

    def __call__(self, x):
        return x * self.scale + self.shift


class ModulatedSineLayer(eqx.Module):
    """
    Sine layer with FiLM modulation for enhanced
    expressivity.

    References
    ----------
    [1] Dupont et al. (2022) "From data to functa: Your data point is a
    function and you can treat it like one",
    https://arxiv.org/abs/2201.12204
    """

    linear: eqx.nn.Linear
    film: FiLM
    omega_0: float
    is_first: bool
    modulate_scale: bool = True
    modulate_shift: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_first: bool,
        omega_0: float,
        modulate_scale: bool = True,
        modulate_shift: bool = True,
        *,
        key: jr.PRNGKey,
    ):
        """
        Initialize a ModulatedSineLayer.

        Parameters
        ----------
        in_features : int
            Dimension of input features.
        out_features : int
            Dimension of output features.
        is_first : bool
            Whether this is the first layer in the network.
        omega_0 : float
            Frequency multiplier for the sine activation.
        modulate_scale : bool, optional
            Whether to apply multiplicative FiLM modulation, by default True.
        modulate_shift : bool, optional
            Whether to apply additive FiLM modulation, by default True.
        key : jr.PRNGKey
            Random key for initialization.
        """
        self.omega_0 = omega_0
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.is_first = is_first

        linear = eqx.nn.Linear(in_features, out_features, key=key)
        self.linear = init_siren_weights(linear, omega_0, is_first, key=key)

        scale = jnp.ones((out_features,)) if modulate_scale else 1.0
        shift = jnp.zeros((out_features,)) if modulate_shift else 0.0

        self.film = FiLM(scale=scale, shift=shift)

    def __call__(self, x):
        x = self.linear(x)
        x = self.film(x)
        return jnp.sin(self.omega_0 * x)


###############################################################################
# Siren and friends
###############################################################################
class Siren(eqx.Module):
    """
    MLP with sine activations with custom weight initialization for learning
    high-quality implicit neural representations [1].

    References
    ----------
    [1] Sitzmann et al. (2020) "Implicit Neural Representations with
    Periodic Activation Functions"
    https://arxiv.org/abs/2006.09661

    """

    layers: List[eqx.Module]
    name: str = "Siren"

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool = False,
        omega_0: float = 30.0,
        *,
        key: jr.PRNGKey,
    ):
        """
        Initialize a Siren network.

        Parameters
        ----------
        in_features : int
            Dimension of input features.
        hidden_features : int
            Dimension of hidden features.
        hidden_layers : int
            Number of hidden layers.
        out_features : int
            Dimension of output features.
        outermost_linear : bool, optional
            Whether to use a linear layer as the output layer, by default False.
        omega_0 : float, optional
            Frequency multiplier for sine activation, by default 30.0.
        key : jr.PRNGKey
            Random key for initialization.
        """

        # First layer
        keys = jr.split(key, hidden_layers + 2)

        self.layers = []

        input_layer = SineLayer(
            in_features,
            hidden_features,
            is_first=True,
            omega_0=omega_0,
            key=keys[0],
        )

        self.layers.append(input_layer)

        # Hidden layers
        for i in range(hidden_layers):
            self.layers.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=omega_0,
                    key=keys[i + 1],
                )
            )

        if outermost_linear:
            final_layer = eqx.nn.Linear(hidden_features, out_features, key=keys[-1])
            # [NOTE] LInear layers require explicit initialization
            final_layer = init_siren_weights(
                final_layer, omega_0=omega_0, is_first=False, key=keys[-1]
            )
        else:
            final_layer = SineLayer(
                hidden_features,
                out_features,
                is_first=False,
                omega_0=omega_0,
                key=keys[-1],
            )

        self.layers.append(final_layer)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x)


###############################################################################
class ResidualSiren(eqx.Module):
    """
    Siren network with skip connections between for improved gradient
    flow and training stability [1].

    References
    ----------
    [1] Pan et al. (2023) "Neural Implicit Flow: a mesh-agnostic
    dimensionality reduction paradigm of spatio-temporal data"
    https://arxiv.org/abs/2204.03216
    """

    layers: List[eqx.Module]
    name: str = "ResidualSiren"
    num_resnet_blocks: int
    num_layers_per_block: int

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        outermost_linear: bool = False,
        num_resnet_blocks: int = 2,
        num_layers_per_block: int = 2,
        omega_0: float = 30.0,
        *,
        key: jr.PRNGKey,
    ):
        """
        Initialize a ResidualSiren network.

        Parameters
        ----------
        in_features : int
            Dimension of input features.
        hidden_features : int
            Dimension of hidden features.
        out_features : int
            Dimension of output features.
        outermost_linear : bool, optional
            Whether to use a linear layer as the output layer, by default False.
        num_resnet_blocks : int, optional
            Number of residual blocks, by default 2.
        num_layers_per_block : int, optional
            Number of sine layers per residual block, by default 2.
        omega_0 : float, optional
            Frequency multiplier for sine activation, by default 30.0.
        key : jr.PRNGKey
            Random key for initialization.
        """

        # Split the keys for initialisation
        num_layers = num_resnet_blocks * num_layers_per_block + 2
        keys = jr.split(key, num_layers)

        self.num_resnet_blocks = num_resnet_blocks
        self.num_layers_per_block = num_layers_per_block

        self.layers = []

        # First layer
        first_layer = SineLayer(
            in_features=in_features,
            out_features=hidden_features,
            is_first=True,
            omega_0=omega_0,
            key=keys[0],
        )

        self.layers.append(first_layer)

        for i in range(num_resnet_blocks):
            for j in range(num_layers_per_block):
                self.layers.append(
                    SineLayer(
                        in_features=hidden_features,
                        out_features=hidden_features,
                        is_first=False,
                        omega_0=omega_0,
                        key=keys[i * num_layers_per_block + j + 1],
                    )
                )

        if outermost_linear:
            output_layer = eqx.nn.Linear(
                in_features=hidden_features,
                out_features=out_features,
                key=keys[-1],
            )

            output_layer = init_siren_weights(
                output_layer, omega_0, False, key=keys[-1]
            )
        else:
            output_layer = SineLayer(
                in_features=hidden_features,
                out_features=out_features,
                is_first=False,
                omega_0=omega_0,
                key=keys[-1],
            )
        self.layers.append(output_layer)

    def __call__(self, x):
        x = self.layers[0](x)

        for i in range(self.num_resnet_blocks):
            residual = x
            start_idx = 1 + i * self.num_layers_per_block
            end_idx = start_idx + self.num_layers_per_block

            for layer in self.layers[start_idx:end_idx]:
                x = layer(x)

            x = 0.5 * (x + residual)

        return self.layers[-1](x)


###############################################################################
class ModulatedSiren(eqx.Module):
    """
    SIREN network with feature-wise linear modulation (FiLM) applied to
    each layer, allowing for increased expressivity. Implementation
    based on [1].

    References
    ----------
    [1] Dupont et al. (2022) "From data to functa: Your data point is a
           function and you can treat it like one"
    """

    layers: List[eqx.Module]
    name: str = "ModulatedSiren"

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool = False,
        omega_0: float = 30.0,
        modulate_scale: bool = True,
        modulate_shift: bool = True,
        *,
        key: jr.PRNGKey,
    ):
        """
        Initialize a ModulatedSiren network.

        Parameters
        ----------
        in_features : int
            Dimension of input features.
        hidden_features : int
            Dimension of hidden features.
        hidden_layers : int
            Number of hidden layers.
        out_features : int
            Dimension of output features.
        outermost_linear : bool, optional
            Whether to use a linear layer as the output layer, by default False.
        omega_0 : float, optional
            Frequency multiplier for sine activation, by default 30.0.
        modulate_scale : bool, optional
            Whether to apply multiplicative modulation, by default True.
        modulate_shift : bool, optional
            Whether to apply additive modulation, by default True.
        key : jr.PRNGKey
            Random key for initialization.
        """
        keys = jr.split(key, hidden_layers + 2)

        self.layers = []

        # First layer
        input_layer = ModulatedSineLayer(
            in_features,
            hidden_features,
            is_first=True,
            omega_0=omega_0,
            modulate_scale=modulate_scale,
            modulate_shift=modulate_shift,
            key=keys[0],
        )
        self.layers.append(input_layer)

        # Hidden layers
        for i in range(hidden_layers):
            self.layers.append(
                ModulatedSineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=omega_0,
                    modulate_scale=modulate_scale,
                    modulate_shift=modulate_shift,
                    key=keys[i + 1],
                )
            )

        # Final layer
        if outermost_linear:
            final_layer = eqx.nn.Linear(hidden_features, out_features, key=keys[-1])

            final_layer = init_siren_weights(final_layer, omega_0, False, key=keys[-1])
        else:
            final_layer = ModulatedSineLayer(
                hidden_features,
                out_features,
                is_first=False,
                omega_0=omega_0,
                modulate_scale=False,
                modulate_shift=False,
                key=keys[-1],
            )
        self.layers.append(final_layer)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


###############################################################################
class LatentModulatedSiren(eqx.Module):
    """
    SIREN network with FiLM modulation parameters generated from a
    latent vector, enabling the network to adapt its behavior based on a
    learned latent representation.

    References
    ----------
    [1] Dupont et al. (2022) "From data to functa: Your data point is a
        function and you can treat it like one"
        https://arxiv.org/abs/2201.12204
    """

    layers: List[eqx.Module]
    latent_vector: jax.Array
    film_generator: eqx.Module
    modulate_scale: bool
    modulate_shift: bool
    n_film_layers: int
    n_film_params_per_layer: int
    name: str = "LatentModulatedSiren"

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        latent_dim: int,
        outermost_linear: bool = False,
        modulate_scale: bool = True,
        modulate_shift: bool = True,
        omega_0: float = 30.0,
        *,
        key: jr.PRNGKey,
    ):
        """
        Initialize a LatentModulatedSiren network.

        Parameters
        ----------
        in_features : int
            Dimension of input features.
        hidden_features : int
            Dimension of hidden features.
        hidden_layers : int
            Number of hidden layers.
        out_features : int
            Dimension of output features.
        latent_dim : int
            Dimension of the latent vector used to generate FiLM parameters.
        outermost_linear : bool, optional
            Whether to use a linear layer as the output layer, by default False.
        modulate_scale : bool, optional
            Whether to apply multiplicative modulation, by default True.
        modulate_shift : bool, optional
            Whether to apply additive modulation, by default True.
        omega_0 : float, optional
            Frequency multiplier for sine activation, by default 30.0.
        key : jr.PRNGKey
            Random key for initialization.
        """
        keys = jr.split(key, 2)  # 1 for FiLM generator, 1 for siren layers

        # Initialize latent vector
        self.latent_vector = jnp.zeros((latent_dim,))

        # At least one of the two should be True
        if not (modulate_scale or modulate_shift):
            raise ValueError(
                "At least one of modulate_scale or modulate_shift must be True"
            )
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift

        # Calculate number of FiLM parameters
        n_film_layers = hidden_layers + 1  # +1 for the input layer
        n_film_params_per_layer = hidden_features * (
            int(modulate_scale) + int(modulate_shift)
        )

        self.n_film_layers = n_film_layers
        self.n_film_params_per_layer = n_film_params_per_layer

        self.film_generator = eqx.nn.Linear(
            in_features=latent_dim,
            out_features=n_film_layers * n_film_params_per_layer,
            key=keys[0],
        )

        self.layers = []
        layer_keys = jr.split(keys[1], hidden_layers + 2)

        # First layer
        input_layer = ModulatedSineLayer(
            in_features,
            hidden_features,
            is_first=True,
            omega_0=omega_0,
            modulate_scale=modulate_scale,
            modulate_shift=modulate_shift,
            key=layer_keys[0],
        )

        self.layers.append(input_layer)

        # Hidden layers
        for i in range(hidden_layers):
            self.layers.append(
                ModulatedSineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=omega_0,
                    modulate_scale=modulate_scale,
                    modulate_shift=modulate_shift,
                    key=layer_keys[i + 1],
                )
            )

        # Final layer
        if outermost_linear:
            final_layer = eqx.nn.Linear(
                hidden_features, out_features, key=layer_keys[-1]
            )
            final_layer = init_siren_weights(
                final_layer, omega_0, False, key=layer_keys[-1]
            )
        else:
            final_layer = ModulatedSineLayer(
                hidden_features,
                out_features,
                is_first=False,
                omega_0=omega_0,
                modulate_scale=False,
                modulate_shift=False,
                key=layer_keys[-1],
            )

        self.layers.append(final_layer)

    def compute_film_params(self, latent_vector):
        all_params = self.film_generator(latent_vector)
        all_params = all_params.reshape((self.n_film_layers, -1))

        def make_film_params(layer_params):
            if self.modulate_scale and self.modulate_shift:
                scale, shift = jnp.split(layer_params, 2)
                return FiLM(scale + 1.0, shift)
            elif self.modulate_scale:
                return FiLM(layer_params + 1.0, 0.0)
            elif self.modulate_shift:
                return FiLM(1.0, layer_params)

        return [make_film_params(p) for p in all_params]

    def __call__(self, x):
        # Compute shifts from latent vector
        film_params = self.compute_film_params(self.latent_vector)

        # Update shifts in ModulatedSineLayers
        model = eqx.tree_at(
            lambda m: [l.film for l in m.layers[:-1]], self, film_params
        )

        # Forward pass
        for layer in model.layers:
            x = layer(x)
        return x
