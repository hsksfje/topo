import equinox as eqx
import jax.numpy as jnp
import jax.random as jr


class FourierFeatures(eqx.Module):
    """
    Fourier feature mapping layer that projects inputs into a higher dimensional
    space using random Fourier features as introduced in Tancik et al. (2020),
    "Fourier Features Let Networks Learn High Frequency Functions in Low
    Dimensional Domains".
    """

    frequencies: jnp.ndarray
    scale: float = 1.0

    def __init__(
        self,
        in_features: int,
        fourier_features: int,
        scale: float = 1.0,
        *,
        key: jr.PRNGKey,
    ):
        """
        Initialize the Fourier feature mapping layer.

        Parameters
        ----------
        in_features : int
            Dimension of input features.
        fourier_features : int
            Number of Fourier feature dimensions (output will be 2x this).
        scale : float, optional
            Scaling factor for the random frequencies, by default 1.0.
        key : jr.PRNGKey
            Random key for initialization.
        """
        self.frequencies = scale * jr.normal(key, (fourier_features, in_features))
        self.scale = scale

    def __call__(self, x):
        projections = 2 * jnp.pi * (x @ self.frequencies.T)

        return jnp.concatenate([jnp.sin(projections), jnp.cos(projections)], axis=-1)


class FourierMLP(eqx.Module):
    """
    Neural network combining random Fourier features with a MLP. This architecture
    enables better learning of high-frequency functions compared to standard MLPs
    through a suitable feature mapping [tancik2020]_.

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    fourier_features : int
        Number of Fourier feature dimensions (pre-concatenation).
    hidden_features : int
        Width of hidden layers in the MLP.
    hidden_layers : int
        Number of hidden layers in the MLP.
    out_features : int
        Dimension of output features.
    fourier_scale : float, optional
        Scaling factor for the random Fourier features, by default 1.0.
    key : jr.PRNGKey
        Random key for initialization.

    References
    ----------
    .. [tancik2020] Tancik et al. (2020) "Fourier Features Let Networks Learn
           High Frequency Functions in Low Dimensional Domains"
    """

    fourier_layer: FourierFeatures
    mlp: eqx.nn.MLP
    name: str = "FourierMLP"

    def __init__(
        self,
        in_features: int,
        fourier_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        fourier_scale: float = 1.0,
        *,
        key: jr.PRNGKey,
    ):
        keys = jr.split(key, 2)

        self.fourier_layer = FourierFeatures(
            in_features=in_features,
            fourier_features=fourier_features,
            scale=fourier_scale,
            key=keys[0],
        )

        self.mlp = eqx.nn.MLP(
            in_size=2 * fourier_features,
            out_size=out_features,
            width_size=hidden_features,
            depth=hidden_layers,
            key=keys[1],
        )

    def __call__(self, x):
        x = self.fourier_layer(x)
        return self.mlp(x)
