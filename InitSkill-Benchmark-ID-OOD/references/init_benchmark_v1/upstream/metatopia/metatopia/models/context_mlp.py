import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class ContextMLP(eqx.Module):
    """
    MLP with concatenation-based conditioning on learnable parameters.
    Used with CAVIA-like meta-learners.
    """

    mlp: eqx.Module
    context_params: jax.Array
    context_features: int

    @property
    def layers(self):
        return self.mlp.layers

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        context_features: int | None = 0,
        *,
        key: jr.PRNGKey,
    ):
        """
        Initialize the context-conditoned MLP.

        Parameters
        ----------
        in_features : int
            Number of input features to the network.
        hidden_features : int
            Number of neurons in each hidden layer.
        hidden_layers : int
            Number of hidden layers in the MLP.
        out_features : int
            Number of output features from the network.
        context_features : int, optional
            Number of learnable context parameters to concatenate with input.
            Defaults to 0, which means no context conditioning.
        key : jr.PRNGKey
            Jax random number generator key for initializing network weights.
        """

        self.context_features = context_features
        self.context_params = jnp.zeros(shape=(context_features,))

        self.mlp = eqx.nn.MLP(
            in_size=in_features + context_features,
            out_size=out_features,
            width_size=hidden_features,
            depth=hidden_layers,
            key=key,
        )

    def __call__(self, x):
        x = jnp.concatenate(
            (x, jnp.resize(self.context_params, (self.context_features,))),
            axis=-1,
        )

        return self.mlp(x)
