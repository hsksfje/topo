import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import matplotlib.pyplot as plt
import metatopia as mtp
import numpy as np
from jax.flatten_util import ravel_pytree
from matplotlib import cm


def get_random_direction(params, key):
    """Generate a random direction with the same structure as params."""
    return jt.map(lambda p: jax.random.normal(key, p.shape, p.dtype), params)


def filter_normalize(direction, ref_params, ord=2):
    def normalization_mapping(d, r):
        if d.ndim == 1:
            # For biases and other parameters, normalize the entire tensor
            d_norm = jnp.linalg.norm(d, ord)
            r_norm = jnp.linalg.norm(r, ord)

            return d * (r_norm / d_norm)

        elif d.ndim >= 2:
            # For weights and other parameters
            filter_normalizer = jax.vmap(
                lambda d_filter, r_filter: d_filter
                * (jnp.linalg.norm(r_filter, ord) / jnp.linalg.norm(d_filter, ord))
            )
            return filter_normalizer(d, r)
        else:
            return d

    return jt.map(normalization_mapping, direction, ref_params)


def make_orthogonal(direction, base_direction, ref_params):
    """Make direction orthogonal to base_direction."""
    flat_dir, unflatten = ravel_pytree(direction)
    flat_base_dir, _ = ravel_pytree(base_direction)

    # Compute dot product
    dot_product = jnp.sum(flat_dir * flat_base_dir)

    # Subtract the projection
    flat_orthogonal = flat_dir - dot_product * flat_base_dir
    orthogonal_dir = unflatten(flat_orthogonal)

    return filter_normalize(orthogonal_dir, ref_params)


def evaluate_loss_landscape(
    params, static, loss_fn, direction1, direction2, n_points=4
):
    """Evaluate loss landscape along two orthogonal directions."""
    mtp.logger.info(f"Creating a loss landscape with {n_points**2} points.")
    alphas = jnp.linspace(-1.0, 1.0, n_points)
    loss_grid = jnp.zeros((n_points, n_points))

    # This should be made more efficient! scan or at least jit
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(alphas):
            # Move along the two directions
            param_perturbation = jt.map(
                lambda p, d1, d2: p + alpha * d1 + beta * d2,
                params,
                direction1,
                direction2,
            )
            # Evaluate loss
            loss, _ = loss_fn(param_perturbation)
            loss_grid = loss_grid.at[i, j].set(loss)

    return loss_grid, alphas


def plot_loss_landscape(loss_grid, alphas, save_path, title="Loss Landscape"):
    """Plot the loss landscape as a 3D surface and a 2D contour plot."""
    X, Y = np.meshgrid(alphas, alphas)

    # 3D surface plot
    fig = plt.figure(figsize=(15, 12))

    # 3D Surface
    ax1 = fig.add_subplot(2, 1, 1, projection="3d")
    surf = ax1.plot_surface(
        X, Y, loss_grid, cmap=cm.coolwarm, linewidth=0, antialiased=True
    )
    ax1.set_xlabel("Direction 1")
    ax1.set_ylabel("Direction 2")
    ax1.set_zlabel("Loss")
    ax1.set_title(f"3D {title}")
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

    # 2D Contour
    ax2 = fig.add_subplot(2, 1, 2)
    contour = ax2.contourf(X, Y, loss_grid, 20, cmap=cm.coolwarm)
    ax2.set_xlabel("Direction 1")
    ax2.set_ylabel("Direction 2")
    ax2.set_title(f"2D Contour of {title}")
    fig.colorbar(contour, ax=ax2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    mtp.logger.info(f"Loss landscape visualization saved to {save_path}")


def visualize_loss_landscape(model, loss_fn, save_path, seed=42, n_points=30):
    """Create and save a loss landscape visualization."""
    # Split model into params and static parts
    params, static = eqx.partition(model, eqx.is_array)

    # Get two random orthogonal directions
    key = jr.key(seed)
    k1, k2, k3 = jr.split(key, 3)

    # Generate first direction
    direction1 = get_random_direction(params, k1)
    direction1 = filter_normalize(direction1, params)

    # Generate second direction and make it orthogonal to the first
    direction2 = get_random_direction(params, k2)
    direction2 = make_orthogonal(direction2, direction1, params)

    # Evaluate loss landscape
    loss_grid, alphas = evaluate_loss_landscape(
        params, static, loss_fn, direction1, direction2, n_points=n_points
    )

    # Plot and save
    plot_loss_landscape(loss_grid, alphas, save_path / "loss_landscape.png")

    # Also save the loss values for potential further analysis
    np.save(save_path / "loss_grid.npy", np.array(loss_grid))
    np.save(save_path / "alphas.npy", np.array(alphas))


# Function to add to your main function
def add_loss_landscape_visualization(model, loss_fn, save_path):
    """Wrapper to add loss landscape visualization to the main function."""
    visualize_loss_landscape(model, loss_fn, save_path)
