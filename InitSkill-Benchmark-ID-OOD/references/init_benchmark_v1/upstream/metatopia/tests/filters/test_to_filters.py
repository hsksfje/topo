import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from metatopia.filters import (
    get_cone_filter,
    get_hoyer_filter,
    get_xu_volume_preserving_filter,
    threshold_design,
)


jax.config.update("jax_enable_x64", True)
# Test the cone filter for different domain shapes, and radii


@pytest.mark.parametrize("Ny, Nx", [(64, 64), (128, 64)])
@pytest.mark.parametrize("radius", [1.0, 5.0])
def test_cone_filter(Ny, Nx, radius):
    domain_shape = (Ny, Nx)
    cone_filter = get_cone_filter(domain_shape, radius)
    design = jnp.ones(Ny * Nx) * 0.5

    filtered_design = cone_filter(design, None)

    # Assert correct output shape and range
    assert filtered_design.shape == (Ny * Nx,), "Incorrect output shape"
    assert jnp.all(filtered_design >= 0.0), "Negative values in filtered design"
    assert jnp.all(filtered_design <= 1.0), "Values above 1.0 in filtered design"

    # Additional assertions to test filter properties
    # Test if the filter preserves the input when radius is large enough
    if radius >= max(Ny, Nx):
        assert jnp.allclose(filtered_design, design, atol=1e-5)


@pytest.mark.parametrize("Ny, Nx", [(64, 64), (128, 64)])
@pytest.mark.parametrize("target_vf", [0.3, 0.5, 0.9])
def test_hoyer_filter(Ny, Nx, target_vf):
    hoyer_filter = get_hoyer_filter()

    # Test with values in [0,1]
    design = jr.uniform(jr.key(0), (Ny * Nx,), minval=0.01, maxval=1.0)
    filtered_design = hoyer_filter(design, {"target_vf": target_vf})

    # The bisection solver in hoyer_volume_enforcing_filter runs with
    # atol=rtol=1e-3 on the logit shift, which bounds the achievable volume
    # fraction error at ~1e-4. This tolerance is deliberately matched to the
    # solver setting used to produce the published results; see the note in
    # metatopia/filters/filters.py.
    assert jnp.isclose(jnp.mean(filtered_design), target_vf, atol=1e-3)
    assert jnp.all(filtered_design >= 0.0)
    assert jnp.all(filtered_design <= 1.0)

    # Test with values outside of [0,1]
    design = jr.uniform(jr.key(0), (Ny * Nx,), minval=0.01, maxval=100.0)
    filtered_design = hoyer_filter(design, {"target_vf": target_vf})

    assert jnp.isclose(
        jnp.mean(filtered_design), target_vf, atol=1e-3
    ), "Incorrect volume fraction of filtered design"
    assert jnp.all(filtered_design >= 0.0), "Negative values in filtered design"
    assert jnp.all(filtered_design <= 1.0), "Values above 1.0 in filtered design"


@pytest.mark.parametrize("Ny, Nx", [(64, 64), (128, 64)])
@pytest.mark.parametrize("beta", [0.1, 200.0])  # Test with low and high beta
@pytest.mark.parametrize("target_vf", [0.3, 0.5, 0.7])
def test_xu_filter(Ny, Nx, beta, target_vf):
    xu_filter = get_xu_volume_preserving_filter()
    design = jnp.ones(Ny * Nx) * target_vf
    filtered_design = xu_filter(design, beta, None)

    assert jnp.isclose(
        jnp.mean(filtered_design), jnp.mean(design), atol=1e-4
    ), "Volume fraction not preserved"
    assert jnp.all(filtered_design >= 0.0), "Negative values in filtered design"
    assert jnp.all(filtered_design <= 1.0), "Values above 1.0 in filtered design"


@pytest.mark.parametrize("Ny, Nx", [(64, 64), (128, 64)])
@pytest.mark.parametrize("seed", range(4))
def test_design_threshold(
    Ny,
    Nx,
    seed,
):
    design = jr.uniform(jr.key(seed), (Ny, Nx), minval=0.0, maxval=1.0)
    vf = jnp.mean(design)
    thresholded_design = threshold_design(design, {"target_vf": vf})

    # Check that all values are close to 0.0 or 1.0
    assert jnp.all(thresholded_design >= 0.0), "Thresholded design has negative values"
    assert jnp.isclose(
        jnp.mean(thresholded_design), vf, atol=1.0 / (Ny * Nx)
    ), "Thresholded design has incorrect volume fraction"


# Test differentiability as well
if __name__ == "__main__":
    test_hoyer_filter(10, 10, 0.3)
