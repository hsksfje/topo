import equinox as eqx
import pytest
from metatopia.learners import get_filters
from metatopia.models import (
    ContextMLP,
    LatentModulatedSiren,
    ModulatedSiren,
    Siren,
)


def test_siren_w_anil_filter(key, base_siren_params):
    """Test that we can partition different models with different filters"""

    # Test ANIL with Siren
    siren = Siren(key=key, **base_siren_params)
    inner_filter, outer_filter = get_filters("ANIL", siren)
    # Verify we can partition the model
    inner_params, inner_static = eqx.partition(siren, inner_filter)
    outer_params, outer_static = eqx.partition(siren, outer_filter)
    # Verify we can combine back
    model_from_outer = eqx.combine(inner_params, inner_static)
    model_from_inner = eqx.combine(outer_params, outer_static)
    assert eqx.tree_equal(model_from_outer, model_from_inner)


def test_modulated_siren_w_film_filter(key, base_siren_params):
    # Test FiLM with ModulatedSiren
    msiren = ModulatedSiren(
        key=key, modulate_scale=True, modulate_shift=True, **base_siren_params
    )
    inner_filter, outer_filter = get_filters("film", msiren)
    # Verify we can partition the model
    # Verify we can partition the model
    inner_params, inner_static = eqx.partition(msiren, inner_filter)
    outer_params, outer_static = eqx.partition(msiren, outer_filter)
    model_from_outer = eqx.combine(inner_params, inner_static)
    model_from_inner = eqx.combine(outer_params, outer_static)
    assert eqx.tree_equal(model_from_outer, model_from_inner)


def test_latent_modulated_siren_w_latent_filter(key, base_siren_params):
    # Test latent with LatentModulatedSiren
    latent_model = LatentModulatedSiren(
        key=key,
        latent_dim=16,
        modulate_scale=True,
        modulate_shift=True,
        **base_siren_params,
    )
    inner_filter, outer_filter = get_filters("latent", latent_model)
    # Verify we can partition the model
    inner_params, inner_static = eqx.partition(latent_model, inner_filter)
    outer_params, outer_static = eqx.partition(latent_model, outer_filter)
    # Verify we can combine back
    model_from_outer = eqx.combine(inner_params, inner_static)
    model_from_inner = eqx.combine(outer_params, outer_static)
    assert eqx.tree_equal(model_from_outer, model_from_inner)


def test_context_mlp_with_cavia_filter(key):
    context_mlp = ContextMLP(
        key=key,
        in_features=2,
        hidden_features=2,
        hidden_layers=1,
        out_features=1,
        context_features=16,
    )

    inner_filter, outer_filter = get_filters("CAVIA", context_mlp)
    # Verify we can partition the model
    inner_params, inner_static = eqx.partition(context_mlp, inner_filter)
    outer_params, outer_static = eqx.partition(context_mlp, outer_filter)
    # Verify we can combine back
    model_from_outer = eqx.combine(inner_params, inner_static)
    model_from_inner = eqx.combine(outer_params, outer_static)
    assert eqx.tree_equal(model_from_outer, model_from_inner)


def test_invalid_filter_name(key, base_siren_params):
    """Test error handling for invalid filter name"""
    model = Siren(key=key, **base_siren_params)
    with pytest.raises(ValueError):
        get_filters("invalid_filter", model)
