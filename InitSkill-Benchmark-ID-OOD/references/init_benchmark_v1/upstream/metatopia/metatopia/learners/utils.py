from typing import Literal, Tuple

import equinox as eqx
import jax.tree as jt
from jaxtyping import PyTree

from metatopia.learners import AbstractMetaLearner, MAML, Reptile
from metatopia.models import FiLM


def get_meta_learner_by_name(learner_name) -> AbstractMetaLearner:
    learners = {
        "MAML": MAML,
        "Reptile": Reptile,
    }
    try:
        return learners[learner_name]
    except KeyError:
        raise ValueError(
            f"Unknown learner: {learner_name}. "
            f"Available names are: {learners.keys()}"
        )


def get_filters(
    name: Literal["default", "ANIL", "CAVIA", "finetune", "latent", "film"],
    model: eqx.Module,
) -> Tuple[PyTree, PyTree]:
    """
    Creates outer and inner filters for common meta-learning
    model architectures.

    Parameters
    ----------
    name : Literal["default", "ANIL", "CAVIA", "finetune", "latent", "film"]
        The type of filter to create:
        - "default": Adapts and meta-learns all parameters
        - "ANIL": Adapts only the last layer, meta-learns all other parameters
        - "CAVIA": Adapts only context parameters, meta-learns base network
        - "finetune": Adapts all parameters, no meta-learning
        - "latent": Adapts only latent vectors, meta-learns base network
        - "film": Adapts FiLM layers, meta-learns all other parameters
    model : eqx.Module
        The model to create filters for. Must have appropriate structure for
        the chosen filter type.

    Returns
    -------
    Tuple[PyTree, PyTree]
        - inner_filter: PyTree matching model structure, True for parameters
          to be adapted during inner loop
        - outer_filter: PyTree matching model structure, True for parameters
          to be meta-learned during outer loop

    Notes
    -----
    Each filter type has specific requirements:
    - ANIL requires model.layers[-1] to exist (final layer)
    - CAVIA requires model.context_params to exist and not be None
    - Latent requires model.latent_vector to exist
    - FiLM requires FiLM layers in the model structure

    The implementation assumes certain model architectures. For custom
    architectures, you may need to implement custom filtering logic.

    References
    ----------
    - ANIL: Rapid Learning or Feature Reuse? Towards Understanding the
      Effectiveness of MAML (Raghu et al., 2020)
    - CAVIA: Fast Context Adaptation via Meta-Learning (Zintgraf et al., 2019)
    - FiLM: FiLM: Visual Reasoning with a General Conditioning Layer
      (Perez et al., 2018)
    """

    base_filter = jt.map(lambda _: False, model)
    learnable_filter = jt.map(eqx.is_array, model)

    def _default_filters(model):
        return eqx.is_array, eqx.is_array

    def _ANIL_filters(model):
        inner_filter = eqx.tree_at(lambda tree: tree.layers[-1], base_filter, True)
        outer_filter = eqx.tree_at(
            lambda tree: tree.layers[-1], learnable_filter, False
        )

        return inner_filter, outer_filter

    def _CAVIA_filters(model):
        if model.context_params is None:
            raise ValueError(
                "CAVIA requires context_params attribute in the model "
                "that is not None."
            )
        inner_filter = jt.map(eqx.is_array, model.context_params)
        inner_filter = eqx.tree_at(
            lambda tree: tree.context_params, base_filter, inner_filter
        )

        outer_filter = eqx.tree_at(
            lambda tree: tree.context_params, learnable_filter, False
        )

        return inner_filter, outer_filter

    def _latent_filter(model):
        inner_filter = jt.map(eqx.is_array, model.latent_vector)
        inner_filter = eqx.tree_at(
            lambda tree: tree.latent_vector, base_filter, inner_filter
        )

        outer_filter = eqx.tree_at(
            lambda tree: tree.latent_vector, learnable_filter, False
        )

        return inner_filter, outer_filter

    def _film_filter(model):
        def is_film(x):
            return isinstance(x, FiLM)

        # True for all film layers
        inner_filter = jt.map(is_film, model, is_leaf=is_film)
        outer_filter = jt.map(
            lambda x: eqx.is_array(x) and not is_film(x),
            model,
            is_leaf=lambda x: is_film(x) or eqx.is_array(x),
        )

        return inner_filter, outer_filter

    filters = {
        "default": _default_filters,
        "ANIL": _ANIL_filters,
        "CAVIA": _CAVIA_filters,
        "film": _film_filter,
        "latent": _latent_filter,
    }

    if name not in filters.keys():
        raise ValueError(
            f"Unknown filter set: {name}. "
            f"Available names are: {list(filters.keys())}"
        )
    return filters[name](model)
