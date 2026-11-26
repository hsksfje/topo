"""
Miscellaneous utilities for the metatopia package.
"""

from .misc import (
    get_grid_coords,
    load_tree,
    process_strain_energy,
    serialise_array,
    serialise_tree,
)
from .plot_utils import (
    BIG_FIG,
    MEDIUM_FIG,
    plot_design,
    plot_losses,
    plot_meta_losses,
    plot_performance_profiles,
    plot_sinusoids,
    SMALL_FIG,
)
from .tree_utils import (
    tree_add_scalar_mul,
    tree_extend,
    tree_l2_norm,
    tree_mean,
    tree_stack,
    tree_sum,
)


__all__ = [
    "get_grid_coords",
    "plot_compliance_comparison",
    "plot_meta_losses",
    "plot_losses",
    "plot_design",
    "plot_performance_profiles",
    "tree_extend",
    "tree_add_scalar_mul",
    "tree_l2_norm",
    "tree_mean",
    "tree_stack",
    "tree_sum",
    "serialise_array",
    "serialise_tree",
    "load_tree",
    "plot_sinusoids",
    "process_strain_energy",
    "SMALL_FIG",
    "MEDIUM_FIG",
    "BIG_FIG",
]
