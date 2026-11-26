"""
Neural topology optimization models.
"""

from .cnn import HoyerCNN
from .context_mlp import ContextMLP
from .fourier_mlp import FourierFeatures, FourierMLP
from .siren import (
    FiLM,
    LatentModulatedSiren,
    ModulatedSineLayer,
    ModulatedSiren,
    ResidualSiren,
    SineLayer,
    Siren,
    siren_weight_init,
)


__all__ = [
    "ContextMLP",
    "FiLM",
    "Siren",
    "ResidualSiren",
    "ModulatedSiren",
    "LatentModulatedSiren",
    "ModulatedSineLayer",
    "SineLayer",
    "FourierFeatures",
    "FourierMLP",
    "siren_weight_init",
    "HoyerCNN",
]
