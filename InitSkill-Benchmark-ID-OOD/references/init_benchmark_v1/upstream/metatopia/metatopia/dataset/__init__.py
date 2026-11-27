"""
Utilities for working with multi-task datasets.
"""

from . import sine as sine, to as to
from .common import AbstractTaskIterator, get_next


__all__ = [
    "get_iterator_fn",
    "get_next",
    "AbstractTaskIterator",
]
