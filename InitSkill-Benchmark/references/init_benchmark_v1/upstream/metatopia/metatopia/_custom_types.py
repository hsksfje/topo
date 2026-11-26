from typing import Any, Callable, Union

from jaxtyping import Array, Float, PyTree


# Add a type var for metrics - PyTree of arrays
InnerMetrics = PyTree[Float[Array, "batch step"]]
InnerLosses = Float[Array, "batch step"]
OuterMetrics = PyTree[Float[Array, "batch step"]]
OuterLosses = Float[Array, "batch step"]
AxisSpec = Union[bool, Callable[[Any], bool]]

TaskBatch = PyTree[Float[Array, "batch *"]]
