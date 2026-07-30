"""TensorScope internal graph representation."""

from tensorscope.graph.model import (
    BufferId,
    GraphModel,
    GraphModelError,
    Operator,
    OperatorId,
    QuantizationParameters,
    Subgraph,
    SubgraphId,
    Tensor,
    TensorDataType,
    TensorId,
)
from tensorscope.graph.tensor_lifetime import (
    GraphLifetimeAnalysis,
    SubgraphLifetimeAnalysis,
    TensorLifetime,
    TensorLifetimeError,
    calculate_graph_lifetimes,
    calculate_subgraph_lifetimes,
)
from tensorscope.graph.tensor_size import (
    TensorSize,
    TensorSizeError,
    bits_per_element,
    bits_to_bytes_rounded_up,
    calculate_element_count,
    calculate_tensor_size,
)
from tensorscope.graph.tflite_converter import (
    convert_tflite_model,
)

__all__ = [
    "BufferId",
    "GraphLifetimeAnalysis",
    "GraphModel",
    "GraphModelError",
    "Operator",
    "OperatorId",
    "QuantizationParameters",
    "Subgraph",
    "SubgraphId",
    "SubgraphLifetimeAnalysis",
    "Tensor",
    "TensorDataType",
    "TensorId",
    "TensorLifetime",
    "TensorLifetimeError",
    "TensorSize",
    "TensorSizeError",
    "bits_per_element",
    "bits_to_bytes_rounded_up",
    "calculate_element_count",
    "calculate_graph_lifetimes",
    "calculate_subgraph_lifetimes",
    "calculate_tensor_size",
    "convert_tflite_model",
]