from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tensorscope.graph import Operator, Subgraph


ComputeCostCategory = Literal["mac", "elementwise", "zero", "unavailable"]

# Short form: paired inline with every rendered MAC/elementwise number, in
# every render surface (text, HTML, JSON). Long form: stated once, in a
# limitations/details context. Both live only here -- see
# render_compute_cost_caveat() -- so this wording cannot drift between
# render surfaces the way the budget-source label drifted twice before a
# shared function existed for it.
_COMPUTE_COST_CAVEAT_SHORT = "Compute cost (MACs) — not a latency or timing estimate."
_COMPUTE_COST_CAVEAT_LONG = (
    "MAC/FLOP counts describe arithmetic volume only. They do not predict "
    "wall-clock latency: real timing depends on target core, clock "
    "frequency, memory wait-states, and whether a reference or "
    "hardware-optimized kernel implementation is used — none of which this "
    "project can determine from the model file alone."
)


def render_compute_cost_caveat(*, long: bool = False) -> str:
    """The single source of truth for the compute-cost caveat wording.

    Every render surface (compact text, --details, HTML, the guidance
    recommendation, JSON) must call this rather than writing the wording
    inline -- the same single-source-of-truth pattern already used for
    render_budget_verdict()/render_budget_source_label(), adopted here from
    the start rather than after the wording drifted across surfaces twice.
    """

    return _COMPUTE_COST_CAVEAT_LONG if long else _COMPUTE_COST_CAVEAT_SHORT


# True MAC-bearing operators: formula derives from weight-tensor shape (not
# builtin_options, which this project does not parse for any operator).
_MAC_BEARING = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "TRANSPOSE_CONV"}

# Pure data movement: genuinely 0 MACs, reported as a real zero, not omitted.
_ZERO_COMPUTE_REASON = "data movement only, no arithmetic"
_ZERO_COMPUTE = {"RESHAPE", "STRIDED_SLICE", "PAD"}

# Real per-element work that is not a multiply-accumulate. Reported
# separately from the MAC total, never summed into it -- the two are not
# commensurate units of cost.
_ELEMENTWISE_REASON = "elementwise op, not a multiply-accumulate"
_ELEMENTWISE = {
    "ADD", "SUB", "MUL", "RELU", "RELU6", "LEAKY_RELU", "LOGISTIC",
    "SOFTMAX", "QUANTIZE", "DEQUANTIZE",
}

# Windowed reduction ops. Real per-element work, not a multiply-accumulate
# (max-pooling is comparisons; average-pooling is additions plus one
# divide) -- reported under the elementwise category, same as ADD/MUL,
# never summed into the MAC total. The filter (kernel) dimensions come from
# Pool2DOptions, read directly by the converter (graph/tflite_converter.py)
# rather than inferred from the input/output shape ratio -- that inference
# would silently assume stride == filter size, a real guess that breaks for
# overlapping pooling windows. If a file's builtin_options are somehow not
# actually Pool2DOptions for one of these two opcodes, the converter leaves
# the filter dims None and this falls back to honestly unavailable rather
# than guessing.
_WINDOWED_REASON = "windowed reduction across a {h}x{w} pooling window per output element"
_UNAVAILABLE_WINDOWED_REASON = (
    "windowed reduction -- kernel size not currently parsed "
    "(requires Pool2DOptions, not extracted by this project)"
)
_WINDOWED = {"MAX_POOL_2D", "AVERAGE_POOL_2D"}


@dataclass(frozen=True)
class OperatorComputeCost:
    """Compute-cost classification for one operator instance."""

    operator_id: int
    operator_name: str
    category: ComputeCostCategory
    mac_count: int | None
    elementwise_op_count: int | None
    note: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "operator_id": self.operator_id,
            "operator_name": self.operator_name,
            "category": self.category,
            "mac_count": self.mac_count,
            "elementwise_op_count": self.elementwise_op_count,
            "note": self.note,
        }


@dataclass(frozen=True)
class ComputeCostSummary:
    """Compute-cost summary for one subgraph."""

    scope: Literal["primary_subgraph"]
    total_mac_count: int
    total_elementwise_ops: int
    mac_bearing_operator_count: int
    unavailable_operator_count: int
    operators: tuple[OperatorComputeCost, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "compute_cost_schema_version": 1,
            "scope": self.scope,
            "total_mac_count": self.total_mac_count,
            "total_elementwise_ops": self.total_elementwise_ops,
            "mac_bearing_operator_count": self.mac_bearing_operator_count,
            "unavailable_operator_count": self.unavailable_operator_count,
            "operators": [item.to_dict() for item in self.operators],
            "caveat": render_compute_cost_caveat(),
        }


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _mac_count_conv2d(subgraph: Subgraph, operator: Operator) -> int:
    weight = subgraph.tensor(operator.inputs[1])
    output = subgraph.tensor(operator.outputs[0])
    c_out, k_h, k_w, c_in = weight.shape
    n, h_out, w_out, _ = output.shape
    return n * h_out * w_out * c_out * k_h * k_w * c_in


def _mac_count_depthwise_conv2d(subgraph: Subgraph, operator: Operator) -> int:
    weight = subgraph.tensor(operator.inputs[1])
    output = subgraph.tensor(operator.outputs[0])
    _, k_h, k_w, c_out = weight.shape
    n, h_out, w_out, _ = output.shape
    return n * h_out * w_out * c_out * k_h * k_w


def _mac_count_fully_connected(subgraph: Subgraph, operator: Operator) -> int:
    weight = subgraph.tensor(operator.inputs[1])
    output = subgraph.tensor(operator.outputs[0])
    _, in_features = weight.shape
    return _product(output.shape) * in_features


def _mac_count_transpose_conv(subgraph: Subgraph, operator: Operator) -> int:
    # Input order is NOT the same as CONV_2D: input[0] is an output-shape
    # tensor (ignored by TFLM, which doesn't support dynamic tensors),
    # input[1] is the filter, input[2] is the actual input activations.
    # Confirmed against the vendored kernel header
    # (kernels/transpose_conv.h), not assumed from general schema
    # knowledge -- see kTransposeConvFilterTensor/kTransposeConvInputTensor.
    weight = subgraph.tensor(operator.inputs[1])
    output = subgraph.tensor(operator.outputs[0])
    c_out, k_h, k_w, c_in = weight.shape
    n, h_out, w_out, _ = output.shape
    return n * h_out * w_out * c_out * k_h * k_w * c_in


_MAC_FORMULAS = {
    "CONV_2D": _mac_count_conv2d,
    "DEPTHWISE_CONV_2D": _mac_count_depthwise_conv2d,
    "FULLY_CONNECTED": _mac_count_fully_connected,
    "TRANSPOSE_CONV": _mac_count_transpose_conv,
}


def _classify_operator(subgraph: Subgraph, operator: Operator) -> OperatorComputeCost:
    name = operator.name
    if name in _MAC_BEARING:
        mac_count = _MAC_FORMULAS[name](subgraph, operator)
        return OperatorComputeCost(operator.id, name, "mac", mac_count, None, None)
    if name in _ZERO_COMPUTE:
        return OperatorComputeCost(operator.id, name, "zero", 0, None, _ZERO_COMPUTE_REASON)
    if name in _ELEMENTWISE:
        output = subgraph.tensor(operator.outputs[0])
        count = _product(output.shape)
        return OperatorComputeCost(operator.id, name, "elementwise", None, count, _ELEMENTWISE_REASON)
    if name in _WINDOWED:
        if operator.pool_filter_height is None or operator.pool_filter_width is None:
            return OperatorComputeCost(operator.id, name, "unavailable", None, None, _UNAVAILABLE_WINDOWED_REASON)
        output = subgraph.tensor(operator.outputs[0])
        count = _product(output.shape) * operator.pool_filter_height * operator.pool_filter_width
        reason = _WINDOWED_REASON.format(h=operator.pool_filter_height, w=operator.pool_filter_width)
        return OperatorComputeCost(operator.id, name, "elementwise", None, count, reason)
    return OperatorComputeCost(
        operator.id, name, "unavailable", None, None,
        f"compute cost is not modeled for {name}",
    )


def compute_subgraph_cost(subgraph: Subgraph) -> ComputeCostSummary:
    """Classify every operator in ``subgraph`` and total the results.

    Static, deterministic, and derived only from data this project already
    parses (tensor shapes, not builtin_options) -- the same category of
    "real, computed, not estimated" quantity as planned arena head. This is
    arithmetic volume, not a timing estimate; see
    render_compute_cost_caveat().
    """

    operators = tuple(
        _classify_operator(subgraph, operator) for operator in subgraph.operators
    )
    return ComputeCostSummary(
        scope="primary_subgraph",
        total_mac_count=sum(item.mac_count or 0 for item in operators if item.category == "mac"),
        total_elementwise_ops=sum(item.elementwise_op_count or 0 for item in operators if item.category == "elementwise"),
        mac_bearing_operator_count=sum(1 for item in operators if item.category == "mac"),
        unavailable_operator_count=sum(1 for item in operators if item.category == "unavailable"),
        operators=operators,
    )
