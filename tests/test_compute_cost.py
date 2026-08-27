from __future__ import annotations

from pathlib import Path

import pytest

from tensorscope.compute_cost import (
    compute_subgraph_cost,
    render_compute_cost_caveat,
)
from tensorscope.graph import convert_tflite_model
from tensorscope.tflite.model_loader import load_tflite_model


CORPUS = Path(__file__).parent / "model_corpus" / "models"
REPOSITORY_ROOT = Path(__file__).parents[1]
TRANSPOSE_CONV_FIXTURE = (
    REPOSITORY_ROOT / "third_party" / "tflite-micro" / "tensorflow" / "lite"
    / "micro" / "integration_tests" / "seanet" / "transpose_conv"
    / "transpose_conv4.tflite"
)


def _cost(model_name: str):
    graph = convert_tflite_model(load_tflite_model(CORPUS / model_name))
    return compute_subgraph_cost(graph.primary_subgraph)


def test_caveat_wording_is_stable_and_short_differs_from_long() -> None:
    short = render_compute_cost_caveat()
    long = render_compute_cost_caveat(long=True)
    assert short == "Compute cost (MACs) — not a latency or timing estimate."
    assert "does not predict" in long or "do not predict" in long
    assert "latency" in short and "latency" in long
    assert short != long


def test_conv2d_mac_formula_against_a_real_model() -> None:
    summary = _cost("conv0.tflite")
    assert summary.total_mac_count == 250_880
    assert len(summary.operators) == 1
    op = summary.operators[0]
    assert op.operator_name == "CONV_2D"
    assert op.category == "mac"
    assert op.mac_count == 250_880


def test_depthwise_conv2d_and_fully_connected_mac_formulas_against_a_real_model() -> None:
    summary = _cost("micro_speech_quantized.tflite")
    by_name = {op.operator_name: op for op in summary.operators}
    assert by_name["DEPTHWISE_CONV_2D"].mac_count == 320_000
    assert by_name["FULLY_CONNECTED"].mac_count == 16_000
    assert summary.total_mac_count == 336_000
    assert by_name["RESHAPE"].category == "zero"
    assert by_name["RESHAPE"].mac_count == 0
    assert by_name["SOFTMAX"].category == "elementwise"
    assert by_name["SOFTMAX"].elementwise_op_count == 4


def test_fully_connected_mac_formula_across_three_chained_layers() -> None:
    summary = _cost("hello_world_int8.tflite")
    macs = [op.mac_count for op in summary.operators]
    assert macs == [16, 256, 16]
    assert summary.total_mac_count == 288
    # Same architecture, different quantization -- MAC count is a function
    # of shape, not of tensor dtype, so this must match exactly.
    assert _cost("hello_world_float.tflite").total_mac_count == 288


def test_transpose_conv_mac_formula_against_the_real_vendored_fixture() -> None:
    # No TRANSPOSE_CONV fixture exists in this project's own corpus (it's
    # registered in the oracle but not arena-head validated -- a known,
    # unrelated scratch-buffer mismatch, see
    # docs/validated_operator_coverage.md). That mismatch is a memory-
    # planning concern; loading the model and reading tensor shapes for a
    # MAC count is unaffected by it, so the real vendored fixture is used
    # directly here instead of skipping verification.
    if not TRANSPOSE_CONV_FIXTURE.is_file():
        pytest.skip("pinned TFLM submodule fixture not available")
    graph = convert_tflite_model(load_tflite_model(TRANSPOSE_CONV_FIXTURE))
    summary = compute_subgraph_cost(graph.primary_subgraph)
    assert len(summary.operators) == 1
    op = summary.operators[0]
    assert op.operator_name == "TRANSPOSE_CONV"
    assert op.category == "mac"
    assert op.mac_count == 995_328


def test_elementwise_ops_report_output_element_count_not_mac() -> None:
    summary = _cost("simple_add_model.tflite")
    op = summary.operators[0]
    assert op.operator_name == "ADD"
    assert op.category == "elementwise"
    assert op.mac_count is None
    assert op.elementwise_op_count == 128 * 128
    assert summary.total_mac_count == 0
    assert summary.total_elementwise_ops == 128 * 128


@pytest.mark.parametrize("model_name", ["pad0.tflite", "strided_slice0.tflite"])
def test_pure_data_movement_ops_report_a_real_zero(model_name: str) -> None:
    summary = _cost(model_name)
    assert summary.total_mac_count == 0
    op = summary.operators[0]
    assert op.category == "zero"
    assert op.mac_count == 0
    assert op.note == "data movement only, no arithmetic"


def test_pooling_reports_a_real_windowed_op_count_from_pool2doptions() -> None:
    # operator_chain_float.tflite's real Pool2DOptions (confirmed directly
    # against the raw FlatBuffer, not assumed): both pooling ops use a 2x2
    # filter with 2x2 stride (non-overlapping windows).
    #   MAX_POOL_2D:     input (1,4,4,1) -> output (1,2,2,1) = 4 output elements
    #                     4 * (2*2) = 16
    #   AVERAGE_POOL_2D:  input (1,2,2,1) -> output (1,1,1,1) = 1 output element
    #                     1 * (2*2) = 4
    summary = _cost("operator_chain_float.tflite")
    by_name = {op.operator_name: op for op in summary.operators}

    max_pool = by_name["MAX_POOL_2D"]
    assert max_pool.category == "elementwise"
    assert max_pool.mac_count is None
    assert max_pool.elementwise_op_count == 16
    assert "2x2 pooling window" in max_pool.note

    avg_pool = by_name["AVERAGE_POOL_2D"]
    assert avg_pool.category == "elementwise"
    assert avg_pool.mac_count is None
    assert avg_pool.elementwise_op_count == 4
    assert "2x2 pooling window" in avg_pool.note

    assert summary.unavailable_operator_count == 0
    # Pooling is real per-element work, never a multiply-accumulate --
    # must not be counted into the MAC total.
    assert summary.total_mac_count == 33  # CONV_2D(16) + DEPTHWISE_CONV_2D(16) + FULLY_CONNECTED(1)
    assert summary.total_elementwise_ops >= 16 + 4


def test_pooling_falls_back_to_unavailable_when_filter_dims_are_not_parsed() -> None:
    # Defensive path: if a file's builtin_options somehow aren't actually
    # Pool2DOptions for a MAX_POOL_2D/AVERAGE_POOL_2D opcode, the converter
    # leaves pool_filter_height/width None -- must fall back to honestly
    # unavailable, never guess or crash.
    from tensorscope.compute_cost import _classify_operator
    from tensorscope.graph import Operator

    graph = convert_tflite_model(load_tflite_model(CORPUS / "operator_chain_float.tflite"))
    subgraph = graph.primary_subgraph
    real_op = next(op for op in subgraph.operators if op.name == "MAX_POOL_2D")
    stripped = Operator(
        id=real_op.id,
        opcode_index=real_op.opcode_index,
        name=real_op.name,
        version=real_op.version,
        inputs=real_op.inputs,
        outputs=real_op.outputs,
        intermediates=real_op.intermediates,
        builtin_code=real_op.builtin_code,
        custom_code=real_op.custom_code,
        pool_filter_height=None,
        pool_filter_width=None,
    )

    result = _classify_operator(subgraph, stripped)

    assert result.category == "unavailable"
    assert result.elementwise_op_count is None
    assert "kernel size not currently parsed" in result.note


def test_every_operator_instance_in_the_full_corpus_lands_in_exactly_one_category() -> None:
    # Same standard as operator coverage validation elsewhere in this
    # project: no silent gaps across the full corpus.
    valid_categories = {"mac", "elementwise", "zero", "unavailable"}
    total_instances = 0
    for path in sorted(CORPUS.glob("*.tflite")):
        graph = convert_tflite_model(load_tflite_model(path))
        summary = compute_subgraph_cost(graph.primary_subgraph)
        for op in summary.operators:
            total_instances += 1
            assert op.category in valid_categories, f"{path.name}: {op.operator_name} uncategorized"
            if op.category == "mac":
                assert op.mac_count is not None and op.mac_count >= 0
            if op.category == "elementwise":
                assert op.elementwise_op_count is not None and op.elementwise_op_count >= 0
            if op.category in ("zero", "unavailable"):
                assert op.note is not None
    assert total_instances > 0
