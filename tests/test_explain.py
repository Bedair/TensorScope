from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorscope.explain import explain_primary_subgraph_memory
from tensorscope.graph import (
    BufferAllocation,
    GraphMemoryPlan,
    GraphModel,
    Operator,
    Subgraph,
    SubgraphMemoryPlan,
    Tensor,
    TensorDataType,
    calculate_graph_lifetimes,
    calculate_graph_memory_plan,
    convert_tflite_model,
)
from tensorscope.text_report import render_ascii_packing, render_memory_explanation
from tensorscope.tflite.model_loader import load_tflite_model


CORPUS = Path(__file__).parent / "model_corpus" / "models"


def _tensor(
    tensor_id: int,
    name: str,
    shape: tuple[int, ...],
    *,
    constant: bool = False,
) -> Tensor:
    logical_size = 4
    for dimension in shape:
        logical_size *= dimension
    return Tensor(
        id=tensor_id,
        name=name,
        data_type=TensorDataType.FLOAT32,
        shape=shape,
        shape_signature=(),
        buffer_id=tensor_id if constant else 0,
        is_variable=False,
        has_constant_data=constant,
        constant_data_size=logical_size if constant else 0,
    )


def _fixture_graph() -> GraphModel:
    return GraphModel(
        source_path=Path("fixture.tflite"),
        schema_version=3,
        description="explanation fixture",
        subgraphs=(
            Subgraph(
                id=0,
                name="main",
                tensors=(
                    _tensor(0, "input", (1,)),
                    _tensor(1, "a_very_long_hidden_tensor_name", (5,)),
                    _tensor(2, "", (1,)),
                    _tensor(3, "output", (1,)),
                    _tensor(4, "weight", (1,), constant=True),
                ),
                operators=(
                    Operator(0, 0, "FIRST", 1, (0, 4), (1,)),
                    Operator(1, 0, "SECOND", 1, (1,), (2,)),
                    Operator(2, 0, "THIRD", 1, (2,), (3,)),
                ),
                inputs=(0,),
                outputs=(3,),
            ),
        ),
    )


def _fixture_plan() -> GraphMemoryPlan:
    def allocation(
        tensor_id: int,
        planner_index: int,
        logical: int,
        aligned: int,
        offset: int,
        first: int,
        last: int,
    ) -> BufferAllocation:
        return BufferAllocation(
            tensor_id=tensor_id,
            planner_index=planner_index,
            logical_size=logical,
            aligned_size=aligned,
            offset=offset,
            first_used=first,
            last_used=last,
        )

    return GraphMemoryPlan(
        subgraphs=(
            SubgraphMemoryPlan(
                subgraph_id=0,
                alignment=16,
                allocations=(
                    allocation(0, 0, 4, 16, 0, 0, 1),
                    allocation(1, 1, 20, 32, 16, 1, 2),
                    allocation(2, 2, 4, 16, 0, 2, 3),
                    allocation(3, 3, 4, 16, 16, 3, 3),
                ),
                maximum_memory_size=48,
            ),
        )
    )


def _explanation(limit: int = 10):
    graph = _fixture_graph()
    return explain_primary_subgraph_memory(
        graph,
        lifetimes=calculate_graph_lifetimes(graph),
        memory_plan=_fixture_plan(),
        largest_limit=limit,
    )


def test_per_tensor_explanation_and_summary() -> None:
    explanation = _explanation()
    first = explanation.allocations[0]

    assert first.tensor_id == 0
    assert first.data_type == "FLOAT32"
    assert first.shape == (1,)
    assert (first.logical_bytes, first.aligned_bytes) == (4, 16)
    assert first.alignment_overhead_bytes == 12
    assert (first.offset, first.end_offset) == (0, 16)
    assert (first.first_used_scope, first.last_used_scope) == (0, 1)
    assert first.lifetime_length == 2
    assert first.is_graph_input
    assert not first.is_graph_output

    summary = explanation.summary
    assert summary.runtime_tensor_count == 4
    assert summary.constant_tensor_count == 1
    assert summary.operator_count == 3
    assert summary.planned_arena_head_bytes == 48
    assert summary.arena_alignment_bytes == 16
    assert summary.logical_runtime_tensor_bytes == 32
    assert summary.aligned_runtime_tensor_bytes == 80
    assert summary.alignment_overhead_bytes == 48


def test_largest_tensor_order_and_limit_are_deterministic() -> None:
    explanation = _explanation(limit=3)
    assert [item.tensor_id for item in explanation.largest_tensors] == [1, 0, 2]


def test_peak_scope_ties_and_live_tensors() -> None:
    explanation = _explanation()

    assert explanation.peak.scope == 1
    assert explanation.peak.tied_scopes == (1, 2)
    assert explanation.peak.operator_id == 0
    assert explanation.peak.operator_name == "FIRST"
    assert explanation.peak.occupied_extent_bytes == 48
    assert explanation.peak.live_aligned_bytes == 48
    assert explanation.peak.live_tensor_ids == (0, 1)
    assert [item.tensor_id for item in explanation.live_tensors_at_peak] == [0, 1]


def test_reuse_requires_memory_overlap_and_disjoint_lifetimes() -> None:
    reuse_pairs = {
        (item.first_tensor_id, item.second_tensor_id)
        for item in _explanation().reuse
    }

    assert reuse_pairs == {(0, 2), (1, 3)}
    assert (0, 1) not in reuse_pairs  # Lifetimes overlap.
    assert (0, 3) not in reuse_pairs  # Memory intervals do not overlap.
    assert len(reuse_pairs) == len(_explanation().reuse)  # No symmetric duplicates.


def test_reuse_blockers_are_conservative_and_name_last_consumer() -> None:
    blockers = {item.tensor_id: item for item in _explanation().reuse_blockers}

    assert blockers[1].overlapping_tensor_ids == (0, 2)
    assert blockers[1].lifetime == (1, 2)
    assert blockers[1].aligned_bytes == 32
    assert blockers[1].last_consumer_operator_id == 1
    assert blockers[1].last_consumer_operator_name == "SECOND"


def test_empty_runtime_plan_is_supported() -> None:
    graph = GraphModel(
        source_path=Path("empty.tflite"),
        schema_version=3,
        description="empty",
        subgraphs=(Subgraph(0, "main", (), (), (), ()),),
    )
    explanation = explain_primary_subgraph_memory(graph)

    assert explanation.summary.runtime_tensor_count == 0
    assert explanation.summary.planned_arena_head_bytes == 0
    assert explanation.peak.scope == 0
    assert explanation.peak.live_tensor_ids == ()
    assert explanation.reuse == ()
    assert "(no runtime allocations)" in render_ascii_packing(explanation)


def test_serialization_and_ascii_are_deterministic_and_handle_names() -> None:
    explanation = _explanation()

    first_json = json.dumps(explanation.to_dict(), sort_keys=True)
    second_json = json.dumps(explanation.to_dict(), sort_keys=True)
    assert first_json == second_json

    first_ascii = render_ascii_packing(explanation, name_width=10)
    second_ascii = render_ascii_packing(explanation, name_width=10)
    assert first_ascii == second_ascii
    assert "a_very_..." in first_ascii
    assert "<unnamed>" in first_ascii
    assert "[0, 16): tensor[0] -> tensor[2]" in first_ascii


def test_detailed_renderer_is_deterministic_and_cautious() -> None:
    explanation = _explanation()
    report = render_memory_explanation(explanation, details=True)

    assert report == render_memory_explanation(explanation, details=True)
    assert "This report covers planned arena head only." in report
    assert "Packing table:" in report
    assert "Reuse blockers (conservative):" in report
    assert "cannot reuse the same memory interval" in report
    assert "caused" not in report


@pytest.mark.parametrize(
    "model_name",
    [
        "hello_world_float.tflite",
        "conv0.tflite",
        "micro_speech_quantized.tflite",
    ],
)
def test_corpus_models_have_explainable_plans(model_name: str) -> None:
    graph = convert_tflite_model(load_tflite_model(CORPUS / model_name))
    explanation = explain_primary_subgraph_memory(graph)

    assert explanation.summary.runtime_tensor_count == len(explanation.allocations)
    assert explanation.summary.planned_arena_head_bytes == (
        calculate_graph_memory_plan(graph).primary_subgraph.maximum_memory_size
    )
    assert explanation.peak.scope in explanation.peak.tied_scopes
    report = render_memory_explanation(explanation)
    assert "This report covers planned arena head only." in report
    if model_name == "conv0.tflite":
        assert "Planned arena head: 10,432 bytes" in report
        assert "scope 1 (operator 0: CONV_2D)" in report


def test_hello_world_float_known_packing_is_preserved() -> None:
    graph = convert_tflite_model(
        load_tflite_model(CORPUS / "hello_world_float.tflite")
    )
    explanation = explain_primary_subgraph_memory(graph)
    by_id = {item.tensor_id: item for item in explanation.allocations}

    assert explanation.summary.planned_arena_head_bytes == 128
    assert (by_id[0].offset, by_id[0].first_used_scope, by_id[0].last_used_scope) == (
        64,
        0,
        1,
    )
    assert (by_id[7].offset, by_id[7].first_used_scope, by_id[7].last_used_scope) == (
        0,
        1,
        2,
    )
    assert (by_id[8].offset, by_id[8].first_used_scope, by_id[8].last_used_scope) == (
        64,
        2,
        3,
    )
    assert (by_id[9].offset, by_id[9].first_used_scope, by_id[9].last_used_scope) == (
        0,
        3,
        3,
    )
