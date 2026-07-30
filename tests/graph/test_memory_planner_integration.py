from __future__ import annotations

from pathlib import Path

import pytest

from tensorscope.graph import (
    MemoryPlannerError,
    TFLM_ARENA_ALIGNMENT,
    calculate_graph_memory_plan,
    convert_tflite_model,
)
from tensorscope.tflite.model_loader import (
    load_tflite_model,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

CORPUS_ROOT = (
    REPOSITORY_ROOT
    / "tests"
    / "model_corpus"
    / "models"
)


def load_graph(model_name: str):
    loaded = load_tflite_model(
        CORPUS_ROOT / model_name
    )

    return convert_tflite_model(loaded)


@pytest.mark.parametrize(
    "model_name",
    [
        "hello_world_float.tflite",
        "hello_world_int8.tflite",
        "simple_add_model.tflite",
        "conv0.tflite",
        "micro_speech_quantized.tflite",
    ],
)
def test_complete_corpus_produces_valid_plan(
    model_name: str,
) -> None:
    graph = load_graph(model_name)

    plan = calculate_graph_memory_plan(
        graph
    )

    assert len(plan.subgraphs) == len(
        graph.subgraphs
    )

    for subgraph_plan in plan.subgraphs:
        assert (
            subgraph_plan.alignment
            == TFLM_ARENA_ALIGNMENT
        )

        assert (
            subgraph_plan.maximum_memory_size
            >= 0
        )

        subgraph_plan.validate_no_conflicts()

        for allocation in subgraph_plan.allocations:
            assert allocation.offset >= 0

            assert (
                allocation.aligned_size
                % TFLM_ARENA_ALIGNMENT
                == 0
            )


def test_hello_world_float_matches_expected_plan() -> None:
    graph = load_graph(
        "hello_world_float.tflite"
    )

    plan = calculate_graph_memory_plan(
        graph
    ).primary_subgraph

    assert plan.maximum_memory_size == 128

    tensor_0 = plan.allocation(0)
    tensor_7 = plan.allocation(7)
    tensor_8 = plan.allocation(8)
    tensor_9 = plan.allocation(9)

    assert (
        tensor_0.logical_size,
        tensor_0.aligned_size,
        tensor_0.offset,
    ) == (4, 16, 64)

    assert (
        tensor_7.logical_size,
        tensor_7.aligned_size,
        tensor_7.offset,
    ) == (64, 64, 0)

    assert (
        tensor_8.logical_size,
        tensor_8.aligned_size,
        tensor_8.offset,
    ) == (64, 64, 64)

    assert (
        tensor_9.logical_size,
        tensor_9.aligned_size,
        tensor_9.offset,
    ) == (4, 16, 0)


def test_hello_world_memory_reuse() -> None:
    graph = load_graph(
        "hello_world_float.tflite"
    )

    plan = calculate_graph_memory_plan(
        graph
    ).primary_subgraph

    tensor_0 = plan.allocation(0)
    tensor_7 = plan.allocation(7)
    tensor_8 = plan.allocation(8)
    tensor_9 = plan.allocation(9)

    assert tensor_0.offset == tensor_8.offset
    assert not tensor_0.overlaps_in_time(
        tensor_8
    )

    assert tensor_7.offset == tensor_9.offset
    assert not tensor_7.overlaps_in_time(
        tensor_9
    )


def test_hello_world_live_bytes_by_scope() -> None:
    graph = load_graph(
        "hello_world_float.tflite"
    )

    plan = calculate_graph_memory_plan(
        graph
    ).primary_subgraph

    assert plan.live_aligned_bytes_at(0) == 16
    assert plan.live_aligned_bytes_at(1) == 80
    assert plan.live_aligned_bytes_at(2) == 128
    assert plan.live_aligned_bytes_at(3) == 80


def test_constants_are_not_in_memory_plan() -> None:
    graph = load_graph(
        "hello_world_float.tflite"
    )

    plan = calculate_graph_memory_plan(
        graph
    ).primary_subgraph

    allocated_ids = {
        allocation.tensor_id
        for allocation in plan.allocations
    }

    assert allocated_ids == {0, 7, 8, 9}

    for constant_tensor_id in (
        1,
        2,
        3,
        4,
        5,
        6,
    ):
        with pytest.raises(
            MemoryPlannerError,
            match="No allocation",
        ):
            plan.allocation(
                constant_tensor_id
            )