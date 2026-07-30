from __future__ import annotations

from pathlib import Path

import pytest

from tensorscope.graph import (
    calculate_graph_lifetimes,
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
def test_corpus_runtime_tensors_have_valid_lifetimes(
    model_name: str,
) -> None:
    graph = load_graph(model_name)
    analysis = calculate_graph_lifetimes(graph)

    assert len(analysis.subgraphs) == len(
        graph.subgraphs
    )

    for subgraph_analysis in analysis.subgraphs:
        for lifetime in (
            subgraph_analysis.plannable_lifetimes
        ):
            assert lifetime.is_initialized
            assert lifetime.first_created is not None
            assert lifetime.last_used is not None
            assert (
                lifetime.first_created
                <= lifetime.last_used
            )


def test_hello_world_float_runtime_lifetimes() -> None:
    graph = load_graph(
        "hello_world_float.tflite"
    )

    analysis = calculate_graph_lifetimes(
        graph
    ).primary_subgraph

    assert analysis.operator_scope_count == 3

    input_tensor = analysis.tensor(0)
    first_activation = analysis.tensor(7)
    second_activation = analysis.tensor(8)
    output_tensor = analysis.tensor(9)

    assert (
        input_tensor.first_created,
        input_tensor.last_used,
    ) == (0, 1)

    assert (
        first_activation.first_created,
        first_activation.last_used,
    ) == (1, 2)

    assert (
        second_activation.first_created,
        second_activation.last_used,
    ) == (2, 3)

    assert (
        output_tensor.first_created,
        output_tensor.last_used,
    ) == (3, 3)


def test_hello_world_float_plannable_tensor_ids() -> None:
    graph = load_graph(
        "hello_world_float.tflite"
    )

    analysis = calculate_graph_lifetimes(
        graph
    ).primary_subgraph

    assert tuple(
        lifetime.tensor_id
        for lifetime in analysis.plannable_lifetimes
    ) == (0, 7, 8, 9)


def test_hello_world_constants_are_not_plannable() -> None:
    graph = load_graph(
        "hello_world_float.tflite"
    )

    analysis = calculate_graph_lifetimes(
        graph
    ).primary_subgraph

    for tensor_id in (1, 2, 3, 4, 5, 6):
        lifetime = analysis.tensor(tensor_id)

        assert not lifetime.needs_allocation


def test_adjacent_hello_world_activations_overlap() -> None:
    graph = load_graph(
        "hello_world_float.tflite"
    )

    analysis = calculate_graph_lifetimes(
        graph
    ).primary_subgraph

    input_tensor = analysis.tensor(0)
    first_activation = analysis.tensor(7)
    second_activation = analysis.tensor(8)
    output_tensor = analysis.tensor(9)

    assert input_tensor.overlaps(first_activation)
    assert first_activation.overlaps(second_activation)
    assert second_activation.overlaps(output_tensor)

    assert not input_tensor.overlaps(
        second_activation
    )

    assert not first_activation.overlaps(
        output_tensor
    )