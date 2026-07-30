from pathlib import Path

import pytest

from tensorscope.graph import (
    TensorDataType,
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
def test_converts_complete_model_corpus(
    model_name: str,
) -> None:
    loaded_model = load_tflite_model(
        CORPUS_ROOT / model_name
    )

    graph_model = convert_tflite_model(
        loaded_model
    )

    assert graph_model.source_path.name == model_name
    assert graph_model.schema_version == 3
    assert len(graph_model.subgraphs) >= 1
    assert graph_model.tensor_count >= 1
    assert graph_model.operator_count >= 1

    primary = graph_model.primary_subgraph

    assert primary.tensors
    assert primary.operators
    assert primary.inputs
    assert primary.outputs

    for tensor in primary.tensors:
        assert tensor.id >= 0
        assert isinstance(
            tensor.data_type,
            TensorDataType,
        )
        assert tensor.buffer_id >= 0
        assert tensor.constant_data_size >= 0

    for operator in primary.operators:
        assert operator.id >= 0
        assert operator.name
        assert operator.version >= 1


def test_hello_world_float_graph() -> None:
    loaded_model = load_tflite_model(
        CORPUS_ROOT / "hello_world_float.tflite"
    )

    graph_model = convert_tflite_model(
        loaded_model
    )

    subgraph = graph_model.primary_subgraph

    assert len(subgraph.inputs) == 1
    assert len(subgraph.outputs) == 1
    assert len(subgraph.operators) >= 1

    operator_names = {
        operator.name
        for operator in subgraph.operators
    }

    assert "FULLY_CONNECTED" in operator_names


def test_hello_world_int8_contains_quantized_tensors() -> None:
    loaded_model = load_tflite_model(
        CORPUS_ROOT / "hello_world_int8.tflite"
    )

    graph_model = convert_tflite_model(
        loaded_model
    )

    quantized_tensors = [
        tensor
        for tensor in graph_model.primary_subgraph.tensors
        if tensor.quantization.is_quantized
    ]

    assert quantized_tensors

    assert any(
        tensor.data_type == TensorDataType.INT8
        for tensor in quantized_tensors
    )


def test_model_contains_constant_buffers() -> None:
    loaded_model = load_tflite_model(
        CORPUS_ROOT / "hello_world_float.tflite"
    )

    graph_model = convert_tflite_model(
        loaded_model
    )

    constant_tensors = [
        tensor
        for tensor in graph_model.primary_subgraph.tensors
        if tensor.has_constant_data
    ]

    assert constant_tensors

    assert all(
        tensor.constant_data_size > 0
        for tensor in constant_tensors
    )