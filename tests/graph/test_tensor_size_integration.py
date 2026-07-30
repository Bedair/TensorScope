from __future__ import annotations

from pathlib import Path

import pytest

from tensorscope.graph import (
    TensorDataType,
    calculate_tensor_size,
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
    loaded_model = load_tflite_model(
        CORPUS_ROOT / model_name
    )

    return convert_tflite_model(
        loaded_model
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
def test_all_fixed_width_corpus_tensors_can_be_sized(
    model_name: str,
) -> None:
    graph = load_graph(model_name)

    for subgraph in graph.subgraphs:
        for tensor in subgraph.tensors:
            size = calculate_tensor_size(tensor)

            assert size.element_count >= 0
            assert size.bits_per_element > 0
            assert size.storage_bits >= 0
            assert size.storage_bytes >= 0


def test_hello_world_float_tensor_sizes() -> None:
    graph = load_graph(
        "hello_world_float.tflite"
    )

    tensors = graph.primary_subgraph.tensors

    input_tensor = tensors[0]
    first_bias = tensors[1]
    first_weights = tensors[4]
    first_activation = tensors[7]
    output_tensor = tensors[9]

    assert input_tensor.data_type == TensorDataType.FLOAT32
    assert input_tensor.shape == (1, 1)
    assert calculate_tensor_size(
        input_tensor
    ).storage_bytes == 4

    assert first_bias.shape == (16,)
    assert calculate_tensor_size(
        first_bias
    ).storage_bytes == 64

    assert first_weights.shape == (16, 1)
    assert calculate_tensor_size(
        first_weights
    ).storage_bytes == 64

    assert first_activation.shape == (1, 16)
    assert calculate_tensor_size(
        first_activation
    ).storage_bytes == 64

    assert output_tensor.shape == (1, 1)
    assert calculate_tensor_size(
        output_tensor
    ).storage_bytes == 4


def test_constant_buffer_size_matches_logical_tensor_size() -> None:
    graph = load_graph(
        "hello_world_float.tflite"
    )

    constant_tensors = [
        tensor
        for tensor in graph.primary_subgraph.tensors
        if tensor.has_constant_data
    ]

    assert constant_tensors

    for tensor in constant_tensors:
        logical_size = calculate_tensor_size(
            tensor
        ).storage_bytes

        assert tensor.constant_data_size == logical_size


def test_int8_model_contains_byte_sized_tensors() -> None:
    graph = load_graph(
        "hello_world_int8.tflite"
    )

    int8_tensors = [
        tensor
        for tensor in graph.primary_subgraph.tensors
        if tensor.data_type == TensorDataType.INT8
    ]

    assert int8_tensors

    for tensor in int8_tensors:
        size = calculate_tensor_size(tensor)

        assert size.bits_per_element == 8
        assert size.storage_bytes == size.element_count