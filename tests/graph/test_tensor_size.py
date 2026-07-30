from __future__ import annotations

import pytest

from tensorscope.graph import (
    Tensor,
    TensorDataType,
    TensorSizeError,
    bits_per_element,
    bits_to_bytes_rounded_up,
    calculate_element_count,
    calculate_tensor_size,
)


def make_tensor(
    data_type: TensorDataType,
    shape: tuple[int, ...],
) -> Tensor:
    return Tensor(
        id=0,
        name="test_tensor",
        data_type=data_type,
        shape=shape,
        shape_signature=(),
        buffer_id=0,
        is_variable=False,
        has_constant_data=False,
        constant_data_size=0,
    )


@pytest.mark.parametrize(
    ("data_type", "expected_bits"),
    [
        (TensorDataType.FLOAT32, 32),
        (TensorDataType.FLOAT16, 16),
        (TensorDataType.INT32, 32),
        (TensorDataType.UINT8, 8),
        (TensorDataType.INT64, 64),
        (TensorDataType.BOOL, 8),
        (TensorDataType.INT16, 16),
        (TensorDataType.COMPLEX64, 64),
        (TensorDataType.INT8, 8),
        (TensorDataType.FLOAT64, 64),
        (TensorDataType.COMPLEX128, 128),
        (TensorDataType.UINT64, 64),
        (TensorDataType.UINT32, 32),
        (TensorDataType.UINT16, 16),
        (TensorDataType.INT4, 4),
        (TensorDataType.BFLOAT16, 16),
    ],
)
def test_fixed_width_tensor_types(
    data_type: TensorDataType,
    expected_bits: int,
) -> None:
    assert bits_per_element(data_type) == expected_bits


@pytest.mark.parametrize(
    "data_type",
    [
        TensorDataType.STRING,
        TensorDataType.RESOURCE,
        TensorDataType.VARIANT,
    ],
)
def test_variable_width_types_are_rejected(
    data_type: TensorDataType,
) -> None:
    with pytest.raises(
        TensorSizeError,
        match="variable-width",
    ):
        bits_per_element(data_type)


@pytest.mark.parametrize(
    ("shape", "expected_count"),
    [
        ((), 1),
        ((1,), 1),
        ((4,), 4),
        ((2, 3), 6),
        ((1, 8, 8, 3), 192),
        ((0,), 0),
        ((2, 0, 4), 0),
    ],
)
def test_calculate_element_count(
    shape: tuple[int, ...],
    expected_count: int,
) -> None:
    assert (
        calculate_element_count(shape)
        == expected_count
    )


@pytest.mark.parametrize(
    "shape",
    [
        (-1,),
        (1, -1, 4),
        (-2, 3),
    ],
)
def test_negative_dimensions_are_rejected(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(
        TensorSizeError,
        match="negative dimensions",
    ):
        calculate_element_count(shape)


@pytest.mark.parametrize(
    ("bit_count", "expected_bytes"),
    [
        (0, 0),
        (1, 1),
        (4, 1),
        (8, 1),
        (9, 2),
        (12, 2),
        (16, 2),
        (17, 3),
    ],
)
def test_bits_to_bytes_rounds_up(
    bit_count: int,
    expected_bytes: int,
) -> None:
    assert (
        bits_to_bytes_rounded_up(bit_count)
        == expected_bytes
    )


def test_negative_bit_count_is_rejected() -> None:
    with pytest.raises(
        TensorSizeError,
        match="non-negative",
    ):
        bits_to_bytes_rounded_up(-1)


@pytest.mark.parametrize(
    (
        "data_type",
        "shape",
        "expected_elements",
        "expected_bits",
        "expected_bytes",
    ),
    [
        (
            TensorDataType.FLOAT32,
            (1, 16),
            16,
            512,
            64,
        ),
        (
            TensorDataType.INT8,
            (1, 16),
            16,
            128,
            16,
        ),
        (
            TensorDataType.INT16,
            (2, 3),
            6,
            96,
            12,
        ),
        (
            TensorDataType.FLOAT64,
            (),
            1,
            64,
            8,
        ),
        (
            TensorDataType.INT4,
            (3,),
            3,
            12,
            2,
        ),
        (
            TensorDataType.INT4,
            (4,),
            4,
            16,
            2,
        ),
        (
            TensorDataType.UINT8,
            (0, 4),
            0,
            0,
            0,
        ),
    ],
)
def test_calculate_tensor_size(
    data_type: TensorDataType,
    shape: tuple[int, ...],
    expected_elements: int,
    expected_bits: int,
    expected_bytes: int,
) -> None:
    size = calculate_tensor_size(
        make_tensor(
            data_type=data_type,
            shape=shape,
        )
    )

    assert size.element_count == expected_elements
    assert size.bits_per_element == bits_per_element(
        data_type
    )
    assert size.storage_bits == expected_bits
    assert size.storage_bytes == expected_bytes


def test_dynamic_tensor_cannot_be_sized() -> None:
    tensor = make_tensor(
        data_type=TensorDataType.FLOAT32,
        shape=(1, -1, 8),
    )

    with pytest.raises(
        TensorSizeError,
        match="negative dimensions",
    ):
        calculate_tensor_size(tensor)


def test_string_tensor_cannot_be_sized() -> None:
    tensor = make_tensor(
        data_type=TensorDataType.STRING,
        shape=(4,),
    )

    with pytest.raises(
        TensorSizeError,
        match="variable-width",
    ):
        calculate_tensor_size(tensor)