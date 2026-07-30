from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul

from tensorscope.graph.model import (
    GraphModelError,
    Tensor,
    TensorDataType,
)


class TensorSizeError(GraphModelError):
    """Raised when a tensor's static storage size cannot be calculated."""


@dataclass(frozen=True)
class TensorSize:
    """Logical storage-size information for one tensor."""

    element_count: int
    bits_per_element: int
    storage_bits: int
    storage_bytes: int

    def __post_init__(self) -> None:
        if self.element_count < 0:
            raise TensorSizeError(
                "Element count must be non-negative"
            )

        if self.bits_per_element <= 0:
            raise TensorSizeError(
                "Bits per element must be positive"
            )

        if self.storage_bits < 0:
            raise TensorSizeError(
                "Storage size in bits must be non-negative"
            )

        if self.storage_bytes < 0:
            raise TensorSizeError(
                "Storage size in bytes must be non-negative"
            )


_FIXED_WIDTH_BITS: dict[TensorDataType, int] = {
    TensorDataType.FLOAT32: 32,
    TensorDataType.FLOAT16: 16,
    TensorDataType.INT32: 32,
    TensorDataType.UINT8: 8,
    TensorDataType.INT64: 64,
    TensorDataType.BOOL: 8,
    TensorDataType.INT16: 16,
    TensorDataType.COMPLEX64: 64,
    TensorDataType.INT8: 8,
    TensorDataType.FLOAT64: 64,
    TensorDataType.COMPLEX128: 128,
    TensorDataType.UINT64: 64,
    TensorDataType.UINT32: 32,
    TensorDataType.UINT16: 16,
    TensorDataType.INT4: 4,
    TensorDataType.BFLOAT16: 16,
}


_VARIABLE_WIDTH_TYPES = {
    TensorDataType.STRING,
    TensorDataType.RESOURCE,
    TensorDataType.VARIANT,
}


def bits_per_element(
    data_type: TensorDataType,
) -> int:
    """
    Return the logical number of storage bits per tensor element.

    Variable-width tensor types cannot be calculated statically.
    """

    if data_type in _VARIABLE_WIDTH_TYPES:
        raise TensorSizeError(
            "Tensor type has variable-width storage and cannot be "
            f"calculated statically: {data_type.name}"
        )

    try:
        return _FIXED_WIDTH_BITS[data_type]
    except KeyError as error:
        raise TensorSizeError(
            "No storage-width definition exists for tensor type: "
            f"{data_type.name}"
        ) from error


def calculate_element_count(
    shape: tuple[int, ...],
) -> int:
    """
    Calculate the number of logical elements represented by a shape.

    An empty shape represents a scalar and therefore contains one
    element. A dimension of zero produces an empty tensor. Negative
    dimensions are dynamic or invalid and cannot be sized statically.
    """

    negative_dimensions = [
        dimension
        for dimension in shape
        if dimension < 0
    ]

    if negative_dimensions:
        raise TensorSizeError(
            "Cannot calculate static tensor size for shape with "
            f"negative dimensions: {shape}"
        )

    if not shape:
        return 1

    return reduce(
        mul,
        shape,
        1,
    )


def bits_to_bytes_rounded_up(
    bit_count: int,
) -> int:
    """Convert bits to bytes, rounding partial bytes upward."""

    if bit_count < 0:
        raise TensorSizeError(
            f"Bit count must be non-negative: {bit_count}"
        )

    return (bit_count + 7) // 8


def calculate_tensor_size(
    tensor: Tensor,
) -> TensorSize:
    """Calculate the logical static storage size of a tensor."""

    element_count = calculate_element_count(
        tensor.shape
    )

    element_bits = bits_per_element(
        tensor.data_type
    )

    storage_bits = (
        element_count
        * element_bits
    )

    storage_bytes = bits_to_bytes_rounded_up(
        storage_bits
    )

    return TensorSize(
        element_count=element_count,
        bits_per_element=element_bits,
        storage_bits=storage_bits,
        storage_bytes=storage_bytes,
    )