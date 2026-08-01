from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import TypeAlias


TensorId: TypeAlias = int
OperatorId: TypeAlias = int
BufferId: TypeAlias = int
SubgraphId: TypeAlias = int


class GraphModelError(ValueError):
    """Raised when an invalid TensorScope graph model is constructed."""


class TensorDataType(IntEnum):
    """Tensor element types used by the TensorFlow Lite schema."""

    FLOAT32 = 0
    FLOAT16 = 1
    INT32 = 2
    UINT8 = 3
    INT64 = 4
    STRING = 5
    BOOL = 6
    INT16 = 7
    COMPLEX64 = 8
    INT8 = 9
    FLOAT64 = 10
    COMPLEX128 = 11
    UINT64 = 12
    RESOURCE = 13
    VARIANT = 14
    UINT32 = 15
    UINT16 = 16
    INT4 = 17
    BFLOAT16 = 18

    @classmethod
    def from_schema_value(
        cls,
        value: int,
    ) -> TensorDataType:
        try:
            return cls(value)
        except ValueError as error:
            raise GraphModelError(
                f"Unsupported TFLite tensor type value: {value}"
            ) from error


@dataclass(frozen=True)
class QuantizationParameters:
    """Quantization metadata associated with a tensor."""

    scales: tuple[float, ...] = ()
    zero_points: tuple[int, ...] = ()
    quantized_dimension: int = 0

    @property
    def is_quantized(self) -> bool:
        return bool(self.scales or self.zero_points)


@dataclass(frozen=True)
class Tensor:
    """A tensor contained in a TensorFlow Lite subgraph."""

    id: TensorId
    name: str
    data_type: TensorDataType
    shape: tuple[int, ...]
    shape_signature: tuple[int, ...]
    buffer_id: BufferId
    is_variable: bool
    has_constant_data: bool
    constant_data_size: int
    quantization: QuantizationParameters = field(
        default_factory=QuantizationParameters
    )

    def __post_init__(self) -> None:
        if self.id < 0:
            raise GraphModelError(
                f"Tensor ID must be non-negative: {self.id}"
            )

        if self.buffer_id < 0:
            raise GraphModelError(
                "Tensor buffer ID must be non-negative: "
                f"{self.buffer_id}"
            )

        if self.constant_data_size < 0:
            raise GraphModelError(
                "Tensor constant-data size must be non-negative: "
                f"{self.constant_data_size}"
            )

        if self.has_constant_data and self.constant_data_size == 0:
            raise GraphModelError(
                "Tensor is marked as constant but its constant-data "
                f"size is zero: tensor {self.id}"
            )

        if not self.has_constant_data and self.constant_data_size != 0:
            raise GraphModelError(
                "Non-constant tensor has a non-zero constant-data "
                f"size: tensor {self.id}"
            )


@dataclass(frozen=True)
class Operator:
    """An operator invocation contained in a subgraph."""

    id: OperatorId
    opcode_index: int
    name: str
    version: int
    inputs: tuple[TensorId, ...]
    outputs: tuple[TensorId, ...]
    intermediates: tuple[TensorId, ...] = ()
    builtin_code: int | None = None
    custom_code: str = ""

    def __post_init__(self) -> None:
        if self.id < 0:
            raise GraphModelError(
                f"Operator ID must be non-negative: {self.id}"
            )

        if self.opcode_index < 0:
            raise GraphModelError(
                "Operator-code index must be non-negative: "
                f"{self.opcode_index}"
            )

        if self.version <= 0:
            raise GraphModelError(
                f"Operator version must be positive: {self.version}"
            )

        if self.builtin_code is not None and self.builtin_code < 0:
            raise GraphModelError(
                f"Builtin operator code must be non-negative: {self.builtin_code}"
            )


@dataclass(frozen=True)
class Subgraph:
    """A TensorFlow Lite computational subgraph."""

    id: SubgraphId
    name: str
    tensors: tuple[Tensor, ...]
    operators: tuple[Operator, ...]
    inputs: tuple[TensorId, ...]
    outputs: tuple[TensorId, ...]

    def __post_init__(self) -> None:
        if self.id < 0:
            raise GraphModelError(
                f"Subgraph ID must be non-negative: {self.id}"
            )

        tensor_ids = tuple(tensor.id for tensor in self.tensors)
        operator_ids = tuple(
            operator.id for operator in self.operators
        )

        if tensor_ids != tuple(range(len(self.tensors))):
            raise GraphModelError(
                "Tensor IDs must be contiguous and match their "
                f"positions: {tensor_ids}"
            )

        if operator_ids != tuple(range(len(self.operators))):
            raise GraphModelError(
                "Operator IDs must be contiguous and match their "
                f"positions: {operator_ids}"
            )

        valid_tensor_ids = set(tensor_ids)

        referenced_tensor_ids: set[int] = set(
            self.inputs + self.outputs
        )

        for operator in self.operators:
            referenced_tensor_ids.update(
                tensor_id
                for tensor_id in (
                    operator.inputs
                    + operator.outputs
                    + operator.intermediates
                )
                if tensor_id >= 0
            )

        invalid_references = sorted(
            referenced_tensor_ids - valid_tensor_ids
        )

        if invalid_references:
            raise GraphModelError(
                "Subgraph references undefined tensor IDs: "
                f"{invalid_references}"
            )

    def tensor(self, tensor_id: TensorId) -> Tensor:
        try:
            return self.tensors[tensor_id]
        except IndexError as error:
            raise GraphModelError(
                f"Unknown tensor ID {tensor_id} in subgraph {self.id}"
            ) from error

    def operator(self, operator_id: OperatorId) -> Operator:
        try:
            return self.operators[operator_id]
        except IndexError as error:
            raise GraphModelError(
                "Unknown operator ID "
                f"{operator_id} in subgraph {self.id}"
            ) from error


@dataclass(frozen=True)
class GraphModel:
    """TensorScope's framework-independent model representation."""

    source_path: Path
    schema_version: int
    description: str
    subgraphs: tuple[Subgraph, ...]

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise GraphModelError(
                "Schema version must be positive: "
                f"{self.schema_version}"
            )

        if not self.subgraphs:
            raise GraphModelError(
                "Graph model must contain at least one subgraph"
            )

        subgraph_ids = tuple(
            subgraph.id for subgraph in self.subgraphs
        )

        if subgraph_ids != tuple(range(len(self.subgraphs))):
            raise GraphModelError(
                "Subgraph IDs must be contiguous and match their "
                f"positions: {subgraph_ids}"
            )

    @property
    def primary_subgraph(self) -> Subgraph:
        return self.subgraphs[0]

    @property
    def tensor_count(self) -> int:
        return sum(
            len(subgraph.tensors)
            for subgraph in self.subgraphs
        )

    @property
    def operator_count(self) -> int:
        return sum(
            len(subgraph.operators)
            for subgraph in self.subgraphs
        )

    def subgraph(
        self,
        subgraph_id: SubgraphId,
    ) -> Subgraph:
        try:
            return self.subgraphs[subgraph_id]
        except IndexError as error:
            raise GraphModelError(
                f"Unknown subgraph ID: {subgraph_id}"
            ) from error
