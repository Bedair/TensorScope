from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tensorscope.graph.model import (
    GraphModel,
    GraphModelError,
    Operator,
    QuantizationParameters,
    Subgraph,
    Tensor,
    TensorDataType,
)
from tensorscope.tflite.model_loader import LoadedModel


def _decode_text(value: bytes | str | None) -> str:
    if value is None:
        return ""

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    return str(value)


def _read_vector(
    length: int,
    reader: Callable[[int], Any],
    converter: Callable[[Any], Any] | None = None,
) -> tuple[Any, ...]:
    if converter is None:
        converter = lambda value: value

    return tuple(
        converter(reader(index))
        for index in range(length)
    )


def _read_optional_int_vector(
    owner: Any,
    length_method_name: str,
    item_method_name: str,
) -> tuple[int, ...]:
    length_method = getattr(
        owner,
        length_method_name,
        None,
    )

    item_method = getattr(
        owner,
        item_method_name,
        None,
    )

    if length_method is None or item_method is None:
        return ()

    return _read_vector(
        int(length_method()),
        item_method,
        int,
    )


def _read_quantization(
    schema_tensor: Any,
) -> QuantizationParameters:
    quantization = schema_tensor.Quantization()

    if quantization is None:
        return QuantizationParameters()

    scales = _read_vector(
        int(quantization.ScaleLength()),
        quantization.Scale,
        float,
    )

    zero_points = _read_vector(
        int(quantization.ZeroPointLength()),
        quantization.ZeroPoint,
        int,
    )

    return QuantizationParameters(
        scales=scales,
        zero_points=zero_points,
        quantized_dimension=int(
            quantization.QuantizedDimension()
        ),
    )


def _buffer_data_size(
    loaded_model: LoadedModel,
    buffer_id: int,
) -> int:
    if buffer_id < 0:
        raise GraphModelError(
            f"Buffer ID must be non-negative: {buffer_id}"
        )

    if buffer_id >= loaded_model.buffer_count:
        raise GraphModelError(
            "Tensor references undefined buffer ID "
            f"{buffer_id}; model has "
            f"{loaded_model.buffer_count} buffers"
        )

    schema_buffer = loaded_model.model.Buffers(
        buffer_id
    )

    if schema_buffer is None:
        return 0

    return int(schema_buffer.DataLength())


def _convert_tensor(
    loaded_model: LoadedModel,
    schema_tensor: Any,
    tensor_id: int,
) -> Tensor:
    buffer_id = int(schema_tensor.Buffer())

    constant_data_size = _buffer_data_size(
        loaded_model,
        buffer_id,
    )

    shape = _read_vector(
        int(schema_tensor.ShapeLength()),
        schema_tensor.Shape,
        int,
    )

    shape_signature = _read_optional_int_vector(
        schema_tensor,
        "ShapeSignatureLength",
        "ShapeSignature",
    )

    return Tensor(
        id=tensor_id,
        name=_decode_text(schema_tensor.Name()),
        data_type=TensorDataType.from_schema_value(
            int(schema_tensor.Type())
        ),
        shape=shape,
        shape_signature=shape_signature,
        buffer_id=buffer_id,
        is_variable=bool(schema_tensor.IsVariable()),
        has_constant_data=constant_data_size > 0,
        constant_data_size=constant_data_size,
        quantization=_read_quantization(
            schema_tensor
        ),
    )


def _builtin_operator_name(
    builtin_code: int,
) -> str:
    try:
        from tensorscope.tflite.schema.schema_generated import (
            BuiltinOperator,
        )
    except ImportError as error:
        raise GraphModelError(
            "Unable to import TFLite BuiltinOperator schema"
        ) from error

    for attribute_name, attribute_value in vars(
        BuiltinOperator
    ).items():
        if attribute_name.startswith("_"):
            continue

        if isinstance(attribute_value, int):
            if attribute_value == builtin_code:
                return attribute_name

    return f"BUILTIN_{builtin_code}"


def _operator_name(
    loaded_model: LoadedModel,
    opcode_index: int,
) -> tuple[str, int]:
    if opcode_index < 0:
        raise GraphModelError(
            "Operator-code index must be non-negative: "
            f"{opcode_index}"
        )

    if opcode_index >= loaded_model.operator_code_count:
        raise GraphModelError(
            "Operator references undefined operator-code index "
            f"{opcode_index}; model has "
            f"{loaded_model.operator_code_count} operator codes"
        )

    operator_code = loaded_model.model.OperatorCodes(
        opcode_index
    )

    if operator_code is None:
        raise GraphModelError(
            f"Operator code {opcode_index} is missing"
        )

    custom_code = _decode_text(
        operator_code.CustomCode()
    )

    builtin_code = int(
        operator_code.BuiltinCode()
    )

    if custom_code:
        name = f"CUSTOM:{custom_code}"
    else:
        name = _builtin_operator_name(
            builtin_code
        )

    version = int(operator_code.Version())

    return name, version


def _convert_operator(
    loaded_model: LoadedModel,
    schema_operator: Any,
    operator_id: int,
) -> Operator:
    opcode_index = int(
        schema_operator.OpcodeIndex()
    )

    name, version = _operator_name(
        loaded_model,
        opcode_index,
    )

    inputs = _read_vector(
        int(schema_operator.InputsLength()),
        schema_operator.Inputs,
        int,
    )

    outputs = _read_vector(
        int(schema_operator.OutputsLength()),
        schema_operator.Outputs,
        int,
    )

    intermediates = _read_optional_int_vector(
        schema_operator,
        "IntermediatesLength",
        "Intermediates",
    )

    return Operator(
        id=operator_id,
        opcode_index=opcode_index,
        name=name,
        version=version,
        inputs=inputs,
        outputs=outputs,
        intermediates=intermediates,
    )


def _convert_subgraph(
    loaded_model: LoadedModel,
    schema_subgraph: Any,
    subgraph_id: int,
) -> Subgraph:
    tensors: list[Tensor] = []

    for tensor_id in range(
        int(schema_subgraph.TensorsLength())
    ):
        schema_tensor = schema_subgraph.Tensors(
            tensor_id
        )

        if schema_tensor is None:
            raise GraphModelError(
                "Missing tensor "
                f"{tensor_id} in subgraph {subgraph_id}"
            )

        tensors.append(
            _convert_tensor(
                loaded_model,
                schema_tensor,
                tensor_id,
            )
        )

    operators: list[Operator] = []

    for operator_id in range(
        int(schema_subgraph.OperatorsLength())
    ):
        schema_operator = schema_subgraph.Operators(
            operator_id
        )

        if schema_operator is None:
            raise GraphModelError(
                "Missing operator "
                f"{operator_id} in subgraph {subgraph_id}"
            )

        operators.append(
            _convert_operator(
                loaded_model,
                schema_operator,
                operator_id,
            )
        )

    inputs = _read_vector(
        int(schema_subgraph.InputsLength()),
        schema_subgraph.Inputs,
        int,
    )

    outputs = _read_vector(
        int(schema_subgraph.OutputsLength()),
        schema_subgraph.Outputs,
        int,
    )

    return Subgraph(
        id=subgraph_id,
        name=_decode_text(
            schema_subgraph.Name()
        ),
        tensors=tuple(tensors),
        operators=tuple(operators),
        inputs=inputs,
        outputs=outputs,
    )


def convert_tflite_model(
    loaded_model: LoadedModel,
) -> GraphModel:
    """Convert a loaded TFLite FlatBuffer into TensorScope IR."""

    subgraphs: list[Subgraph] = []

    for subgraph_id in range(
        loaded_model.subgraph_count
    ):
        schema_subgraph = (
            loaded_model.model.Subgraphs(
                subgraph_id
            )
        )

        if schema_subgraph is None:
            raise GraphModelError(
                f"Missing subgraph {subgraph_id}"
            )

        subgraphs.append(
            _convert_subgraph(
                loaded_model,
                schema_subgraph,
                subgraph_id,
            )
        )

    return GraphModel(
        source_path=loaded_model.path,
        schema_version=loaded_model.schema_version,
        description=_decode_text(
            loaded_model.model.Description()
        ),
        subgraphs=tuple(subgraphs),
    )