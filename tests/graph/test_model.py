from pathlib import Path

import pytest

from tensorscope.graph.model import (
    GraphModel,
    GraphModelError,
    Operator,
    Subgraph,
    Tensor,
    TensorDataType,
)


def make_tensor(
    tensor_id: int,
) -> Tensor:
    return Tensor(
        id=tensor_id,
        name=f"tensor_{tensor_id}",
        data_type=TensorDataType.FLOAT32,
        shape=(1, 4),
        shape_signature=(),
        buffer_id=0,
        is_variable=False,
        has_constant_data=False,
        constant_data_size=0,
    )


def test_graph_model_summary_properties() -> None:
    subgraph = Subgraph(
        id=0,
        name="main",
        tensors=(
            make_tensor(0),
            make_tensor(1),
        ),
        operators=(
            Operator(
                id=0,
                opcode_index=0,
                name="ADD",
                version=1,
                inputs=(0, 1),
                outputs=(1,),
            ),
        ),
        inputs=(0,),
        outputs=(1,),
    )

    model = GraphModel(
        source_path=Path("model.tflite"),
        schema_version=3,
        description="test",
        subgraphs=(subgraph,),
    )

    assert model.primary_subgraph is subgraph
    assert model.tensor_count == 2
    assert model.operator_count == 1
    assert model.subgraph(0) is subgraph
    assert subgraph.tensor(1).name == "tensor_1"
    assert subgraph.operator(0).name == "ADD"


def test_non_contiguous_tensor_ids_are_rejected() -> None:
    with pytest.raises(
        GraphModelError,
        match="Tensor IDs must be contiguous",
    ):
        Subgraph(
            id=0,
            name="main",
            tensors=(
                make_tensor(0),
                make_tensor(2),
            ),
            operators=(),
            inputs=(0,),
            outputs=(2,),
        )


def test_invalid_tensor_reference_is_rejected() -> None:
    with pytest.raises(
        GraphModelError,
        match="undefined tensor IDs",
    ):
        Subgraph(
            id=0,
            name="main",
            tensors=(make_tensor(0),),
            operators=(
                Operator(
                    id=0,
                    opcode_index=0,
                    name="ADD",
                    version=1,
                    inputs=(0, 5),
                    outputs=(0,),
                ),
            ),
            inputs=(0,),
            outputs=(0,),
        )


def test_unknown_tensor_type_is_rejected() -> None:
    with pytest.raises(
        GraphModelError,
        match="Unsupported TFLite tensor type",
    ):
        TensorDataType.from_schema_value(999)


def test_invalid_constant_metadata_is_rejected() -> None:
    with pytest.raises(
        GraphModelError,
        match="marked as constant",
    ):
        Tensor(
            id=0,
            name="invalid",
            data_type=TensorDataType.INT8,
            shape=(4,),
            shape_signature=(),
            buffer_id=1,
            is_variable=False,
            has_constant_data=True,
            constant_data_size=0,
        )