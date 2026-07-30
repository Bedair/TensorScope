from __future__ import annotations

from pathlib import Path

import pytest

from tensorscope.graph import (
    GraphModel,
    Operator,
    Subgraph,
    Tensor,
    TensorDataType,
    TensorLifetime,
    TensorLifetimeError,
    calculate_graph_lifetimes,
    calculate_subgraph_lifetimes,
)


def make_tensor(
    tensor_id: int,
    *,
    shape: tuple[int, ...] = (1,),
    constant: bool = False,
    variable: bool = False,
) -> Tensor:
    return Tensor(
        id=tensor_id,
        name=f"tensor_{tensor_id}",
        data_type=TensorDataType.FLOAT32,
        shape=shape,
        shape_signature=(),
        buffer_id=tensor_id if constant else 0,
        is_variable=variable,
        has_constant_data=constant,
        constant_data_size=4 if constant else 0,
    )


def test_linear_graph_lifetimes() -> None:
    subgraph = Subgraph(
        id=0,
        name="main",
        tensors=(
            make_tensor(0),
            make_tensor(1),
            make_tensor(2),
        ),
        operators=(
            Operator(
                id=0,
                opcode_index=0,
                name="OP_0",
                version=1,
                inputs=(0,),
                outputs=(1,),
            ),
            Operator(
                id=1,
                opcode_index=0,
                name="OP_1",
                version=1,
                inputs=(1,),
                outputs=(2,),
            ),
        ),
        inputs=(0,),
        outputs=(2,),
    )

    analysis = calculate_subgraph_lifetimes(
        subgraph
    )

    assert analysis.operator_scope_count == 2

    assert analysis.tensor(0).first_created == 0
    assert analysis.tensor(0).last_used == 1

    assert analysis.tensor(1).first_created == 1
    assert analysis.tensor(1).last_used == 2

    assert analysis.tensor(2).first_created == 2
    assert analysis.tensor(2).last_used == 2


def test_operator_input_and_output_overlap() -> None:
    first = TensorLifetime(
        tensor_id=0,
        first_created=0,
        last_used=1,
        needs_allocation=True,
        is_subgraph_input=True,
        is_subgraph_output=False,
    )

    second = TensorLifetime(
        tensor_id=1,
        first_created=1,
        last_used=2,
        needs_allocation=True,
        is_subgraph_input=False,
        is_subgraph_output=True,
    )

    assert first.overlaps(second)
    assert first.is_live_at(1)
    assert second.is_live_at(1)


def test_disjoint_lifetimes_do_not_overlap() -> None:
    first = TensorLifetime(
        tensor_id=0,
        first_created=0,
        last_used=1,
        needs_allocation=True,
        is_subgraph_input=True,
        is_subgraph_output=False,
    )

    second = TensorLifetime(
        tensor_id=1,
        first_created=2,
        last_used=3,
        needs_allocation=True,
        is_subgraph_input=False,
        is_subgraph_output=True,
    )

    assert not first.overlaps(second)


def test_lifetime_duration_is_inclusive() -> None:
    lifetime = TensorLifetime(
        tensor_id=0,
        first_created=1,
        last_used=3,
        needs_allocation=True,
        is_subgraph_input=False,
        is_subgraph_output=False,
    )

    assert lifetime.duration == 3


def test_unused_subgraph_input_has_scope_zero_lifetime() -> None:
    subgraph = Subgraph(
        id=0,
        name="main",
        tensors=(make_tensor(0),),
        operators=(),
        inputs=(0,),
        outputs=(),
    )

    analysis = calculate_subgraph_lifetimes(
        subgraph
    )

    lifetime = analysis.tensor(0)

    assert lifetime.first_created == 0
    assert lifetime.last_used == 0
    assert lifetime.needs_allocation


def test_empty_subgraph_output_gets_valid_lifetime() -> None:
    subgraph = Subgraph(
        id=0,
        name="main",
        tensors=(make_tensor(0),),
        operators=(),
        inputs=(0,),
        outputs=(0,),
    )

    lifetime = calculate_subgraph_lifetimes(
        subgraph
    ).tensor(0)

    assert lifetime.first_created == 0
    assert lifetime.last_used == 0
    assert lifetime.is_subgraph_input
    assert lifetime.is_subgraph_output


def test_optional_negative_input_is_ignored() -> None:
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
                name="OPTIONAL_INPUT_OP",
                version=1,
                inputs=(0, -1),
                outputs=(1,),
            ),
        ),
        inputs=(0,),
        outputs=(1,),
    )

    analysis = calculate_subgraph_lifetimes(
        subgraph
    )

    assert analysis.tensor(0).last_used == 1
    assert analysis.tensor(1).first_created == 1


def test_constant_tensor_is_not_plannable() -> None:
    subgraph = Subgraph(
        id=0,
        name="main",
        tensors=(
            make_tensor(0),
            make_tensor(1, constant=True),
            make_tensor(2),
        ),
        operators=(
            Operator(
                id=0,
                opcode_index=0,
                name="ADD",
                version=1,
                inputs=(0, 1),
                outputs=(2,),
            ),
        ),
        inputs=(0,),
        outputs=(2,),
    )

    analysis = calculate_subgraph_lifetimes(
        subgraph
    )

    constant = analysis.tensor(1)

    assert constant.first_created is None
    assert constant.last_used == 1
    assert not constant.needs_allocation


def test_variable_tensor_is_not_plannable() -> None:
    subgraph = Subgraph(
        id=0,
        name="main",
        tensors=(
            make_tensor(0, variable=True),
        ),
        operators=(),
        inputs=(0,),
        outputs=(0,),
    )

    lifetime = calculate_subgraph_lifetimes(
        subgraph
    ).tensor(0)

    assert lifetime.is_initialized
    assert not lifetime.needs_allocation


def test_zero_sized_tensor_is_not_plannable() -> None:
    subgraph = Subgraph(
        id=0,
        name="main",
        tensors=(
            make_tensor(0, shape=(0,)),
        ),
        operators=(),
        inputs=(),
        outputs=(),
    )

    lifetime = calculate_subgraph_lifetimes(
        subgraph
    ).tensor(0)

    assert not lifetime.is_initialized
    assert not lifetime.needs_allocation


def test_unconnected_runtime_tensor_is_rejected() -> None:
    subgraph = Subgraph(
        id=0,
        name="main",
        tensors=(make_tensor(0),),
        operators=(),
        inputs=(),
        outputs=(),
    )

    with pytest.raises(
        TensorLifetimeError,
        match="not connected",
    ):
        calculate_subgraph_lifetimes(subgraph)


def test_plannable_lifetimes_filters_constants() -> None:
    subgraph = Subgraph(
        id=0,
        name="main",
        tensors=(
            make_tensor(0),
            make_tensor(1, constant=True),
            make_tensor(2),
        ),
        operators=(
            Operator(
                id=0,
                opcode_index=0,
                name="ADD",
                version=1,
                inputs=(0, 1),
                outputs=(2,),
            ),
        ),
        inputs=(0,),
        outputs=(2,),
    )

    analysis = calculate_subgraph_lifetimes(
        subgraph
    )

    assert tuple(
        lifetime.tensor_id
        for lifetime in analysis.plannable_lifetimes
    ) == (0, 2)


def test_graph_lifetime_analysis() -> None:
    subgraph = Subgraph(
        id=0,
        name="main",
        tensors=(make_tensor(0),),
        operators=(),
        inputs=(0,),
        outputs=(0,),
    )

    graph = GraphModel(
        source_path=Path("model.tflite"),
        schema_version=3,
        description="test",
        subgraphs=(subgraph,),
    )

    analysis = calculate_graph_lifetimes(graph)

    assert analysis.primary_subgraph.subgraph_id == 0
    assert analysis.subgraph(0).tensor(0).duration == 1


def test_negative_scope_is_rejected() -> None:
    lifetime = TensorLifetime(
        tensor_id=0,
        first_created=0,
        last_used=0,
        needs_allocation=True,
        is_subgraph_input=True,
        is_subgraph_output=True,
    )

    with pytest.raises(
        TensorLifetimeError,
        match="non-negative",
    ):
        lifetime.is_live_at(-1)