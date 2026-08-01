from __future__ import annotations

from tensorscope.graph import Operator, Subgraph, Tensor, TensorDataType
from tensorscope.graph.memory_planner import calculate_subgraph_memory_plan
from tensorscope.graph.tensor_lifetime import calculate_subgraph_lifetimes


def _tensor(index: int, *, constant: bool = False) -> Tensor:
    return Tensor(index, f"t{index}", TensorDataType.FLOAT32, (4,), (),
                  index if constant else 0, False, constant, 16 if constant else 0)


def test_fanout_merge_skip_and_late_input_have_deterministic_plan() -> None:
    # t0 fans out into two branches; t1 is a graph input consumed only at the
    # merge; t2 also skips over op 2. t6 is constant and never enters the plan.
    subgraph = Subgraph(
        0, "branches", tuple(_tensor(i, constant=(i == 6)) for i in range(7)),
        (
            Operator(0, 0, "RELU", 1, (0,), (2,)),
            Operator(1, 0, "RELU6", 1, (2,), (3,)),
            Operator(2, 0, "LOGISTIC", 1, (0,), (4,)),
            Operator(3, 0, "ADD", 1, (3, 4, 1, 2, 6), (5,)),
        ),
        (0, 1), (5,),
    )
    lifetimes = calculate_subgraph_lifetimes(subgraph)
    assert (lifetimes.tensor(0).first_created, lifetimes.tensor(0).last_used) == (0, 3)
    assert (lifetimes.tensor(1).first_created, lifetimes.tensor(1).last_used) == (0, 4)
    assert lifetimes.tensor(2).last_used == 4
    assert not lifetimes.tensor(6).needs_allocation
    plan = calculate_subgraph_memory_plan(subgraph, lifetimes)
    assert plan.maximum_memory_size == 80
    assert [allocation.tensor_id for allocation in plan.allocations] == [0, 1, 2, 3, 4, 5]
