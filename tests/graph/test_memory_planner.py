from __future__ import annotations

import pytest

from tensorscope.graph.memory_planner import (
    BufferAllocation,
    BufferRequirement,
    MemoryPlannerError,
    SubgraphMemoryPlan,
    align_size_up,
    greedy_plan_requirements,
)


def make_requirement(
    planner_index: int,
    *,
    size: int,
    first: int,
    last: int,
    tensor_id: int | None = None,
    alignment: int = 16,
) -> BufferRequirement:
    if tensor_id is None:
        tensor_id = planner_index

    return BufferRequirement(
        tensor_id=tensor_id,
        logical_size=size,
        aligned_size=align_size_up(
            size,
            alignment,
        ),
        first_used=first,
        last_used=last,
        planner_index=planner_index,
    )


@pytest.mark.parametrize(
    ("size", "alignment", "expected"),
    [
        (0, 16, 0),
        (1, 16, 16),
        (4, 16, 16),
        (15, 16, 16),
        (16, 16, 16),
        (17, 16, 32),
        (31, 16, 32),
        (32, 16, 32),
        (33, 16, 48),
        (1, 8, 8),
        (9, 8, 16),
        (16, 8, 16),
    ],
)
def test_align_size_up(
    size: int,
    alignment: int,
    expected: int,
) -> None:
    assert align_size_up(
        size,
        alignment,
    ) == expected


@pytest.mark.parametrize(
    "alignment",
    [
        0,
        -1,
        3,
        6,
        12,
    ],
)
def test_invalid_alignment_is_rejected(
    alignment: int,
) -> None:
    with pytest.raises(
        MemoryPlannerError,
        match="Alignment",
    ):
        align_size_up(
            8,
            alignment,
        )


def test_negative_size_is_rejected() -> None:
    with pytest.raises(
        MemoryPlannerError,
        match="non-negative",
    ):
        align_size_up(-1)


def test_single_buffer_is_placed_at_zero() -> None:
    allocations = greedy_plan_requirements(
        (
            make_requirement(
                0,
                size=20,
                first=0,
                last=1,
            ),
        )
    )

    assert len(allocations) == 1
    assert allocations[0].offset == 0
    assert allocations[0].aligned_size == 32


def test_simultaneously_live_buffers_are_separated() -> None:
    allocations = greedy_plan_requirements(
        (
            make_requirement(
                0,
                size=64,
                first=0,
                last=2,
            ),
            make_requirement(
                1,
                size=32,
                first=1,
                last=3,
            ),
        )
    )

    assert allocations[0].offset == 0
    assert allocations[1].offset == 64


def test_disjoint_buffers_reuse_offset_zero() -> None:
    allocations = greedy_plan_requirements(
        (
            make_requirement(
                0,
                size=64,
                first=0,
                last=1,
            ),
            make_requirement(
                1,
                size=32,
                first=2,
                last=3,
            ),
        )
    )

    assert allocations[0].offset == 0
    assert allocations[1].offset == 0


def test_inclusive_lifetimes_overlap() -> None:
    allocations = greedy_plan_requirements(
        (
            make_requirement(
                0,
                size=16,
                first=0,
                last=1,
            ),
            make_requirement(
                1,
                size=16,
                first=1,
                last=2,
            ),
        )
    )

    assert allocations[0].offset == 0
    assert allocations[1].offset == 16


def test_larger_buffers_are_planned_first() -> None:
    allocations = greedy_plan_requirements(
        (
            make_requirement(
                0,
                size=16,
                first=0,
                last=3,
            ),
            make_requirement(
                1,
                size=64,
                first=0,
                last=3,
            ),
            make_requirement(
                2,
                size=32,
                first=0,
                last=3,
            ),
        )
    )

    assert allocations[1].offset == 0
    assert allocations[2].offset == 64
    assert allocations[0].offset == 96


def test_equal_sizes_keep_original_order() -> None:
    allocations = greedy_plan_requirements(
        (
            make_requirement(
                0,
                size=32,
                first=0,
                last=2,
            ),
            make_requirement(
                1,
                size=32,
                first=0,
                last=2,
            ),
        )
    )

    assert allocations[0].offset == 0
    assert allocations[1].offset == 32


def test_first_fitting_gap_is_used() -> None:
    allocations = greedy_plan_requirements(
        (
            make_requirement(
                0,
                size=64,
                first=0,
                last=4,
            ),
            make_requirement(
                1,
                size=32,
                first=0,
                last=1,
            ),
            make_requirement(
                2,
                size=16,
                first=2,
                last=4,
            ),
        )
    )

    assert allocations[0].offset == 0
    assert allocations[1].offset == 64
    assert allocations[2].offset == 64


def test_plan_detects_time_and_memory_conflict() -> None:
    with pytest.raises(
        MemoryPlannerError,
        match="overlap",
    ):
        SubgraphMemoryPlan(
            subgraph_id=0,
            alignment=16,
            allocations=(
                BufferAllocation(
                    tensor_id=0,
                    planner_index=0,
                    logical_size=16,
                    aligned_size=16,
                    offset=0,
                    first_used=0,
                    last_used=2,
                ),
                BufferAllocation(
                    tensor_id=1,
                    planner_index=1,
                    logical_size=16,
                    aligned_size=16,
                    offset=0,
                    first_used=1,
                    last_used=3,
                ),
            ),
            maximum_memory_size=16,
        )


def test_same_memory_is_valid_for_disjoint_lifetimes() -> None:
    plan = SubgraphMemoryPlan(
        subgraph_id=0,
        alignment=16,
        allocations=(
            BufferAllocation(
                tensor_id=0,
                planner_index=0,
                logical_size=16,
                aligned_size=16,
                offset=0,
                first_used=0,
                last_used=1,
            ),
            BufferAllocation(
                tensor_id=1,
                planner_index=1,
                logical_size=16,
                aligned_size=16,
                offset=0,
                first_used=2,
                last_used=3,
            ),
        ),
        maximum_memory_size=16,
    )

    assert plan.maximum_memory_size == 16


def test_live_byte_counters() -> None:
    plan = SubgraphMemoryPlan(
        subgraph_id=0,
        alignment=16,
        allocations=(
            BufferAllocation(
                tensor_id=0,
                planner_index=0,
                logical_size=4,
                aligned_size=16,
                offset=0,
                first_used=0,
                last_used=1,
            ),
            BufferAllocation(
                tensor_id=1,
                planner_index=1,
                logical_size=20,
                aligned_size=32,
                offset=16,
                first_used=1,
                last_used=2,
            ),
        ),
        maximum_memory_size=48,
    )

    assert plan.live_logical_bytes_at(0) == 4
    assert plan.live_aligned_bytes_at(0) == 16

    assert plan.live_logical_bytes_at(1) == 24
    assert plan.live_aligned_bytes_at(1) == 48

    assert plan.live_logical_bytes_at(2) == 20
    assert plan.live_aligned_bytes_at(2) == 32