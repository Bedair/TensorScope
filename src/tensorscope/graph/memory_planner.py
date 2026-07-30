from __future__ import annotations

from dataclasses import dataclass

from tensorscope.graph.model import (
    GraphModel,
    GraphModelError,
    Subgraph,
    TensorId,
)
from tensorscope.graph.tensor_lifetime import (
    GraphLifetimeAnalysis,
    SubgraphLifetimeAnalysis,
    calculate_graph_lifetimes,
)
from tensorscope.graph.tensor_size import (
    calculate_tensor_size,
)


TFLM_ARENA_ALIGNMENT = 16


class MemoryPlannerError(GraphModelError):
    """Raised when a valid memory plan cannot be created."""


def align_size_up(
    size: int,
    alignment: int = TFLM_ARENA_ALIGNMENT,
) -> int:
    """Round a byte count upward to the requested alignment."""

    if size < 0:
        raise MemoryPlannerError(
            f"Size must be non-negative: {size}"
        )

    if alignment <= 0:
        raise MemoryPlannerError(
            f"Alignment must be positive: {alignment}"
        )

    if alignment & (alignment - 1):
        raise MemoryPlannerError(
            "Alignment must be a power of two: "
            f"{alignment}"
        )

    return (size + alignment - 1) & ~(alignment - 1)


@dataclass(frozen=True)
class BufferRequirement:
    """One runtime tensor buffer passed to the greedy planner."""

    tensor_id: TensorId
    logical_size: int
    aligned_size: int
    first_used: int
    last_used: int
    planner_index: int

    def __post_init__(self) -> None:
        if self.tensor_id < 0:
            raise MemoryPlannerError(
                "Tensor ID must be non-negative: "
                f"{self.tensor_id}"
            )

        if self.logical_size <= 0:
            raise MemoryPlannerError(
                "Logical size must be positive: "
                f"{self.logical_size}"
            )

        if self.aligned_size < self.logical_size:
            raise MemoryPlannerError(
                "Aligned size cannot be smaller than logical size"
            )

        if self.first_used < 0:
            raise MemoryPlannerError(
                "First-used scope must be non-negative: "
                f"{self.first_used}"
            )

        if self.last_used < self.first_used:
            raise MemoryPlannerError(
                "Last-used scope cannot precede first-used scope: "
                f"{self.first_used}..{self.last_used}"
            )

        if self.planner_index < 0:
            raise MemoryPlannerError(
                "Planner index must be non-negative: "
                f"{self.planner_index}"
            )

    def overlaps_in_time(
        self,
        other: BufferRequirement,
    ) -> bool:
        return not (
            self.last_used < other.first_used
            or other.last_used < self.first_used
        )


@dataclass(frozen=True)
class BufferAllocation:
    """Final arena-head placement for one runtime tensor."""

    tensor_id: TensorId
    planner_index: int
    logical_size: int
    aligned_size: int
    offset: int
    first_used: int
    last_used: int

    def __post_init__(self) -> None:
        if self.tensor_id < 0:
            raise MemoryPlannerError(
                "Tensor ID must be non-negative: "
                f"{self.tensor_id}"
            )

        if self.planner_index < 0:
            raise MemoryPlannerError(
                "Planner index must be non-negative: "
                f"{self.planner_index}"
            )

        if self.logical_size <= 0:
            raise MemoryPlannerError(
                "Logical size must be positive: "
                f"{self.logical_size}"
            )

        if self.aligned_size < self.logical_size:
            raise MemoryPlannerError(
                "Aligned size cannot be smaller than logical size"
            )

        if self.offset < 0:
            raise MemoryPlannerError(
                f"Offset must be non-negative: {self.offset}"
            )

        if self.first_used < 0:
            raise MemoryPlannerError(
                "First-used scope must be non-negative: "
                f"{self.first_used}"
            )

        if self.last_used < self.first_used:
            raise MemoryPlannerError(
                "Last-used scope cannot precede first-used scope"
            )

    @property
    def end_offset(self) -> int:
        return self.offset + self.aligned_size

    def overlaps_in_time(
        self,
        other: BufferAllocation,
    ) -> bool:
        return not (
            self.last_used < other.first_used
            or other.last_used < self.first_used
        )

    def overlaps_in_memory(
        self,
        other: BufferAllocation,
    ) -> bool:
        return not (
            self.end_offset <= other.offset
            or other.end_offset <= self.offset
        )

    def conflicts_with(
        self,
        other: BufferAllocation,
    ) -> bool:
        return (
            self.overlaps_in_time(other)
            and self.overlaps_in_memory(other)
        )


@dataclass(frozen=True)
class SubgraphMemoryPlan:
    """Greedy memory plan for one subgraph."""

    subgraph_id: int
    alignment: int
    allocations: tuple[BufferAllocation, ...]
    maximum_memory_size: int

    def __post_init__(self) -> None:
        if self.subgraph_id < 0:
            raise MemoryPlannerError(
                "Subgraph ID must be non-negative: "
                f"{self.subgraph_id}"
            )

        if self.alignment <= 0:
            raise MemoryPlannerError(
                "Alignment must be positive: "
                f"{self.alignment}"
            )

        if self.maximum_memory_size < 0:
            raise MemoryPlannerError(
                "Maximum memory size must be non-negative"
            )

        expected_size = max(
            (
                allocation.end_offset
                for allocation in self.allocations
            ),
            default=0,
        )

        if self.maximum_memory_size != expected_size:
            raise MemoryPlannerError(
                "Maximum memory size does not match the "
                "allocation high-water mark: "
                f"expected {expected_size}, "
                f"got {self.maximum_memory_size}"
            )

        self.validate_no_conflicts()

    def allocation(
        self,
        tensor_id: TensorId,
    ) -> BufferAllocation:
        for allocation in self.allocations:
            if allocation.tensor_id == tensor_id:
                return allocation

        raise MemoryPlannerError(
            "No allocation exists for tensor "
            f"{tensor_id} in subgraph {self.subgraph_id}"
        )

    def validate_no_conflicts(self) -> None:
        for index, first in enumerate(self.allocations):
            for second in self.allocations[index + 1 :]:
                if first.conflicts_with(second):
                    raise MemoryPlannerError(
                        "Buffers overlap in both time and memory: "
                        f"tensor {first.tensor_id} and "
                        f"tensor {second.tensor_id}"
                    )

    def live_allocations_at(
        self,
        scope: int,
    ) -> tuple[BufferAllocation, ...]:
        if scope < 0:
            raise MemoryPlannerError(
                f"Scope must be non-negative: {scope}"
            )

        return tuple(
            allocation
            for allocation in self.allocations
            if (
                allocation.first_used
                <= scope
                <= allocation.last_used
            )
        )

    def live_logical_bytes_at(
        self,
        scope: int,
    ) -> int:
        return sum(
            allocation.logical_size
            for allocation in self.live_allocations_at(scope)
        )

    def live_aligned_bytes_at(
        self,
        scope: int,
    ) -> int:
        return sum(
            allocation.aligned_size
            for allocation in self.live_allocations_at(scope)
        )


@dataclass(frozen=True)
class GraphMemoryPlan:
    """Memory plans for all graph subgraphs."""

    subgraphs: tuple[SubgraphMemoryPlan, ...]

    @property
    def primary_subgraph(self) -> SubgraphMemoryPlan:
        if not self.subgraphs:
            raise MemoryPlannerError(
                "Graph memory plan contains no subgraphs"
            )

        return self.subgraphs[0]

    @property
    def maximum_memory_size(self) -> int:
        return max(
            (
                subgraph.maximum_memory_size
                for subgraph in self.subgraphs
            ),
            default=0,
        )

    def subgraph(
        self,
        subgraph_id: int,
    ) -> SubgraphMemoryPlan:
        try:
            return self.subgraphs[subgraph_id]
        except IndexError as error:
            raise MemoryPlannerError(
                f"Unknown subgraph memory-plan ID: {subgraph_id}"
            ) from error


def _build_requirements(
    subgraph: Subgraph,
    lifetimes: SubgraphLifetimeAnalysis,
    alignment: int,
) -> tuple[BufferRequirement, ...]:
    requirements: list[BufferRequirement] = []

    for planner_index, lifetime in enumerate(
        lifetimes.plannable_lifetimes
    ):
        if not lifetime.is_initialized:
            raise MemoryPlannerError(
                "Plannable tensor has no initialized lifetime: "
                f"tensor {lifetime.tensor_id}"
            )

        assert lifetime.first_created is not None
        assert lifetime.last_used is not None

        tensor = subgraph.tensor(
            lifetime.tensor_id
        )

        logical_size = calculate_tensor_size(
            tensor
        ).storage_bytes

        if logical_size <= 0:
            raise MemoryPlannerError(
                "Plannable tensor has non-positive size: "
                f"tensor {tensor.id}, size {logical_size}"
            )

        requirements.append(
            BufferRequirement(
                tensor_id=tensor.id,
                logical_size=logical_size,
                aligned_size=align_size_up(
                    logical_size,
                    alignment,
                ),
                first_used=lifetime.first_created,
                last_used=lifetime.last_used,
                planner_index=planner_index,
            )
        )

    return tuple(requirements)


def _sort_requirements(
    requirements: tuple[BufferRequirement, ...],
) -> tuple[BufferRequirement, ...]:
    return tuple(
        sorted(
            requirements,
            key=lambda requirement: (
                -requirement.aligned_size
            ),
        )
    )


def _requirement_overlaps_allocation(
    requirement: BufferRequirement,
    allocation: BufferAllocation,
) -> bool:
    return not (
        requirement.last_used < allocation.first_used
        or allocation.last_used < requirement.first_used
    )


def _find_offset(
    requirement: BufferRequirement,
    placed_by_offset: list[BufferAllocation],
) -> int:
    candidate_offset = 0

    for allocation in placed_by_offset:
        if not _requirement_overlaps_allocation(
            requirement,
            allocation,
        ):
            continue

        gap_size = allocation.offset - candidate_offset

        if gap_size >= requirement.aligned_size:
            return candidate_offset

        if allocation.end_offset > candidate_offset:
            candidate_offset = allocation.end_offset

    return candidate_offset


def _insert_by_offset(
    allocations: list[BufferAllocation],
    new_allocation: BufferAllocation,
) -> None:
    insertion_index = len(allocations)

    for index, allocation in enumerate(allocations):
        if allocation.offset > new_allocation.offset:
            insertion_index = index
            break

    allocations.insert(
        insertion_index,
        new_allocation,
    )


def greedy_plan_requirements(
    requirements: tuple[BufferRequirement, ...],
) -> tuple[BufferAllocation, ...]:
    """Place runtime buffers using a greedy first-fit algorithm."""

    if not requirements:
        return ()

    sorted_requirements = _sort_requirements(
        requirements
    )

    placed_by_offset: list[BufferAllocation] = []
    allocations_by_planner_index: dict[
        int,
        BufferAllocation,
    ] = {}

    for requirement in sorted_requirements:
        offset = _find_offset(
            requirement,
            placed_by_offset,
        )

        allocation = BufferAllocation(
            tensor_id=requirement.tensor_id,
            planner_index=requirement.planner_index,
            logical_size=requirement.logical_size,
            aligned_size=requirement.aligned_size,
            offset=offset,
            first_used=requirement.first_used,
            last_used=requirement.last_used,
        )

        _insert_by_offset(
            placed_by_offset,
            allocation,
        )

        allocations_by_planner_index[
            requirement.planner_index
        ] = allocation

    expected_indices = set(
        range(len(requirements))
    )

    if set(allocations_by_planner_index) != expected_indices:
        raise MemoryPlannerError(
            "Planner did not produce exactly one allocation "
            "for every requirement"
        )

    return tuple(
        allocations_by_planner_index[index]
        for index in range(len(requirements))
    )


def calculate_subgraph_memory_plan(
    subgraph: Subgraph,
    lifetimes: SubgraphLifetimeAnalysis,
    *,
    alignment: int = TFLM_ARENA_ALIGNMENT,
) -> SubgraphMemoryPlan:
    """Create a greedy memory plan for one subgraph."""

    if subgraph.id != lifetimes.subgraph_id:
        raise MemoryPlannerError(
            "Subgraph and lifetime-analysis IDs differ: "
            f"{subgraph.id} != {lifetimes.subgraph_id}"
        )

    requirements = _build_requirements(
        subgraph,
        lifetimes,
        alignment,
    )

    allocations = greedy_plan_requirements(
        requirements
    )

    maximum_memory_size = max(
        (
            allocation.end_offset
            for allocation in allocations
        ),
        default=0,
    )

    return SubgraphMemoryPlan(
        subgraph_id=subgraph.id,
        alignment=alignment,
        allocations=allocations,
        maximum_memory_size=maximum_memory_size,
    )


def calculate_graph_memory_plan(
    graph: GraphModel,
    lifetimes: GraphLifetimeAnalysis | None = None,
    *,
    alignment: int = TFLM_ARENA_ALIGNMENT,
) -> GraphMemoryPlan:
    """Create greedy memory plans for all graph subgraphs."""

    if lifetimes is None:
        lifetimes = calculate_graph_lifetimes(
            graph
        )

    if len(graph.subgraphs) != len(
        lifetimes.subgraphs
    ):
        raise MemoryPlannerError(
            "Graph and lifetime analysis have different "
            "subgraph counts"
        )

    return GraphMemoryPlan(
        subgraphs=tuple(
            calculate_subgraph_memory_plan(
                subgraph,
                lifetime_analysis,
                alignment=alignment,
            )
            for subgraph, lifetime_analysis in zip(
                graph.subgraphs,
                lifetimes.subgraphs,
                strict=True,
            )
        )
    )