from __future__ import annotations

from dataclasses import asdict, dataclass

from tensorscope.graph import (
    GraphLifetimeAnalysis,
    GraphMemoryPlan,
    GraphModel,
    Subgraph,
    SubgraphLifetimeAnalysis,
    SubgraphMemoryPlan,
    calculate_graph_lifetimes,
    calculate_graph_memory_plan,
)


@dataclass(frozen=True)
class MemorySummary:
    runtime_tensor_count: int
    constant_tensor_count: int
    constant_tensor_bytes: int
    operator_count: int
    planned_arena_head_bytes: int
    arena_alignment_bytes: int
    logical_runtime_tensor_bytes: int
    aligned_runtime_tensor_bytes: int
    alignment_overhead_bytes: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class TensorExplanation:
    tensor_id: int
    name: str
    data_type: str
    shape: tuple[int, ...]
    logical_bytes: int
    aligned_bytes: int
    alignment_overhead_bytes: int
    offset: int
    end_offset: int
    first_used_scope: int
    last_used_scope: int
    lifetime_length: int
    is_graph_input: bool
    is_graph_output: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["shape"] = list(self.shape)
        return result


@dataclass(frozen=True)
class ScopeExplanation:
    scope: int
    scope_kind: str
    operator_id: int | None
    operator_name: str | None
    occupied_extent_bytes: int
    live_aligned_bytes: int
    live_tensor_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["live_tensor_ids"] = list(self.live_tensor_ids)
        return result


@dataclass(frozen=True)
class PeakExplanation:
    scope: int
    tied_scopes: tuple[int, ...]
    scope_kind: str
    operator_id: int | None
    operator_name: str | None
    occupied_extent_bytes: int
    live_aligned_bytes: int
    live_tensor_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["tied_scopes"] = list(self.tied_scopes)
        result["live_tensor_ids"] = list(self.live_tensor_ids)
        return result


@dataclass(frozen=True)
class ReuseRelationship:
    first_tensor_id: int
    first_tensor_name: str
    first_lifetime: tuple[int, int]
    second_tensor_id: int
    second_tensor_name: str
    second_lifetime: tuple[int, int]
    overlap_start: int
    overlap_end: int
    execution_order: tuple[int, int]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["first_lifetime"] = list(self.first_lifetime)
        result["second_lifetime"] = list(self.second_lifetime)
        result["execution_order"] = list(self.execution_order)
        return result


@dataclass(frozen=True)
class ReuseBlocker:
    tensor_id: int
    tensor_name: str
    lifetime: tuple[int, int]
    aligned_bytes: int
    overlapping_tensor_ids: tuple[int, ...]
    last_consumer_operator_id: int | None
    last_consumer_operator_name: str | None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["lifetime"] = list(self.lifetime)
        result["overlapping_tensor_ids"] = list(
            self.overlapping_tensor_ids
        )
        return result


def describe_reuse_blocker(blocker: ReuseBlocker) -> str:
    """Render the deterministic, plain-text explanation for one conservative
    reuse blocker.

    Shared by the text renderer, the HTML prose list, and the HTML arena
    chart's SVG tooltips, so the reasoning can never drift between them. SVG
    ``<title>`` elements cannot contain markup, so this returns plain text;
    callers that want HTML apply their own escaping.
    """

    name = blocker.tensor_name or "<unnamed>"
    overlapping = ", ".join(
        f"tensor[{tensor_id}]"
        for tensor_id in blocker.overlapping_tensor_ids
    )
    through = (
        f"operator {blocker.last_consumer_operator_id} "
        f"({blocker.last_consumer_operator_name})"
        if blocker.last_consumer_operator_id is not None
        else f"scope {blocker.lifetime[1]}"
    )
    return (
        f"tensor[{blocker.tensor_id}] {name} ({blocker.aligned_bytes:,} aligned bytes) "
        f"remains live through {through}. Its lifetime "
        f"{blocker.lifetime[0]}..{blocker.lifetime[1]} overlaps with {overlapping}, "
        "so those tensors cannot reuse the same memory interval."
    )


@dataclass(frozen=True)
class MemoryExplanation:
    summary: MemorySummary
    largest_tensors: tuple[TensorExplanation, ...]
    peak: PeakExplanation
    scopes: tuple[ScopeExplanation, ...]
    allocations: tuple[TensorExplanation, ...]
    live_tensors_at_peak: tuple[TensorExplanation, ...]
    reuse: tuple[ReuseRelationship, ...]
    reuse_blockers: tuple[ReuseBlocker, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary.to_dict(),
            "largest_tensors": [item.to_dict() for item in self.largest_tensors],
            "peak": self.peak.to_dict(),
            "scopes": [item.to_dict() for item in self.scopes],
            "allocations": [item.to_dict() for item in self.allocations],
            "live_tensors_at_peak": [
                item.to_dict() for item in self.live_tensors_at_peak
            ],
            "reuse": [item.to_dict() for item in self.reuse],
            "reuse_blockers": [item.to_dict() for item in self.reuse_blockers],
        }


def _scope_context(subgraph: Subgraph, scope: int) -> tuple[str, int | None, str | None]:
    if scope == 0:
        return "subgraph_input", None, None
    operator = subgraph.operator(scope - 1)
    return "operator", operator.id, operator.name


def _tensor_explanations(
    subgraph: Subgraph,
    plan: SubgraphMemoryPlan,
) -> tuple[TensorExplanation, ...]:
    input_ids = set(subgraph.inputs)
    output_ids = set(subgraph.outputs)
    result: list[TensorExplanation] = []
    for allocation in sorted(plan.allocations, key=lambda item: item.tensor_id):
        tensor = subgraph.tensor(allocation.tensor_id)
        result.append(
            TensorExplanation(
                tensor_id=tensor.id,
                name=tensor.name,
                data_type=tensor.data_type.name,
                shape=tensor.shape,
                logical_bytes=allocation.logical_size,
                aligned_bytes=allocation.aligned_size,
                alignment_overhead_bytes=(
                    allocation.aligned_size - allocation.logical_size
                ),
                offset=allocation.offset,
                end_offset=allocation.end_offset,
                first_used_scope=allocation.first_used,
                last_used_scope=allocation.last_used,
                lifetime_length=(allocation.last_used - allocation.first_used + 1),
                is_graph_input=tensor.id in input_ids,
                is_graph_output=tensor.id in output_ids,
            )
        )
    return tuple(result)


def _scope_explanations(
    subgraph: Subgraph,
    lifetimes: SubgraphLifetimeAnalysis,
    plan: SubgraphMemoryPlan,
) -> tuple[ScopeExplanation, ...]:
    result: list[ScopeExplanation] = []
    for scope in range(lifetimes.operator_scope_count + 1):
        live = sorted(
            plan.live_allocations_at(scope),
            key=lambda item: (item.offset, item.end_offset, item.tensor_id),
        )
        kind, operator_id, operator_name = _scope_context(subgraph, scope)
        result.append(
            ScopeExplanation(
                scope=scope,
                scope_kind=kind,
                operator_id=operator_id,
                operator_name=operator_name,
                occupied_extent_bytes=max(
                    (allocation.end_offset for allocation in live), default=0
                ),
                live_aligned_bytes=sum(
                    allocation.aligned_size for allocation in live
                ),
                live_tensor_ids=tuple(item.tensor_id for item in live),
            )
        )
    return tuple(result)


def _reuse_relationships(
    allocations: tuple[TensorExplanation, ...],
) -> tuple[ReuseRelationship, ...]:
    result: list[ReuseRelationship] = []
    for index, left in enumerate(allocations):
        for right in allocations[index + 1 :]:
            overlap_start = max(left.offset, right.offset)
            overlap_end = min(left.end_offset, right.end_offset)
            memory_overlaps = overlap_start < overlap_end
            time_overlaps = not (
                left.last_used_scope < right.first_used_scope
                or right.last_used_scope < left.first_used_scope
            )
            if not memory_overlaps or time_overlaps:
                continue
            first, second = sorted(
                (left, right),
                key=lambda item: (item.first_used_scope, item.tensor_id),
            )
            result.append(
                ReuseRelationship(
                    first_tensor_id=first.tensor_id,
                    first_tensor_name=first.name,
                    first_lifetime=(first.first_used_scope, first.last_used_scope),
                    second_tensor_id=second.tensor_id,
                    second_tensor_name=second.name,
                    second_lifetime=(
                        second.first_used_scope,
                        second.last_used_scope,
                    ),
                    overlap_start=overlap_start,
                    overlap_end=overlap_end,
                    execution_order=(first.tensor_id, second.tensor_id),
                )
            )
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.overlap_start,
                item.overlap_end,
                item.first_tensor_id,
                item.second_tensor_id,
            ),
        )
    )


def _last_consumer(
    subgraph: Subgraph,
    tensor: TensorExplanation,
) -> tuple[int | None, str | None]:
    if tensor.last_used_scope == 0:
        return None, None
    operator = subgraph.operator(tensor.last_used_scope - 1)
    if tensor.tensor_id not in operator.inputs:
        return None, None
    return operator.id, operator.name


def _reuse_blockers(
    subgraph: Subgraph,
    allocations: tuple[TensorExplanation, ...],
) -> tuple[ReuseBlocker, ...]:
    result: list[ReuseBlocker] = []
    for tensor in allocations:
        overlapping = tuple(
            other.tensor_id
            for other in allocations
            if other.tensor_id != tensor.tensor_id
            and not (
                tensor.last_used_scope < other.first_used_scope
                or other.last_used_scope < tensor.first_used_scope
            )
        )
        if not overlapping:
            continue
        operator_id, operator_name = _last_consumer(subgraph, tensor)
        result.append(
            ReuseBlocker(
                tensor_id=tensor.tensor_id,
                tensor_name=tensor.name,
                lifetime=(tensor.first_used_scope, tensor.last_used_scope),
                aligned_bytes=tensor.aligned_bytes,
                overlapping_tensor_ids=overlapping,
                last_consumer_operator_id=operator_id,
                last_consumer_operator_name=operator_name,
            )
        )
    return tuple(
        sorted(
            result,
            key=lambda item: (-item.aligned_bytes, item.tensor_id),
        )
    )


def explain_primary_subgraph_memory(
    graph: GraphModel,
    *,
    lifetimes: GraphLifetimeAnalysis | None = None,
    memory_plan: GraphMemoryPlan | None = None,
    largest_limit: int = 10,
) -> MemoryExplanation:
    """Explain the existing arena-head plan for the primary subgraph."""

    if largest_limit < 0:
        raise ValueError("Largest-tensor limit must be non-negative")
    if lifetimes is None:
        lifetimes = calculate_graph_lifetimes(graph)
    if memory_plan is None:
        memory_plan = calculate_graph_memory_plan(graph, lifetimes)

    subgraph = graph.primary_subgraph
    lifetime_analysis = lifetimes.primary_subgraph
    plan = memory_plan.primary_subgraph
    allocations = _tensor_explanations(subgraph, plan)
    scopes = _scope_explanations(subgraph, lifetime_analysis, plan)
    peak_extent = max((scope.occupied_extent_bytes for scope in scopes), default=0)
    tied_scopes = tuple(
        scope.scope for scope in scopes if scope.occupied_extent_bytes == peak_extent
    )
    selected_scope = next(scope for scope in scopes if scope.scope == tied_scopes[0])
    by_id = {item.tensor_id: item for item in allocations}
    live_at_peak = tuple(
        sorted(
            (by_id[tensor_id] for tensor_id in selected_scope.live_tensor_ids),
            key=lambda item: (item.offset, item.end_offset, item.tensor_id),
        )
    )
    largest = tuple(
        sorted(
            allocations,
            key=lambda item: (-item.aligned_bytes, -item.logical_bytes, item.tensor_id),
        )[:largest_limit]
    )
    logical_total = sum(item.logical_bytes for item in allocations)
    aligned_total = sum(item.aligned_bytes for item in allocations)
    return MemoryExplanation(
        summary=MemorySummary(
            runtime_tensor_count=len(allocations),
            constant_tensor_count=sum(
                1 for tensor in subgraph.tensors if tensor.has_constant_data
            ),
            constant_tensor_bytes=sum(
                tensor.constant_data_size for tensor in subgraph.tensors if tensor.has_constant_data
            ),
            operator_count=len(subgraph.operators),
            planned_arena_head_bytes=plan.maximum_memory_size,
            arena_alignment_bytes=plan.alignment,
            logical_runtime_tensor_bytes=logical_total,
            aligned_runtime_tensor_bytes=aligned_total,
            alignment_overhead_bytes=aligned_total - logical_total,
        ),
        largest_tensors=largest,
        peak=PeakExplanation(
            scope=selected_scope.scope,
            tied_scopes=tied_scopes,
            scope_kind=selected_scope.scope_kind,
            operator_id=selected_scope.operator_id,
            operator_name=selected_scope.operator_name,
            occupied_extent_bytes=selected_scope.occupied_extent_bytes,
            live_aligned_bytes=selected_scope.live_aligned_bytes,
            live_tensor_ids=selected_scope.live_tensor_ids,
        ),
        scopes=scopes,
        allocations=allocations,
        live_tensors_at_peak=live_at_peak,
        reuse=_reuse_relationships(allocations),
        reuse_blockers=_reuse_blockers(subgraph, allocations),
    )
