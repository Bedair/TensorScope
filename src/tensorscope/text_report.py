from __future__ import annotations

from tensorscope.explain import MemoryExplanation, TensorExplanation


DEFAULT_TABLE_LIMIT = 10
DEFAULT_NAME_WIDTH = 20
DEFAULT_ASCII_NAME_WIDTH = 12


def _name(value: str, width: int = DEFAULT_NAME_WIDTH) -> str:
    rendered = value or "<unnamed>"
    if len(rendered) <= width:
        return rendered
    if width <= 3:
        return rendered[:width]
    return f"{rendered[: width - 3]}..."


def _shape(shape: tuple[int, ...]) -> str:
    return "[" + ",".join(str(dimension) for dimension in shape) + "]"


def _scope_label(
    scope: int,
    scope_kind: str,
    operator_id: int | None,
    operator_name: str | None,
) -> str:
    if scope_kind == "subgraph_input":
        return f"scope {scope} (subgraph input)"
    return f"scope {scope} (operator {operator_id}: {operator_name or '<unnamed>'})"


def _tensor_table(
    tensors: tuple[TensorExplanation, ...],
    *,
    include_offset: bool,
) -> list[str]:
    if not tensors:
        return ["  (none)"]
    if include_offset:
        lines = [
            "  Tensor  Name                  Offset       End   Logical   Aligned  Lifetime"
        ]
        for item in tensors:
            lines.append(
                f"  {item.tensor_id:>6}  {_name(item.name):<20}  "
                f"{item.offset:>10,}  {item.end_offset:>8,}  "
                f"{item.logical_bytes:>8,}  {item.aligned_bytes:>8,}  "
                f"{item.first_used_scope}..{item.last_used_scope}"
            )
        return lines
    lines = [
        "  Tensor  Name                  Type       Shape          Logical   Aligned  Overhead"
    ]
    for item in tensors:
        lines.append(
            f"  {item.tensor_id:>6}  {_name(item.name):<20}  "
            f"{item.data_type:<9}  {_shape(item.shape):<13}  "
            f"{item.logical_bytes:>8,}  {item.aligned_bytes:>8,}  "
            f"{item.alignment_overhead_bytes:>8,}"
        )
    return lines


def render_ascii_packing(
    explanation: MemoryExplanation,
    *,
    limit: int | None = DEFAULT_TABLE_LIMIT,
    name_width: int = DEFAULT_ASCII_NAME_WIDTH,
) -> str:
    """Render deterministic allocation regions and safe reuse chains."""

    allocations = tuple(
        sorted(
            explanation.allocations,
            key=lambda item: (item.offset, item.end_offset, item.tensor_id),
        )
    )
    shown = allocations if limit is None else allocations[:limit]
    lines = [
        f"Arena-head packing: {explanation.summary.planned_arena_head_bytes:,} bytes",
        "  Offset range             Tensor                    Lifetime",
    ]
    if not shown:
        lines.append("  (no runtime allocations)")
    for item in shown:
        tensor_label = f"tensor[{item.tensor_id}] {_name(item.name, name_width)}"
        lines.append(
            f"  [{item.offset:>8,}, {item.end_offset:>8,})  "
            f"{tensor_label:<26}  {item.first_used_scope}..{item.last_used_scope}"
        )
    if len(shown) < len(allocations):
        lines.append(f"  ... {len(allocations) - len(shown)} more allocations (use --details)")

    lines.append("Memory reuse:")
    reuse = explanation.reuse if limit is None else explanation.reuse[:limit]
    if not reuse:
        lines.append("  (no overlapping memory regions reused)")
    for item in reuse:
        lines.append(
            f"  [{item.overlap_start:,}, {item.overlap_end:,}): "
            f"tensor[{item.first_tensor_id}] -> tensor[{item.second_tensor_id}] "
            f"(lifetimes {item.first_lifetime[0]}..{item.first_lifetime[1]} -> "
            f"{item.second_lifetime[0]}..{item.second_lifetime[1]})"
        )
    if len(reuse) < len(explanation.reuse):
        lines.append(f"  ... {len(explanation.reuse) - len(reuse)} more reuse pairs")
    return "\n".join(lines)


def render_memory_explanation(
    explanation: MemoryExplanation,
    *,
    details: bool = False,
    include_ascii: bool = True,
) -> str:
    """Render an arena-head explanation without terminal-specific styling."""

    summary = explanation.summary
    peak = explanation.peak
    lines = [
        "",
        "Memory explanation",
        "This report covers planned arena head only.",
        "",
        "Model summary:",
        f"  Runtime tensors planned: {summary.runtime_tensor_count:,}",
        f"  Constant tensors: {summary.constant_tensor_count:,}",
        f"  Operators: {summary.operator_count:,}",
        f"  Planned arena head: {summary.planned_arena_head_bytes:,} bytes",
        f"  Arena alignment: {summary.arena_alignment_bytes:,} bytes",
        f"  Sum of logical runtime-tensor sizes: {summary.logical_runtime_tensor_bytes:,} bytes",
        f"  Sum of aligned runtime-tensor sizes: {summary.aligned_runtime_tensor_bytes:,} bytes",
        f"  Per-tensor alignment overhead: {summary.alignment_overhead_bytes:,} bytes",
        "  Tensor-size sums may exceed planned head because safe regions are reused.",
        "",
        f"Largest tensors (top {len(explanation.largest_tensors)}):",
    ]
    lines.extend(_tensor_table(explanation.largest_tensors, include_offset=False))
    lines.extend(
        [
            "",
            "Peak execution point:",
            "  "
            + _scope_label(
                peak.scope,
                peak.scope_kind,
                peak.operator_id,
                peak.operator_name,
            ),
            f"  Occupied arena extent: {peak.occupied_extent_bytes:,} bytes",
            f"  Sum of live aligned tensor sizes: {peak.live_aligned_bytes:,} bytes",
            f"  Tied peak scopes: {', '.join(str(scope) for scope in peak.tied_scopes)}",
            "",
            "Live tensors at peak:",
        ]
    )
    lines.extend(_tensor_table(explanation.live_tensors_at_peak, include_offset=True))

    ordered_allocations = tuple(
        sorted(
            explanation.allocations,
            key=lambda item: (item.offset, item.end_offset, item.tensor_id),
        )
    )
    table_limit = None if details else DEFAULT_TABLE_LIMIT
    shown_allocations = (
        ordered_allocations
        if table_limit is None
        else ordered_allocations[:table_limit]
    )
    lines.extend(["", "Packing table:"])
    lines.extend(_tensor_table(shown_allocations, include_offset=True))
    if len(shown_allocations) < len(ordered_allocations):
        lines.append(
            f"  ... {len(ordered_allocations) - len(shown_allocations)} more "
            "allocations (use --details)"
        )

    lines.extend(
        [
            "",
            "Reuse summary:",
            f"  Safe memory-overlap reuse pairs: {len(explanation.reuse):,}",
            f"  Tensors with overlapping lifetimes: {len(explanation.reuse_blockers):,}",
        ]
    )
    if details:
        lines.append("Reuse blockers (conservative):")
        if not explanation.reuse_blockers:
            lines.append("  (none)")
        for blocker in explanation.reuse_blockers:
            consumers = ", ".join(
                f"tensor[{tensor_id}]"
                for tensor_id in blocker.overlapping_tensor_ids
            )
            through = (
                f"operator {blocker.last_consumer_operator_id} "
                f"({blocker.last_consumer_operator_name})"
                if blocker.last_consumer_operator_id is not None
                else f"scope {blocker.lifetime[1]}"
            )
            lines.append(
                f"  Tensor {blocker.tensor_id} ({_name(blocker.tensor_name)}) "
                f"remains live through {through}; its lifetime "
                f"{blocker.lifetime[0]}..{blocker.lifetime[1]} overlaps in time "
                f"with {consumers}, so those tensors cannot reuse the same "
                "memory interval."
            )
    if include_ascii:
        lines.extend(
            [
                "",
                render_ascii_packing(
                    explanation,
                    limit=None if details else DEFAULT_TABLE_LIMIT,
                ),
            ]
        )
    return "\n".join(lines)
