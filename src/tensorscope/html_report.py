from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import os
from pathlib import Path
import tempfile

from tensorscope.compute_cost import ComputeCostSummary, OperatorComputeCost, render_compute_cost_caveat
from tensorscope.explain import (
    MemoryExplanation,
    ReuseBlocker,
    ReuseRelationship,
    TensorExplanation,
    describe_reuse_blocker,
)
from tensorscope.memory_budget import (
    BUDGET_STATUS_LABELS,
    ArenaHeadBudgetResult,
    render_budget_source_label,
    render_budget_verdict,
)
from tensorscope.recommendations import MemoryRiskAssessment


class HTMLReportError(OSError):
    """Raised when a rendered report cannot be written."""


def _text(value: object) -> str:
    return escape(str(value), quote=True)


def _name(value: str) -> str:
    return _text(value or "<unnamed>")


def _bytes(value: int | None) -> str:
    return "Not estimated" if value is None else f"{value:,} bytes"


def _shape(value: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(dimension) for dimension in value) + "]"


def _scope_context(
    scope: int,
    kind: str,
    operator_id: int | None,
    operator_name: str | None,
) -> str:
    if kind == "subgraph_input":
        return f"Scope {scope} · subgraph input"
    return (
        f"Scope {scope} · operator {operator_id} · "
        f"{operator_name or '<unnamed>'}"
    )


#
# validate_palette.js (dataviz skill), --mode light, adjacent pairs: all
# checks PASS. Slots 6 and 8 were nudged from their original values --
# #237b9f -> #127ca3 (chroma only, hue/lightness held) and #8d4b35 ->
# #82371d (hue held; lightness had to move too since raising chroma alone
# at the original hue made its CVD separation from slot 7 worse, not
# better) -- to clear the OKLCH chroma floor and the adjacent-pair CVD/
# normal-vision separation threshold (issue #39). The other six slots are
# unchanged.
_SVG_PALETTE = (
    "#3257d5",
    "#00876c",
    "#9c4f00",
    "#7b4ab5",
    "#b3345c",
    "#127ca3",
    "#6b6f00",
    "#82371d",
)


def _short_name(value: str, limit: int = 18) -> str:
    rendered = value or "<unnamed>"
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."


def _timestamp(value: datetime | None) -> str:
    generated = datetime.now(timezone.utc) if value is None else value
    if generated.tzinfo is None or generated.utcoffset() is None:
        raise ValueError("Report timestamp must include an explicit timezone")
    utc = generated.astimezone(timezone.utc).replace(microsecond=0)
    return utc.isoformat().replace("+00:00", "Z")


def _reuse_handoff_title(relationship: ReuseRelationship) -> str:
    return (
        f"tensor[{relationship.second_tensor_id}] reuses tensor[{relationship.first_tensor_id}]'s "
        f"memory interval [{relationship.overlap_start:,}, {relationship.overlap_end:,}) -- "
        f"tensor[{relationship.first_tensor_id}]'s lifetime ends at scope {relationship.first_lifetime[1]}, "
        f"tensor[{relationship.second_tensor_id}] does not start until scope {relationship.second_lifetime[0]}, "
        "so the planner safely hands the freed slot over."
    )


def render_packing_svg(explanation: MemoryExplanation) -> str:
    """Render scope-by-offset tensor placement as a responsive inline SVG.

    Beyond placement, this marks two things the planner already knows:
    dashed hand-off markers between ``explanation.reuse`` pairs (a chevron
    where the columns are too close together for a legible line), and a
    small badge on every ``explanation.reuse_blockers`` tensor whose tooltip
    and click-to-jump link explain why it could not share a slot.
    """

    width = 1000
    height = 620
    left = 105
    right = 30
    top = 45
    bottom = 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    scope_count = max(len(explanation.scopes), 1)
    column_width = plot_width / scope_count
    arena_bytes = explanation.summary.planned_arena_head_bytes
    denominator = max(arena_bytes, 1)
    allocations = sorted(
        explanation.allocations,
        key=lambda item: (
            item.first_used_scope,
            item.last_used_scope,
            item.offset,
            item.tensor_id,
        ),
    )
    blockers_by_tensor_id: dict[int, ReuseBlocker] = {
        blocker.tensor_id: blocker for blocker in explanation.reuse_blockers
    }

    elements = [
        '<svg id="arena-packing-svg" class="packing-svg" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="arena-packing-title arena-packing-description">',
        '<title id="arena-packing-title">Arena placement across execution scopes</title>',
        '<desc id="arena-packing-description">Each tensor rectangle spans its inclusive execution lifetime horizontally and its planned arena-head byte interval vertically. A dashed marker between two rectangles shows a safe reuse hand-off; a badge on a rectangle shows it could not share a slot with an overlapping tensor.</desc>',
        '<defs><marker id="reuse-arrowhead-marker" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" class="reuse-arrowhead" /></marker></defs>',
        f'<rect class="plot-background" x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" />',
    ]

    for scope in range(scope_count + 1):
        x = left + scope * column_width
        elements.append(
            f'<line class="grid-line" x1="{x:.3f}" y1="{top}" x2="{x:.3f}" y2="{top + plot_height}" />'
        )
    for scope in explanation.scopes:
        x = left + (scope.scope + 0.5) * column_width
        elements.append(
            f'<text class="axis-tick scope-tick" x="{x:.3f}" y="{top + plot_height + 24}" text-anchor="middle">{scope.scope}</text>'
        )

    offset_ticks = 5
    for index in range(offset_ticks + 1):
        value = round(arena_bytes * index / offset_ticks) if arena_bytes else 0
        y = top + plot_height * index / offset_ticks
        elements.append(
            f'<line class="grid-line" x1="{left}" y1="{y:.3f}" x2="{left + plot_width}" y2="{y:.3f}" />'
        )
        elements.append(
            f'<text class="axis-tick offset-tick" x="{left - 10}" y="{y + 4:.3f}" text-anchor="end">{value:,}</text>'
        )

    rects: dict[int, tuple[float, float, float, float]] = {}
    if allocations:
        for tensor in allocations:
            x = left + tensor.first_used_scope * column_width + 2
            rectangle_width = (
                (tensor.last_used_scope - tensor.first_used_scope + 1)
                * column_width
                - 4
            )
            y = top + tensor.offset * plot_height / denominator
            rectangle_height = tensor.aligned_bytes * plot_height / denominator
            rects[tensor.tensor_id] = (x, y, rectangle_width, rectangle_height)
            color = _SVG_PALETTE[tensor.tensor_id % len(_SVG_PALETTE)]
            title = (
                f"tensor[{tensor.tensor_id}] {tensor.name or '<unnamed>'}; "
                f"lifetime {tensor.first_used_scope}..{tensor.last_used_scope}; "
                f"memory [{tensor.offset}, {tensor.end_offset}) bytes"
            )
            blocker = blockers_by_tensor_id.get(tensor.tensor_id)
            if blocker is not None:
                title = f"{title} -- reuse-blocked: {describe_reuse_blocker(blocker)}"

            wrap_open, wrap_close = "", ""
            if blocker is not None:
                wrap_open = f'<a href="#reuse-blocker-{tensor.tensor_id}" class="blocked-link">'
                wrap_close = "</a>"

            elements.append(
                f'{wrap_open}<g id="tensor-allocation-{tensor.tensor_id}" class="tensor-allocation{" is-blocked" if blocker is not None else ""}" data-tensor-id="{tensor.tensor_id}">'
                f'<rect class="tensor-rect" x="{x:.3f}" y="{y:.3f}" width="{max(rectangle_width, 1):.3f}" height="{max(rectangle_height, 1):.3f}" fill="{color}" stroke="#172033" stroke-width="1">'
                f'<title>{_text(title)}</title></rect>'
                f'<text class="tensor-label" x="{x + 5:.3f}" y="{y + min(max(rectangle_height, 14), 22):.3f}">#{tensor.tensor_id} {_text(_short_name(tensor.name))}</text>'
            )
            if blocker is not None:
                badge_size = 9.0
                badge_x = x + max(rectangle_width - badge_size - 2, 2)
                badge_y = y + 3
                elements.append(
                    f'<polygon class="blocker-badge" points='
                    f'"{badge_x:.3f},{badge_y + badge_size:.3f} '
                    f'{badge_x + badge_size / 2:.3f},{badge_y:.3f} '
                    f'{badge_x + badge_size:.3f},{badge_y + badge_size:.3f}">'
                    f'<title>{_text("Reuse-blocked: " + describe_reuse_blocker(blocker))}</title>'
                    "</polygon>"
                )
            elements.append(f"</g>{wrap_close}")
    else:
        elements.append(
            f'<text class="empty-svg" x="{left + plot_width / 2:.3f}" y="{top + plot_height / 2:.3f}" text-anchor="middle">No runtime allocations</text>'
        )

    for relationship in explanation.reuse:
        first_rect = rects.get(relationship.first_tensor_id)
        second_rect = rects.get(relationship.second_tensor_id)
        if first_rect is None or second_rect is None:
            continue
        first_x, _, first_width, _ = first_rect
        second_x, _, _, _ = second_rect
        y_mid = top + (
            (relationship.overlap_start + relationship.overlap_end)
            / 2
            * plot_height
            / denominator
        )
        gap_start = first_x + first_width
        gap_end = second_x
        title = _text(_reuse_handoff_title(relationship))
        if gap_end - gap_start > 20:
            elements.append(
                f'<line class="reuse-arrow" x1="{gap_start:.3f}" y1="{y_mid:.3f}" '
                f'x2="{gap_end - 3:.3f}" y2="{y_mid:.3f}" marker-end="url(#reuse-arrowhead-marker)">'
                f"<title>{title}</title></line>"
            )
        else:
            # Adjacent columns leave too little room for a legible line -- a
            # fixed-size hand-off chevron stays visible regardless of gap width.
            mid_x = (gap_start + gap_end) / 2
            elements.append(
                f'<g class="reuse-handoff"><title>{title}</title>'
                f'<circle cx="{mid_x:.3f}" cy="{y_mid:.3f}" r="7" />'
                f'<path d="M {mid_x - 2.5:.3f} {y_mid - 3.5:.3f} L {mid_x + 2.5:.3f} {y_mid:.3f} '
                f'L {mid_x - 2.5:.3f} {y_mid + 3.5:.3f}" class="reuse-chevron" /></g>'
            )

    peak_x = left + (explanation.peak.scope + 0.5) * column_width
    elements.extend(
        [
            f'<line id="peak-scope-marker" class="peak-marker" x1="{peak_x:.3f}" y1="{top}" x2="{peak_x:.3f}" y2="{top + plot_height}" />',
            f'<text class="peak-label" x="{peak_x:.3f}" y="{top - 12}" text-anchor="middle">selected peak scope {explanation.peak.scope}</text>',
            f'<line id="arena-head-boundary" class="head-boundary" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" />',
            f'<text class="boundary-label" x="{left + plot_width}" y="{top + plot_height - 7}" text-anchor="end">planned arena head: {arena_bytes:,} bytes</text>',
            f'<text class="axis-label" x="{left + plot_width / 2:.3f}" y="{height - 18}" text-anchor="middle">Execution scope →</text>',
            f'<text class="axis-label" transform="translate(22 {top + plot_height / 2:.3f}) rotate(-90)" text-anchor="middle">Arena offset (bytes) →</text>',
            "</svg>",
        ]
    )
    return "".join(elements)


def _highlight_badges(tensor_id: int, peak_ids: frozenset[int], largest_ids: frozenset[int]) -> str:
    badges = []
    if tensor_id in peak_ids:
        badges.append('<span class="tag tag-peak">PEAK</span>')
    if tensor_id in largest_ids:
        badges.append('<span class="tag tag-largest">LARGEST</span>')
    return "".join(badges) or "—"


def _tensor_rows(
    tensors: tuple[TensorExplanation, ...],
    *,
    peak_ids: frozenset[int] = frozenset(),
    largest_ids: frozenset[int] = frozenset(),
) -> str:
    if not tensors:
        return '<tr><td colspan="13" class="empty">No runtime tensors</td></tr>'
    rows: list[str] = []
    for tensor in tensors:
        roles = ", ".join(
            role
            for role, enabled in (
                ("input", tensor.is_graph_input),
                ("output", tensor.is_graph_output),
            )
            if enabled
        ) or "—"
        rows.append(
            "<tr>"
            f"<td>{tensor.tensor_id}</td>"
            f"<td class=\"name\">{_name(tensor.name)}</td>"
            f"<td>{_text(tensor.data_type)}</td>"
            f"<td>{_text(_shape(tensor.shape))}</td>"
            f"<td>{tensor.logical_bytes:,}</td>"
            f"<td>{tensor.aligned_bytes:,}</td>"
            f"<td>{tensor.alignment_overhead_bytes:,}</td>"
            f"<td>{tensor.offset:,}</td>"
            f"<td>{tensor.end_offset:,}</td>"
            f"<td>{tensor.first_used_scope}..{tensor.last_used_scope}</td>"
            f"<td>{tensor.lifetime_length}</td>"
            f"<td>{roles}</td>"
            f"<td class=\"tags\">{_highlight_badges(tensor.tensor_id, peak_ids, largest_ids)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _tensor_table(
    tensors: tuple[TensorExplanation, ...],
    *,
    peak_ids: frozenset[int] = frozenset(),
    largest_ids: frozenset[int] = frozenset(),
) -> str:
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Tensor</th><th>Name</th><th>Type</th><th>Shape</th>"
        "<th>Logical</th><th>Aligned</th><th>Overhead</th>"
        "<th>Offset</th><th>End</th><th>Lifetime</th>"
        "<th>Scopes</th><th>Role</th><th>Highlights</th>"
        "</tr></thead><tbody>"
        + _tensor_rows(tensors, peak_ids=peak_ids, largest_ids=largest_ids)
        + "</tbody></table></div>"
    )


def _packing_rows(explanation: MemoryExplanation) -> str:
    arena_bytes = explanation.summary.planned_arena_head_bytes
    allocations = sorted(
        explanation.allocations,
        key=lambda item: (item.offset, item.end_offset, item.tensor_id),
    )
    if not allocations:
        return '<p class="empty">No runtime allocations.</p>'
    rows: list[str] = []
    for tensor in allocations:
        if arena_bytes:
            left = tensor.offset * 100 / arena_bytes
            width = tensor.aligned_bytes * 100 / arena_bytes
        else:
            left = 0.0
            width = 0.0
        rows.append(
            '<div class="packing-row">'
            '<div class="packing-label">'
            f"<strong>tensor[{tensor.tensor_id}]</strong> {_name(tensor.name)}"
            f"<span>[{tensor.offset:,}, {tensor.end_offset:,}) · "
            f"scope {tensor.first_used_scope}..{tensor.last_used_scope}</span>"
            "</div>"
            '<div class="packing-track" aria-label="arena allocation">'
            f'<div class="packing-block" style="left:{left:.6f}%;width:{width:.6f}%">'
            f"{tensor.tensor_id}</div></div></div>"
        )
    return "".join(rows)


def _reuse_rows(explanation: MemoryExplanation) -> str:
    if not explanation.reuse:
        return '<tr><td colspan="5" class="empty">No overlapping memory regions are reused.</td></tr>'
    return "".join(
        "<tr>"
        f"<td>tensor[{item.first_tensor_id}] {_name(item.first_tensor_name)}</td>"
        f"<td>{item.first_lifetime[0]}..{item.first_lifetime[1]}</td>"
        f"<td>[{item.overlap_start:,}, {item.overlap_end:,})</td>"
        f"<td>tensor[{item.second_tensor_id}] {_name(item.second_tensor_name)}</td>"
        f"<td>{item.second_lifetime[0]}..{item.second_lifetime[1]}</td>"
        "</tr>"
        for item in explanation.reuse
    )


def _blocker_items(explanation: MemoryExplanation) -> str:
    if not explanation.reuse_blockers:
        return '<p class="empty">No overlapping runtime lifetimes.</p>'
    items = [
        f'<li id="reuse-blocker-{blocker.tensor_id}">{_text(describe_reuse_blocker(blocker))}</li>'
        for blocker in explanation.reuse_blockers
    ]
    return '<ul class="blockers">' + "".join(items) + "</ul>"


def render_html_report(
    result: dict[str, object],
    explanation: MemoryExplanation,
    *,
    tool_version: str,
    generated_at: datetime | None = None,
    budget: ArenaHeadBudgetResult | None = None,
    guidance: MemoryRiskAssessment | None = None,
    target_clause: str | None = None,
    compute_cost: ComputeCostSummary | None = None,
) -> str:
    """Render one deterministic, dependency-free HTML analysis report.

    ``target_clause`` is only ever supplied when ``budget`` came from
    ``--target`` (a real, cited MCU/dev-kit); see
    tensorscope.target_profiles.render_target_verdict_clause. A generic
    --mcu-profile/--arena-head-budget report is unaffected.
    """

    head = result["arena_head"]
    tail = result["arena_tail"]
    total = result["arena_total"]
    assert isinstance(head, dict)
    assert isinstance(tail, dict)
    assert isinstance(total, dict)
    summary = explanation.summary
    peak = explanation.peak
    model_path = str(result["model_path"])
    model = result.get("model")
    if isinstance(model, dict):
        model_filename = str(model.get("filename", Path(model_path).name))
        schema_version = model.get("schema_version")
    else:
        model_filename = Path(model_path).name
        schema_version = None
    generated_timestamp = _timestamp(generated_at)
    tied = ", ".join(str(scope) for scope in peak.tied_scopes)
    peak_context = _scope_context(
        peak.scope,
        peak.scope_kind,
        peak.operator_id,
        peak.operator_name,
    )
    scope_rows = "".join(
        "<tr>"
        f"<td>{scope.scope}</td>"
        f"<td>{_text(_scope_context(scope.scope, scope.scope_kind, scope.operator_id, scope.operator_name))}</td>"
        f"<td>{scope.occupied_extent_bytes:,}</td>"
        f"<td>{scope.live_aligned_bytes:,}</td>"
        f"<td>{_text(', '.join(str(item) for item in scope.live_tensor_ids) or '—')}</td>"
        "</tr>"
        for scope in explanation.scopes
    )
    verdict_banner = _verdict_banner(budget, target_clause=target_clause) if budget is not None else ""
    budget_section = _budget_section(budget, target_clause=target_clause) if budget is not None else ""
    guidance_section = _guidance_section(guidance) if guidance is not None else ""
    compute_cost_section = _compute_cost_section(compute_cost) if compute_cost is not None else ""
    views_section = _analysis_views_section(result)
    peak_ids = frozenset(item.tensor_id for item in explanation.live_tensors_at_peak)
    largest_ids = frozenset(item.tensor_id for item in explanation.largest_tensors)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TensorScope analysis · {_text(model_filename)}</title>
<style>
:root {{ color-scheme: light; --ink:#172033; --muted:#5f6b7a; --line:#d8dee8; --paper:#fff; --panel:#f6f8fb; --accent:#3257d5; --good:#176b45; --warn:#8a4b08; --bad:#a51f31; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#eef1f6; color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:28px 18px 56px; }}
h1 {{ margin:0 0 4px; font-size:28px; }} h2 {{ margin:30px 0 12px; font-size:20px; }} h3 {{ margin:20px 0 8px; font-size:16px; }}
.subtitle,.muted {{ color:var(--muted); }} .subtitle {{ overflow-wrap:anywhere; }}
.lede {{ margin:16px 0 0; font-size:16px; max-width:70ch; }}
.notice {{ margin:16px 0 0; padding:14px 16px; border-left:4px solid var(--accent); background:#edf2ff; }}
.notice p {{ margin:3px 0; max-width:70ch; }}
.budget-status {{ display:inline-block; padding:4px 9px; border:2px solid currentColor; border-radius:4px; font-weight:750; }}
.verdict-banner {{ margin:16px 0 0; padding:16px 18px; border:2px solid currentColor; border-radius:8px; }}
.verdict-banner.verdict-fits {{ color:var(--good); background:#e9f7f1; }}
.verdict-banner.verdict-exact_fit {{ color:var(--warn); background:#fdf3e7; }}
.verdict-banner.verdict-exceeds {{ color:var(--bad); background:#fbe9ec; }}
.verdict-headline {{ display:flex; align-items:center; gap:10px; font-size:21px; font-weight:800; line-height:1.3; }}
.verdict-icon {{ font-size:24px; line-height:1; }}
.verdict-secondary {{ margin:6px 0 0 34px; font-size:13px; font-weight:500; opacity:.82; max-width:70ch; }}
.tag-peak {{ border-color:var(--accent); color:var(--accent); font-weight:700; }}
.tag-largest {{ border-color:var(--warn); color:var(--warn); font-weight:700; }}
details {{ margin-top:10px; }} details > summary {{ cursor:pointer; font-weight:650; color:var(--accent); padding:5px 0; }}
details[open] > summary {{ margin-bottom:6px; }}
.guidance-item {{ margin:12px 0; padding:13px; background:var(--panel); border-radius:6px; }} .guidance-item h3 {{ margin:0 0 5px; }}
.finding-summary {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin-top:20px; }}
.card,section {{ background:var(--paper); border:1px solid var(--line); border-radius:8px; }}
.card {{ padding:15px; }} .card .label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
.card .value {{ display:block; margin:4px 0; font-size:22px; font-weight:700; }}
.tags {{ display:flex; flex-wrap:wrap; gap:5px; }} .tag {{ border-radius:99px; background:var(--panel); border:1px solid var(--line); padding:2px 7px; font-size:12px; }}
section {{ margin-top:16px; padding:18px; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:8px 18px; margin:0; }}
.metrics div {{ padding:7px 0; border-bottom:1px solid var(--line); min-width:0; }} .metrics dt {{ color:var(--muted); }} .metrics dd {{ margin:1px 0 0; font-weight:650; overflow-wrap:anywhere; }}
.table-wrap {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
th,td {{ padding:8px 9px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap; }} th {{ background:var(--panel); font-size:12px; }} td.name {{ max-width:300px; overflow:hidden; text-overflow:ellipsis; }}
.empty {{ color:var(--muted); font-style:italic; }}
.peak {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; }} .peak div {{ background:var(--panel); padding:12px; border-radius:6px; }}
.packing-row {{ display:grid; grid-template-columns:minmax(220px,1fr) 2fr; gap:14px; align-items:center; margin:8px 0; }}
.packing-label {{ min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }} .packing-label span {{ display:block; color:var(--muted); font-size:12px; }}
.packing-track {{ position:relative; height:25px; border:1px solid var(--line); border-radius:4px; background:repeating-linear-gradient(90deg,#fafbfc 0,#fafbfc 9.8%,#e7ebf1 10%); overflow:hidden; }}
.packing-block {{ position:absolute; top:0; bottom:0; min-width:2px; display:flex; align-items:center; justify-content:center; overflow:hidden; color:#fff; background:var(--accent); font-size:11px; }}
.packing-svg {{ display:block; width:100%; height:auto; min-height:320px; border:1px solid var(--line); border-radius:6px; background:#fff; }}
.plot-background {{ fill:#fafbfc; stroke:var(--line); }} .grid-line {{ stroke:#dfe4ec; stroke-width:1; }}
.tensor-label {{ fill:#fff; font-size:11px; font-weight:700; pointer-events:none; paint-order:stroke; stroke:#172033; stroke-width:2px; stroke-linejoin:round; }}
.axis-tick {{ fill:var(--muted); font-size:11px; }} .axis-label {{ fill:var(--ink); font-size:13px; font-weight:650; }}
.peak-marker {{ stroke:#c22e3f; stroke-width:3; stroke-dasharray:8 5; }} .peak-label {{ fill:#a51f31; font-size:12px; font-weight:700; }}
.head-boundary {{ stroke:#172033; stroke-width:3; }} .boundary-label {{ fill:#172033; font-size:11px; font-weight:700; }} .empty-svg {{ fill:var(--muted); font-size:18px; }}
.tensor-rect {{ transition:opacity .12s; }} .tensor-allocation:hover .tensor-rect {{ opacity:.85; }}
.blocked-link {{ cursor:pointer; }} .blocker-badge {{ fill:var(--warn); stroke:#fff; stroke-width:1; }}
.reuse-arrow, .reuse-chevron {{ stroke:var(--good); stroke-width:2.4; fill:none; stroke-linecap:round; stroke-linejoin:round; }}
.reuse-arrow {{ stroke-dasharray:6 4; }} .reuse-arrowhead {{ fill:var(--good); }}
.reuse-handoff circle {{ fill:#fff; stroke:var(--good); stroke-width:2; }} .reuse-handoff:hover circle {{ fill:#e9f7f1; }}
.chart-legend {{ display:flex; flex-wrap:wrap; gap:16px; align-items:center; padding:9px 13px; background:var(--panel); border:1px solid var(--line); border-radius:6px; margin:0 0 12px; font-size:12.5px; }}
.chart-legend .item {{ display:flex; align-items:center; gap:6px; white-space:nowrap; }}
.chart-legend .swatch-reuse {{ width:18px; height:0; border-top:2px dashed var(--good); display:inline-block; }}
.chart-legend .swatch-blocked {{ width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent; border-bottom:9px solid var(--warn); display:inline-block; }}
.chart-legend .swatch-peak {{ width:0; height:14px; border-left:2px dashed #c22e3f; display:inline-block; }}
.chart-legend .swatch-boundary {{ width:18px; height:3px; background:#172033; display:inline-block; }}
figure {{ margin:12px 0 0; }} figcaption {{ margin-top:8px; color:var(--muted); }}
.blockers {{ padding-left:22px; }} .blockers li {{ margin:8px 0; scroll-margin-top:16px; }}
.blockers li:target {{ outline:2px solid var(--warn); outline-offset:3px; background:#fdf3e7; border-radius:4px; }}
footer {{ margin-top:24px; color:var(--muted); font-size:12px; }}
@media (max-width:680px) {{ main {{ padding:16px 10px 36px; }} .packing-row {{ grid-template-columns:1fr; gap:4px; }} section {{ padding:13px; }} }}
@media print {{ body {{ background:#fff; }} main {{ max-width:none; padding:0; }} section,.card {{ break-inside:avoid; }} }}
</style>
</head>
<body><main>
<header><h1>TensorScope arena-head analysis</h1><div class="subtitle"><strong>{_text(model_filename)}</strong><br>{_text(model_path)}</div></header>
<p class="lede">This report tells you whether <strong>{_text(model_filename)}</strong>'s planned runtime memory (arena head) fits your target's available RAM, and shows exactly why — tensor by tensor.</p>
<div class="notice" role="note">
<p><strong>Arena head is computed by exact, deterministic static analysis</strong> — not an estimate. Arena-head validation is not run by <code>analyze</code>; validation applies only when the TFLM oracle is run.</p>
<p>This report covers planned arena head only. Arena tail is not statically estimated. Complete arena total is not statically estimated.</p>
<p class="muted">Together, arena tail, firmware stack, and other non-model RAM sit outside this report's scope — on their own, this does not prove complete MCU or firmware fit.</p>
</div>
{verdict_banner}
<div class="cards">
<div class="card"><span class="label">Arena head</span><span class="value">{_bytes(head.get('bytes'))}</span><div class="tags"><span class="tag">scope: {_text(head.get('scope'))}</span><span class="tag">confidence: {_text(head.get('confidence'))}</span><span class="tag">source: {_text(head.get('source'))}</span><span class="tag">validation: {_text(head.get('validation_state'))}</span></div></div>
<div class="card"><span class="label">Arena tail</span><span class="value">{_bytes(tail.get('bytes'))}</span><div class="tags"><span class="tag">scope: {_text(tail.get('scope'))}</span><span class="tag">confidence: {_text(tail.get('confidence'))}</span><span class="tag">validation: {_text(tail.get('validation_state'))}</span></div></div>
<div class="card"><span class="label">Complete arena total</span><span class="value">{_bytes(total.get('bytes'))}</span><div class="tags"><span class="tag">scope: {_text(total.get('scope'))}</span><span class="tag">confidence: {_text(total.get('confidence'))}</span><span class="tag">validation: {_text(total.get('validation_state'))}</span></div></div>
</div>
{budget_section}
{guidance_section}
{compute_cost_section}
{views_section}
<section><h2>Report metadata</h2><dl class="metrics">
<div><dt>Model filename</dt><dd>{_text(model_filename)}</dd></div><div><dt>Model path</dt><dd>{_text(model_path)}</dd></div><div><dt>TFLite schema version</dt><dd>{_text(schema_version if schema_version is not None else 'Unavailable')}</dd></div><div><dt>Generated at</dt><dd>{_text(generated_timestamp)}</dd></div><div><dt>TensorScope version</dt><dd>{_text(tool_version)}</dd></div><div><dt>Analysis scope</dt><dd>Primary subgraph · planned arena head only</dd></div>
</dl></section>
<section><h2>Model summary</h2><dl class="metrics">
<div><dt>Runtime tensors planned</dt><dd>{summary.runtime_tensor_count:,}</dd></div><div><dt>Constant tensors</dt><dd>{summary.constant_tensor_count:,}</dd></div><div><dt>Operators</dt><dd>{summary.operator_count:,}</dd></div><div><dt>Planned arena head</dt><dd>{summary.planned_arena_head_bytes:,} bytes</dd></div><div><dt>Arena alignment</dt><dd>{summary.arena_alignment_bytes:,} bytes</dd></div><div><dt>Logical runtime-tensor sum</dt><dd>{summary.logical_runtime_tensor_bytes:,} bytes</dd></div><div><dt>Aligned runtime-tensor sum</dt><dd>{summary.aligned_runtime_tensor_bytes:,} bytes</dd></div><div><dt>Per-tensor alignment overhead</dt><dd>{summary.alignment_overhead_bytes:,} bytes</dd></div><div><dt>Safe reuse relationships</dt><dd>{len(explanation.reuse):,}</dd></div><div><dt>Conservative reuse blockers</dt><dd>{len(explanation.reuse_blockers):,}</dd></div>
</dl><p class="muted">Tensor-size sums are not the planned head: safely reused regions can reduce the plan.</p></section>
<section><h2>Peak execution point</h2><div class="peak"><div><strong>{_text(peak_context)}</strong></div><div>Occupied arena extent<br><strong>{peak.occupied_extent_bytes:,} bytes</strong></div><div>Live aligned tensor sum<br><strong>{peak.live_aligned_bytes:,} bytes</strong></div><div>Tied peak scopes<br><strong>{_text(tied)}</strong></div></div></section>
<section><h2>Runtime tensors</h2><p class="muted">Every planned runtime tensor; <span class="tag tag-peak">PEAK</span> marks a tensor live at the selected peak scope, <span class="tag tag-largest">LARGEST</span> marks one of the top {len(explanation.largest_tensors)} by aligned size.</p>{_tensor_table(explanation.allocations, peak_ids=peak_ids, largest_ids=largest_ids)}</section>
<section><h2>Arena placement across execution scopes</h2><div class="chart-legend"><span class="item"><span class="swatch-reuse"></span> safe reuse hand-off (hover for detail)</span><span class="item"><span class="swatch-blocked"></span> reuse-blocked (hover, or click to jump to why)</span><span class="item"><span class="swatch-peak"></span> selected peak scope</span><span class="item"><span class="swatch-boundary"></span> planned arena-head boundary</span></div><figure>{render_packing_svg(explanation)}<figcaption>Horizontal position represents inclusive execution lifetime; vertical position represents the planned arena-head memory interval. Tensors occupying the same memory interval at disjoint execution scopes represent safe reuse. Rectangle labels and outlines ensure the view does not rely on color alone.</figcaption></figure><details><summary>Compact offset view (alternate bar layout)</summary><p class="muted">Horizontal position and width below correspond to planned byte offsets.</p>{_packing_rows(explanation)}</details></section>
<section><h2>Execution scopes</h2><details><summary>Show all {len(explanation.scopes)} execution scopes</summary><div class="table-wrap"><table><thead><tr><th>Scope</th><th>Context</th><th>Occupied extent</th><th>Live aligned sum</th><th>Live tensor IDs</th></tr></thead><tbody>{scope_rows}</tbody></table></div></details></section>
<section><h2>Safe memory reuse</h2><details><summary>Show reuse table ({len(explanation.reuse)} pairs) and reuse blockers ({len(explanation.reuse_blockers)})</summary><div class="table-wrap"><table><thead><tr><th>Earlier tensor</th><th>Earlier lifetime</th><th>Overlap interval</th><th>Later tensor</th><th>Later lifetime</th></tr></thead><tbody>{_reuse_rows(explanation)}</tbody></table></div><h3>Conservative reuse blockers</h3>{_blocker_items(explanation)}</details></section>
<section id="limitations"><h2>Limitations</h2><ul>
<li>Analysis covers the primary subgraph and planned arena head only.</li>
<li>Arena tail is not statically estimated.</li>
<li>Complete arena total is not statically estimated.</li>
<li>Not estimated: Firmware stack usage, heap usage, DMA buffers, RTOS memory, Application memory.</li>
<li>The report does not establish complete MCU fit.</li>
<li>Reuse blockers are conservative explanations and do not represent proven counterfactual byte costs.</li>
<li>Results reflect the planner behavior implemented and pinned by TensorScope.</li>
</ul></section>
<footer>Generated by TensorScope {_text(tool_version)} at {_text(generated_timestamp)} · static report · no external assets</footer>
</main></body></html>
"""


_BUDGET_STATUS_ICONS = {"fits": "✓", "exact_fit": "⚠", "exceeds": "✕"}


def _verdict_banner(budget: ArenaHeadBudgetResult, *, target_clause: str | None = None) -> str:
    """Render the FITS/EXACT FIT/EXCEEDS verdict as a two-tier banner above the
    always-visible summary cards: a bold headline (status, byte fraction,
    target) and a smaller secondary line (citation, a compact tail note).

    ``render_budget_verdict()`` remains the single source of truth for the
    full sentence used in JSON and text output -- unchanged here. The
    headline is built from the same typed ``budget``/``target_clause``
    fields it uses internally (same numbers, same status label), just laid
    out in two tiers instead of one run-on sentence. The full canonical
    sentence is still carried verbatim in this banner's ``aria-label``, so
    it stays intact for assistive tech and for anything (like a screenshot
    crop) that only sees this element.
    """

    label = BUDGET_STATUS_LABELS[budget.status]
    icon = _BUDGET_STATUS_ICONS[budget.status]
    headline = f"{label} — {budget.planned_arena_head_bytes:,} / {budget.effective_budget_bytes:,} bytes"
    citation = ""
    if target_clause:
        on_part, _, citation = target_clause.partition(", ")
        headline = f"{headline} {on_part}"
    secondary_bits = [citation] if citation else []
    secondary_bits.append("tail not estimated — run `tensorscope validate`")
    secondary = "; ".join(secondary_bits)
    full_verdict = render_budget_verdict(budget, target_clause=target_clause)
    return (
        f'<div class="verdict-banner verdict-{budget.status}" role="note" '
        f'aria-label="Arena-head budget result: {_text(full_verdict)}">'
        f'<div class="verdict-headline"><span class="verdict-icon" aria-hidden="true">{icon}</span>'
        f'<span>{_text(headline)}</span></div>'
        f'<p class="verdict-secondary">{_text(secondary)}</p>'
        "</div>"
    )


def _budget_section(budget: ArenaHeadBudgetResult, *, target_clause: str | None = None) -> str:
    profile_fields = ""
    if budget.profile_name is not None:
        profile_fields = (
            f"<div><dt>Profile</dt><dd>{_text(budget.profile_name)} ({_text(budget.profile_id)})</dd></div>"
            f"<div><dt>Profile RAM</dt><dd>{budget.profile_ram_bytes:,} bytes</dd></div>"
        )
    if budget.remaining_bytes >= 0:
        difference = f"<div><dt>Remaining budget</dt><dd>{budget.remaining_bytes:,} bytes</dd></div>"
    else:
        difference = f"<div><dt>Exceeded by</dt><dd>{-budget.remaining_bytes:,} bytes</dd></div>"
    utilization = (
        "Not defined for a zero-byte budget"
        if budget.utilization_percent is None
        else f"{budget.utilization_percent:.2f}%"
    )
    source = render_budget_source_label(budget, target_clause=target_clause)
    return (
        '<section id="arena-head-budget"><h2>Arena-head budget details</h2>'
        '<dl class="metrics">'
        f"<div><dt>Budget source</dt><dd>{source}</dd></div>"
        f"{profile_fields}"
        f"<div><dt>Reserved RAM</dt><dd>{budget.reserve_bytes:,} bytes</dd></div>"
        f"<div><dt>Effective arena-head budget</dt><dd>{budget.effective_budget_bytes:,} bytes</dd></div>"
        f"<div><dt>Planned arena head</dt><dd>{budget.planned_arena_head_bytes:,} bytes</dd></div>"
        f"{difference}"
        f"<div><dt>Utilization</dt><dd>{utilization}</dd></div>"
        f"<div><dt>Scope</dt><dd>{_text(budget.scope)}</dd></div>"
        "</dl>"
        "<p><strong>This check covers planned arena head only.</strong></p>"
        "<p>This is not a complete MCU or firmware memory-fit conclusion.</p>"
        "</section>"
    )


def _guidance_section(guidance: MemoryRiskAssessment) -> str:
    recommendations = {item.recommendation_id: item for item in guidance.recommendations}
    rendered: list[str] = []
    for index, finding in enumerate(guidance.findings, start=1):
        tensor_ids = ", ".join(str(item) for item in finding.affected_tensor_ids) or "None"
        operator_ids = ", ".join(str(item) for item in finding.affected_operator_ids) or "None"
        evidence = "".join(
            f"<div><dt>{_text(key.replace('_', ' ').title())}</dt><dd>{_text(value if value is not None else 'Not available')}</dd></div>"
            for key, value in finding.evidence
        )
        linked = "".join(
            "<div class=\"guidance-item\">"
            f"<strong>Recommendation {_text(recommendation.recommendation_id)}</strong> "
            f"[{_text(recommendation.priority)}, {_text(recommendation.confidence)}]"
            f"<p>{_text(recommendation.action)} {_text(recommendation.expected_effect)}</p>"
            f"<p class=\"muted\">Caveat: {_text('; '.join(recommendation.caveats))}</p></div>"
            for recommendation_id in finding.recommendation_ids
            for recommendation in (recommendations[recommendation_id],)
        )
        rendered.append(
            '<details class="guidance-item">'
            "<summary><span class=\"finding-summary\">"
            f"<strong>{index}. {_text(finding.title)}</strong>"
            f"<span class=\"tag\">severity: {_text(finding.severity)}</span>"
            f"<span class=\"tag\">confidence: {_text(finding.confidence)}</span>"
            f"<span class=\"tag\">category: {_text(finding.category)}</span>"
            "</span></summary>"
            f"<p>{_text(finding.explanation)}</p>"
            f"<p class=\"muted\">Affected tensors: {_text(tensor_ids)} · Affected operators: {_text(operator_ids)}</p>"
            f"<dl class=\"metrics\">{evidence}</dl>{linked}</details>"
        )
    if not rendered:
        rendered.append('<p class="muted">No material model-level arena-head optimization finding was detected by the current rules.</p>')
    summary = guidance.to_dict()["summary"]
    assert isinstance(summary, dict)
    return (
        '<section id="memory-guidance"><h2>Memory risk and optimization guidance</h2>'
        f'<p><span class="budget-status">Overall risk: {_text(guidance.overall_risk.upper())}</span></p>'
        '<div class="cards">'
        f'<div class="card"><span class="label">Findings</span><span class="value">{summary["finding_count"]}</span></div>'
        f'<div class="card"><span class="label">Recommendations</span><span class="value">{summary["recommendation_count"]}</span></div>'
        f'<div class="card"><span class="label">High or critical</span><span class="value">{summary["high_or_critical_count"]}</span></div></div>'
        + "".join(rendered)
        + "<p><strong>Recommendations are evidence-based suggestions, not guaranteed byte savings.</strong></p>"
        "<p>Model accuracy, operator support, and graph semantics must be revalidated after any model change.</p>"
        "</section>"
    )


def _compute_cost_row(item: OperatorComputeCost) -> str:
    if item.category == "mac":
        detail = f"{item.mac_count:,} MACs"
    elif item.category == "elementwise":
        detail = f"{item.elementwise_op_count:,} elementwise ops ({_text(item.note)})"
    elif item.category == "zero":
        detail = f"0 ({_text(item.note)})"
    else:
        detail = f"unavailable ({_text(item.note)})"
    return (
        "<tr>"
        f"<td>{item.operator_id}</td>"
        f"<td>{_text(item.operator_name)}</td>"
        f"<td>{_text(item.category)}</td>"
        f"<td>{detail}</td>"
        "</tr>"
    )


def _compute_cost_section(compute_cost: ComputeCostSummary) -> str:
    rows = "".join(_compute_cost_row(item) for item in compute_cost.operators)
    unavailable_note = (
        f"<p class=\"muted\">{compute_cost.unavailable_operator_count} operator(s) have unavailable compute cost — see the table below.</p>"
        if compute_cost.unavailable_operator_count
        else ""
    )
    return (
        '<section id="compute-cost"><h2>Compute cost</h2>'
        f'<p><strong>{_text(render_compute_cost_caveat())}</strong></p>'
        '<div class="cards">'
        f'<div class="card"><span class="label">Total MACs</span><span class="value">{compute_cost.total_mac_count:,}</span></div>'
        f'<div class="card"><span class="label">Total elementwise ops</span><span class="value">{compute_cost.total_elementwise_ops:,}</span></div>'
        "</div>"
        f"{unavailable_note}"
        "<details><summary>Per-operator compute cost</summary>"
        '<div class="table-wrap"><table><thead><tr><th>Operator</th><th>Name</th><th>Category</th><th>Detail</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
        f'<p class="muted">{_text(render_compute_cost_caveat(long=True))}</p>'
        "</details>"
        "</section>"
    )


def _analysis_views_section(result: dict[str, object]) -> str:
    views = result.get("analysis_views")
    if not isinstance(views, dict):
        return ""
    attribution = views["operator_attribution"]
    timeline = views["execution_timeline"]
    graph = views["graph_view"]
    operators = attribution["operators"]
    rows = "".join(
        f"<tr><td>{item['operator_id']}</td><td>{_text(item['operator_name'])}</td>"
        f"<td>{item['represented_input_aligned_bytes']:,}</td><td>{item['represented_output_aligned_bytes']:,}</td>"
        f"<td>{item['live_aligned_bytes_at_scope']:,}</td><td>{item['occupied_extent_bytes_at_scope']:,}</td>"
        f"<td>{_text(item['pressure'])}</td></tr>" for item in operators
    )
    scopes = timeline["scopes"]
    maximum = max((item["occupied_extent_bytes"] for item in scopes), default=1) or 1
    points_live = " ".join(f"{40 + index * 80},{210 - item['live_aligned_bytes'] * 170 / maximum:.2f}" for index, item in enumerate(scopes))
    points_extent = " ".join(f"{40 + index * 80},{210 - item['occupied_extent_bytes'] * 170 / maximum:.2f}" for index, item in enumerate(scopes))
    timeline_svg = (
        '<svg id="execution-timeline-svg" viewBox="0 0 760 240" role="img" aria-labelledby="timeline-title timeline-desc">'
        '<title id="timeline-title">Execution memory timeline</title><desc id="timeline-desc">Labeled lines show live aligned bytes and occupied arena extent by execution scope.</desc>'
        f'<polyline points="{points_extent}" fill="none" stroke="#3257d5" stroke-width="3"/><polyline points="{points_live}" fill="none" stroke="#00876c" stroke-width="3"/>'
        '<text x="40" y="232">Blue: occupied extent · Green: live aligned bytes</text></svg>'
    )
    nodes = graph["operators"]
    graph_nodes = "".join(
        f'<g><rect x="{30 + (index % 6) * 120}" y="{25 + (index // 6) * 70}" width="105" height="44" fill="#edf2ff" stroke="#3257d5"/>'
        f'<text x="{36 + (index % 6) * 120}" y="{43 + (index // 6) * 70}">#{item["operator_id"]}</text>'
        f'<text x="{36 + (index % 6) * 120}" y="{59 + (index // 6) * 70}">{_text(str(item["operator_name"])[:14])}</text></g>'
        for index, item in enumerate(nodes)
    )
    height = max(100, 30 + ((len(nodes) + 5) // 6) * 70)
    graph_svg = (
        f'<svg id="model-graph-svg" viewBox="0 0 760 {height}" role="img" aria-labelledby="graph-title graph-desc">'
        '<title id="graph-title">Primary subgraph operator view</title><desc id="graph-desc">Operator nodes are shown in deterministic execution order; tensor relationships remain available in the JSON data.</desc>'
        f'{graph_nodes}</svg>'
    )
    truncation_note = (
        '<p class="muted">Node list truncated; not every operator is shown.</p>'
        if graph["truncated"]
        else ""
    )
    return (
        '<section id="analysis-views"><h2>Additional analysis views</h2>'
        '<details id="operator-attribution"><summary>Operator-level arena-head pressure</summary>'
        '<p>Represented live-set metrics are not independently additive contributions to planned arena head.</p>'
        '<div class="table-wrap"><table><thead><tr><th>ID</th><th>Operator</th><th>Input bytes</th><th>Output bytes</th><th>Live bytes</th><th>Occupied extent</th><th>Pressure</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div></details>'
        f'<details id="execution-timeline"><summary>Execution timeline</summary>{timeline_svg}</details>'
        f'<details id="model-graph"><summary>Primary subgraph graph view</summary>{graph_svg}{truncation_note}</details>'
        "</section>"
    )


def write_html_report(path: str | Path, content: str) -> Path:
    """Write UTF-8 HTML and return the resolved output path."""

    destination = Path(path).expanduser().resolve()
    temporary_name: str | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    except OSError as error:
        raise HTMLReportError(
            f"Unable to write HTML report {destination}: {error}"
        ) from error
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except OSError:
                pass
    return destination
