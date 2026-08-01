from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import os
from pathlib import Path
import tempfile

from tensorscope.explain import MemoryExplanation, TensorExplanation
from tensorscope.memory_budget import ArenaHeadBudgetResult


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


_SVG_PALETTE = (
    "#3257d5",
    "#00876c",
    "#9c4f00",
    "#7b4ab5",
    "#b3345c",
    "#237b9f",
    "#6b6f00",
    "#8d4b35",
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


def render_packing_svg(explanation: MemoryExplanation) -> str:
    """Render scope-by-offset tensor placement as a responsive inline SVG."""

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

    elements = [
        '<svg id="arena-packing-svg" class="packing-svg" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="arena-packing-title arena-packing-description">',
        '<title id="arena-packing-title">Arena placement across execution scopes</title>',
        '<desc id="arena-packing-description">Each tensor rectangle spans its inclusive execution lifetime horizontally and its planned arena-head byte interval vertically.</desc>',
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
            color = _SVG_PALETTE[tensor.tensor_id % len(_SVG_PALETTE)]
            title = (
                f"tensor[{tensor.tensor_id}] {tensor.name or '<unnamed>'}; "
                f"lifetime {tensor.first_used_scope}..{tensor.last_used_scope}; "
                f"memory [{tensor.offset}, {tensor.end_offset}) bytes"
            )
            elements.append(
                f'<g id="tensor-allocation-{tensor.tensor_id}" class="tensor-allocation" data-tensor-id="{tensor.tensor_id}">'
                f'<rect class="tensor-rect" x="{x:.3f}" y="{y:.3f}" width="{max(rectangle_width, 1):.3f}" height="{max(rectangle_height, 1):.3f}" fill="{color}" stroke="#172033" stroke-width="1">'
                f'<title>{_text(title)}</title></rect>'
                f'<text class="tensor-label" x="{x + 5:.3f}" y="{y + min(max(rectangle_height, 14), 22):.3f}">#{tensor.tensor_id} {_text(_short_name(tensor.name))}</text>'
                "</g>"
            )
    else:
        elements.append(
            f'<text class="empty-svg" x="{left + plot_width / 2:.3f}" y="{top + plot_height / 2:.3f}" text-anchor="middle">No runtime allocations</text>'
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


def _tensor_rows(tensors: tuple[TensorExplanation, ...]) -> str:
    if not tensors:
        return '<tr><td colspan="12" class="empty">No runtime tensors</td></tr>'
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
            "</tr>"
        )
    return "".join(rows)


def _tensor_table(tensors: tuple[TensorExplanation, ...]) -> str:
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Tensor</th><th>Name</th><th>Type</th><th>Shape</th>"
        "<th>Logical</th><th>Aligned</th><th>Overhead</th>"
        "<th>Offset</th><th>End</th><th>Lifetime</th>"
        "<th>Scopes</th><th>Role</th>"
        "</tr></thead><tbody>"
        + _tensor_rows(tensors)
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
    items: list[str] = []
    for blocker in explanation.reuse_blockers:
        overlapping = ", ".join(
            f"tensor[{tensor_id}]"
            for tensor_id in blocker.overlapping_tensor_ids
        )
        through = (
            f"operator {blocker.last_consumer_operator_id} "
            f"({_text(blocker.last_consumer_operator_name)})"
            if blocker.last_consumer_operator_id is not None
            else f"scope {blocker.lifetime[1]}"
        )
        items.append(
            "<li>"
            f"<strong>tensor[{blocker.tensor_id}] {_name(blocker.tensor_name)}</strong> "
            f"({blocker.aligned_bytes:,} aligned bytes) remains live through {through}. "
            f"Its lifetime {blocker.lifetime[0]}..{blocker.lifetime[1]} overlaps "
            f"with {_text(overlapping)}, so those tensors cannot reuse the same "
            "memory interval."
            "</li>"
        )
    return '<ul class="blockers">' + "".join(items) + "</ul>"


def render_html_report(
    result: dict[str, object],
    explanation: MemoryExplanation,
    *,
    tool_version: str,
    generated_at: datetime | None = None,
    budget: ArenaHeadBudgetResult | None = None,
) -> str:
    """Render one deterministic, dependency-free HTML analysis report."""

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
    budget_section = _budget_section(budget) if budget is not None else ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TensorScope analysis · {_text(model_filename)}</title>
<style>
:root {{ color-scheme: light; --ink:#172033; --muted:#5f6b7a; --line:#d8dee8; --paper:#fff; --panel:#f6f8fb; --accent:#3257d5; --good:#176b45; --warn:#8a4b08; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#eef1f6; color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:28px 18px 56px; }}
h1 {{ margin:0 0 4px; font-size:28px; }} h2 {{ margin:30px 0 12px; font-size:20px; }} h3 {{ margin:20px 0 8px; font-size:16px; }}
.subtitle,.muted {{ color:var(--muted); }} .subtitle {{ overflow-wrap:anywhere; }}
.notice {{ margin:20px 0; padding:14px 16px; border-left:4px solid var(--accent); background:#edf2ff; }}
.notice p {{ margin:3px 0; }}
.budget-status {{ display:inline-block; padding:4px 9px; border:2px solid currentColor; border-radius:4px; font-weight:750; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }}
.card,section {{ background:var(--paper); border:1px solid var(--line); border-radius:8px; }}
.card {{ padding:15px; }} .card .label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
.card .value {{ display:block; margin:4px 0; font-size:22px; font-weight:700; }}
.tags {{ display:flex; flex-wrap:wrap; gap:5px; }} .tag {{ border-radius:99px; background:var(--panel); border:1px solid var(--line); padding:2px 7px; font-size:12px; }}
section {{ margin-top:16px; padding:18px; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:8px 18px; margin:0; }}
.metrics div {{ padding:7px 0; border-bottom:1px solid var(--line); }} .metrics dt {{ color:var(--muted); }} .metrics dd {{ margin:1px 0 0; font-weight:650; }}
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
figure {{ margin:12px 0 0; }} figcaption {{ margin-top:8px; color:var(--muted); }}
.blockers {{ padding-left:22px; }} .blockers li {{ margin:8px 0; }}
footer {{ margin-top:24px; color:var(--muted); font-size:12px; }}
@media (max-width:680px) {{ main {{ padding:16px 10px 36px; }} .packing-row {{ grid-template-columns:1fr; gap:4px; }} section {{ padding:13px; }} }}
@media print {{ body {{ background:#fff; }} main {{ max-width:none; padding:0; }} section,.card {{ break-inside:avoid; }} }}
</style>
</head>
<body><main>
<header><h1>TensorScope arena-head analysis</h1><div class="subtitle"><strong>{_text(model_filename)}</strong><br>{_text(model_path)}</div></header>
<div class="notice" role="note">
<p><strong>This report covers planned arena head only.</strong></p>
<p>Arena head is statically calculated. Arena-head validation is not run by <code>analyze</code>; validation applies only when the TFLM oracle is run.</p>
<p>Arena tail is not statically estimated. Complete arena total is not statically estimated.</p>
<p>This report does not prove complete MCU or firmware fit.</p>
</div>
<div class="cards">
<div class="card"><span class="label">Arena head</span><span class="value">{_bytes(head.get('bytes'))}</span><div class="tags"><span class="tag">scope: {_text(head.get('scope'))}</span><span class="tag">confidence: {_text(head.get('confidence'))}</span><span class="tag">source: {_text(head.get('source'))}</span><span class="tag">validation: {_text(head.get('validation_state'))}</span></div></div>
<div class="card"><span class="label">Arena tail</span><span class="value">{_bytes(tail.get('bytes'))}</span><div class="tags"><span class="tag">scope: {_text(tail.get('scope'))}</span><span class="tag">confidence: {_text(tail.get('confidence'))}</span><span class="tag">validation: {_text(tail.get('validation_state'))}</span></div></div>
<div class="card"><span class="label">Complete arena total</span><span class="value">{_bytes(total.get('bytes'))}</span><div class="tags"><span class="tag">scope: {_text(total.get('scope'))}</span><span class="tag">confidence: {_text(total.get('confidence'))}</span><span class="tag">validation: {_text(total.get('validation_state'))}</span></div></div>
</div>
{budget_section}
<section><h2>Report metadata</h2><dl class="metrics">
<div><dt>Model filename</dt><dd>{_text(model_filename)}</dd></div><div><dt>Model path</dt><dd>{_text(model_path)}</dd></div><div><dt>TFLite schema version</dt><dd>{_text(schema_version if schema_version is not None else 'Unavailable')}</dd></div><div><dt>Generated at</dt><dd>{_text(generated_timestamp)}</dd></div><div><dt>TensorScope version</dt><dd>{_text(tool_version)}</dd></div><div><dt>Analysis scope</dt><dd>Primary subgraph · planned arena head only</dd></div>
</dl></section>
<section><h2>Model summary</h2><dl class="metrics">
<div><dt>Runtime tensors planned</dt><dd>{summary.runtime_tensor_count:,}</dd></div><div><dt>Constant tensors</dt><dd>{summary.constant_tensor_count:,}</dd></div><div><dt>Operators</dt><dd>{summary.operator_count:,}</dd></div><div><dt>Planned arena head</dt><dd>{summary.planned_arena_head_bytes:,} bytes</dd></div><div><dt>Arena alignment</dt><dd>{summary.arena_alignment_bytes:,} bytes</dd></div><div><dt>Logical runtime-tensor sum</dt><dd>{summary.logical_runtime_tensor_bytes:,} bytes</dd></div><div><dt>Aligned runtime-tensor sum</dt><dd>{summary.aligned_runtime_tensor_bytes:,} bytes</dd></div><div><dt>Per-tensor alignment overhead</dt><dd>{summary.alignment_overhead_bytes:,} bytes</dd></div><div><dt>Safe reuse relationships</dt><dd>{len(explanation.reuse):,}</dd></div><div><dt>Conservative reuse blockers</dt><dd>{len(explanation.reuse_blockers):,}</dd></div>
</dl><p class="muted">Tensor-size sums are not the planned head: safely reused regions can reduce the plan.</p></section>
<section><h2>Peak execution point</h2><div class="peak"><div><strong>{_text(peak_context)}</strong></div><div>Occupied arena extent<br><strong>{peak.occupied_extent_bytes:,} bytes</strong></div><div>Live aligned tensor sum<br><strong>{peak.live_aligned_bytes:,} bytes</strong></div><div>Tied peak scopes<br><strong>{_text(tied)}</strong></div></div><h3>Live tensors at peak</h3>{_tensor_table(explanation.live_tensors_at_peak)}</section>
<section><h2>Largest runtime tensors</h2>{_tensor_table(explanation.largest_tensors)}</section>
<section><h2>Arena placement across execution scopes</h2><figure>{render_packing_svg(explanation)}<figcaption>Horizontal position represents inclusive execution lifetime; vertical position represents the planned arena-head memory interval. Tensors occupying the same memory interval at disjoint execution scopes represent safe reuse. Rectangle labels and outlines ensure the view does not rely on color alone.</figcaption></figure><h3>Compact offset view</h3><p class="muted">Horizontal position and width below correspond to planned byte offsets.</p>{_packing_rows(explanation)}</section>
<section><h2>All planned runtime tensors</h2>{_tensor_table(explanation.allocations)}</section>
<section><h2>Execution scopes</h2><div class="table-wrap"><table><thead><tr><th>Scope</th><th>Context</th><th>Occupied extent</th><th>Live aligned sum</th><th>Live tensor IDs</th></tr></thead><tbody>{scope_rows}</tbody></table></div></section>
<section><h2>Safe memory reuse</h2><div class="table-wrap"><table><thead><tr><th>Earlier tensor</th><th>Earlier lifetime</th><th>Overlap interval</th><th>Later tensor</th><th>Later lifetime</th></tr></thead><tbody>{_reuse_rows(explanation)}</tbody></table></div><h3>Conservative reuse blockers</h3>{_blocker_items(explanation)}</section>
<section id="limitations"><h2>Limitations</h2><ul>
<li>Analysis covers the primary subgraph only.</li>
<li>Analysis covers planned arena head only.</li>
<li>Arena tail is not statically estimated.</li>
<li>Complete arena total is not statically estimated.</li>
<li>Firmware stack usage is not estimated.</li>
<li>General heap usage is not estimated.</li>
<li>DMA buffers are not estimated.</li>
<li>RTOS memory is not estimated.</li>
<li>Application memory is not estimated.</li>
<li>The report does not establish complete MCU fit.</li>
<li>Reuse blockers are conservative explanations and do not represent proven counterfactual byte costs.</li>
<li>Results reflect the planner behavior implemented and pinned by TensorScope.</li>
</ul></section>
<footer>Generated by TensorScope {_text(tool_version)} at {_text(generated_timestamp)} · static report · no external assets</footer>
</main></body></html>
"""


def _budget_section(budget: ArenaHeadBudgetResult) -> str:
    status = {
        "fits": "FITS",
        "exact_fit": "EXACT FIT",
        "exceeds": "EXCEEDS BUDGET",
    }[budget.status]
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
    source = "Direct arena-head budget" if budget.source == "direct" else "Generic MCU planning profile"
    return (
        '<section id="arena-head-budget"><h2>Arena-head budget check</h2>'
        f'<p><span class="budget-status">Arena-head budget result: {status}</span></p>'
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
