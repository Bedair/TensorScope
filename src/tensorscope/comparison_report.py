from __future__ import annotations

from html import escape

from tensorscope.comparison import ModelComparison


def _text(value: object) -> str:
    return escape(str(value), quote=True)


def render_comparison_html(comparison: ModelComparison, *, tool_version: str) -> str:
    metrics = dict(comparison.metrics)
    head = metrics["planned_arena_head_bytes"]
    maximum = max(int(head.baseline or 0), int(head.candidate or 0), 1)
    baseline_width = int(head.baseline or 0) * 600 / maximum
    candidate_width = int(head.candidate or 0) * 600 / maximum
    tensor_rows = "".join(
        "<tr>"
        f"<td>{_text(item.status)}</td><td>{_text(item.baseline_name or '—')}</td>"
        f"<td>{_text(item.candidate_name or '—')}</td><td>{_text(item.match_confidence)}</td>"
        f"<td>{_text(item.aligned_bytes.delta if item.aligned_bytes.delta is not None else '—')}</td>"
        f"<td>{item.impact_score}</td></tr>"
        for item in comparison.tensor_deltas
    ) or '<tr><td colspan="6">No runtime tensor changes</td></tr>'
    operator = dict(comparison.operator_comparison)
    guidance = dict(comparison.guidance_comparison)
    quantization = dict(comparison.quantization_comparison)
    peak = dict(comparison.peak_comparison)
    reasons = "".join(f"<li>{_text(item)}</li>" for item in comparison.regression.reasons) or "<li>No deterministic regression rule was triggered.</li>"
    budget = ""
    if comparison.budget_comparison is not None:
        value = dict(comparison.budget_comparison)
        budget = (
            '<section id="budget-comparison"><h2>Arena-head budget comparison</h2>'
            f"<p>Baseline: <strong>{_text(value['baseline_status'])}</strong> · Candidate: "
            f"<strong>{_text(value['candidate_status'])}</strong> · Status change: {_text(value['status_change'])}</p>"
            "<p>Budget results do not establish complete MCU or firmware memory fit.</p></section>"
        )
    metric_cards = "".join(
        '<div class="card">'
        f"<span>{_text(name.replace('_', ' '))}</span><strong>{_text(delta.delta)}</strong>"
        f"<small>{_text(delta.direction)}</small></div>"
        for name, delta in comparison.metrics
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TensorScope model comparison</title><style>
:root {{ --ink:#172033; --muted:#5f6b7a; --line:#d8dee8; --paper:#fff; --panel:#f6f8fb; --accent:#3257d5; }}
* {{ box-sizing:border-box }} body {{ margin:0;background:#eef1f6;color:var(--ink);font:14px/1.5 system-ui,sans-serif }}
main {{ max-width:1100px;margin:auto;padding:28px 18px }} section {{ background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:18px;margin-top:16px }}
h1 {{ margin:0 }} h2 {{ margin:0 0 12px }} .path {{ overflow-wrap:anywhere;color:var(--muted) }} .status {{ display:inline-block;border:2px solid currentColor;border-radius:4px;padding:5px 10px;font-weight:750 }}
.cards {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px }} .card {{ background:var(--panel);padding:12px;border-radius:6px;display:grid }} .card span,.card small {{ color:var(--muted) }} .card strong {{ font-size:20px }}
table {{ width:100%;border-collapse:collapse }} th,td {{ text-align:left;padding:8px;border-bottom:1px solid var(--line) }} .table {{ overflow:auto }}
svg {{ width:100%;height:auto }} .bar-label {{ fill:var(--ink);font-size:14px }} footer {{ color:var(--muted);margin-top:20px }}
</style></head><body><main>
<header><h1>Model comparison</h1><p class="path">Baseline: {_text(comparison.baseline_model)}<br>Candidate: {_text(comparison.candidate_model)}</p><span class="status">Comparison status: {_text(comparison.status.upper())}</span></header>
<section><h2>Arena-head summary</h2><p>Baseline: <strong>{head.baseline:,} bytes</strong> · Candidate: <strong>{head.candidate:,} bytes</strong> · Delta: <strong>{head.delta:+,} bytes</strong></p>
<svg id="arena-head-comparison-svg" viewBox="0 0 760 120" role="img" aria-labelledby="comparison-chart-title comparison-chart-desc"><title id="comparison-chart-title">Baseline and candidate planned arena head</title><desc id="comparison-chart-desc">Labeled horizontal bars compare planned arena-head byte counts.</desc><text class="bar-label" x="0" y="27">Baseline</text><rect x="110" y="10" width="{baseline_width:.3f}" height="24" fill="#3257d5"/><text class="bar-label" x="720" y="27" text-anchor="end">{head.baseline} bytes</text><text class="bar-label" x="0" y="77">Candidate</text><rect x="110" y="60" width="{candidate_width:.3f}" height="24" fill="#00876c"/><text class="bar-label" x="720" y="77" text-anchor="end">{head.candidate} bytes</text></svg></section>
<section><h2>Key metric changes</h2><div class="cards">{metric_cards}</div></section>
<section><h2>Peak comparison</h2><p>Baseline scope {_text(peak['baseline_scope'])} ({_text(peak['baseline_operator_name'] or 'subgraph input')}); candidate scope {_text(peak['candidate_scope'])} ({_text(peak['candidate_operator_name'] or 'subgraph input')}). Peak moved: <strong>{'yes' if peak['peak_moved'] else 'no'}</strong>.</p></section>
<section><h2>Tensor changes</h2><div class="table"><table><thead><tr><th>Status</th><th>Baseline</th><th>Candidate</th><th>Match</th><th>Aligned delta</th><th>Impact</th></tr></thead><tbody>{tensor_rows}</tbody></table></div><p>Tensor matching is deterministic but does not prove semantic equivalence.</p></section>
<section><h2>Operator comparison</h2><p>Added name counts: {_text(operator['added_name_counts'])}</p><p>Removed name counts: {_text(operator['removed_name_counts'])}</p><p>Operator-name sequences equal: {_text(operator['sequences_equal'])}</p></section>
<section><h2>Guidance comparison</h2><p>Introduced categories: {_text(guidance['introduced_categories'])}</p><p>Resolved categories: {_text(guidance['resolved_categories'])}</p><p>Severity changes: {_text(guidance['severity_changes'])}</p></section>
<section><h2>Quantization comparison</h2><p>Changed tensors: {_text(len(quantization['tensor_changes']))}</p><p>{_text('; '.join(quantization['warnings']))}</p></section>
{budget}
<section><h2>Regression assessment</h2><ul>{reasons}</ul></section>
<section id="limitations"><h2>Limitations</h2><ul><li>Comparison covers planned arena head only.</li><li>Tensor matching does not establish semantic equivalence.</li><li>Model accuracy, operator support, and graph semantics must be validated separately.</li><li>Static arena tail and complete arena total are not compared.</li><li>Budget results do not establish complete MCU or firmware memory fit.</li></ul></section>
<footer>Generated by TensorScope {_text(tool_version)} · deterministic static comparison · no external assets</footer>
</main></body></html>"""
