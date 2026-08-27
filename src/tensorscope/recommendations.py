from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tensorscope.compute_cost import ComputeCostSummary, render_compute_cost_caveat
from tensorscope.explain import MemoryExplanation, TensorExplanation
from tensorscope.graph import GraphModel
from tensorscope.memory_budget import ArenaHeadBudgetResult


Severity = Literal["info", "low", "medium", "high", "critical"]
Confidence = Literal["exact", "high", "medium", "low"]
Category = Literal[
    "peak_concentration", "long_lived_tensor", "reuse_blocker",
    "alignment_overhead", "graph_input_retention", "graph_output_retention",
    "branch_merge_pressure", "budget_pressure", "oracle_context",
    "compute_concentration", "general_summary",
]
EvidenceValue = int | float | str | None

_SEVERITY_RANK: dict[Severity, int] = {
    "info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4,
}
_VALID_CONFIDENCE = {"exact", "high", "medium", "low"}


@dataclass(frozen=True)
class MemoryFinding:
    finding_id: str
    category: Category
    severity: Severity
    confidence: Confidence
    title: str
    explanation: str
    evidence: tuple[tuple[str, EvidenceValue], ...] = ()
    affected_tensor_ids: tuple[int, ...] = ()
    affected_operator_ids: tuple[int, ...] = ()
    recommendation_ids: tuple[str, ...] = ()
    impact_score: int = 0

    def __post_init__(self) -> None:
        if not self.finding_id or self.severity not in _SEVERITY_RANK:
            raise ValueError("Finding requires a stable ID and valid severity")
        if self.confidence not in _VALID_CONFIDENCE:
            raise ValueError("Finding confidence is invalid")
        if self.impact_score < 0:
            raise ValueError("Finding impact score must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "finding_id": self.finding_id,
            "category": self.category,
            "severity": self.severity,
            "confidence": self.confidence,
            "title": self.title,
            "explanation": self.explanation,
            "evidence": dict(self.evidence),
            "affected_tensor_ids": list(self.affected_tensor_ids),
            "affected_operator_ids": list(self.affected_operator_ids),
            "recommendation_ids": list(self.recommendation_ids),
        }


@dataclass(frozen=True)
class MemoryRecommendation:
    recommendation_id: str
    category: Category
    priority: Severity
    confidence: Confidence
    action: str
    rationale: str
    expected_effect: str
    caveats: tuple[str, ...]
    linked_finding_ids: tuple[str, ...]
    impact_score: int = 0

    def __post_init__(self) -> None:
        if not self.recommendation_id or self.priority not in _SEVERITY_RANK:
            raise ValueError("Recommendation requires a stable ID and valid priority")
        if self.confidence not in _VALID_CONFIDENCE:
            raise ValueError("Recommendation confidence is invalid")
        if not self.linked_finding_ids:
            raise ValueError("Recommendation must link at least one finding")

    def to_dict(self) -> dict[str, object]:
        return {
            "recommendation_id": self.recommendation_id,
            "category": self.category,
            "priority": self.priority,
            "confidence": self.confidence,
            "action": self.action,
            "rationale": self.rationale,
            "expected_effect": self.expected_effect,
            "caveats": list(self.caveats),
            "linked_finding_ids": list(self.linked_finding_ids),
        }


@dataclass(frozen=True)
class MemoryRiskAssessment:
    findings: tuple[MemoryFinding, ...]
    recommendations: tuple[MemoryRecommendation, ...]
    scope: Literal["arena_head"] = "arena_head"

    def __post_init__(self) -> None:
        finding_ids = {item.finding_id for item in self.findings}
        recommendation_ids = {item.recommendation_id for item in self.recommendations}
        if len(finding_ids) != len(self.findings) or len(recommendation_ids) != len(self.recommendations):
            raise ValueError("Finding and recommendation IDs must be unique")
        for item in self.findings:
            if not set(item.recommendation_ids) <= recommendation_ids:
                raise ValueError("Finding links an unknown recommendation")
        for item in self.recommendations:
            if not set(item.linked_finding_ids) <= finding_ids:
                raise ValueError("Recommendation links an unknown finding")

    @property
    def overall_risk(self) -> Severity:
        return max(
            (item.severity for item in self.findings),
            key=lambda value: _SEVERITY_RANK[value],
            default="info",
        )

    def to_dict(self) -> dict[str, object]:
        high_count = sum(item.severity in ("high", "critical") for item in self.findings)
        return {
            "scope": self.scope,
            "overall_risk": self.overall_risk,
            "summary": {
                "finding_count": len(self.findings),
                "recommendation_count": len(self.recommendations),
                "high_or_critical_count": high_count,
            },
            "findings": [item.to_dict() for item in self.findings],
            "recommendations": [item.to_dict() for item in self.recommendations],
        }


def _percent(numerator: int, denominator: int) -> float | None:
    return numerator * 100 / denominator if denominator else None


def _sort_findings(items: list[MemoryFinding]) -> tuple[MemoryFinding, ...]:
    return tuple(sorted(items, key=lambda item: (
        -_SEVERITY_RANK[item.severity], -item.impact_score, item.category,
        item.affected_tensor_ids[0] if item.affected_tensor_ids else 1 << 30,
        item.affected_operator_ids[0] if item.affected_operator_ids else 1 << 30,
        item.finding_id,
    )))


def _sort_recommendations(items: list[MemoryRecommendation]) -> tuple[MemoryRecommendation, ...]:
    return tuple(sorted(items, key=lambda item: (
        -_SEVERITY_RANK[item.priority], -item.impact_score, item.category,
        item.recommendation_id,
    )))


def assess_memory_risk(
    graph: GraphModel,
    explanation: MemoryExplanation,
    *,
    budget: ArenaHeadBudgetResult | None = None,
    compute_cost: ComputeCostSummary | None = None,
) -> MemoryRiskAssessment:
    findings: list[MemoryFinding] = []
    recommendations: list[MemoryRecommendation] = []
    allocations = {item.tensor_id: item for item in explanation.allocations}
    head = explanation.summary.planned_arena_head_bytes

    def add(
        *, finding_id: str, recommendation_id: str | None, category: Category,
        severity: Severity, title: str, explanation_text: str,
        evidence: tuple[tuple[str, EvidenceValue], ...], tensors: tuple[int, ...] = (),
        operators: tuple[int, ...] = (), impact: int = 0, action: str = "",
        rationale: str = "", expected: str = "", caveat: str = "",
    ) -> None:
        recommendation_ids = (recommendation_id,) if recommendation_id else ()
        findings.append(MemoryFinding(
            finding_id, category, severity, "exact", title, explanation_text,
            evidence, tensors, operators, recommendation_ids, impact,
        ))
        if recommendation_id:
            recommendations.append(MemoryRecommendation(
                recommendation_id, category, severity, "medium", action, rationale,
                expected, (caveat,), (finding_id,), impact,
            ))

    peak_total = explanation.peak.live_aligned_bytes
    peak_tensors = sorted(explanation.live_tensors_at_peak, key=lambda item: (-item.aligned_bytes, item.tensor_id))
    if peak_total:
        cumulative = 0
        selected: list[TensorExplanation] = []
        threshold = 0
        for count, tensor in enumerate(peak_tensors[:3], start=1):
            cumulative += tensor.aligned_bytes
            selected.append(tensor)
            target = {1: 50, 2: 75, 3: 90}[count]
            if cumulative * 100 >= target * peak_total:
                threshold = target
                break
        if threshold:
            ids = tuple(item.tensor_id for item in selected)
            share = _percent(cumulative, peak_total)
            add(
                finding_id=f"peak-concentration-t{'-'.join(map(str, ids))}",
                recommendation_id="review-peak-tensors", category="peak_concentration",
                severity="high" if len(ids) == 1 else "medium",
                title=f"Peak memory is concentrated in {len(ids)} tensor{'s' if len(ids) != 1 else ''}",
                explanation_text=f"Tensors {', '.join(map(str, ids))} account for {share:.2f}% of live aligned bytes at the selected peak.",
                evidence=(("peak_live_aligned_bytes", peak_total), ("tensor_aligned_bytes", cumulative),
                          ("share_percent", share), ("threshold_percent", threshold)),
                tensors=ids, operators=((explanation.peak.operator_id,) if explanation.peak.operator_id is not None else ()),
                impact=cumulative, action="Review the shapes and widths of the dominant peak tensors.",
                rationale="These tensors account for most known live bytes at the selected peak.",
                expected="Narrower tensors or less simultaneous liveness may reduce planned arena head.",
                caveat="Any shape, scheduling, or fusion change must preserve model semantics, accuracy, and supported operators.",
            )

    if compute_cost is not None and compute_cost.total_mac_count:
        mac_operators = sorted(
            (item for item in compute_cost.operators if item.category == "mac"),
            key=lambda item: (-(item.mac_count or 0), item.operator_id),
        )
        total_macs = compute_cost.total_mac_count
        cumulative_macs = 0
        selected_ops: list[int] = []
        mac_threshold = 0
        for count, item in enumerate(mac_operators[:3], start=1):
            cumulative_macs += item.mac_count or 0
            selected_ops.append(item.operator_id)
            target = {1: 50, 2: 75, 3: 90}[count]
            if cumulative_macs * 100 >= target * total_macs:
                mac_threshold = target
                break
        if mac_threshold:
            share = _percent(cumulative_macs, total_macs)
            add(
                finding_id=f"compute-concentration-op{'-'.join(map(str, selected_ops))}",
                recommendation_id=f"review-compute-op{'-'.join(map(str, selected_ops))}",
                category="compute_concentration",
                severity="info",
                title=f"Compute is concentrated in {len(selected_ops)} operator{'s' if len(selected_ops) != 1 else ''}",
                explanation_text=f"Operators {', '.join(map(str, selected_ops))} account for {share:.2f}% of total MACs.",
                evidence=(("total_mac_count", total_macs), ("concentrated_mac_count", cumulative_macs),
                          ("share_percent", share), ("threshold_percent", mac_threshold)),
                operators=tuple(selected_ops),
                impact=cumulative_macs,
                action="Review the shapes and channel counts of the dominant compute operators.",
                rationale="These operators account for most of the model's known multiply-accumulate volume.",
                expected="Smaller kernels, fewer channels, or reduced spatial dimensions may reduce compute cost.",
                caveat=render_compute_cost_caveat(long=True),
            )

    for tensor in sorted(allocations.values(), key=lambda item: item.tensor_id):
        if head and tensor.aligned_bytes * 4 >= head and tensor.lifetime_length >= 3:
            fid = f"long-lived-tensor-{tensor.tensor_id}"
            rid = f"shorten-lifetime-tensor-{tensor.tensor_id}"
            add(
                finding_id=fid, recommendation_id=rid, category="long_lived_tensor",
                severity="high" if tensor.aligned_bytes * 2 >= head else "medium",
                title=f"Large tensor {tensor.tensor_id} is long-lived",
                explanation_text=f"Tensor {tensor.tensor_id} occupies {tensor.aligned_bytes} aligned bytes across scopes {tensor.first_used_scope}..{tensor.last_used_scope}.",
                evidence=(("aligned_bytes", tensor.aligned_bytes), ("head_share_percent", _percent(tensor.aligned_bytes, head)),
                          ("lifetime_length", tensor.lifetime_length), ("first_scope", tensor.first_used_scope),
                          ("last_scope", tensor.last_used_scope), ("is_graph_input", int(tensor.is_graph_input)),
                          ("is_graph_output", int(tensor.is_graph_output))), tensors=(tensor.tensor_id,),
                impact=tensor.aligned_bytes * tensor.lifetime_length,
                action="Review whether this tensor's last use can move earlier or its shape can be narrowed.",
                rationale="Large, multi-scope lifetimes reduce opportunities for other buffers to reuse its region.",
                expected="Shortening this tensor's lifetime may create additional reuse opportunities.",
                caveat="Reordering or fusing operations may help only if model semantics and kernel support allow it.",
            )

    for blocker in explanation.reuse_blockers:
        blocked = tuple(sorted(blocker.overlapping_tensor_ids))
        blocked_bytes = sum(allocations[item].aligned_bytes for item in blocked if item in allocations)
        if len(blocked) >= 2 and (not head or blocked_bytes * 4 >= head):
            impact = blocker.aligned_bytes * len(blocked) + blocked_bytes
            add(
                finding_id=f"reuse-blocker-tensor-{blocker.tensor_id}",
                recommendation_id=f"review-reuse-blocker-{blocker.tensor_id}", category="reuse_blocker",
                severity="medium", title=f"Tensor {blocker.tensor_id} blocks multiple reuse opportunities",
                explanation_text=f"Its lifetime overlaps {len(blocked)} runtime tensors totaling {blocked_bytes} aligned bytes.",
                evidence=(("aligned_bytes", blocker.aligned_bytes), ("blocked_tensor_count", len(blocked)),
                          ("blocked_aligned_bytes", blocked_bytes), ("impact_score", impact)),
                tensors=(blocker.tensor_id, *blocked),
                operators=((blocker.last_consumer_operator_id,) if blocker.last_consumer_operator_id is not None else ()),
                impact=impact, action="Inspect the last consumer and adjacent operations for a semantics-preserving shorter lifetime.",
                rationale="The exact lifetime overlap is known, although planner changes after a rewrite are not.",
                expected="Shortening this tensor's lifetime may create additional reuse opportunities.",
                caveat="This is not an exact byte-saving estimate; the complete plan must be recalculated after changes.",
            )

    summary = explanation.summary
    tiny = tuple(item.tensor_id for item in allocations.values() if item.logical_bytes < summary.arena_alignment_bytes)
    overhead_percent = _percent(summary.alignment_overhead_bytes, summary.aligned_runtime_tensor_bytes)
    high_individual = tuple(item.tensor_id for item in allocations.values() if item.aligned_bytes and item.alignment_overhead_bytes * 2 >= item.aligned_bytes)
    if (overhead_percent is not None and overhead_percent >= 10) or len(tiny) >= 3 or high_individual:
        add(
            finding_id="alignment-overhead", recommendation_id="reduce-tiny-intermediates",
            category="alignment_overhead", severity="low" if (overhead_percent or 0) < 20 else "medium",
            title="Runtime tensors incur material alignment overhead",
            explanation_text=f"Alignment adds {summary.alignment_overhead_bytes} bytes across runtime tensors; {len(tiny)} tensors are smaller than one alignment block.",
            evidence=(("overhead_bytes", summary.alignment_overhead_bytes), ("overhead_percent", overhead_percent),
                      ("tiny_tensor_count", len(tiny)), ("high_overhead_tensor_count", len(high_individual)),
                      ("alignment_bytes", summary.arena_alignment_bytes)), tensors=tuple(sorted(set(tiny + high_individual))),
            impact=summary.alignment_overhead_bytes, action="Review whether model semantics permit fewer tiny intermediate tensors.",
            rationale="Many small buffers can each round up to the planner's required alignment.",
            expected="Reducing tiny intermediates may reduce cumulative alignment overhead.",
            caveat="Allocator alignment is required by current TensorScope/TFLM assumptions and should not be changed casually.",
        )

    for tensor in sorted(allocations.values(), key=lambda item: item.tensor_id):
        if tensor.is_graph_input and tensor.lifetime_length >= 3 and (not head or tensor.aligned_bytes * 10 >= head):
            add(
                finding_id=f"graph-input-retention-{tensor.tensor_id}", recommendation_id=f"review-input-retention-{tensor.tensor_id}",
                category="graph_input_retention", severity="medium", title=f"Graph input {tensor.tensor_id} remains live for late use",
                explanation_text=f"The input remains live through scope {tensor.last_used_scope} and occupies {tensor.aligned_bytes} aligned bytes.",
                evidence=(("aligned_bytes", tensor.aligned_bytes), ("lifetime_length", tensor.lifetime_length),
                          ("last_scope", tensor.last_used_scope), ("head_share_percent", _percent(tensor.aligned_bytes, head))),
                tensors=(tensor.tensor_id,), impact=tensor.aligned_bytes * tensor.lifetime_length,
                action="Check whether a late branch or skip connection requires this input's final use.",
                rationale="The current graph retains the input until its last represented consumer.",
                expected="Removing an unnecessary late use may permit earlier reuse of its allocation.",
                caveat="Do not release the input earlier unless the model architecture no longer requires that use.",
            )
        if tensor.is_graph_output and head and tensor.aligned_bytes * 4 >= head:
            add(
                finding_id=f"graph-output-retention-{tensor.tensor_id}", recommendation_id=f"review-output-shape-{tensor.tensor_id}",
                category="graph_output_retention", severity="low", title=f"Graph output {tensor.tensor_id} is a substantial arena-head allocation",
                explanation_text=f"The required output occupies {tensor.aligned_bytes} aligned bytes ({_percent(tensor.aligned_bytes, head):.2f}% of planned head).",
                evidence=(("aligned_bytes", tensor.aligned_bytes), ("head_share_percent", _percent(tensor.aligned_bytes, head))),
                tensors=(tensor.tensor_id,), impact=tensor.aligned_bytes,
                action="Expose only required outputs and review whether output dimensionality can be reduced.",
                rationale="Graph outputs are retained by current planner semantics.",
                expected="A smaller required output may reduce planned arena-head pressure.",
                caveat="TensorScope cannot free a required graph output early; moving post-processing off-model is application-dependent.",
            )

    for operator in graph.primary_subgraph.operators:
        if operator.name not in {"ADD", "MUL"}:
            continue
        inputs = tuple(sorted(item for item in operator.inputs if item in allocations))
        live_sum = sum(allocations[item].aligned_bytes for item in inputs)
        if len(inputs) >= 2 and head and live_sum * 2 >= head:
            add(
                finding_id=f"branch-merge-{operator.name.lower()}-{operator.id}", recommendation_id=f"narrow-merge-{operator.id}",
                category="branch_merge_pressure", severity="medium", title=f"{operator.name} merge has substantial simultaneous input pressure",
                explanation_text=f"Its {len(inputs)} runtime inputs total {live_sum} aligned bytes at merge scope {operator.id + 1}.",
                evidence=(("merge_operator", operator.name), ("merge_scope", operator.id + 1),
                          ("input_aligned_bytes", live_sum), ("head_share_percent", _percent(live_sum, head))),
                tensors=inputs, operators=(operator.id,), impact=live_sum,
                action="Review branch widths and skip-connection placement around this merge.",
                rationale="The represented merge requires its input tensors simultaneously live.",
                expected="Narrower branch tensors may reduce simultaneous live bytes.",
                caveat="Fusion or branch changes are conditional on model semantics, accuracy, and operator support.",
            )

    if budget is not None:
        if budget.status == "exceeds":
            severity, label = "critical", "exceeds the selected arena-head budget"
        elif budget.status == "exact_fit":
            severity, label = "high", "exactly fills the selected arena-head budget"
        elif budget.utilization_percent is None:
            severity, label = "high", "has unavailable utilization for the zero-byte budget"
        elif budget.utilization_percent >= 90:
            severity, label = "high", "is near the selected arena-head limit"
        elif budget.utilization_percent >= 75:
            severity, label = "medium", "has moderate arena-head pressure"
        else:
            severity, label = "info", "has comfortable arena-head budget headroom"
        add(
            finding_id="budget-pressure", recommendation_id=("reduce-arena-head-for-budget" if severity != "info" else None),
            category="budget_pressure", severity=severity, title=f"Planned arena head {label}",
            explanation_text=f"The plan uses {budget.planned_arena_head_bytes} of {budget.effective_budget_bytes} budget bytes.",
            evidence=(("effective_budget_bytes", budget.effective_budget_bytes),
                      ("planned_arena_head_bytes", budget.planned_arena_head_bytes),
                      ("remaining_bytes", budget.remaining_bytes),
                      ("utilization_percent", budget.utilization_percent), ("budget_status", budget.status)),
            impact=abs(budget.remaining_bytes), action="Prioritize the highest-ranked tensor and lifetime findings before deployment planning.",
            rationale="The selected budget applies specifically to planned arena head.",
            expected="Model-level reductions may improve arena-head margin.",
            caveat="This budget check does not establish complete MCU or firmware memory fit.",
        )

    return MemoryRiskAssessment(_sort_findings(findings), _sort_recommendations(recommendations))


def render_memory_guidance(
    guidance: MemoryRiskAssessment,
    *,
    details: bool = False,
    limit: int = 5,
) -> str:
    findings = guidance.findings if details else guidance.findings[:limit]
    recommendations = {item.recommendation_id: item for item in guidance.recommendations}
    lines = [
        "Memory risk and optimization guidance",
        f"Risk summary: {guidance.overall_risk.upper()}",
        f"Findings: {len(guidance.findings)}",
        f"Recommendations: {len(guidance.recommendations)}",
    ]
    if not findings:
        lines.append("No material model-level arena-head optimization finding was detected by the current rules.")
    for index, finding in enumerate(findings, start=1):
        lines.extend([
            "",
            f"{index}. {finding.title} [{finding.severity}, {finding.confidence}]",
            f"   {finding.explanation}",
        ])
        for recommendation_id in finding.recommendation_ids:
            recommendation = recommendations[recommendation_id]
            lines.extend([
                "   Recommendation:",
                f"   {recommendation.action} {recommendation.expected_effect}",
                f"   Caveat: {recommendation.caveats[0]}",
            ])
    if not details and len(guidance.findings) > limit:
        lines.append(f"\nShowing {limit} of {len(guidance.findings)} findings; use --details for all findings.")
    lines.extend([
        "",
        "Recommendations are evidence-based suggestions, not guaranteed byte savings.",
        "This guidance covers planned arena head only.",
        "Model accuracy, operator support, and graph semantics must be revalidated after any model change.",
    ])
    return "\n".join(lines)
