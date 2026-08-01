from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from typing import Literal

from tensorscope.explain import MemoryExplanation, TensorExplanation
from tensorscope.graph import GraphModel
from tensorscope.memory_budget import ArenaHeadBudgetResult
from tensorscope.recommendations import MemoryRiskAssessment


Direction = Literal["increase", "decrease", "unchanged", "unavailable"]
MatchConfidence = Literal["exact", "high", "medium", "unmatched"]
ComparisonStatus = Literal["improved", "unchanged", "mixed", "regressed"]


@dataclass(frozen=True)
class MetricDelta:
    baseline: int | float | None
    candidate: int | float | None
    delta: int | float | None
    percent_delta: float | None
    direction: Direction

    @classmethod
    def calculate(cls, baseline: int | float | None, candidate: int | float | None) -> MetricDelta:
        if baseline is None or candidate is None:
            return cls(baseline, candidate, None, None, "unavailable")
        delta = candidate - baseline
        direction: Direction = "increase" if delta > 0 else "decrease" if delta < 0 else "unchanged"
        percent = delta * 100 / baseline if baseline != 0 else None
        return cls(baseline, candidate, delta, percent, direction)

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline, "candidate": self.candidate,
            "delta": self.delta, "percent_delta": self.percent_delta,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class TensorMatch:
    baseline_tensor_id: int
    candidate_tensor_id: int
    confidence: MatchConfidence
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_tensor_id": self.baseline_tensor_id,
            "candidate_tensor_id": self.candidate_tensor_id,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TensorDelta:
    status: Literal["matched", "added", "removed"]
    baseline_tensor_id: int | None
    candidate_tensor_id: int | None
    baseline_name: str | None
    candidate_name: str | None
    match_confidence: MatchConfidence
    match_reason: str
    logical_bytes: MetricDelta
    aligned_bytes: MetricDelta
    alignment_overhead_bytes: MetricDelta
    first_scope: MetricDelta
    last_scope: MetricDelta
    lifetime_length: MetricDelta
    allocation_offset: MetricDelta
    graph_input_changed: bool | None
    graph_output_changed: bool | None
    impact_score: int

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "baseline_tensor_id": self.baseline_tensor_id,
            "candidate_tensor_id": self.candidate_tensor_id,
            "baseline_name": self.baseline_name,
            "candidate_name": self.candidate_name,
            "match_confidence": self.match_confidence,
            "match_reason": self.match_reason,
            "logical_bytes": self.logical_bytes.to_dict(),
            "aligned_bytes": self.aligned_bytes.to_dict(),
            "alignment_overhead_bytes": self.alignment_overhead_bytes.to_dict(),
            "first_scope": self.first_scope.to_dict(),
            "last_scope": self.last_scope.to_dict(),
            "lifetime_length": self.lifetime_length.to_dict(),
            "allocation_offset": self.allocation_offset.to_dict(),
            "graph_input_changed": self.graph_input_changed,
            "graph_output_changed": self.graph_output_changed,
            "impact_score": self.impact_score,
        }


@dataclass(frozen=True)
class RegressionAssessment:
    is_regression: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"is_regression": self.is_regression, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class ModelComparison:
    baseline_model: str
    candidate_model: str
    status: ComparisonStatus
    regression: RegressionAssessment
    metrics: tuple[tuple[str, MetricDelta], ...]
    matches: tuple[TensorMatch, ...]
    tensor_deltas: tuple[TensorDelta, ...]
    peak_comparison: tuple[tuple[str, object], ...]
    operator_comparison: tuple[tuple[str, object], ...]
    guidance_comparison: tuple[tuple[str, object], ...]
    budget_comparison: tuple[tuple[str, object], ...] | None
    comparison_schema_version: int = 1

    def __post_init__(self) -> None:
        baseline_ids = [item.baseline_tensor_id for item in self.matches]
        candidate_ids = [item.candidate_tensor_id for item in self.matches]
        if len(set(baseline_ids)) != len(baseline_ids) or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Tensor matches must be one-to-one")

    def to_dict(self) -> dict[str, object]:
        matched = len(self.matches)
        added = sum(item.status == "added" for item in self.tensor_deltas)
        removed = sum(item.status == "removed" for item in self.tensor_deltas)
        return {
            "comparison_schema_version": self.comparison_schema_version,
            "baseline_model": self.baseline_model,
            "candidate_model": self.candidate_model,
            "status": self.status,
            "regression": self.regression.to_dict(),
            "metrics": {key: value.to_dict() for key, value in self.metrics},
            "tensor_matching": {
                "matched_count": matched, "added_count": added, "removed_count": removed,
                "matches": [item.to_dict() for item in self.matches],
            },
            "tensor_deltas": [item.to_dict() for item in self.tensor_deltas],
            "peak_comparison": dict(self.peak_comparison),
            "operator_comparison": dict(self.operator_comparison),
            "guidance_comparison": dict(self.guidance_comparison),
            "budget_comparison": dict(self.budget_comparison) if self.budget_comparison is not None else None,
        }


@dataclass(frozen=True)
class ComparisonInput:
    model_path: str
    graph: GraphModel
    explanation: MemoryExplanation
    guidance: MemoryRiskAssessment
    budget: ArenaHeadBudgetResult | None = None


def normalize_tensor_name(value: str) -> str:
    rendered = re.sub(r"\s+", " ", value.strip()).casefold()
    return rendered if rendered and rendered not in {"<unnamed>", "unnamed"} else ""


def _producer_consumers(graph: GraphModel) -> tuple[dict[int, str], dict[int, tuple[str, ...]]]:
    producers: dict[int, str] = {}
    consumers: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for operator in graph.primary_subgraph.operators:
        for tensor_id in operator.outputs:
            producers[tensor_id] = operator.name
        for tensor_id in operator.inputs:
            if tensor_id >= 0:
                consumers[tensor_id].append((operator.id, operator.name))
    return producers, {
        tensor_id: tuple(name for _, name in sorted(items))
        for tensor_id, items in consumers.items()
    }


def match_tensors(
    baseline_graph: GraphModel,
    baseline: tuple[TensorExplanation, ...],
    candidate_graph: GraphModel,
    candidate: tuple[TensorExplanation, ...],
) -> tuple[TensorMatch, ...]:
    unmatched_baseline = {item.tensor_id: item for item in baseline}
    unmatched_candidate = {item.tensor_id: item for item in candidate}
    matches: list[TensorMatch] = []
    baseline_producers, baseline_consumers = _producer_consumers(baseline_graph)
    candidate_producers, candidate_consumers = _producer_consumers(candidate_graph)

    def apply_tier(key_function, confidence: MatchConfidence, reason: str) -> None:
        baseline_groups: dict[object, list[TensorExplanation]] = defaultdict(list)
        candidate_groups: dict[object, list[TensorExplanation]] = defaultdict(list)
        for tensor in unmatched_baseline.values():
            key = key_function(tensor, baseline_producers, baseline_consumers)
            if key is not None:
                baseline_groups[key].append(tensor)
        for tensor in unmatched_candidate.values():
            key = key_function(tensor, candidate_producers, candidate_consumers)
            if key is not None:
                candidate_groups[key].append(tensor)
        for key in sorted(set(baseline_groups) & set(candidate_groups), key=repr):
            left, right = baseline_groups[key], candidate_groups[key]
            if len(left) != 1 or len(right) != 1:
                continue
            baseline_tensor, candidate_tensor = left[0], right[0]
            matches.append(TensorMatch(baseline_tensor.tensor_id, candidate_tensor.tensor_id, confidence, reason))
            del unmatched_baseline[baseline_tensor.tensor_id]
            del unmatched_candidate[candidate_tensor.tensor_id]

    def exact_key(tensor, producers, consumers):
        name = normalize_tensor_name(tensor.name)
        return (name, tensor.data_type, tensor.shape) if name else None

    def name_key(tensor, producers, consumers):
        return normalize_tensor_name(tensor.name) or None

    def structural_key(tensor, producers, consumers):
        return (
            tensor.data_type, tensor.shape, tensor.is_graph_input, tensor.is_graph_output,
            producers.get(tensor.tensor_id), consumers.get(tensor.tensor_id, ()),
        )

    apply_tier(exact_key, "exact", "normalized_name_type_shape")
    apply_tier(name_key, "high", "unique_normalized_name")
    apply_tier(structural_key, "medium", "unique_structural_signature")
    return tuple(sorted(matches, key=lambda item: (item.baseline_tensor_id, item.candidate_tensor_id)))


def _tensor_delta(
    baseline: TensorExplanation | None,
    candidate: TensorExplanation | None,
    match: TensorMatch | None,
) -> TensorDelta:
    def metric(attribute: str) -> MetricDelta:
        return MetricDelta.calculate(
            getattr(baseline, attribute) if baseline is not None else None,
            getattr(candidate, attribute) if candidate is not None else None,
        )

    if baseline is None:
        status, confidence, reason = "added", "unmatched", "candidate_only"
        impact = candidate.aligned_bytes if candidate else 0
    elif candidate is None:
        status, confidence, reason = "removed", "unmatched", "baseline_only"
        impact = baseline.aligned_bytes
    else:
        status, confidence, reason = "matched", match.confidence, match.reason  # type: ignore[union-attr]
        aligned_delta = abs(candidate.aligned_bytes - baseline.aligned_bytes)
        lifetime_delta = abs(candidate.lifetime_length - baseline.lifetime_length)
        impact = aligned_delta + lifetime_delta * max(baseline.aligned_bytes, candidate.aligned_bytes)
    return TensorDelta(
        status, baseline.tensor_id if baseline else None, candidate.tensor_id if candidate else None,
        baseline.name if baseline else None, candidate.name if candidate else None,
        confidence, reason, metric("logical_bytes"), metric("aligned_bytes"),
        metric("alignment_overhead_bytes"), metric("first_used_scope"),
        metric("last_used_scope"), metric("lifetime_length"), metric("offset"),
        (baseline.is_graph_input != candidate.is_graph_input) if baseline and candidate else None,
        (baseline.is_graph_output != candidate.is_graph_output) if baseline and candidate else None,
        impact,
    )


def _metrics(explanation: MemoryExplanation) -> dict[str, int]:
    summary = explanation.summary
    return {
        "planned_arena_head_bytes": summary.planned_arena_head_bytes,
        "peak_occupied_extent_bytes": explanation.peak.occupied_extent_bytes,
        "peak_live_aligned_bytes": explanation.peak.live_aligned_bytes,
        "runtime_tensor_count": summary.runtime_tensor_count,
        "constant_tensor_count": summary.constant_tensor_count,
        "operator_count": summary.operator_count,
        "logical_runtime_tensor_bytes": summary.logical_runtime_tensor_bytes,
        "aligned_runtime_tensor_bytes": summary.aligned_runtime_tensor_bytes,
        "alignment_overhead_bytes": summary.alignment_overhead_bytes,
        "safe_reuse_pair_count": len(explanation.reuse),
        "reuse_blocker_count": len(explanation.reuse_blockers),
    }


def _peak_comparison(
    baseline: MemoryExplanation,
    candidate: MemoryExplanation,
    matches: tuple[TensorMatch, ...],
) -> tuple[tuple[str, object], ...]:
    mapping = {item.baseline_tensor_id: item.candidate_tensor_id for item in matches}
    baseline_peak = set(baseline.peak.live_tensor_ids)
    candidate_peak = set(candidate.peak.live_tensor_ids)
    retained = tuple(sorted((left, right) for left, right in mapping.items() if left in baseline_peak and right in candidate_peak))
    baseline_only = tuple(sorted(item for item in baseline_peak if item not in {left for left, _ in retained}))
    candidate_only = tuple(sorted(item for item in candidate_peak if item not in {right for _, right in retained}))
    moved = (
        baseline.peak.scope != candidate.peak.scope
        or baseline.peak.operator_name != candidate.peak.operator_name
    )
    return (
        ("baseline_scope", baseline.peak.scope), ("candidate_scope", candidate.peak.scope),
        ("baseline_operator_name", baseline.peak.operator_name),
        ("candidate_operator_name", candidate.peak.operator_name), ("peak_moved", moved),
        ("retained_peak_tensor_matches", [list(item) for item in retained]),
        ("baseline_only_peak_tensor_ids", list(baseline_only)),
        ("candidate_only_peak_tensor_ids", list(candidate_only)),
    )


def _operator_comparison(baseline: GraphModel, candidate: GraphModel) -> tuple[tuple[str, object], ...]:
    left = Counter(item.name for item in baseline.primary_subgraph.operators)
    right = Counter(item.name for item in candidate.primary_subgraph.operators)
    names = sorted(set(left) | set(right))
    added = {name: right[name] - left[name] for name in names if right[name] > left[name]}
    removed = {name: left[name] - right[name] for name in names if left[name] > right[name]}
    unchanged = {name: min(left[name], right[name]) for name in names if min(left[name], right[name])}
    return (
        ("added_name_counts", added), ("removed_name_counts", removed),
        ("unchanged_name_counts", unchanged),
        ("baseline_sequence", [item.name for item in baseline.primary_subgraph.operators]),
        ("candidate_sequence", [item.name for item in candidate.primary_subgraph.operators]),
        ("sequences_equal", tuple(item.name for item in baseline.primary_subgraph.operators) == tuple(item.name for item in candidate.primary_subgraph.operators)),
    )


def _guidance_comparison(baseline: MemoryRiskAssessment, candidate: MemoryRiskAssessment) -> tuple[tuple[str, object], ...]:
    severity_rank = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    left: dict[str, str] = {}
    right: dict[str, str] = {}
    for item in baseline.findings:
        if item.category not in left or severity_rank[item.severity] > severity_rank[left[item.category]]:
            left[item.category] = item.severity
    for item in candidate.findings:
        if item.category not in right or severity_rank[item.severity] > severity_rank[right[item.category]]:
            right[item.category] = item.severity
    shared = sorted(set(left) & set(right))
    changes = [
        {"category": category, "baseline_severity": left[category], "candidate_severity": right[category],
         "direction": "increase" if severity_rank[right[category]] > severity_rank[left[category]] else "decrease"}
        for category in shared if left[category] != right[category]
    ]
    return (
        ("introduced_categories", sorted(set(right) - set(left))),
        ("resolved_categories", sorted(set(left) - set(right))),
        ("severity_changes", changes),
        ("recommendation_count_delta", len(candidate.recommendations) - len(baseline.recommendations)),
        ("high_or_critical_count_delta", sum(item.severity in {"high", "critical"} for item in candidate.findings) - sum(item.severity in {"high", "critical"} for item in baseline.findings)),
    )


def _budget_comparison(
    baseline: ArenaHeadBudgetResult | None,
    candidate: ArenaHeadBudgetResult | None,
) -> tuple[tuple[str, object], ...] | None:
    if baseline is None or candidate is None:
        return None
    rank = {"fits": 0, "exact_fit": 1, "exceeds": 2}
    direction = "regressed" if rank[candidate.status] > rank[baseline.status] else "improved" if rank[candidate.status] < rank[baseline.status] else "unchanged"
    utilization_delta = (
        candidate.utilization_percent - baseline.utilization_percent
        if candidate.utilization_percent is not None and baseline.utilization_percent is not None else None
    )
    return (
        ("scope", "arena_head"), ("effective_budget_bytes", baseline.effective_budget_bytes),
        ("baseline_status", baseline.status), ("candidate_status", candidate.status),
        ("baseline_utilization_percent", baseline.utilization_percent),
        ("candidate_utilization_percent", candidate.utilization_percent),
        ("utilization_delta_percent", utilization_delta), ("status_change", direction),
    )


def compare_models(baseline: ComparisonInput, candidate: ComparisonInput) -> ModelComparison:
    baseline_metrics, candidate_metrics = _metrics(baseline.explanation), _metrics(candidate.explanation)
    metric_deltas = tuple(
        (name, MetricDelta.calculate(baseline_metrics[name], candidate_metrics[name]))
        for name in baseline_metrics
    )
    matches = match_tensors(
        baseline.graph, baseline.explanation.allocations,
        candidate.graph, candidate.explanation.allocations,
    )
    baseline_by_id = {item.tensor_id: item for item in baseline.explanation.allocations}
    candidate_by_id = {item.tensor_id: item for item in candidate.explanation.allocations}
    matched_baseline = {item.baseline_tensor_id for item in matches}
    matched_candidate = {item.candidate_tensor_id for item in matches}
    deltas = [
        _tensor_delta(baseline_by_id[item.baseline_tensor_id], candidate_by_id[item.candidate_tensor_id], item)
        for item in matches
    ]
    deltas.extend(_tensor_delta(None, item, None) for item in candidate.explanation.allocations if item.tensor_id not in matched_candidate)
    deltas.extend(_tensor_delta(item, None, None) for item in baseline.explanation.allocations if item.tensor_id not in matched_baseline)
    tensor_deltas = tuple(sorted(deltas, key=lambda item: (
        -item.impact_score, {"added": 0, "removed": 1, "matched": 2}[item.status],
        item.baseline_tensor_id if item.baseline_tensor_id is not None else 1 << 30,
        item.candidate_tensor_id if item.candidate_tensor_id is not None else 1 << 30,
    )))
    budget_comparison = _budget_comparison(baseline.budget, candidate.budget)
    head = dict(metric_deltas)["planned_arena_head_bytes"]
    reasons: list[str] = []
    qualifies_increase = bool(
        head.delta is not None and head.delta >= 256
        and head.percent_delta is not None and head.percent_delta >= 5
    )
    if qualifies_increase:
        reasons.append("candidate planned arena head increased by at least 256 bytes and 5%")
    if baseline.budget and candidate.budget and baseline.budget.status in {"fits", "exact_fit"} and candidate.budget.status == "exceeds":
        reasons.append("candidate arena-head budget status changed from fitting to exceeds")
    baseline_critical_budget = any(item.category == "budget_pressure" and item.severity == "critical" for item in baseline.guidance.findings)
    candidate_critical_budget = any(item.category == "budget_pressure" and item.severity == "critical" for item in candidate.guidance.findings)
    if candidate_critical_budget and not baseline_critical_budget:
        reasons.append("candidate introduced a critical budget-pressure finding")
    regression = RegressionAssessment(bool(reasons), tuple(reasons))
    qualifies_improvement = bool(
        head.delta is not None and head.delta <= -256
        and head.percent_delta is not None and head.percent_delta <= -5
        and not regression.is_regression
    )
    guidance = _guidance_comparison(baseline.guidance, candidate.guidance)
    operators = _operator_comparison(baseline.graph, candidate.graph)
    any_change = any(item.direction != "unchanged" for _, item in metric_deltas) or any(
        item.status != "matched" or item.impact_score for item in tensor_deltas
    ) or not dict(operators)["sequences_equal"] or any(dict(guidance)[key] for key in ("introduced_categories", "resolved_categories", "severity_changes"))
    status: ComparisonStatus = "regressed" if regression.is_regression else "improved" if qualifies_improvement else "mixed" if any_change else "unchanged"
    return ModelComparison(
        baseline.model_path, candidate.model_path, status, regression, metric_deltas,
        matches, tensor_deltas, _peak_comparison(baseline.explanation, candidate.explanation, matches),
        operators, guidance, budget_comparison,
    )


def render_comparison_text(comparison: ModelComparison, *, details: bool = False, limit: int = 5) -> str:
    metrics = dict(comparison.metrics)
    head = metrics["planned_arena_head_bytes"]
    percent = "not available" if head.percent_delta is None else f"{head.percent_delta:+.2f}%"
    lines = [
        "Model comparison", "", f"Baseline: {comparison.baseline_model}",
        f"Candidate: {comparison.candidate_model}", f"Comparison status: {comparison.status.upper()}",
        "", "Arena-head summary", f"  Baseline: {head.baseline:,} bytes",
        f"  Candidate: {head.candidate:,} bytes", f"  Delta: {head.delta:+,} bytes ({percent})",
        "", "Key metric changes",
    ]
    labels = {
        "peak_occupied_extent_bytes": "Peak occupied extent", "peak_live_aligned_bytes": "Peak live aligned bytes",
        "runtime_tensor_count": "Runtime tensors", "constant_tensor_count": "Constant tensors",
        "operator_count": "Operators", "logical_runtime_tensor_bytes": "Logical runtime bytes",
        "aligned_runtime_tensor_bytes": "Aligned runtime bytes", "alignment_overhead_bytes": "Alignment overhead",
        "safe_reuse_pair_count": "Safe reuse pairs", "reuse_blocker_count": "Reuse blockers",
    }
    for name, label in labels.items():
        delta = metrics[name]
        lines.append(f"  {label}: {delta.delta:+}")
    peak = dict(comparison.peak_comparison)
    lines.extend([
        "", "Peak change",
        f"  Baseline peak: scope {peak['baseline_scope']} ({peak['baseline_operator_name'] or 'subgraph input'})",
        f"  Candidate peak: scope {peak['candidate_scope']} ({peak['candidate_operator_name'] or 'subgraph input'})",
        f"  Peak moved: {'yes' if peak['peak_moved'] else 'no'}", "", "Top tensor changes",
    ])
    selected = comparison.tensor_deltas if details else comparison.tensor_deltas[:limit]
    for item in selected:
        name = item.candidate_name if item.candidate_name is not None else item.baseline_name
        delta = item.aligned_bytes.delta
        rendered_delta = "added" if item.status == "added" else "removed" if item.status == "removed" else f"{delta:+} aligned bytes"
        lines.append(f"  {item.status.upper()}: {name or '<unnamed>'} — {rendered_delta} [{item.match_confidence}]" )
    if not details and len(comparison.tensor_deltas) > limit:
        lines.append(f"  Showing {limit} of {len(comparison.tensor_deltas)} tensor changes; use --details for all changes.")
    operator = dict(comparison.operator_comparison)
    guidance = dict(comparison.guidance_comparison)
    lines.extend([
        "", "Operator changes",
        f"  Added name counts: {operator['added_name_counts']}",
        f"  Removed name counts: {operator['removed_name_counts']}",
        "", "Guidance changes",
        f"  Introduced categories: {guidance['introduced_categories']}",
        f"  Resolved categories: {guidance['resolved_categories']}",
        f"  Severity changes: {guidance['severity_changes']}",
    ])
    if comparison.regression.reasons:
        lines.extend(["", "Regression reasons", *(f"  - {item}" for item in comparison.regression.reasons)])
    if comparison.budget_comparison is not None:
        budget = dict(comparison.budget_comparison)
        lines.extend(["", "Arena-head budget comparison", f"  Baseline: {budget['baseline_status']}", f"  Candidate: {budget['candidate_status']}", f"  Status change: {budget['status_change']}"])
    lines.extend([
        "", "Comparison covers planned arena head only.",
        "Tensor matching is deterministic but does not prove semantic equivalence.",
        "Model accuracy, operator support, and graph semantics must be validated separately.",
        "Budget results do not establish complete MCU or firmware memory fit.",
    ])
    return "\n".join(lines)
