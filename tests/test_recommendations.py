from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from tensorscope.cli import analyze_model
from tensorscope.explain import explain_primary_subgraph_memory
from tensorscope.graph import convert_tflite_model
from tensorscope.memory_budget import evaluate_direct_budget
from tensorscope.recommendations import (
    MemoryFinding,
    MemoryRecommendation,
    MemoryRiskAssessment,
    assess_memory_risk,
    render_memory_guidance,
)
from tensorscope.tflite.model_loader import load_tflite_model


MODELS = Path(__file__).parent / "model_corpus" / "models"


def _analysis(name: str = "hello_world_float.tflite"):
    graph = convert_tflite_model(load_tflite_model(MODELS / name))
    return graph, explain_primary_subgraph_memory(graph)


def _finding(**changes: object) -> MemoryFinding:
    values = {
        "finding_id": "finding-a", "category": "general_summary", "severity": "low",
        "confidence": "exact", "title": "Title", "explanation": "Evidence",
        "recommendation_ids": ("recommendation-a",),
    }
    values.update(changes)
    return MemoryFinding(**values)  # type: ignore[arg-type]


def _recommendation(**changes: object) -> MemoryRecommendation:
    values = {
        "recommendation_id": "recommendation-a", "category": "general_summary",
        "priority": "low", "confidence": "medium", "action": "Review evidence.",
        "rationale": "Known data.", "expected_effect": "May reduce pressure.",
        "caveats": ("Revalidate.",), "linked_finding_ids": ("finding-a",),
    }
    values.update(changes)
    return MemoryRecommendation(**values)  # type: ignore[arg-type]


def test_domain_models_are_immutable_serializable_and_linked() -> None:
    result = MemoryRiskAssessment((_finding(),), (_recommendation(),))
    assert result.overall_risk == "low"
    assert list(result.to_dict()) == ["scope", "overall_risk", "summary", "findings", "recommendations"]
    assert result.to_dict()["findings"][0]["finding_id"] == "finding-a"
    with pytest.raises(FrozenInstanceError):
        result.scope = "other"  # type: ignore[misc,assignment]
    with pytest.raises(ValueError, match="unknown recommendation"):
        MemoryRiskAssessment((_finding(recommendation_ids=("missing",)),), (_recommendation(),))


def test_empty_result_is_informational() -> None:
    result = MemoryRiskAssessment((), ())
    assert result.overall_risk == "info"
    assert result.to_dict()["summary"] == {
        "finding_count": 0, "recommendation_count": 0, "high_or_critical_count": 0,
    }
    assert "No material model-level" in render_memory_guidance(result)


@pytest.mark.parametrize("field", ["severity", "confidence"])
def test_invalid_severity_or_confidence_is_rejected(field: str) -> None:
    with pytest.raises(ValueError):
        _finding(**{field: "invalid"})


def test_peak_concentration_alignment_and_reuse_are_deterministic() -> None:
    graph, explanation = _analysis()
    first = assess_memory_risk(graph, explanation)
    second = assess_memory_risk(graph, explanation)
    assert first.to_dict() == second.to_dict()
    ids = [item.finding_id for item in first.findings]
    assert ids[:4] == [
        "peak-concentration-t7", "reuse-blocker-tensor-7",
        "reuse-blocker-tensor-8", "alignment-overhead",
    ]
    peak = first.findings[0]
    assert dict(peak.evidence)["share_percent"] == 50.0
    assert all("exact byte" not in item.expected_effect.lower() for item in first.recommendations)


def test_zero_peak_is_safe_and_distributed_peak_uses_top_three() -> None:
    graph, explanation = _analysis()
    zero = replace(
        explanation, peak=replace(explanation.peak, live_aligned_bytes=0),
        live_tensors_at_peak=(), summary=replace(explanation.summary, planned_arena_head_bytes=0),
    )
    assert not any(item.category == "peak_concentration" for item in assess_memory_risk(graph, zero).findings)

    tensors = tuple(replace(item, aligned_bytes=10, logical_bytes=10) for item in explanation.allocations[:3])
    distributed = replace(explanation, peak=replace(explanation.peak, live_aligned_bytes=30), live_tensors_at_peak=tensors)
    finding = next(item for item in assess_memory_risk(graph, distributed).findings if item.category == "peak_concentration")
    assert len(finding.affected_tensor_ids) == 3


def test_long_lived_threshold_distinguishes_size_and_lifetime() -> None:
    graph, explanation = _analysis()
    base = explanation.allocations[1]
    large_long = replace(base, aligned_bytes=64, first_used_scope=0, last_used_scope=3, lifetime_length=4, is_graph_input=True)
    small_long = replace(explanation.allocations[0], aligned_bytes=16, first_used_scope=0, last_used_scope=3, lifetime_length=4)
    short = replace(explanation.allocations[2], aligned_bytes=64, lifetime_length=2)
    changed = replace(explanation, allocations=(large_long, small_long, short, explanation.allocations[3]))
    result = assess_memory_risk(graph, changed)
    long_ids = [item.finding_id for item in result.findings if item.category == "long_lived_tensor"]
    assert long_ids == [f"long-lived-tensor-{large_long.tensor_id}"]
    assert any(item.category == "graph_input_retention" for item in result.findings)


@pytest.mark.parametrize(
    ("budget", "severity", "status"),
    [(64, "critical", "exceeds"), (128, "high", "exact_fit"), (140, "high", "fits"),
     (160, "medium", "fits"), (256, "info", "fits")],
)
def test_budget_pressure_thresholds(budget: int, severity: str, status: str) -> None:
    graph, explanation = _analysis()
    result = assess_memory_risk(graph, explanation, budget=evaluate_direct_budget(128, budget))
    finding = next(item for item in result.findings if item.category == "budget_pressure")
    assert finding.severity == severity
    assert dict(finding.evidence)["budget_status"] == status


def test_zero_and_missing_budget_are_handled() -> None:
    graph, explanation = _analysis()
    assert not any(item.category == "budget_pressure" for item in assess_memory_risk(graph, explanation).findings)
    finding = next(item for item in assess_memory_risk(
        graph, explanation, budget=evaluate_direct_budget(128, 0)
    ).findings if item.category == "budget_pressure")
    assert dict(finding.evidence)["utilization_percent"] is None


def test_add_merge_and_large_required_output_are_reported() -> None:
    graph, explanation = _analysis("simple_add_model.tflite")
    result = assess_memory_risk(graph, explanation)
    merge = next(item for item in result.findings if item.category == "branch_merge_pressure")
    output = next(item for item in result.findings if item.category == "graph_output_retention")
    assert merge.finding_id == "branch-merge-add-0"
    assert dict(merge.evidence)["merge_operator"] == "ADD"
    recommendation = next(item for item in result.recommendations if item.recommendation_id in output.recommendation_ids)
    assert "required graph output" in recommendation.caveats[0]


def test_mul_merge_uses_exact_input_evidence() -> None:
    graph, explanation = _analysis("operator_chain_float.tflite")
    merge = next(operator for operator in graph.primary_subgraph.operators if operator.name == "MUL")
    runtime_inputs = (17, 19)
    operators = tuple(
        replace(operator, inputs=runtime_inputs) if operator.id == merge.id else operator
        for operator in graph.primary_subgraph.operators
    )
    graph = replace(graph, subgraphs=(replace(graph.primary_subgraph, operators=operators),))
    changed_allocations = tuple(
        replace(item, aligned_bytes=64) if item.tensor_id in runtime_inputs else item
        for item in explanation.allocations
    )
    changed = replace(explanation, allocations=changed_allocations)
    result = assess_memory_risk(graph, changed)
    finding = next(item for item in result.findings if item.finding_id == f"branch-merge-mul-{merge.id}")
    assert finding.affected_tensor_ids == runtime_inputs
    assert dict(finding.evidence)["input_aligned_bytes"] == 128


@pytest.mark.parametrize(
    ("model", "finding", "recommendation"),
    [
        ("hello_world_float.tflite", "peak-concentration-t7", "review-peak-tensors"),
        ("micro_speech_quantized.tflite", "peak-concentration-t3", "review-reuse-blocker-2"),
        ("conv0.tflite", "graph-output-retention-3", "review-output-shape-3"),
        ("operator_chain_float.tflite", "reuse-blocker-tensor-3", "review-reuse-blocker-3"),
        ("simple_add_model.tflite", "branch-merge-add-0", "narrow-merge-0"),
    ],
)
def test_corpus_guidance_stable_ids(model: str, finding: str, recommendation: str) -> None:
    guidance = analyze_model(MODELS / model)["memory_guidance"]
    assert finding in {item["finding_id"] for item in guidance["findings"]}
    assert recommendation in {item["recommendation_id"] for item in guidance["recommendations"]}
