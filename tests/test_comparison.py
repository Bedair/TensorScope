from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tensorscope.comparison import (
    ComparisonInput, MetricDelta, ModelComparison, RegressionAssessment,
    compare_models, match_tensors,
)
from tensorscope.comparison_report import render_comparison_html
from tensorscope.explain import explain_primary_subgraph_memory
from tensorscope.graph import convert_tflite_model
from tensorscope.memory_budget import evaluate_direct_budget
from tensorscope.recommendations import assess_memory_risk
from tensorscope.tflite.model_loader import load_tflite_model


ROOT = Path(__file__).parents[1]
MODELS = ROOT / "tests" / "model_corpus" / "models"


def _input(name: str, budget: int | None = None) -> ComparisonInput:
    path = MODELS / name
    graph = convert_tflite_model(load_tflite_model(path))
    explanation = explain_primary_subgraph_memory(graph)
    budget_result = evaluate_direct_budget(explanation.summary.planned_arena_head_bytes, budget) if budget is not None else None
    guidance = assess_memory_risk(graph, explanation, budget=budget_result)
    return ComparisonInput(str(path.resolve()), graph, explanation, guidance, budget_result)


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "tensorscope", *arguments], cwd=ROOT,
        env=environment, text=True, capture_output=True, check=False,
    )


@pytest.mark.parametrize(
    ("baseline", "candidate", "direction", "percent"),
    [(100, 120, "increase", 20.0), (100, 80, "decrease", -20.0),
     (100, 100, "unchanged", 0.0), (0, 10, "increase", None)],
)
def test_metric_delta_math(baseline: int, candidate: int, direction: str, percent: float | None) -> None:
    delta = MetricDelta.calculate(baseline, candidate)
    assert delta.direction == direction
    assert delta.percent_delta == percent
    assert list(delta.to_dict()) == ["baseline", "candidate", "delta", "percent_delta", "direction"]


def test_domain_is_immutable_validates_unique_matches_and_serializes() -> None:
    comparison = compare_models(_input("hello_world_float.tflite"), _input("hello_world_float.tflite"))
    assert comparison.status == "unchanged"
    assert list(comparison.to_dict())[0] == "comparison_schema_version"
    with pytest.raises(FrozenInstanceError):
        comparison.status = "mixed"  # type: ignore[misc,assignment]
    with pytest.raises(ValueError, match="one-to-one"):
        replace(comparison, matches=(comparison.matches[0], comparison.matches[0]))


def test_tensor_matching_exact_unique_name_structural_and_changed_ids() -> None:
    baseline = _input("hello_world_float.tflite")
    allocations = baseline.explanation.allocations
    exact = match_tensors(baseline.graph, allocations, baseline.graph, allocations)
    assert all(item.confidence == "exact" for item in exact)

    candidate = tuple(
        replace(item, data_type="INT8", shape=(99,)) if item.tensor_id == allocations[0].tensor_id else item
        for item in allocations
    )
    unique_name = match_tensors(baseline.graph, allocations, baseline.graph, candidate)
    assert next(item for item in unique_name if item.baseline_tensor_id == allocations[0].tensor_id).confidence == "high"

    renamed = tuple(replace(item, name=f"candidate-{item.tensor_id}") for item in allocations)
    structural = match_tensors(baseline.graph, allocations, baseline.graph, renamed)
    assert {item.confidence for item in structural} == {"medium"}


def test_ambiguous_empty_structural_tensors_remain_unmatched() -> None:
    source = _input("hello_world_float.tflite")
    tensor = source.explanation.allocations[0]
    baseline = (replace(tensor, tensor_id=100, name=""), replace(tensor, tensor_id=101, name=""))
    candidate = (replace(tensor, tensor_id=200, name=""), replace(tensor, tensor_id=201, name=""))
    assert match_tensors(source.graph, baseline, source.graph, candidate) == ()


def test_tensor_deltas_capture_added_removed_size_lifetime_offset_and_roles() -> None:
    baseline = _input("hello_world_float.tflite")
    candidate_explanation = baseline.explanation
    first = candidate_explanation.allocations[0]
    changed = replace(
        first, aligned_bytes=first.aligned_bytes + 16, logical_bytes=first.logical_bytes + 4,
        lifetime_length=first.lifetime_length + 1, last_used_scope=first.last_used_scope + 1,
        offset=first.offset + 16, is_graph_output=not first.is_graph_output,
    )
    candidate_explanation = replace(candidate_explanation, allocations=(changed, *candidate_explanation.allocations[1:]))
    candidate = replace(
        baseline, model_path="candidate", explanation=candidate_explanation,
        guidance=assess_memory_risk(baseline.graph, candidate_explanation),
    )
    result = compare_models(baseline, candidate)
    delta = next(item for item in result.tensor_deltas if item.baseline_tensor_id == first.tensor_id)
    assert delta.aligned_bytes.delta == 16
    assert delta.lifetime_length.delta == 1
    assert delta.allocation_offset.delta == 16
    assert delta.graph_output_changed is True


def test_peak_operator_and_guidance_comparisons_for_real_pair() -> None:
    result = compare_models(_input("hello_world_float.tflite"), _input("operator_chain_float.tflite"))
    peak = dict(result.peak_comparison)
    operators = dict(result.operator_comparison)
    guidance = dict(result.guidance_comparison)
    assert isinstance(peak["peak_moved"], bool)
    assert operators["added_name_counts"]
    assert operators["sequences_equal"] is False
    assert guidance["severity_changes"] == [{
        "category": "alignment_overhead", "baseline_severity": "low",
        "candidate_severity": "medium", "direction": "increase",
    }]
    assert isinstance(guidance["recommendation_count_delta"], int)


def test_budget_comparison_fit_regression_and_improvement() -> None:
    regressed = compare_models(_input("micro_speech_quantized.tflite", 8192), _input("conv0.tflite", 8192))
    budget = dict(regressed.budget_comparison or ())
    assert budget["baseline_status"] == "fits"
    assert budget["candidate_status"] == "exceeds"
    assert budget["status_change"] == "regressed"
    assert budget["utilization_delta_percent"] > 0
    improved = compare_models(_input("conv0.tflite", 8192), _input("micro_speech_quantized.tflite", 8192))
    assert dict(improved.budget_comparison or ())["status_change"] == "improved"
    assert compare_models(_input("conv0.tflite"), _input("conv0.tflite")).budget_comparison is None


def test_regression_thresholds_improvement_unchanged_and_mixed() -> None:
    base = _input("simple_add_model.tflite")

    def with_head(value: int) -> ComparisonInput:
        explanation = replace(base.explanation, summary=replace(base.explanation.summary, planned_arena_head_bytes=value))
        return replace(base, model_path=str(value), explanation=explanation, guidance=assess_memory_risk(base.graph, explanation))

    assert compare_models(with_head(10000), with_head(10255)).regression.is_regression is False
    assert compare_models(with_head(10000), with_head(10400)).regression.is_regression is False
    assert compare_models(with_head(10000), with_head(10500)).status == "regressed"
    assert compare_models(with_head(10000), with_head(9400)).status == "improved"
    assert compare_models(with_head(10000), with_head(10000)).status == "unchanged"
    assert compare_models(with_head(10000), with_head(10100)).status == "mixed"


@pytest.mark.parametrize(
    ("baseline", "candidate"),
    [("hello_world_float.tflite", "hello_world_float.tflite"),
     ("hello_world_float.tflite", "hello_world_int8.tflite"),
     ("hello_world_float.tflite", "operator_chain_float.tflite"),
     ("micro_speech_quantized.tflite", "conv0.tflite")],
)
def test_corpus_pairs_are_deterministic(baseline: str, candidate: str) -> None:
    first = compare_models(_input(baseline), _input(candidate)).to_dict()
    second = compare_models(_input(baseline), _input(candidate)).to_dict()
    assert first == second


def test_cli_text_json_failure_and_output_before_exit(tmp_path: Path) -> None:
    baseline = str(MODELS / "micro_speech_quantized.tflite")
    candidate = str(MODELS / "conv0.tflite")
    text = _run("compare", baseline, candidate, "--arena-head-budget", "8KiB")
    assert text.returncode == 0
    assert "Model comparison" in text.stdout and "Comparison status: REGRESSED" in text.stdout
    assert "Delta: +4,464 bytes (+74.80%)" in text.stdout
    assert "Peak moved:" in text.stdout and "Tensor matching is deterministic" in text.stdout
    assert "Showing 5 of" in text.stdout
    detailed = _run("compare", baseline, candidate, "--details")
    assert "Showing 5 of" not in detailed.stdout
    assert "Operator changes" in detailed.stdout and "Guidance changes" in detailed.stdout
    failed = _run("compare", baseline, candidate, "--fail-on-regression")
    assert failed.returncode == 7
    assert "Comparison status: REGRESSED" in failed.stdout
    encoded = _run("compare", baseline, candidate, "--json")
    result = json.loads(encoded.stdout)
    assert result["comparison_schema_version"] == 1
    assert isinstance(result["metrics"]["planned_arena_head_bytes"]["delta"], int)
    assert result["budget_comparison"] is None


def test_cli_invalid_path_and_html_written_before_regression_exit(tmp_path: Path) -> None:
    invalid = _run("compare", "missing.tflite", str(MODELS / "conv0.tflite"))
    assert invalid.returncode == 2 and "Error (unsupported_input)" in invalid.stderr
    destination = tmp_path / "comparison.html"
    completed = _run(
        "compare", str(MODELS / "micro_speech_quantized.tflite"), str(MODELS / "conv0.tflite"),
        "--html", str(destination), "--fail-on-regression",
    )
    assert completed.returncode == 7 and destination.is_file()


def test_html_is_escaped_self_contained_deterministic_and_complete() -> None:
    result = compare_models(_input("micro_speech_quantized.tflite", 8192), _input("conv0.tflite", 8192))
    changed_tensor = replace(result.tensor_deltas[0], candidate_name='<img src=x onerror="bad">')
    dangerous = replace(result, baseline_model='<script>alert("model")</script>', tensor_deltas=(changed_tensor, *result.tensor_deltas[1:]))
    first = render_comparison_html(dangerous, tool_version="test")
    assert first == render_comparison_html(dangerous, tool_version="test")
    assert dangerous.baseline_model not in first
    assert "&lt;script&gt;alert(&quot;model&quot;)&lt;/script&gt;" in first
    assert "&lt;img src=x onerror=&quot;bad&quot;&gt;" in first
    assert '<svg id="arena-head-comparison-svg"' in first
    for phrase in ("Peak comparison", "Tensor changes", "Operator comparison", "Guidance comparison", "Arena-head budget comparison", "Regression assessment", "Limitations"):
        assert phrase in first
    assert "http://" not in first.lower() and "https://" not in first.lower()
    assert "<script" not in first.lower()
