from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import pytest

from tensorscope import __version__
from tensorscope.cli import analyze_model
from tensorscope.explain import explain_primary_subgraph_memory
from tensorscope.graph import convert_tflite_model
from tensorscope.html_report import (
    HTMLReportError,
    render_packing_svg,
    render_html_report,
    write_html_report,
)
from tensorscope.memory_budget import evaluate_direct_budget, evaluate_profile_budget, get_mcu_profile
from tensorscope.tflite.model_loader import load_tflite_model


CORPUS = Path(__file__).parent / "model_corpus" / "models"
FIXED_TIME = datetime(2026, 8, 1, 12, 34, 56, tzinfo=timezone.utc)


class _DocumentParser(HTMLParser):
    pass


class _SVGParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.svg_attributes: list[dict[str, str | None]] = []
        self.tensor_ids: list[int] = []
        self.tensor_rectangles = 0
        self.tensor_titles = 0
        self._inside_main_svg = False
        self._inside_tensor_rect = False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if tag == "svg" and attributes.get("id") == "arena-packing-svg":
            self._inside_main_svg = True
            self.svg_attributes.append(attributes)
        elif self._inside_main_svg and tag == "g" and "data-tensor-id" in attributes:
            self.tensor_ids.append(int(attributes["data-tensor-id"] or "-1"))
        elif self._inside_main_svg and tag == "rect" and attributes.get("class") == "tensor-rect":
            self.tensor_rectangles += 1
            self._inside_tensor_rect = True
        elif self._inside_tensor_rect and tag == "title":
            self.tensor_titles += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "rect":
            self._inside_tensor_rect = False
        elif tag == "svg":
            self._inside_main_svg = False


def _report(model_name: str = "hello_world_float.tflite") -> str:
    model = CORPUS / model_name
    graph = convert_tflite_model(load_tflite_model(model))
    explanation = explain_primary_subgraph_memory(graph)
    return render_html_report(
        analyze_model(model),
        explanation,
        tool_version=__version__,
        generated_at=FIXED_TIME,
    )


def test_report_is_complete_self_contained_html() -> None:
    report = _report()
    parser = _DocumentParser()
    parser.feed(report)
    parser.close()

    assert report.startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in report
    assert "<style>" in report
    assert "<script" not in report.lower()
    assert "http://" not in report.lower()
    assert "https://" not in report.lower()
    assert " src=" not in report.lower()
    assert " href=" not in report.lower()


def test_report_states_scope_confidence_and_fit_limitations() -> None:
    report = _report()

    assert "This report covers planned arena head only." in report
    assert "Arena head is statically calculated." in report
    assert "validation applies only when the TFLM oracle is run" in report
    assert "Arena tail is not statically estimated." in report
    assert "Complete arena total is not statically estimated." in report
    assert "does not prove complete MCU or firmware fit" in report
    assert "confidence: exact" in report
    assert "source: static_analysis" in report
    assert "validation: not_validated" in report


def test_report_contains_explainability_sections_and_known_plan() -> None:
    report = _report()

    assert "Planned arena head</dt><dd>128 bytes" in report
    assert "Peak execution point" in report
    assert "Live tensors at peak" in report
    assert "Largest runtime tensors" in report
    assert "Arena placement across execution scopes" in report
    assert "All planned runtime tensors" in report
    assert "Execution scopes" in report
    assert "Safe memory reuse" in report
    assert "[0, 16)" in report
    assert "[64, 80)" in report


def test_model_summary_contains_existing_reuse_counts() -> None:
    report = _report()

    assert "Safe reuse relationships</dt><dd>2" in report
    assert "Conservative reuse blockers</dt><dd>4" in report


def test_report_has_complete_visible_limitations_section() -> None:
    report = _report()

    assert '<section id="limitations"><h2>Limitations</h2>' in report
    required_phrases = (
        "primary subgraph",
        "planned arena head",
        "Arena tail",
        "Complete arena total",
        "Firmware stack",
        "heap usage",
        "DMA buffers",
        "RTOS memory",
        "Application memory",
        "complete MCU fit",
        "Reuse blockers are conservative explanations",
        "planner behavior implemented and pinned by TensorScope",
    )
    for phrase in required_phrases:
        assert phrase in report


def test_report_is_deterministic() -> None:
    assert _report() == _report()


@pytest.mark.parametrize(
    ("planned", "budget", "label"),
    [(127, 128, "FITS"), (128, 128, "EXACT FIT"), (129, 128, "EXCEEDS BUDGET")],
)
def test_budget_section_renders_every_textual_status(planned: int, budget: int, label: str) -> None:
    model = CORPUS / "hello_world_float.tflite"
    graph = convert_tflite_model(load_tflite_model(model))
    report = render_html_report(
        analyze_model(model), explain_primary_subgraph_memory(graph), tool_version=__version__,
        generated_at=FIXED_TIME, budget=evaluate_direct_budget(planned, budget),
    )
    assert '<section id="arena-head-budget">' in report
    assert f"Arena-head budget result: {label}" in report
    assert "This check covers planned arena head only." in report
    assert "This is not a complete MCU or firmware memory-fit conclusion." in report
    assert '<svg id="arena-packing-svg"' in report


def test_profile_budget_section_includes_escaped_profile_and_reserve() -> None:
    model = CORPUS / "hello_world_float.tflite"
    graph = convert_tflite_model(load_tflite_model(model))
    profile = replace(get_mcu_profile("cortex-m4-128k"), display_name='<Profile & "name">')
    report = render_html_report(
        analyze_model(model), explain_primary_subgraph_memory(graph), tool_version=__version__,
        generated_at=FIXED_TIME, budget=evaluate_profile_budget(128, profile, 32768),
    )
    assert "&lt;Profile &amp; &quot;name&quot;&gt;" in report
    assert "Profile RAM</dt><dd>131,072 bytes" in report
    assert "Reserved RAM</dt><dd>32,768 bytes" in report
    assert "Utilization</dt><dd>0.13%" in report


def test_report_without_budget_preserves_absence() -> None:
    assert 'id="arena-head-budget"' not in _report()


def test_main_svg_has_accessible_deterministic_tensor_rectangles() -> None:
    report = _report()
    parser = _SVGParser()
    parser.feed(report)

    assert len(parser.svg_attributes) == 1
    assert parser.svg_attributes[0]["viewbox"] == "0 0 1000 620"
    assert parser.tensor_ids == [0, 7, 8, 9]
    assert parser.tensor_rectangles == 4
    assert parser.tensor_titles == 4
    assert 'id="peak-scope-marker"' in report
    assert 'id="arena-head-boundary"' in report
    assert "Execution scope →" in report
    assert "Arena offset (bytes) →" in report
    assert "represent safe reuse" in report


def test_model_content_is_html_escaped() -> None:
    model = CORPUS / "hello_world_float.tflite"
    graph = convert_tflite_model(load_tflite_model(model))
    explanation = explain_primary_subgraph_memory(graph)
    dangerous = '<script>alert("tensor")</script>'
    changed = replace(explanation.allocations[0], name=dangerous)
    explanation = replace(
        explanation,
        allocations=(changed, *explanation.allocations[1:]),
    )

    report = render_html_report(
        analyze_model(model),
        explanation,
        tool_version=__version__,
        generated_at=FIXED_TIME,
    )

    assert dangerous not in report
    assert "&lt;script&gt;alert(&quot;tensor&quot;)&lt;/script&gt;" in report


def test_empty_allocations_produce_graceful_svg() -> None:
    model = CORPUS / "hello_world_float.tflite"
    graph = convert_tflite_model(load_tflite_model(model))
    explanation = explain_primary_subgraph_memory(graph)
    empty = replace(
        explanation,
        summary=replace(
            explanation.summary,
            runtime_tensor_count=0,
            planned_arena_head_bytes=0,
            logical_runtime_tensor_bytes=0,
            aligned_runtime_tensor_bytes=0,
            alignment_overhead_bytes=0,
        ),
        allocations=(),
    )

    svg = render_packing_svg(empty)

    assert svg.startswith('<svg id="arena-packing-svg"')
    assert 'viewBox="0 0 1000 620"' in svg
    assert "No runtime allocations" in svg
    assert 'class="tensor-rect"' not in svg
    assert 'id="peak-scope-marker"' in svg


def test_header_metadata_uses_fixed_utc_timestamp_and_schema() -> None:
    report = _report()

    assert "hello_world_float.tflite" in report
    assert str((CORPUS / "hello_world_float.tflite").resolve()) in report
    assert "TFLite schema version</dt><dd>3" in report
    assert "Generated at</dt><dd>2026-08-01T12:34:56Z" in report
    assert f"TensorScope version</dt><dd>{__version__}" in report
    assert "Primary subgraph · planned arena head only" in report


def test_write_report_uses_utf8_and_returns_resolved_path(tmp_path: Path) -> None:
    destination = tmp_path / "report.html"
    report = _report()

    written = write_html_report(destination, report)

    assert written == destination.resolve()
    assert destination.read_text(encoding="utf-8") == report


def test_write_report_failure_is_clear(tmp_path: Path) -> None:
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")
    destination = blocked_parent / "report.html"

    with pytest.raises(HTMLReportError, match="Unable to write HTML report"):
        write_html_report(destination, _report())


@pytest.mark.parametrize(
    "model_name",
    [
        "hello_world_float.tflite",
        "conv0.tflite",
        "micro_speech_quantized.tflite",
        "operator_chain_float.tflite",
        "quantize_dequantize_int8.tflite",
    ],
)
def test_corpus_reports_render(model_name: str) -> None:
    report = _report(model_name)

    assert f"TensorScope analysis · {model_name}" in report
    assert "This report covers planned arena head only." in report
    assert "Arena placement across execution scopes" in report
