from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tensorscope.cli import (
    EXIT_BUDGET_EXCEEDED,
    EXIT_INPUT_ERROR,
    EXIT_REPORT_ERROR,
    EXIT_SUCCESS,
    TFLM_REVISION,
    analyze_model,
    main,
    validate_model,
)


MODEL = (
    Path(__file__).parent
    / "model_corpus"
    / "models"
    / "hello_world_float.tflite"
)
OPERATOR_CHAIN_MODEL = Path(__file__).parent / "model_corpus" / "models" / "operator_chain_float.tflite"
CONV0_MODEL = Path(__file__).parent / "model_corpus" / "models" / "conv0.tflite"
MICRO_SPEECH_MODEL = Path(__file__).parent / "model_corpus" / "models" / "micro_speech_quantized.tflite"
REPOSITORY_ROOT = Path(__file__).parents[1]


def _run_package(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = (
        os.pathsep.join((source_root, existing_pythonpath))
        if existing_pythonpath
        else source_root
    )
    return subprocess.run(
        [sys.executable, "-m", "tensorscope", *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_package_module_help_executes_successfully() -> None:
    completed = _run_package("--help")

    assert completed.returncode == EXIT_SUCCESS
    assert completed.stderr == ""
    assert "usage: tensorscope" in completed.stdout
    assert "{analyze,validate}" in completed.stdout


def test_package_module_analyze_executes_successfully() -> None:
    completed = _run_package("analyze", str(MODEL))

    assert completed.returncode == EXIT_SUCCESS
    assert completed.stderr == ""
    assert "Arena head: 128 bytes" in completed.stdout
    assert "Arena tail is not yet statically estimated." in completed.stdout


def test_new_operator_model_supports_text_json_validation_and_html(tmp_path: Path) -> None:
    text_result = _run_package("analyze", str(OPERATOR_CHAIN_MODEL))
    assert text_result.returncode == EXIT_SUCCESS
    assert "Arena head: 128 bytes" in text_result.stdout

    json_result = _run_package("analyze", str(OPERATOR_CHAIN_MODEL), "--json")
    assert json_result.returncode == EXIT_SUCCESS
    assert json.loads(json_result.stdout)["analysis"]["summary"]["planned_arena_head_bytes"] == 128

    validation = _run_package("validate", str(OPERATOR_CHAIN_MODEL))
    assert validation.returncode == EXIT_SUCCESS
    assert "Arena-head validation: EXACT MATCH" in validation.stdout

    destination = tmp_path / "operator-chain.html"
    html_result = _run_package("analyze", str(OPERATOR_CHAIN_MODEL), "--html", str(destination))
    assert html_result.returncode == EXIT_SUCCESS
    report = destination.read_text(encoding="utf-8")
    assert report.startswith("<!doctype html>")
    assert "http://" not in report.lower() and "https://" not in report.lower()


def test_analyze_result_is_confidence_aware() -> None:
    result = analyze_model(MODEL)

    assert result["arena_head"] == {
        "bytes": 128,
        "scope": "arena_head",
        "confidence": "exact",
        "source": "static_analysis",
        "validation_state": "not_validated",
        "validated": False,
        "validated_tflm_revision": None,
    }
    assert result["arena_tail"]["bytes"] is None
    assert result["arena_tail"]["scope"] == "arena_tail"
    assert result["arena_tail"]["confidence"] == "not_estimated"
    assert result["arena_tail"]["source"] is None
    assert result["arena_tail"]["validation_state"] == "not_validated"
    assert result["arena_total"]["bytes"] is None
    assert result["arena_total"]["scope"] == "arena_total"
    assert result["arena_total"]["confidence"] == "not_estimated"
    assert result["arena_total"]["source"] is None
    assert result["arena_total"]["validation_state"] == "not_validated"


def test_analyze_json_is_stable(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["analyze", str(MODEL), "--json"]) == EXIT_SUCCESS

    output = capsys.readouterr()
    result = json.loads(output.out)
    assert output.err == ""
    assert result["command"] == "analyze"
    assert result["model"]["filename"] == "hello_world_float.tflite"
    assert result["model"]["schema_version"] == 3
    assert result["arena_head"]["scope"] == "arena_head"
    assert result["arena_head"]["confidence"] == "exact"
    assert result["arena_head"]["source"] == "static_analysis"
    assert result["arena_head"]["validation_state"] == "not_validated"
    analysis = result["analysis"]
    assert analysis["summary"]["planned_arena_head_bytes"] == 128
    assert analysis["summary"]["arena_alignment_bytes"] == 16
    assert analysis["peak"]["live_tensor_ids"]
    assert analysis["allocations"]
    assert "reuse" in analysis
    assert "reuse_blockers" in analysis


def test_human_output_names_every_scope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["analyze", str(MODEL)]) == EXIT_SUCCESS

    output = capsys.readouterr().out
    assert "Arena head: 128 bytes [exact; static_analysis; not_validated]" in output
    assert "Arena tail is not yet statically estimated." in output
    assert "Complete arena total is not yet statically estimated." in output
    assert "None" not in output
    assert "EXACT MATCH" not in output


def test_analyze_detailed_text_includes_explanation_and_ascii(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["analyze", str(MODEL), "--details"]) == EXIT_SUCCESS

    output = capsys.readouterr().out
    assert "This report covers planned arena head only." in output
    assert "Largest tensors" in output
    assert "Peak execution point:" in output
    assert "Live tensors at peak:" in output
    assert "Packing table:" in output
    assert "Reuse summary:" in output
    assert "Reuse blockers (conservative):" in output
    assert "Arena-head packing: 128 bytes" in output
    assert "Arena tail is not yet statically estimated." in output
    assert "Complete arena total is not yet statically estimated." in output


def test_analyze_no_ascii_preserves_text_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["analyze", str(MODEL), "--no-ascii"]) == EXIT_SUCCESS

    output = capsys.readouterr().out
    assert "Packing table:" in output
    assert "Arena-head packing:" not in output


def test_top_tensors_option_controls_json_ranking(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(["analyze", str(MODEL), "--top-tensors", "2", "--json"])
        == EXIT_SUCCESS
    )

    result = json.loads(capsys.readouterr().out)
    assert len(result["analysis"]["largest_tensors"]) == 2


def test_analyze_html_writes_self_contained_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "analysis.html"

    assert main(["analyze", str(MODEL), "--html", str(destination)]) == EXIT_SUCCESS

    output = capsys.readouterr()
    assert output.err == ""
    assert output.out.strip() == f"HTML report written: {destination.resolve()}"
    report = destination.read_text(encoding="utf-8")
    assert report.startswith("<!doctype html>")
    assert "Planned arena head</dt><dd>128 bytes" in report
    assert "Arena tail is not statically estimated." in report


def test_analyze_html_write_error_has_stable_exit_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")
    destination = blocked_parent / "analysis.html"

    assert (
        main(["analyze", str(MODEL), "--html", str(destination)])
        == EXIT_REPORT_ERROR
    )

    output = capsys.readouterr()
    assert output.out == ""
    assert "Error (report_write_error):" in output.err


@pytest.mark.parametrize(
    ("budget", "status", "returncode"),
    [("129", "FITS", EXIT_SUCCESS), ("128", "EXACT FIT", EXIT_SUCCESS), ("127", "EXCEEDS BUDGET", EXIT_SUCCESS)],
)
def test_direct_budget_has_qualified_status(budget: str, status: str, returncode: int) -> None:
    completed = _run_package("analyze", str(MODEL), "--arena-head-budget", budget)
    assert completed.returncode == returncode
    assert f"Arena-head budget result: {status}" in completed.stdout
    assert "This check covers planned arena head only." in completed.stdout
    assert "This is not a complete MCU or firmware memory-fit conclusion." in completed.stdout


def test_profile_reserve_appears_in_json() -> None:
    completed = _run_package(
        "analyze", str(MODEL), "--mcu-profile", "cortex-m4-256k", "--reserve", "64KiB", "--json"
    )
    assert completed.returncode == EXIT_SUCCESS
    budget = json.loads(completed.stdout)["arena_head_budget"]
    assert budget == {
        "source": "profile", "profile_id": "cortex-m4-256k",
        "profile_name": "Cortex-M4 class — 256 KiB RAM", "profile_ram_bytes": 262144,
        "reserve_bytes": 65536, "effective_budget_bytes": 196608,
        "planned_arena_head_bytes": 128, "remaining_bytes": 196480,
        "utilization_ratio": 128 / 196608, "utilization_percent": 128 / 196608 * 100,
        "status": "fits", "scope": "arena_head",
    }


def test_analyze_without_budget_has_no_budget_object() -> None:
    completed = _run_package("analyze", str(MODEL), "--json")
    assert "arena_head_budget" not in json.loads(completed.stdout)


@pytest.mark.parametrize(
    "arguments",
    [
        ("--arena-head-budget", "1", "--mcu-profile", "cortex-m0-32k"),
        ("--reserve", "1"),
        ("--fail-on-budget-exceeded",),
        ("--arena-head-budget", "1KB"),
        ("--mcu-profile", "unknown"),
        ("--mcu-profile", "cortex-m0-32k", "--reserve", "33KiB"),
    ],
)
def test_invalid_budget_cli_inputs_fail(arguments: tuple[str, ...]) -> None:
    completed = _run_package("analyze", str(MODEL), *arguments)
    assert completed.returncode == EXIT_INPUT_ERROR
    assert completed.stderr


def test_profile_listing_needs_no_model_and_is_deterministic() -> None:
    first = _run_package("analyze", "--list-mcu-profiles")
    second = _run_package("analyze", "--list-mcu-profiles")
    assert first.returncode == EXIT_SUCCESS
    assert first.stdout == second.stdout
    assert "cortex-m0-32k\tCortex-M0 class — 32 KiB RAM\t32768 bytes" in first.stdout
    assert "generic planning presets, not specifications" in first.stdout


def test_fail_on_exceeded_uses_dedicated_code_but_exact_fit_succeeds() -> None:
    exceeded = _run_package("analyze", str(MODEL), "--arena-head-budget", "127", "--fail-on-budget-exceeded")
    exact = _run_package("analyze", str(MODEL), "--arena-head-budget", "128", "--fail-on-budget-exceeded")
    assert exceeded.returncode == EXIT_BUDGET_EXCEEDED
    assert "EXCEEDS BUDGET" in exceeded.stdout
    assert exact.returncode == EXIT_SUCCESS


def test_html_is_written_before_budget_failure(tmp_path: Path) -> None:
    destination = tmp_path / "failed-budget.html"
    completed = _run_package(
        "analyze", str(MODEL), "--arena-head-budget", "127",
        "--fail-on-budget-exceeded", "--html", str(destination),
    )
    assert completed.returncode == EXIT_BUDGET_EXCEEDED
    assert destination.is_file()
    assert "Arena-head budget result: EXCEEDS BUDGET" in destination.read_text(encoding="utf-8")


def test_budget_checks_cover_required_integration_corpus() -> None:
    operator_chain = _run_package("analyze", str(OPERATOR_CHAIN_MODEL), "--arena-head-budget", "128", "--json")
    conv0 = _run_package("analyze", str(CONV0_MODEL), "--arena-head-budget", "1KiB", "--json")
    micro_speech = _run_package(
        "analyze", str(MICRO_SPEECH_MODEL), "--mcu-profile", "cortex-m4-128k",
        "--reserve", "32KiB", "--json",
    )
    assert json.loads(operator_chain.stdout)["arena_head_budget"]["status"] == "exact_fit"
    assert json.loads(conv0.stdout)["arena_head_budget"]["status"] == "exceeds"
    assert json.loads(micro_speech.stdout)["arena_head_budget"]["status"] == "fits"


def test_analyze_text_renders_ranked_memory_guidance_and_disclaimers() -> None:
    completed = _run_package("analyze", str(MODEL))
    assert completed.returncode == EXIT_SUCCESS
    assert "Memory risk and optimization guidance" in completed.stdout
    assert "Risk summary: HIGH" in completed.stdout
    assert "1. Peak memory is concentrated" in completed.stdout
    assert "Recommendations are evidence-based suggestions, not guaranteed byte savings." in completed.stdout
    assert "This guidance covers planned arena head only." in completed.stdout
    assert "Model accuracy, operator support, and graph semantics must be revalidated" in completed.stdout


def test_analyze_json_has_stable_numeric_guidance_with_and_without_budget() -> None:
    plain = json.loads(_run_package("analyze", str(MODEL), "--json").stdout)
    budgeted = json.loads(_run_package(
        "analyze", str(MODEL), "--arena-head-budget", "128", "--json"
    ).stdout)
    assert plain["memory_guidance"]["scope"] == "arena_head"
    assert plain["memory_guidance"]["findings"][0]["finding_id"] == "peak-concentration-t7"
    assert isinstance(plain["memory_guidance"]["findings"][0]["evidence"]["share_percent"], float)
    budget_finding = next(
        item for item in budgeted["memory_guidance"]["findings"]
        if item["category"] == "budget_pressure"
    )
    assert budget_finding["severity"] == "high"
    assert budget_finding["evidence"]["utilization_percent"] == 100.0
    assert plain["arena_tail"]["bytes"] is None


def test_details_shows_all_guidance_without_default_limit() -> None:
    default = _run_package("analyze", str(OPERATOR_CHAIN_MODEL))
    detailed = _run_package("analyze", str(OPERATOR_CHAIN_MODEL), "--details")
    assert "Showing 5 of" in default.stdout
    assert "Showing 5 of" not in detailed.stdout
    assert "Tensor 20 blocks multiple reuse opportunities" in detailed.stdout


def test_json_and_html_are_mutually_exclusive(tmp_path: Path) -> None:
    completed = _run_package(
        "analyze",
        str(MODEL),
        "--json",
        "--html",
        str(tmp_path / "analysis.html"),
    )

    assert completed.returncode == 2
    assert "not allowed with argument --json" in completed.stderr


def test_unsupported_input_has_stable_json_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid = tmp_path / "invalid.tflite"
    invalid.write_bytes(b"not a model")

    assert main(["analyze", str(invalid), "--json"]) == EXIT_INPUT_ERROR
    error = json.loads(capsys.readouterr().err)
    assert error["confidence"] == "unsupported"
    assert error["error_type"] == "unsupported_input"
    assert error["exit_code"] == EXIT_INPUT_ERROR


@pytest.mark.skipif(
    not (
        Path(__file__).parents[1]
        / "tools"
        / "tflm_oracle"
        / "build"
        / "tflm_oracle"
    ).is_file(),
    reason="TFLM oracle is not built",
)
def test_validate_reports_explicit_head_match() -> None:
    result, exact_match = validate_model(MODEL)

    assert exact_match
    assert result["arena_head"]["confidence"] == "exact"
    assert result["arena_head"]["scope"] == "arena_head"
    assert result["arena_head"]["source"] == "static_analysis"
    assert result["arena_head"]["validation_state"] == "exact_match"
    assert result["arena_head"]["validated"] is True
    assert result["arena_head"]["validated_tflm_revision"] == TFLM_REVISION
    assert result["validation"]["scope"] == "arena_head"
    assert result["validation"]["state"] == "exact_match"
    assert result["validation"]["delta_bytes"] == 0
    assert result["validation"]["tflm_revision"] == TFLM_REVISION
    assert result["arena_tail"]["bytes"] is None
    assert result["arena_tail"]["confidence"] == "not_estimated"
    assert result["arena_tail"]["source"] is None
    assert result["arena_tail"]["validation_state"] == "not_validated"
    assert result["arena_total"]["bytes"] is None
    assert result["arena_total"]["confidence"] == "not_estimated"
    assert result["arena_total"]["source"] is None
    assert result["arena_total"]["validation_state"] == "not_validated"
    observation = result["oracle_arena_observation"]
    assert observation["source"] == "tflm_oracle"
    assert observation["observation_scope"] == "host_allocator_run"
    assert observation["head_bytes"] == 128
    assert observation["tail_bytes"] == 1296
    assert observation["used_bytes"] == 1424
    assert observation["capacity_bytes"] == 2097152
    assert observation["remaining_bytes"] == 2095728
    assert observation["temporary_bytes"] is None


@pytest.mark.skipif(
    not (
        Path(__file__).parents[1]
        / "tools"
        / "tflm_oracle"
        / "build"
        / "tflm_oracle"
    ).is_file(),
    reason="TFLM oracle is not built",
)
def test_validate_json_reports_exact_head_match(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["validate", str(MODEL), "--json"]) == EXIT_SUCCESS

    output = capsys.readouterr()
    result = json.loads(output.out)
    assert output.err == ""
    assert result["arena_head"]["scope"] == "arena_head"
    assert result["arena_head"]["confidence"] == "exact"
    assert result["arena_head"]["source"] == "static_analysis"
    assert result["arena_head"]["validation_state"] == "exact_match"
    assert result["validation"]["state"] == "exact_match"
    assert result["validation"]["delta_bytes"] == 0
    assert result["arena_tail"]["bytes"] is None
    assert result["arena_total"]["bytes"] is None
    observation = result["oracle_arena_observation"]
    assert observation["source"] == "tflm_oracle"
    assert observation["head_bytes"] == result["validation"]["tflm_oracle_bytes"]
    assert observation["tail_bytes"] == 1296


@pytest.mark.skipif(
    not (
        Path(__file__).parents[1]
        / "tools"
        / "tflm_oracle"
        / "build"
        / "tflm_oracle"
    ).is_file(),
    reason="TFLM oracle is not built",
)
def test_validate_human_label_is_not_generic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["validate", str(MODEL)]) == EXIT_SUCCESS

    output = capsys.readouterr().out
    assert "Arena-head validation: EXACT MATCH" in output
    assert "Validation scope: arena head only." in output
    assert all(line.strip() != "EXACT MATCH" for line in output.splitlines())
    assert "Arena tail is not yet statically estimated." in output
    assert "Complete arena total is not yet statically estimated." in output
    assert f"Validated TFLM revision: {TFLM_REVISION}" in output
    assert "TFLM oracle memory observation" in output
    assert "Observation source: tflm_oracle" in output
    assert "Observed arena used: 1,424 bytes" in output
    assert "Observed arena head: 128 bytes" in output
    assert "Observed arena tail: 1,296 bytes" in output
    assert "Temporary bytes: not available" in output
    assert "Arena tail and complete arena usage are oracle observations, not static TensorScope estimates." in output
    assert "pinned host-side TFLM allocator run" in output
    assert "tail validation" not in output.lower()
