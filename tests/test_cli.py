from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tensorscope.cli import (
    EXIT_INPUT_ERROR,
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
    assert result["arena_head"]["scope"] == "arena_head"
    assert result["arena_head"]["confidence"] == "exact"
    assert result["arena_head"]["source"] == "static_analysis"
    assert result["arena_head"]["validation_state"] == "not_validated"


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
