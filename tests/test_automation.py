from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from tensorscope import __version__
from tensorscope.automation import (
    analysis_sarif, check_baseline, create_baseline_manifest, evaluate_policy,
    load_policy, parse_gnu_map, resolve_models,
)
from tensorscope.cli import (
    EXIT_BASELINE_DRIFT, EXIT_FIRMWARE_CHECK_FAILURE, EXIT_POLICY_FAILURE,
    EXIT_SUCCESS, analyze_model, main,
)


MODELS = Path(__file__).parent / "model_corpus" / "models"
HELLO = MODELS / "hello_world_float.tflite"


def test_policy_rejects_unknown_keys_and_has_deterministic_failures(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("schema_version: 1\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown policy keys: unknown"):
        load_policy(invalid)
    policy = {
        "schema_version": 1, "maximum_arena_head_bytes": 1,
        "forbidden_operators": ["FULLY_CONNECTED"], "maximum_risk": "low",
    }
    result = evaluate_policy(policy, analyze_model(HELLO))
    assert result["status"] == "failed"
    assert [item["rule_id"] for item in result["failures"]] == sorted(item["rule_id"] for item in result["failures"])


def test_policy_outputs_json_and_sarif_before_failure(tmp_path: Path) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text("schema_version: 1\nmaximum_arena_head_bytes: 1\n", encoding="utf-8")
    output = tmp_path / "policy.json"
    sarif = tmp_path / "policy.sarif.json"
    assert main(["check", str(HELLO), "--policy", str(policy), "--json", str(output), "--sarif", str(sarif)]) == EXIT_POLICY_FAILURE
    assert json.loads(output.read_text())["status"] == "failed"
    assert json.loads(sarif.read_text())["version"] == "2.1.0"


def test_baseline_is_deterministic_and_detects_hash_drift(tmp_path: Path) -> None:
    analysis = analyze_model(HELLO)
    first = create_baseline_manifest(HELLO, analysis, tool_version=__version__)
    assert first == create_baseline_manifest(HELLO, analysis, tool_version=__version__)
    assert "timestamp" not in first
    assert check_baseline(first, analysis, HELLO)["status"] == "passed"
    changed = json.loads(json.dumps(first))
    changed["model"]["sha256"] = "0" * 64
    assert check_baseline(changed, analysis, HELLO)["status"] == "failed"


def test_baseline_cli_writes_before_drift_exit(tmp_path: Path) -> None:
    manifest = tmp_path / "baseline.json"
    assert main(["baseline", "create", str(HELLO), "--output", str(manifest)]) == EXIT_SUCCESS
    value = json.loads(manifest.read_text())
    value["metrics"]["planned_arena_head_bytes"] = 0
    manifest.write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "check.json"
    assert main(["baseline", "check", str(HELLO), "--baseline", str(manifest), "--json", str(output)]) == EXIT_BASELINE_DRIFT
    assert json.loads(output.read_text())["reasons"]


def test_policy_evaluates_baseline_growth_and_drift(tmp_path: Path) -> None:
    manifest = create_baseline_manifest(HELLO, analyze_model(HELLO), tool_version=__version__)
    manifest["metrics"]["planned_arena_head_bytes"] = 1
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(manifest), encoding="utf-8")
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "schema_version: 1\nbaseline_manifest: baseline.json\nmaximum_growth_bytes: 1\nfail_on_regression: true\n",
        encoding="utf-8",
    )
    output = tmp_path / "result.json"
    assert main(["check", str(HELLO), "--policy", str(policy), "--json", str(output)]) == EXIT_POLICY_FAILURE
    rules = {item["rule_id"] for item in json.loads(output.read_text())["failures"]}
    assert {"TS-POLICY-BASELINE-DRIFT", "TS-POLICY-GROWTH-BYTES"} <= rules


def test_analysis_sarif_is_valid_stable_and_has_no_line_numbers() -> None:
    first = analysis_sarif(HELLO, analyze_model(HELLO))
    assert first == analysis_sarif(HELLO, analyze_model(HELLO))
    assert first["version"] == "2.1.0"
    assert "region" not in json.dumps(first)
    assert first["runs"][0]["tool"]["driver"]["rules"]


def test_analysis_views_are_non_additive_deterministic_and_diagnostic() -> None:
    first = analyze_model(HELLO)
    second = analyze_model(HELLO)
    assert first["analysis_views"] == second["analysis_views"]
    views = first["analysis_views"]
    assert views["operator_attribution"]["non_additive"] is True
    assert views["operator_attribution"]["operators"][0]["scratch_observation"]["availability"] == "unavailable"
    assert views["execution_timeline"]["scopes"]
    assert views["graph_view"]["operators"]
    diagnostics = first["model_diagnostics"]
    assert diagnostics["overall_combination_status"] == "single_subgraph"
    assert diagnostics["persistent_buffer_estimate"] is None


def test_batch_resolves_lexically_deduplicates_and_writes_aggregates(tmp_path: Path) -> None:
    assert resolve_models([HELLO, MODELS], recursive=False) == tuple(sorted(set(resolve_models([MODELS], recursive=False)), key=str))
    output = tmp_path / "batch"
    assert main(["batch", str(HELLO), str(HELLO), "--output-dir", str(output), "--sarif"]) == EXIT_SUCCESS
    aggregate = json.loads((output / "aggregate.json").read_text())
    assert aggregate["model_count"] == aggregate["success_count"] == 1
    assert (output / "aggregate.csv").read_text().startswith("model,status,")
    assert json.loads((output / "aggregate.sarif.json").read_text())["version"] == "2.1.0"


def test_gnu_map_subset_missing_duplicate_and_overflow() -> None:
    valid = "RAM 0x20000000 0x00010000 xrw\n0x20000100 tensor_arena\n"
    parsed = parse_gnu_map(valid, "tensor_arena")
    assert parsed["arena_address"] == 0x20000100
    with pytest.raises(ValueError, match="not found"):
        parse_gnu_map(valid, "missing")
    with pytest.raises(ValueError, match="duplicated"):
        parse_gnu_map(valid + "0x20000200 tensor_arena\n", "tensor_arena")
    with pytest.raises(ValueError, match="outside"):
        parse_gnu_map("RAM 0x20000000 0x100 xrw\n0x30000000 tensor_arena\n", "tensor_arena")


def test_firmware_check_and_deployment_header_compile(tmp_path: Path) -> None:
    map_file = tmp_path / "firmware.map"
    map_file.write_text("RAM 0x20000000 0x00010000 xrw\n0x20000100 tensor_arena\n", encoding="utf-8")
    output = tmp_path / "firmware.json"
    assert main(["firmware-check", str(HELLO), "--map-file", str(map_file), "--arena-symbol", "tensor_arena", "--arena-size", "127", "--json", str(output)]) == EXIT_FIRMWARE_CHECK_FAILURE
    assert output.is_file() and json.loads(output.read_text())["status"] == "exceeds"
    deploy = tmp_path / "deploy"
    assert main(["deploy-report", str(HELLO), "--output-dir", str(deploy)]) == EXIT_SUCCESS
    source = tmp_path / "smoke.cc"
    source.write_text('#include "tensorscope_deployment.h"\nint main(){return kTensorScopePlannedArenaHeadBytes == 128 ? 0 : 1;}\n', encoding="utf-8")
    subprocess.run(["g++", "-std=c++17", "-I", str(deploy), str(source), "-o", str(tmp_path / "smoke")], check=True)
