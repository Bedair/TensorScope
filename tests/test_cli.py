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
    EXIT_VALIDATION_UNAVAILABLE,
    TFLM_REVISION,
    analyze_model,
    main,
    validate_model,
)
from tensorscope.oracle import oracle_is_runnable


MODEL = (
    Path(__file__).parent
    / "model_corpus"
    / "models"
    / "hello_world_float.tflite"
)
OPERATOR_CHAIN_MODEL = Path(__file__).parent / "model_corpus" / "models" / "operator_chain_float.tflite"
HELLO_WORLD_INT8_MODEL = Path(__file__).parent / "model_corpus" / "models" / "hello_world_int8.tflite"
CONV0_MODEL = Path(__file__).parent / "model_corpus" / "models" / "conv0.tflite"
MICRO_SPEECH_MODEL = Path(__file__).parent / "model_corpus" / "models" / "micro_speech_quantized.tflite"
REPOSITORY_ROOT = Path(__file__).parents[1]
ORACLE_EXECUTABLE = REPOSITORY_ROOT / "tools" / "tflm_oracle" / "build" / "tflm_oracle"
# Real, already-vendored fixture whose op (UNIDIRECTIONAL_SEQUENCE_LSTM)
# TFLM genuinely implements but the oracle's resolver has not registered.
# PAD used to serve this role, but the corpus expansion registered it (see
# tests/model_corpus/models/pad0.tflite) so it no longer demonstrates a
# coverage gap -- see tests/oracle/test_tflm_oracle.py for why this one.
UNREGISTERED_OPERATOR_MODEL = (
    REPOSITORY_ROOT
    / "third_party"
    / "tflite-micro"
    / "tensorflow"
    / "lite"
    / "micro"
    / "examples"
    / "mnist_lstm"
    / "trained_lstm_int8.tflite"
)


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
    assert "{analyze,validate,compare,check,baseline,batch,firmware-check,deploy-report,list-profiles,list-targets}" in completed.stdout


def test_package_module_analyze_executes_successfully() -> None:
    completed = _run_package("analyze", str(MODEL))

    assert completed.returncode == EXIT_SUCCESS
    assert completed.stderr == ""
    assert "RAM (arena head)" in completed.stdout
    assert "128 B" in completed.stdout
    assert "Arena tail" in completed.stdout
    assert "unavailable" in completed.stdout


def test_new_operator_model_supports_text_json_and_html(tmp_path: Path) -> None:
    # Deliberately oracle-free: analyze/--json/--html never need the TFLM
    # oracle, so this runs on every platform. The matching validate check
    # (which does need the oracle) lives in the skip-guarded test below.
    text_result = _run_package("analyze", str(OPERATOR_CHAIN_MODEL))
    assert text_result.returncode == EXIT_SUCCESS
    assert "RAM (arena head)" in text_result.stdout
    assert "128 B" in text_result.stdout

    json_result = _run_package("analyze", str(OPERATOR_CHAIN_MODEL), "--json")
    assert json_result.returncode == EXIT_SUCCESS
    assert json.loads(json_result.stdout)["analysis"]["summary"]["planned_arena_head_bytes"] == 128

    destination = tmp_path / "operator-chain.html"
    html_result = _run_package("analyze", str(OPERATOR_CHAIN_MODEL), "--html", str(destination))
    assert html_result.returncode == EXIT_SUCCESS
    report = destination.read_text(encoding="utf-8")
    assert report.startswith("<!doctype html>")
    assert "http://" not in report.lower() and "https://" not in report.lower()


@pytest.mark.skipif(
    not oracle_is_runnable(ORACLE_EXECUTABLE),
    reason=(
        "TFLM oracle is not available on this platform (not built, or the "
        "committed Linux binary cannot run here); run "
        "'make -C tools/tflm_oracle' from Linux/WSL, or set "
        "TENSORSCOPE_TFLM_ORACLE to a working binary"
    ),
)
def test_new_operator_model_validates_against_the_oracle() -> None:
    validation = _run_package("validate", str(OPERATOR_CHAIN_MODEL))
    assert validation.returncode == EXIT_SUCCESS
    assert "Arena-head validation: EXACT MATCH" in validation.stdout


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
    # Per-figure confidence/source/validation labeling is a --details
    # concern -- the compact default doesn't show it.
    assert main(["analyze", str(MODEL), "--details"]) == EXIT_SUCCESS

    output = capsys.readouterr().out
    assert "Arena head: 128 bytes [exact; static_analysis; not_validated]" in output
    assert "Arena tail is not yet statically estimated." in output
    assert "Complete arena total is not yet statically estimated." in output
    assert "None" not in output
    assert "EXACT MATCH" not in output


def test_analyze_detailed_text_includes_explanation_and_ascii(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Confirms --details still produces the full, unchanged legacy output
    # (tensor tables, packing, peak execution point, reuse summary) exactly
    # as it did before the compact default existed.
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


def test_compact_view_is_the_default_and_omits_the_full_breakdown() -> None:
    completed = _run_package("analyze", str(MODEL))

    assert completed.returncode == EXIT_SUCCESS
    assert "Model: hello_world_float.tflite" in completed.stdout
    assert "Memory" in completed.stdout
    assert "RAM (arena head)" in completed.stdout
    assert "128 B" in completed.stdout
    assert "Arena tail" in completed.stdout
    assert "unavailable" in completed.stdout
    assert "Run `tensorscope analyze ... --details`" in completed.stdout
    # None of the full-breakdown content leaks into the compact default.
    assert "Largest tensors" not in completed.stdout
    assert "Peak execution point:" not in completed.stdout
    assert "Packing table:" not in completed.stdout
    assert "Memory risk and optimization guidance" not in completed.stdout
    assert "Operator-level arena-head pressure" not in completed.stdout


def test_compact_view_shows_flash_row_for_a_real_target_with_correct_bytes() -> None:
    completed = _run_package(
        "analyze", str(HELLO_WORLD_INT8_MODEL), "--target", "stm32u585",
    )

    assert completed.returncode == EXIT_SUCCESS
    assert "Model: hello_world_int8.tflite" in completed.stdout
    assert "Target: STM32U585 (STMicroelectronics)" in completed.stdout
    assert "Flash (model)" in completed.stdout
    # 420 bytes is the real sum of constant-tensor bytes in this model, not
    # a guess -- confirmed independently against the graph in
    # test_explain.py-style direct computation; regressions here would mean
    # the constant-bytes wiring or the flash citation drifted.
    assert "420 B / 2,097,152 B" in completed.stdout
    assert "RAM (arena head)" in completed.stdout
    assert "32 B / 804,864 B" in completed.stdout
    assert "FITS" in completed.stdout


def test_compact_view_omits_flash_row_for_a_generic_mcu_profile() -> None:
    completed = _run_package(
        "analyze", str(HELLO_WORLD_INT8_MODEL), "--mcu-profile", "cortex-m4-256k",
    )

    assert completed.returncode == EXIT_SUCCESS
    assert "Profile: Cortex-M4 class — 256 KiB RAM" in completed.stdout
    assert "Flash" not in completed.stdout
    assert "RAM (arena head)" in completed.stdout
    assert "32 B / 262,144 B" in completed.stdout
    assert "FITS" in completed.stdout


def test_compact_view_omits_flash_row_for_esp32s3_since_flash_is_module_dependent() -> None:
    # ESP32-S3's total_flash_bytes is honestly null (see target_profiles.py
    # tests for why) -- confirm the compact view says so instead of
    # printing a blank or a guessed number.
    completed = _run_package(
        "analyze", str(HELLO_WORLD_INT8_MODEL), "--target", "esp32-s3",
    )

    assert completed.returncode == EXIT_SUCCESS
    assert "Flash (model)" in completed.stdout
    assert "420 B / unavailable" in completed.stdout
    assert "module-dependent" in completed.stdout.lower() or "not a single well-defined" in completed.stdout.lower()


def test_compact_view_omits_flash_row_when_no_target_or_profile_given() -> None:
    completed = _run_package("analyze", str(HELLO_WORLD_INT8_MODEL))

    assert completed.returncode == EXIT_SUCCESS
    assert "Flash" not in completed.stdout
    assert "RAM (arena head)" in completed.stdout
    # No budget check requested at all -- bare bytes, no comparison or verdict.
    assert "32 B" in completed.stdout
    assert "FITS" not in completed.stdout


def test_analyze_no_ascii_preserves_text_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The packing table only appears in the --details breakdown.
    assert main(["analyze", str(MODEL), "--details", "--no-ascii"]) == EXIT_SUCCESS

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
    # The full budget-check block ("Arena-head budget result: ...", the two
    # scope-caveat lines) only appears in the --details breakdown; the
    # compact default shows a plain FITS/EXACT FIT/EXCEEDS BUDGET word.
    completed = _run_package("analyze", str(MODEL), "--arena-head-budget", budget, "--details")
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
        "verdict": (
            "FITS (head only — 128 / 196,608 bytes; arena tail is not "
            "estimated here — run `tensorscope validate` for an "
            "oracle-observed tail)"
        ),
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
    assert "analyze MODEL --mcu-profile" in first.stdout


def test_list_profiles_subcommand_also_points_at_mcu_profile_usage() -> None:
    completed = _run_package("list-profiles")
    assert completed.returncode == EXIT_SUCCESS
    assert "analyze MODEL --mcu-profile" in completed.stdout


def test_help_text_points_list_mcu_profiles_at_mcu_profile() -> None:
    completed = _run_package("analyze", "--help")
    assert completed.returncode == EXIT_SUCCESS
    assert "--mcu-profile <id>" in completed.stdout


def test_fail_on_exceeded_uses_dedicated_code_but_exact_fit_succeeds() -> None:
    # The full verdict sentence (with its parenthetical caveat) is a
    # --details concern; the exit code itself is unaffected by render mode.
    exceeded = _run_package("analyze", str(MODEL), "--arena-head-budget", "127", "--fail-on-budget-exceeded", "--details")
    exact = _run_package("analyze", str(MODEL), "--arena-head-budget", "128", "--fail-on-budget-exceeded")
    assert exceeded.returncode == EXIT_BUDGET_EXCEEDED
    assert "EXCEEDS BUDGET (head only — 128 / 127 bytes; arena tail is not estimated here" in exceeded.stdout
    assert "tensorscope validate" in exceeded.stdout
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


@pytest.mark.parametrize(
    ("target_name", "expected_ram_bytes"),
    [
        ("STM32U585", 804864),
        ("nucleo-u575zi-q", 804864),
        ("nRF52840", 262144),
        ("nrf52840-dk", 262144),
        ("Arduino Nano 33 BLE", 262144),
        ("arduino nano 33 ble", 262144),
        ("Arduino Nano 33 BLE Sense", 262144),
        ("adafruit feather nrf52840 sense", 262144),
        ("ESP32-S3", 524288),
        ("esp32-s3-devkitc-1", 524288),
        ("CY8C624ABZI-S2D44", 1048576),
        ("cy8ckit-062s2-ai", 1048576),
    ],
)
def test_target_resolves_real_mcu_parts_and_dev_kit_aliases_case_insensitively(
    target_name: str, expected_ram_bytes: int,
) -> None:
    completed = _run_package("analyze", str(MODEL), "--target", target_name, "--json")
    assert completed.returncode == EXIT_SUCCESS
    budget = json.loads(completed.stdout)["arena_head_budget"]
    assert budget["profile_ram_bytes"] == expected_ram_bytes
    assert budget["status"] == "fits"
    assert budget["source"] == "profile"


def test_unknown_target_fails_clearly_and_lists_known_names() -> None:
    completed = _run_package("analyze", str(MODEL), "--target", "STM32F4")
    assert completed.returncode == EXIT_INPUT_ERROR
    assert "Unknown target 'STM32F4'" in completed.stderr
    assert "STM32U585" in completed.stderr
    assert "nRF52840" in completed.stderr
    assert "ESP32-S3" in completed.stderr
    assert "CY8C624ABZI-S2D44" in completed.stderr


def test_target_does_not_partially_or_fuzzy_match() -> None:
    # A near-miss must fail, not silently resolve to the closest name.
    completed = _run_package("analyze", str(MODEL), "--target", "STM32U58")
    assert completed.returncode == EXIT_INPUT_ERROR
    assert "Unknown target" in completed.stderr


@pytest.mark.parametrize("other_flag", [["--mcu-profile", "cortex-m4-256k"], ["--arena-head-budget", "1MiB"]])
def test_target_is_mutually_exclusive_with_mcu_profile_and_arena_head_budget(
    other_flag: list[str],
) -> None:
    completed = _run_package("analyze", str(MODEL), "--target", "esp32-s3", *other_flag)
    assert completed.returncode != EXIT_SUCCESS
    assert "not allowed with argument" in completed.stderr


def test_reserve_and_fail_on_budget_exceeded_both_accept_target() -> None:
    # 262,144 byte target minus a 262,000 byte reserve leaves 144 bytes --
    # MODEL's 128-byte head still fits that, so force conv0 (10,432 bytes)
    # to actually exceed it and confirm --target reaches the same dedicated
    # exit code --mcu-profile and --arena-head-budget already use.
    exceeded = _run_package(
        "analyze", str(CONV0_MODEL), "--target", "nrf52840",
        "--reserve", "262000", "--fail-on-budget-exceeded", "--details",
    )
    fits = _run_package(
        "analyze", str(MODEL), "--target", "nrf52840",
        "--reserve", "262000", "--fail-on-budget-exceeded",
    )
    assert exceeded.returncode == EXIT_BUDGET_EXCEEDED
    assert "EXCEEDS BUDGET (head only — 10,432 / 144 bytes" in exceeded.stdout
    assert fits.returncode == EXIT_SUCCESS


def test_reserve_without_mcu_profile_or_target_is_rejected() -> None:
    completed = _run_package("analyze", str(MODEL), "--reserve", "1KiB")
    assert completed.returncode == EXIT_INPUT_ERROR
    assert "--reserve may be used only with --mcu-profile or --target" in completed.stderr


def test_list_targets_subcommand_and_analyze_list_targets_agree() -> None:
    subcommand = _run_package("list-targets")
    analyze_flag = _run_package("analyze", "--list-targets")
    assert subcommand.returncode == EXIT_SUCCESS
    assert analyze_flag.returncode == EXIT_SUCCESS
    assert subcommand.stdout == analyze_flag.stdout
    assert "STM32U585" in subcommand.stdout
    assert "nRF52840" in subcommand.stdout
    assert "ESP32-S3" in subcommand.stdout
    assert "CY8C624ABZI-S2D44" in subcommand.stdout
    assert "--target" in subcommand.stdout


def test_list_targets_needs_no_model() -> None:
    completed = _run_package("list-targets")
    assert completed.returncode == EXIT_SUCCESS


def test_target_html_report_includes_the_inline_verdict(tmp_path: Path) -> None:
    destination = tmp_path / "target-report.html"
    completed = _run_package(
        "analyze", str(MODEL), "--target", "STM32U585", "--html", str(destination),
    )
    assert completed.returncode == EXIT_SUCCESS
    report = destination.read_text(encoding="utf-8")
    assert "FITS (head only — 128 / 804,864 bytes" in report
    assert "STM32U585" in report


def test_target_verdict_cites_the_resolved_part_and_vendor_in_all_three_modes(
    tmp_path: Path,
) -> None:
    expected_clause = "on STM32U585, per STMicroelectronics datasheet"

    # The full citation clause is a --details concern in text mode; the
    # compact default shows the Target's part/vendor but not the full
    # verdict sentence. JSON and HTML are untouched by the compact-view
    # change and still carry it unconditionally.
    text = _run_package("analyze", str(MODEL), "--target", "STM32U585", "--details")
    as_json = _run_package("analyze", str(MODEL), "--target", "STM32U585", "--json")
    destination = tmp_path / "cited-report.html"
    as_html = _run_package(
        "analyze", str(MODEL), "--target", "STM32U585", "--html", str(destination),
    )

    assert expected_clause in text.stdout
    assert expected_clause in json.loads(as_json.stdout)["arena_head_budget"]["verdict"]
    assert expected_clause in destination.read_text(encoding="utf-8")


def test_budget_source_label_agrees_across_text_and_html_for_every_budget_kind(
    tmp_path: Path,
) -> None:
    # Regression test: the "Budget source" label was fixed in the HTML
    # renderer (mislabeling a --target result "Generic MCU planning
    # profile") without a shared helper, so the exact same bug shipped
    # again, independently, in the text renderer. All three budget kinds
    # are covered here so a future fix to one render surface can't drift
    # from the others the same way, for any of them.
    cases = [
        (("--target", "STM32U585"), "Real MCU/dev-kit target (cited datasheet)"),
        (("--mcu-profile", "cortex-m4-256k"), "Generic MCU planning profile"),
        (("--arena-head-budget", "1MiB"), "Direct arena-head budget"),
    ]
    for budget_flags, expected_label in cases:
        # "Budget source" is a --details concern in text mode now; HTML is
        # untouched by the compact-view change and always shows it.
        text = _run_package("analyze", str(MODEL), *budget_flags, "--details")
        destination = tmp_path / f"{'-'.join(budget_flags)}.html"
        as_html = _run_package(
            "analyze", str(MODEL), *budget_flags, "--html", str(destination),
        )

        assert f"Budget source: {expected_label}" in text.stdout
        assert f"<dt>Budget source</dt><dd>{expected_label}</dd>" in destination.read_text(encoding="utf-8")

    # JSON has no separate "Budget source" label field to drift -- it
    # exposes the raw source enum plus profile_id and the citation-bearing
    # verdict instead. Confirmed here so JSON isn't silently assumed safe.
    as_json = _run_package("analyze", str(MODEL), "--target", "STM32U585", "--json")
    budget_json = json.loads(as_json.stdout)["arena_head_budget"]
    assert budget_json["source"] == "profile"
    assert budget_json["profile_id"] == "stm32u585"
    assert "on STM32U585, per STMicroelectronics datasheet" in budget_json["verdict"]


def test_target_verdict_cites_correctly_when_resolved_via_a_dev_kit_alias() -> None:
    # The citation names the resolved MCU, not the alias string the user typed.
    completed = _run_package(
        "analyze", str(MODEL), "--target", "Arduino Nano 33 BLE Sense", "--json",
    )
    verdict = json.loads(completed.stdout)["arena_head_budget"]["verdict"]
    assert "on nRF52840, per Nordic Semiconductor datasheet" in verdict
    assert "Arduino" not in verdict


def test_arduino_nano_33_ble_without_sense_resolves_to_the_same_nrf52840_citation() -> None:
    # Both boards are aliases of the same chip and must cite identically.
    sense = _run_package(
        "analyze", str(MODEL), "--target", "Arduino Nano 33 BLE Sense", "--json",
    )
    plain = _run_package(
        "analyze", str(MODEL), "--target", "Arduino Nano 33 BLE", "--json",
    )
    sense_budget = json.loads(sense.stdout)["arena_head_budget"]
    plain_budget = json.loads(plain.stdout)["arena_head_budget"]
    assert sense_budget["verdict"] == plain_budget["verdict"]
    assert sense_budget["profile_ram_bytes"] == plain_budget["profile_ram_bytes"] == 262144


def test_psoc6_target_resolves_and_cites_infineon_via_either_name() -> None:
    by_part = _run_package("analyze", str(MODEL), "--target", "CY8C624ABZI-S2D44", "--json")
    by_kit = _run_package("analyze", str(MODEL), "--target", "CY8CKIT-062S2-AI", "--json")
    part_budget = json.loads(by_part.stdout)["arena_head_budget"]
    kit_budget = json.loads(by_kit.stdout)["arena_head_budget"]
    assert part_budget["verdict"] == kit_budget["verdict"]
    assert part_budget["profile_ram_bytes"] == 1048576
    assert "on CY8C624ABZI-S2D44, per Infineon Technologies datasheet" in part_budget["verdict"]


@pytest.mark.parametrize(
    "budget_flags",
    [
        ["--mcu-profile", "cortex-m4-256k"],
        ["--arena-head-budget", "1MiB"],
    ],
)
def test_generic_budget_verdicts_carry_no_datasheet_citation(budget_flags: list[str]) -> None:
    completed = _run_package("analyze", str(MODEL), *budget_flags, "--json")
    verdict = json.loads(completed.stdout)["arena_head_budget"]["verdict"]
    assert "datasheet" not in verdict
    assert " per " not in verdict


def test_analyze_text_renders_ranked_memory_guidance_and_disclaimers() -> None:
    # Guidance text is a --details concern in the compact default.
    completed = _run_package("analyze", str(MODEL), "--details")
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
    # The compact default no longer shows guidance findings at all (that's
    # exclusively a --details concern now, see test_recommendations.py for
    # direct coverage of the underlying 5-finding truncation itself).
    # At the CLI level, confirm the compact default omits guidance findings
    # entirely and --details shows every one of them, unrestricted.
    default = _run_package("analyze", str(OPERATOR_CHAIN_MODEL))
    detailed = _run_package("analyze", str(OPERATOR_CHAIN_MODEL), "--details")
    assert "Memory risk and optimization guidance" not in default.stdout
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
    not oracle_is_runnable(ORACLE_EXECUTABLE),
    reason=(
        "TFLM oracle is not available on this platform (not built, or the "
        "committed Linux binary cannot run here); run "
        "'make -C tools/tflm_oracle' from Linux/WSL, or set "
        "TENSORSCOPE_TFLM_ORACLE to a working binary"
    ),
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
    not oracle_is_runnable(ORACLE_EXECUTABLE),
    reason=(
        "TFLM oracle is not available on this platform (not built, or the "
        "committed Linux binary cannot run here); run "
        "'make -C tools/tflm_oracle' from Linux/WSL, or set "
        "TENSORSCOPE_TFLM_ORACLE to a working binary"
    ),
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
    not oracle_is_runnable(ORACLE_EXECUTABLE),
    reason=(
        "TFLM oracle is not available on this platform (not built, or the "
        "committed Linux binary cannot run here); run "
        "'make -C tools/tflm_oracle' from Linux/WSL, or set "
        "TENSORSCOPE_TFLM_ORACLE to a working binary"
    ),
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


@pytest.mark.skipif(
    not oracle_is_runnable(ORACLE_EXECUTABLE) or not UNREGISTERED_OPERATOR_MODEL.is_file(),
    reason=(
        "TFLM oracle is not available on this platform (not built, or the "
        "committed Linux binary cannot run here), or the pinned TFLM "
        "submodule fixture is missing; run 'make -C tools/tflm_oracle' "
        "from Linux/WSL and check out third_party/tflite-micro"
    ),
)
def test_validate_reports_unregistered_operator_category(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["validate", str(UNREGISTERED_OPERATOR_MODEL), "--json"])

    assert exit_code == EXIT_VALIDATION_UNAVAILABLE
    output = capsys.readouterr()
    assert output.out == ""
    payload = json.loads(output.err)
    assert payload["error_type"] == "unregistered_operator"
    assert payload["exit_code"] == EXIT_VALIDATION_UNAVAILABLE
    assert "coverage gap" in payload["explanation"]
