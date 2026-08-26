from __future__ import annotations

import json

import pytest

from tensorscope.memory_budget import (
    MCU_PROFILES,
    PROFILE_DISCLAIMER,
    PROFILE_SOURCE,
    PROFILE_USAGE_HINT,
    evaluate_direct_budget,
    evaluate_profile_budget,
    get_mcu_profile,
    parse_size,
    render_budget_verdict,
    render_profile_listing,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0", 0),
        ("42", 42),
        ("42B", 42),
        ("2KiB", 2048),
        ("2MiB", 2 * 1024 * 1024),
        (" 2 kib ", 2048),
        ("1MIB", 1024 * 1024),
    ],
)
def test_parse_size(text: str, expected: int) -> None:
    assert parse_size(text) == expected


@pytest.mark.parametrize("text", ["-1", "1.5KiB", "1KB", "1MB", "", "KiB", "1 bytes", "+1", "garbage"])
def test_parse_size_rejects_invalid_or_ambiguous_values(text: str) -> None:
    with pytest.raises(ValueError, match="Invalid size"):
        parse_size(text)


def test_profile_catalog_is_deterministic_unique_and_exact() -> None:
    assert [item.profile_id for item in MCU_PROFILES] == [
        "cortex-m0-32k",
        "cortex-m4-128k",
        "cortex-m4-256k",
        "cortex-m7-512k",
        "cortex-m7-1m",
    ]
    assert [item.ram_bytes for item in MCU_PROFILES] == [32768, 131072, 262144, 524288, 1048576]
    assert len({item.profile_id for item in MCU_PROFILES}) == len(MCU_PROFILES)
    assert all(item.source_classification == PROFILE_SOURCE for item in MCU_PROFILES)
    assert all(item.notes == PROFILE_DISCLAIMER for item in MCU_PROFILES)
    assert get_mcu_profile("cortex-m4-256k") is MCU_PROFILES[2]


def test_unknown_profile_is_clear() -> None:
    with pytest.raises(ValueError, match="Unknown MCU profile 'unknown'"):
        get_mcu_profile("unknown")


@pytest.mark.parametrize(
    ("planned", "budget", "status", "remaining"),
    [(127, 128, "fits", 1), (128, 128, "exact_fit", 0), (129, 128, "exceeds", -1)],
)
def test_direct_budget_statuses(planned: int, budget: int, status: str, remaining: int) -> None:
    result = evaluate_direct_budget(planned, budget)
    assert result.source == "direct"
    assert result.profile_id is None
    assert result.status == status
    assert result.remaining_bytes == remaining
    assert result.scope == "arena_head"


def test_profile_budget_reserve_and_utilization() -> None:
    profile = get_mcu_profile("cortex-m0-32k")
    result = evaluate_profile_budget(8192, profile, 16384)
    assert result.effective_budget_bytes == 16384
    assert result.utilization_ratio == 0.5
    assert result.utilization_percent == 50.0
    assert result.profile_ram_bytes == 32768
    assert result.reserve_bytes == 16384


def test_reserve_equal_to_ram_and_zero_budget_cases() -> None:
    profile = get_mcu_profile("cortex-m0-32k")
    exact = evaluate_profile_budget(0, profile, profile.ram_bytes)
    exceeded = evaluate_profile_budget(1, profile, profile.ram_bytes)
    assert exact.status == "exact_fit"
    assert exceeded.status == "exceeds"
    assert exceeded.remaining_bytes == -1
    assert exact.utilization_ratio is exact.utilization_percent is None
    assert exceeded.utilization_ratio is exceeded.utilization_percent is None


def test_reserve_greater_than_ram_is_rejected() -> None:
    profile = get_mcu_profile("cortex-m0-32k")
    with pytest.raises(ValueError, match="exceeds profile RAM"):
        evaluate_profile_budget(0, profile, profile.ram_bytes + 1)


def test_serialization_is_deterministic() -> None:
    result = evaluate_direct_budget(128, 256)
    expected_keys = [
        "source", "profile_id", "profile_name", "profile_ram_bytes", "reserve_bytes",
        "effective_budget_bytes", "planned_arena_head_bytes", "remaining_bytes",
        "utilization_ratio", "utilization_percent", "status", "scope", "verdict",
    ]
    assert list(result.to_dict()) == expected_keys
    assert json.dumps(result.to_dict()) == json.dumps(result.to_dict())


def test_profile_listing_points_at_how_to_use_a_profile() -> None:
    listing = render_profile_listing()

    assert PROFILE_USAGE_HINT in listing
    assert "--mcu-profile" in listing
    assert "--arena-head-budget" in listing
    # The hint comes after the enumerated profiles and the disclaimer, not
    # buried in the middle of the profile table.
    assert listing.index(PROFILE_DISCLAIMER) < listing.index(PROFILE_USAGE_HINT)


@pytest.mark.parametrize(
    ("planned", "budget", "expected_label"),
    [(127, 128, "FITS"), (128, 128, "EXACT FIT"), (129, 128, "EXCEEDS BUDGET")],
)
def test_verdict_states_head_only_scope_inline_and_points_to_validate(
    planned: int, budget: int, expected_label: str,
) -> None:
    result = evaluate_direct_budget(planned, budget)

    verdict = render_budget_verdict(result)

    assert verdict.startswith(expected_label)
    assert "head only" in verdict
    assert f"{planned:,}" in verdict
    assert f"{budget:,}" in verdict
    assert "tail is not estimated" in verdict
    assert "tensorscope validate" in verdict
    assert result.to_dict()["verdict"] == verdict


def test_target_clause_is_spliced_between_the_byte_fraction_and_the_tail_caveat() -> None:
    result = evaluate_direct_budget(12480, 804864)

    verdict = render_budget_verdict(
        result, target_clause="on STM32U585, per STMicroelectronics datasheet"
    )

    assert verdict == (
        "FITS (head only — 12,480 / 804,864 bytes on STM32U585, per "
        "STMicroelectronics datasheet; arena tail is not estimated here — "
        "run `tensorscope validate` for an oracle-observed tail)"
    )


def test_target_clause_defaults_to_none_and_changes_nothing_when_omitted() -> None:
    result = evaluate_direct_budget(128, 256)

    assert render_budget_verdict(result) == render_budget_verdict(result, target_clause=None)


def test_to_dict_verdict_has_no_target_clause_by_default() -> None:
    # ArenaHeadBudgetResult.to_dict() itself never receives a target_clause
    # -- only the CLI overrides the JSON verdict, and only for --target
    # results. This documents that the dataclass's own shape is unaffected.
    result = evaluate_profile_budget(128, get_mcu_profile("cortex-m4-256k"))

    assert result.to_dict()["verdict"] == render_budget_verdict(result)
    assert "datasheet" not in result.to_dict()["verdict"]
