from __future__ import annotations

import json

import pytest

from tensorscope.memory_budget import MCUProfile
from tensorscope.target_profiles import (
    TARGET_DISCLAIMER,
    TARGET_USAGE_HINT,
    ProfileSource,
    TargetProfile,
    TargetProfileError,
    _check_for_collisions,
    _parse_profile,
    as_mcu_profile,
    load_target_profiles,
    render_target_listing,
    render_target_verdict_clause,
    resolve_target,
)


def _source(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "datasheet_title": "Example Datasheet",
        "revision": None,
        "section": None,
        "page": None,
        "url": None,
    }
    base.update(overrides)
    return base


def _profile_json(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": 1,
        "id": "example",
        "mcu_part": "EXAMPLE-MCU",
        "vendor": "Example Vendor",
        "architecture": "Example Core",
        "total_sram_bytes": 1024,
        "total_flash_bytes": 2048,
        "dev_kit_aliases": ["EXAMPLE-DK"],
        "source": _source(),
        "notes": "",
    }
    base.update(overrides)
    return base


def _synthetic_profile(
    *,
    id: str = "example",
    mcu_part: str = "EXAMPLE-MCU",
    dev_kit_aliases: tuple[str, ...] = ("EXAMPLE-DK",),
    total_flash_bytes: int | None = 2048,
    file_name: str = "example.json",
) -> TargetProfile:
    return TargetProfile(
        id=id,
        mcu_part=mcu_part,
        vendor="Example Vendor",
        architecture="Example Core",
        total_sram_bytes=1024,
        total_flash_bytes=total_flash_bytes,
        dev_kit_aliases=dev_kit_aliases,
        source=ProfileSource("Example Datasheet", None, None, None, None),
        notes="",
        default_firmware_reserve_bytes=None,
        file_name=file_name,
    )


# --- Real bundled catalog -------------------------------------------------


def test_bundled_catalog_loads_exactly_the_shipped_targets() -> None:
    profiles = load_target_profiles()

    assert {profile.id for profile in profiles} == {
        "stm32u585", "nrf52840", "esp32-s3", "cy8c624abzi-s2d44",
        "stm32h743zi", "stm32h747xi", "esp32-s3-wroom-1-n8", "esp32-s3-wroom-1-n16r8",
        "nrf52832", "rp2040", "rp2350a", "stm32f746zg", "stm32f767zi", "imxrt1062",
        "nrf5340", "efr32mg24", "efr32mg26", "apollo4-plus", "apollo4-blue-plus",
        "ra8d1", "hx6537-a", "cxd5602", "cc1352p",
    }
    assert len(profiles) == 23


def test_bundled_catalog_is_sorted_by_file_name_and_deterministic() -> None:
    first = load_target_profiles()
    second = load_target_profiles()

    assert first == second
    assert [profile.file_name for profile in first] == sorted(
        profile.file_name for profile in first
    )


@pytest.mark.parametrize(
    ("target_id", "mcu_part", "total_sram_bytes", "total_flash_bytes", "vendor"),
    [
        ("stm32u585", "STM32U585", 804864, 2097152, "STMicroelectronics"),
        ("nrf52840", "nRF52840", 262144, 1048576, "Nordic Semiconductor"),
        ("esp32-s3", "ESP32-S3", 524288, None, "Espressif Systems"),
        ("cy8c624abzi-s2d44", "CY8C624ABZI-S2D44", 1048576, 2097152, "Infineon Technologies"),
        ("stm32h743zi", "STM32H743ZI", 1085440, 2097152, "STMicroelectronics"),
        ("stm32h747xi", "STM32H747XI", 1085440, 2097152, "STMicroelectronics"),
        ("esp32-s3-wroom-1-n8", "ESP32-S3-WROOM-1-N8", 524288, 8388608, "Espressif Systems"),
        ("esp32-s3-wroom-1-n16r8", "ESP32-S3-WROOM-1-N16R8", 524288, 16777216, "Espressif Systems"),
        ("nrf52832", "nRF52832", 65536, 524288, "Nordic Semiconductor"),
        ("rp2040", "RP2040", 270336, None, "Raspberry Pi Ltd"),
        ("rp2350a", "RP2350A", 532480, None, "Raspberry Pi Ltd"),
        ("stm32f746zg", "STM32F746ZG", 348160, 1048576, "STMicroelectronics"),
        ("stm32f767zi", "STM32F767ZI", 544768, 2097152, "STMicroelectronics"),
        ("imxrt1062", "i.MX RT1062", 1048576, None, "NXP Semiconductors"),
        ("nrf5340", "nRF5340", 524288, 1048576, "Nordic Semiconductor"),
        ("efr32mg24", "EFR32MG24", 262144, 1572864, "Silicon Labs"),
        ("efr32mg26", "EFR32MG26", 524288, 3276800, "Silicon Labs"),
        ("apollo4-plus", "Apollo4 Plus", 2883584, 2097152, "Ambiq Micro"),
        ("apollo4-blue-plus", "Apollo4 Blue Plus", 2883584, 2097152, "Ambiq Micro"),
        ("ra8d1", "RA8D1", 1048576, 2097152, "Renesas Electronics"),
        ("hx6537-a", "HX6537-A", 2162688, None, "Himax Technology"),
        ("cxd5602", "CXD5602", 1572864, None, "Sony Semiconductor Solutions"),
        ("cc1352p", "CC1352P", 81920, 360448, "Texas Instruments"),
    ],
)
def test_each_shipped_target_has_the_expected_datasheet_figures(
    target_id: str, mcu_part: str, total_sram_bytes: int, total_flash_bytes: int | None, vendor: str,
) -> None:
    profiles = {profile.id: profile for profile in load_target_profiles()}
    profile = profiles[target_id]

    assert profile.mcu_part == mcu_part
    assert profile.total_sram_bytes == total_sram_bytes
    assert profile.total_flash_bytes == total_flash_bytes
    assert profile.vendor == vendor
    # Every shipped profile must carry a real, non-empty citation title even
    # when revision/section/page/url are honestly null rather than guessed.
    assert profile.source.datasheet_title


def test_every_shipped_target_has_a_dev_kit_alias_or_a_documented_reason_it_lacks_one() -> None:
    # Most profiles carry at least one real dev-kit alias. A few module-level
    # or recently-announced parts genuinely don't have one confirmed yet --
    # that's honest, but it must be stated in notes, not silently absent,
    # so a real alias found later doesn't go unnoticed as "still missing".
    for profile in load_target_profiles():
        if not profile.dev_kit_aliases:
            assert "alias" in profile.notes.lower(), profile.id


def test_esp32s3_flash_is_honestly_null_with_an_explanatory_note() -> None:
    # The bare ESP32-S3 die has no embedded flash (external SPI only, chosen
    # per module/board); the dev-kit alias itself ships in multiple
    # flash-size SKUs with no single default. Must not guess a figure.
    profiles = {profile.id: profile for profile in load_target_profiles()}
    esp32s3 = profiles["esp32-s3"]

    assert esp32s3.total_flash_bytes is None
    assert "module-dependent" in esp32s3.notes or "external" in esp32s3.notes.lower()
    assert "WROOM" in esp32s3.notes


# --- Resolution -------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected_id"),
    [
        ("STM32U585", "stm32u585"),
        ("stm32u585", "stm32u585"),
        ("  stm32u585  ", "stm32u585"),
        ("NUCLEO-U575ZI-Q", "stm32u585"),
        ("nucleo-u575zi-q", "stm32u585"),
        ("nRF52840", "nrf52840"),
        ("nrf52840-dk", "nrf52840"),
        ("Arduino Nano 33 BLE", "nrf52840"),
        ("arduino nano 33 ble", "nrf52840"),
        ("Arduino Nano 33 BLE Sense", "nrf52840"),
        ("arduino nano 33 ble sense", "nrf52840"),
        ("Adafruit Feather nRF52840 Sense", "nrf52840"),
        ("ESP32-S3", "esp32-s3"),
        ("esp32-s3-devkitc-1", "esp32-s3"),
        ("CY8C624ABZI-S2D44", "cy8c624abzi-s2d44"),
        ("cy8c624abzi-s2d44", "cy8c624abzi-s2d44"),
        ("CY8CKIT-062S2-AI", "cy8c624abzi-s2d44"),
        ("cy8ckit-062s2-ai", "cy8c624abzi-s2d44"),
        ("NUCLEO-H743ZI", "stm32h743zi"),
        ("Portenta H7", "stm32h747xi"),
        ("Nicla Vision", "stm32h747xi"),
        ("OpenMV H7", "stm32h747xi"),
        ("STM32U575AI", "stm32u585"),
        ("stm32u575zi", "stm32u585"),
        ("STM32U585ZI", "stm32u585"),
        ("Raspberry Pi Pico", "rp2040"),
        ("Raspberry Pi Pico 2", "rp2350a"),
        ("Teensy 4.1", "imxrt1062"),
        ("nRF5340-DK", "nrf5340"),
        ("xG24-DK2601B", "efr32mg24"),
        ("LAUNCHXL-CC1352P", "cc1352p"),
    ],
)
def test_resolve_target_matches_id_mcu_part_and_alias_case_insensitively(
    query: str, expected_id: str,
) -> None:
    assert resolve_target(query).id == expected_id


def test_resolve_target_fails_clearly_on_unknown_name_no_fuzzy_matching() -> None:
    with pytest.raises(TargetProfileError, match="Unknown target 'STM32F4'"):
        resolve_target("STM32F4")


def test_resolve_target_error_lists_every_known_name() -> None:
    with pytest.raises(TargetProfileError) as excinfo:
        resolve_target("does-not-exist")

    message = str(excinfo.value)
    for profile in load_target_profiles():
        assert profile.mcu_part in message
        for alias in profile.dev_kit_aliases:
            assert alias in message


def test_resolve_target_does_not_partially_or_prefix_match() -> None:
    # "STM32U58" is a genuine prefix of "STM32U585" but must not resolve --
    # exact match only, never a guess.
    with pytest.raises(TargetProfileError):
        resolve_target("STM32U58")


def test_target_profile_error_is_a_value_error() -> None:
    # So it flows through the CLI's existing input-error handling without a
    # new except clause needing to be added.
    assert issubclass(TargetProfileError, ValueError)


# --- Adapter into the existing MCUProfile-shaped budget pipeline ------


def test_as_mcu_profile_carries_ram_bytes_and_identity_through_unchanged() -> None:
    target = resolve_target("esp32-s3")

    adapted = as_mcu_profile(target)

    assert isinstance(adapted, MCUProfile)
    assert adapted.ram_bytes == target.total_sram_bytes == 524288
    assert adapted.profile_id == target.id
    assert target.mcu_part in adapted.display_name
    assert target.vendor in adapted.display_name
    assert "vendor_datasheet" == adapted.source_classification


def test_as_mcu_profile_notes_cite_the_source_when_revision_present() -> None:
    target = resolve_target("esp32-s3")

    adapted = as_mcu_profile(target)

    assert target.source.datasheet_title in adapted.notes
    assert target.source.revision in adapted.notes


def test_as_mcu_profile_omits_revision_from_citation_when_null() -> None:
    target = resolve_target("stm32u585")
    assert target.source.revision is None

    adapted = as_mcu_profile(target)

    assert adapted.notes == f"Source: {target.source.datasheet_title}"


# --- Inline verdict citation clause --------------------------------------


def test_render_target_verdict_clause_names_the_part_and_vendor() -> None:
    target = resolve_target("stm32u585")

    clause = render_target_verdict_clause(target)

    assert clause == "on STM32U585, per STMicroelectronics datasheet"


def test_render_target_verdict_clause_covers_every_shipped_target() -> None:
    for target in load_target_profiles():
        clause = render_target_verdict_clause(target)

        assert clause == f"on {target.mcu_part}, per {target.vendor} datasheet"
        assert "datasheet" in clause


# --- Listing ------------------------------------------------------------


def test_render_target_listing_includes_every_target_and_usage_hint() -> None:
    listing = render_target_listing()

    for profile in load_target_profiles():
        assert profile.mcu_part in listing
        assert str(profile.total_sram_bytes) in listing
        for alias in profile.dev_kit_aliases:
            assert alias in listing
    assert TARGET_DISCLAIMER in listing
    assert TARGET_USAGE_HINT in listing
    assert "--target" in listing


# --- Schema validation (via the parsing function directly -- this is the
# core validation logic with many independent failure branches, better
# covered granularly than only indirectly through file I/O) ---------------


def test_parse_profile_accepts_a_well_formed_minimal_profile() -> None:
    profile = _parse_profile("example.json", json.dumps(_profile_json()))

    assert profile.id == "example"
    assert profile.default_firmware_reserve_bytes is None


def test_parse_profile_accepts_a_null_total_flash_bytes() -> None:
    # Honest "not a single well-defined figure" case, e.g. an
    # external-flash-only chip -- must not be forced to guess a number.
    data = _profile_json(total_flash_bytes=None)

    profile = _parse_profile("example.json", json.dumps(data))

    assert profile.total_flash_bytes is None


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda d: d.pop("mcu_part"), "missing required field"),
        (lambda d: d.pop("source"), "missing required field"),
        (lambda d: d.update(schema_version=2), "unsupported schema_version"),
        (lambda d: d.update(id=""), "id.*non-empty string"),
        (lambda d: d.update(id=123), "id.*non-empty string"),
        (lambda d: d.update(total_sram_bytes=0), "total_sram_bytes must be a positive integer"),
        (lambda d: d.update(total_sram_bytes=-1), "total_sram_bytes must be a positive integer"),
        (lambda d: d.update(total_sram_bytes=1.5), "total_sram_bytes must be a positive integer"),
        (lambda d: d.update(total_sram_bytes=True), "total_sram_bytes must be a positive integer"),
        (lambda d: d.pop("total_flash_bytes"), "missing required field"),
        (lambda d: d.update(total_flash_bytes=0), "total_flash_bytes must be a positive integer or null"),
        (lambda d: d.update(total_flash_bytes=-1), "total_flash_bytes must be a positive integer or null"),
        (lambda d: d.update(total_flash_bytes=1.5), "total_flash_bytes must be a positive integer or null"),
        (lambda d: d.update(total_flash_bytes=True), "total_flash_bytes must be a positive integer or null"),
        (lambda d: d.update(dev_kit_aliases="EXAMPLE-DK"), "dev_kit_aliases must be a list"),
        (lambda d: d.update(dev_kit_aliases=[""]), "dev_kit_aliases must be a list"),
        (lambda d: d.update(dev_kit_aliases=[1]), "dev_kit_aliases must be a list"),
        (lambda d: d.update(notes=None), "'notes' must be a string"),
        (lambda d: d.update(default_firmware_reserve_bytes=-1), "default_firmware_reserve_bytes"),
    ],
)
def test_parse_profile_rejects_malformed_top_level_fields(mutation, match: str) -> None:
    data = _profile_json()
    mutation(data)

    with pytest.raises(TargetProfileError, match=match):
        _parse_profile("example.json", json.dumps(data))


@pytest.mark.parametrize(
    ("source_mutation", "match"),
    [
        (lambda s: s.pop("revision"), "source missing field"),
        (lambda s: s.update(datasheet_title=""), "datasheet_title.*non-empty string"),
        (lambda s: s.update(revision=123), r"source\.revision must be a string or null"),
        (lambda s: s.update(section=123), r"source\.section must be a string or null"),
        (lambda s: s.update(url=123), r"source\.url must be a string or null"),
        (lambda s: s.update(page="38"), r"source\.page must be an integer or null"),
        (lambda s: s.update(page=1.5), r"source\.page must be an integer or null"),
    ],
)
def test_parse_profile_rejects_malformed_source_fields(source_mutation, match: str) -> None:
    data = _profile_json()
    source_mutation(data["source"])

    with pytest.raises(TargetProfileError, match=match):
        _parse_profile("example.json", json.dumps(data))


def test_parse_profile_rejects_invalid_json() -> None:
    with pytest.raises(TargetProfileError, match="invalid JSON"):
        _parse_profile("example.json", "{not json")


def test_parse_profile_rejects_a_json_array_at_top_level() -> None:
    with pytest.raises(TargetProfileError, match="must be a JSON object"):
        _parse_profile("example.json", "[]")


def test_parse_profile_accepts_a_fully_null_source_except_title() -> None:
    # Exactly the shape used for nRF52840/STM32U585 today: title present,
    # everything else honestly null rather than guessed.
    data = _profile_json(source=_source(revision=None, section=None, page=None, url=None))

    profile = _parse_profile("example.json", json.dumps(data))

    assert profile.source.revision is None
    assert profile.source.section is None
    assert profile.source.page is None
    assert profile.source.url is None


# --- Catalog-wide ambiguity detection -------------------------------------


def test_check_for_collisions_accepts_a_catalog_with_no_overlap() -> None:
    profiles = [
        _synthetic_profile(id="a", mcu_part="MCU-A", dev_kit_aliases=("DK-A",), file_name="a.json"),
        _synthetic_profile(id="b", mcu_part="MCU-B", dev_kit_aliases=("DK-B",), file_name="b.json"),
    ]
    _check_for_collisions(profiles)  # must not raise


def test_check_for_collisions_rejects_duplicate_ids() -> None:
    profiles = [
        _synthetic_profile(id="dup", mcu_part="MCU-A", file_name="a.json"),
        _synthetic_profile(id="dup", mcu_part="MCU-B", file_name="b.json"),
    ]
    with pytest.raises(TargetProfileError, match="Ambiguous target name 'dup'"):
        _check_for_collisions(profiles)


def test_check_for_collisions_rejects_alias_colliding_with_another_mcu_part() -> None:
    profiles = [
        _synthetic_profile(id="a", mcu_part="SHARED-NAME", file_name="a.json"),
        _synthetic_profile(
            id="b", mcu_part="MCU-B", dev_kit_aliases=("SHARED-NAME",), file_name="b.json",
        ),
    ]
    with pytest.raises(TargetProfileError, match="Ambiguous target name"):
        _check_for_collisions(profiles)


def test_check_for_collisions_is_case_insensitive() -> None:
    profiles = [
        _synthetic_profile(id="a", mcu_part="Shared-Name", file_name="a.json"),
        _synthetic_profile(
            id="b", mcu_part="MCU-B", dev_kit_aliases=("shared-name",), file_name="b.json",
        ),
    ]
    with pytest.raises(TargetProfileError, match="Ambiguous target name"):
        _check_for_collisions(profiles)


def test_the_real_shipped_catalog_has_no_collisions() -> None:
    # load_target_profiles() already runs this at load time; asserting it
    # here directly documents the invariant against the real bundled files.
    _check_for_collisions(list(load_target_profiles()))
