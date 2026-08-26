from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json

from tensorscope.memory_budget import MCUProfile


TARGET_PROFILE_SCHEMA_VERSION = 1
TARGET_SOURCE_CLASSIFICATION = "vendor_datasheet"

TARGET_USAGE_HINT = (
    "Check a model against one: `tensorscope analyze MODEL --target <mcu-part-or-dev-kit-name>` "
    "(matches case-insensitively; run with a name that isn't listed here to see the exact error)."
)
TARGET_DISCLAIMER = (
    "Each figure is sourced from the named vendor datasheet or reference manual; see "
    "each profile's `source` field for the citation this project could actually verify."
)

_REQUIRED_FIELDS = (
    "schema_version", "id", "mcu_part", "vendor", "architecture",
    "total_sram_bytes", "dev_kit_aliases", "source", "notes",
)
_REQUIRED_SOURCE_FIELDS = ("datasheet_title", "revision", "section", "page", "url")
_NULLABLE_STRING_SOURCE_FIELDS = ("revision", "section", "url")


class TargetProfileError(ValueError):
    """Raised when a target profile file is malformed or the catalog is ambiguous.

    Subclasses ValueError so it flows through the CLI's existing input-error
    handling without a new exception type needing to be wired in separately.
    """


@dataclass(frozen=True)
class ProfileSource:
    datasheet_title: str
    revision: str | None
    section: str | None
    page: int | None
    url: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "datasheet_title": self.datasheet_title,
            "revision": self.revision,
            "section": self.section,
            "page": self.page,
            "url": self.url,
        }


@dataclass(frozen=True)
class TargetProfile:
    id: str
    mcu_part: str
    vendor: str
    architecture: str
    total_sram_bytes: int
    dev_kit_aliases: tuple[str, ...]
    source: ProfileSource
    notes: str
    default_firmware_reserve_bytes: int | None
    file_name: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "mcu_part": self.mcu_part,
            "vendor": self.vendor,
            "architecture": self.architecture,
            "total_sram_bytes": self.total_sram_bytes,
            "dev_kit_aliases": list(self.dev_kit_aliases),
            "source": self.source.to_dict(),
            "notes": self.notes,
            "default_firmware_reserve_bytes": self.default_firmware_reserve_bytes,
        }


def _require_non_empty_string(data: dict[str, object], field: str, file_name: str) -> str:
    value = data[field]
    if not isinstance(value, str) or not value:
        raise TargetProfileError(f"{file_name}: field {field!r} must be a non-empty string")
    return value


def _parse_profile(file_name: str, raw_text: str) -> TargetProfile:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise TargetProfileError(f"{file_name}: invalid JSON ({error})") from error

    if not isinstance(data, dict):
        raise TargetProfileError(f"{file_name}: profile must be a JSON object")

    missing = [field for field in _REQUIRED_FIELDS if field not in data]
    if missing:
        raise TargetProfileError(f"{file_name}: missing required field(s): {', '.join(missing)}")

    if data["schema_version"] != TARGET_PROFILE_SCHEMA_VERSION:
        raise TargetProfileError(
            f"{file_name}: unsupported schema_version {data['schema_version']!r}; "
            f"expected {TARGET_PROFILE_SCHEMA_VERSION}"
        )

    profile_id = _require_non_empty_string(data, "id", file_name)
    mcu_part = _require_non_empty_string(data, "mcu_part", file_name)
    vendor = _require_non_empty_string(data, "vendor", file_name)
    architecture = _require_non_empty_string(data, "architecture", file_name)

    if not isinstance(data["notes"], str):
        raise TargetProfileError(f"{file_name}: field 'notes' must be a string")

    total_sram_bytes = data["total_sram_bytes"]
    if (
        not isinstance(total_sram_bytes, int)
        or isinstance(total_sram_bytes, bool)
        or total_sram_bytes <= 0
    ):
        raise TargetProfileError(f"{file_name}: total_sram_bytes must be a positive integer")

    aliases = data["dev_kit_aliases"]
    if not isinstance(aliases, list) or not all(
        isinstance(item, str) and item for item in aliases
    ):
        raise TargetProfileError(
            f"{file_name}: dev_kit_aliases must be a list of non-empty strings"
        )

    source_data = data["source"]
    if not isinstance(source_data, dict):
        raise TargetProfileError(f"{file_name}: field 'source' must be a JSON object")
    missing_source = [field for field in _REQUIRED_SOURCE_FIELDS if field not in source_data]
    if missing_source:
        raise TargetProfileError(
            f"{file_name}: source missing field(s): {', '.join(missing_source)}"
        )
    datasheet_title = _require_non_empty_string(source_data, "datasheet_title", file_name)
    for field in _NULLABLE_STRING_SOURCE_FIELDS:
        value = source_data[field]
        if value is not None and not isinstance(value, str):
            raise TargetProfileError(f"{file_name}: source.{field} must be a string or null")
    page = source_data["page"]
    if page is not None and (not isinstance(page, int) or isinstance(page, bool)):
        raise TargetProfileError(f"{file_name}: source.page must be an integer or null")

    reserve = data.get("default_firmware_reserve_bytes")
    if reserve is not None and (
        not isinstance(reserve, int) or isinstance(reserve, bool) or reserve < 0
    ):
        raise TargetProfileError(
            f"{file_name}: default_firmware_reserve_bytes must be a non-negative integer or null"
        )

    return TargetProfile(
        id=profile_id,
        mcu_part=mcu_part,
        vendor=vendor,
        architecture=architecture,
        total_sram_bytes=total_sram_bytes,
        dev_kit_aliases=tuple(aliases),
        source=ProfileSource(
            datasheet_title=datasheet_title,
            revision=source_data["revision"],
            section=source_data["section"],
            page=page,
            url=source_data["url"],
        ),
        notes=data["notes"],
        default_firmware_reserve_bytes=reserve,
        file_name=file_name,
    )


def _match_names(profile: TargetProfile) -> tuple[str, ...]:
    """Every string --target may resolve against for this profile."""
    return (profile.id, profile.mcu_part, *profile.dev_kit_aliases)


def _check_for_collisions(profiles: list[TargetProfile]) -> None:
    seen: dict[str, str] = {}
    for profile in profiles:
        for name in _match_names(profile):
            key = name.strip().lower()
            if key in seen and seen[key] != profile.file_name:
                raise TargetProfileError(
                    f"Ambiguous target name {name!r}: matches both "
                    f"{seen[key]} and {profile.file_name}"
                )
            seen[key] = profile.file_name


def load_target_profiles() -> tuple[TargetProfile, ...]:
    """Load and validate every target profile bundled with the package.

    Fails loudly on a malformed file or a catalog-wide ambiguity (the same
    id, mcu_part, or dev_kit_alias claimed by two files) rather than
    silently skipping it -- a profile that failed to load unnoticed would be
    worse than one that was never there, since its absence from --target
    would look like "unknown name" instead of "broken file".
    """

    package_files = resources.files("tensorscope") / "profiles" / "mcu"
    profiles: list[TargetProfile] = []
    for entry in sorted(package_files.iterdir(), key=lambda item: item.name):
        if entry.name.endswith(".json"):
            profiles.append(_parse_profile(entry.name, entry.read_text(encoding="utf-8")))

    _check_for_collisions(profiles)
    return tuple(profiles)


def resolve_target(
    name: str,
    profiles: tuple[TargetProfile, ...] | None = None,
) -> TargetProfile:
    """Resolve --target <name> against every profile's id, mcu_part, and
    dev_kit_aliases, case-insensitively. Exact match only: no partial,
    prefix, or fuzzy matching, so a resolved target is never a guess.
    """

    if profiles is None:
        profiles = load_target_profiles()

    normalized = name.strip().lower()
    for profile in profiles:
        if any(candidate.strip().lower() == normalized for candidate in _match_names(profile)):
            return profile

    known = sorted(
        {candidate for profile in profiles for candidate in (profile.mcu_part, *profile.dev_kit_aliases)},
        key=str.lower,
    )
    raise TargetProfileError(
        f"Unknown target {name!r}; known MCU parts and dev-kit aliases: {', '.join(known)}"
    )


def as_mcu_profile(target: TargetProfile) -> MCUProfile:
    """Adapt a real per-vendor TargetProfile into the MCUProfile shape the
    existing budget-evaluation code (evaluate_profile_budget) already
    consumes, so that code doesn't need a second implementation for real
    targets versus the generic Cortex-class presets.
    """

    citation = target.source.datasheet_title
    if target.source.revision:
        citation = f"{citation}, {target.source.revision}"

    return MCUProfile(
        profile_id=target.id,
        display_name=f"{target.mcu_part} ({target.vendor})",
        ram_bytes=target.total_sram_bytes,
        family=target.architecture,
        notes=f"Source: {citation}",
        source_classification=TARGET_SOURCE_CLASSIFICATION,
    )


def render_target_verdict_clause(target: TargetProfile) -> str:
    """The inline "on <part>, per <vendor> datasheet" clause spliced into a
    --target budget verdict (see memory_budget.render_budget_verdict's
    target_clause parameter), naming the resolved part and citing where its
    RAM figure came from -- so the verdict states which real chip and which
    real source backed it, not just the generic head-only caveat every
    budget verdict already carries. A generic --mcu-profile verdict has no
    real citation to show and never calls this.
    """

    return f"on {target.mcu_part}, per {target.vendor} datasheet"


def render_target_listing() -> str:
    profiles = load_target_profiles()
    lines = []
    for profile in profiles:
        aliases = ", ".join(profile.dev_kit_aliases) if profile.dev_kit_aliases else "(none)"
        lines.append(
            f"{profile.mcu_part}\t{profile.vendor}, {profile.architecture}\t"
            f"{profile.total_sram_bytes} bytes\taliases: {aliases}"
        )
    return "\n".join((*lines, TARGET_DISCLAIMER, TARGET_USAGE_HINT))
