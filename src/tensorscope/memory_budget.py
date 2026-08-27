from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


PROFILE_DISCLAIMER = (
    "These are generic planning presets, not specifications for every MCU using "
    "the named processor core."
)
PROFILE_USAGE_HINT = (
    "Check a model against one: `tensorscope analyze MODEL --mcu-profile <id>` "
    "(or --arena-head-budget SIZE for a direct byte budget instead of a profile)."
)
PROFILE_SOURCE = "generic_planning_preset"


@dataclass(frozen=True)
class MCUProfile:
    profile_id: str
    display_name: str
    ram_bytes: int
    family: str
    notes: str
    source_classification: str = PROFILE_SOURCE


MCU_PROFILES: tuple[MCUProfile, ...] = (
    MCUProfile("cortex-m0-32k", "Cortex-M0 class — 32 KiB RAM", 32 * 1024, "Arm Cortex-M0 class", PROFILE_DISCLAIMER),
    MCUProfile("cortex-m4-128k", "Cortex-M4 class — 128 KiB RAM", 128 * 1024, "Arm Cortex-M4 class", PROFILE_DISCLAIMER),
    MCUProfile("cortex-m4-256k", "Cortex-M4 class — 256 KiB RAM", 256 * 1024, "Arm Cortex-M4 class", PROFILE_DISCLAIMER),
    MCUProfile("cortex-m7-512k", "Cortex-M7 class — 512 KiB RAM", 512 * 1024, "Arm Cortex-M7 class", PROFILE_DISCLAIMER),
    MCUProfile("cortex-m7-1m", "Cortex-M7 class — 1 MiB RAM", 1024 * 1024, "Arm Cortex-M7 class", PROFILE_DISCLAIMER),
)

_PROFILE_BY_ID = {profile.profile_id: profile for profile in MCU_PROFILES}
_SIZE_PATTERN = re.compile(r"^([0-9]+)\s*(B|KiB|MiB)?$", re.IGNORECASE)


def parse_size(value: str) -> int:
    """Parse a non-negative integral byte quantity using binary units."""

    rendered = value.strip()
    match = _SIZE_PATTERN.fullmatch(rendered)
    if match is None:
        raise ValueError(
            f"Invalid size {value!r}; use a whole number followed by bytes, B, KiB, or MiB"
        )
    quantity = int(match.group(1))
    suffix = (match.group(2) or "").lower()
    multiplier = {"": 1, "b": 1, "kib": 1024, "mib": 1024 * 1024}[suffix]
    return quantity * multiplier


def get_mcu_profile(profile_id: str) -> MCUProfile:
    try:
        return _PROFILE_BY_ID[profile_id]
    except KeyError as error:
        available = ", ".join(profile.profile_id for profile in MCU_PROFILES)
        raise ValueError(
            f"Unknown MCU profile {profile_id!r}; available profiles: {available}"
        ) from error


BudgetSource = Literal["direct", "profile"]
BudgetStatus = Literal["fits", "exact_fit", "exceeds"]


@dataclass(frozen=True)
class ArenaHeadBudgetResult:
    source: BudgetSource
    profile_id: str | None
    profile_name: str | None
    profile_ram_bytes: int | None
    reserve_bytes: int
    effective_budget_bytes: int
    planned_arena_head_bytes: int
    remaining_bytes: int
    utilization_ratio: float | None
    utilization_percent: float | None
    status: BudgetStatus
    scope: Literal["arena_head"] = "arena_head"

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "profile_id": self.profile_id,
            "profile_name": self.profile_name,
            "profile_ram_bytes": self.profile_ram_bytes,
            "reserve_bytes": self.reserve_bytes,
            "effective_budget_bytes": self.effective_budget_bytes,
            "planned_arena_head_bytes": self.planned_arena_head_bytes,
            "remaining_bytes": self.remaining_bytes,
            "utilization_ratio": self.utilization_ratio,
            "utilization_percent": self.utilization_percent,
            "status": self.status,
            "scope": self.scope,
            "verdict": render_budget_verdict(self),
        }


_BUDGET_STATUS_LABELS: dict[BudgetStatus, str] = {
    "fits": "FITS",
    "exact_fit": "EXACT FIT",
    "exceeds": "EXCEEDS BUDGET",
}


def render_budget_verdict(
    budget: ArenaHeadBudgetResult,
    *,
    target_clause: str | None = None,
) -> str:
    """Render the FITS/EXACT FIT/EXCEEDS BUDGET verdict with its arena-head-only
    scope stated inline, so the caveat cannot be missed at the point the verdict
    itself is read -- not only in a separate field elsewhere in the same result.

    Shared by text, JSON (via to_dict), and HTML rendering so the wording can't
    drift between them.

    ``target_clause`` is optional and only ever supplied for a --target result
    (see tensorscope.target_profiles.render_target_verdict_clause) -- a generic
    --mcu-profile/--arena-head-budget verdict has no real datasheet citation to
    show and is unaffected by this parameter. Kept as a rendering-time argument
    rather than a new ArenaHeadBudgetResult field, so that dataclass's shape
    stays exactly what every existing caller already expects.
    """

    label = _BUDGET_STATUS_LABELS[budget.status]
    clause = f" {target_clause}" if target_clause else ""
    return (
        f"{label} (head only — {budget.planned_arena_head_bytes:,} / "
        f"{budget.effective_budget_bytes:,} bytes{clause}; arena tail is not estimated "
        "here — run `tensorscope validate` for an oracle-observed tail)"
    )


def render_budget_source_label(
    budget: ArenaHeadBudgetResult,
    *,
    target_clause: str | None = None,
) -> str:
    """Render the human-readable "Budget source" label for text/HTML output.

    The single source of truth for this label, shared by the text renderer
    (cli.py's ``_render_budget_text``) and the HTML renderer
    (html_report.py's ``_budget_section``) -- both call this instead of each
    keeping their own copy of the ``budget.source == "direct"`` /
    ``target_clause is not None`` branching. A prior fix applied only to the
    HTML copy (mislabeling a --target result "Generic MCU planning
    profile") went unnoticed in the text renderer's independent copy for
    exactly this reason; a single shared function makes that class of drift
    impossible between whichever render surfaces call it.

    JSON output does not render this label at all -- it exposes the raw
    ``budget.source`` enum ("direct"/"profile") plus ``profile_id`` and the
    citation-bearing ``verdict`` string instead, so it was never subject to
    this particular mislabeling.
    """

    if budget.source == "direct":
        return "Direct arena-head budget"
    if target_clause is not None:
        return "Real MCU/dev-kit target (cited datasheet)"
    return "Generic MCU planning profile"


def _evaluate(
    planned_arena_head_bytes: int,
    effective_budget_bytes: int,
    *,
    source: BudgetSource,
    profile: MCUProfile | None,
    reserve_bytes: int,
) -> ArenaHeadBudgetResult:
    if planned_arena_head_bytes < 0 or effective_budget_bytes < 0 or reserve_bytes < 0:
        raise ValueError("Memory byte values must be non-negative")
    remaining = effective_budget_bytes - planned_arena_head_bytes
    status: BudgetStatus = (
        "fits" if remaining > 0 else "exact_fit" if remaining == 0 else "exceeds"
    )
    ratio = (
        planned_arena_head_bytes / effective_budget_bytes
        if effective_budget_bytes
        else None
    )
    return ArenaHeadBudgetResult(
        source=source,
        profile_id=profile.profile_id if profile else None,
        profile_name=profile.display_name if profile else None,
        profile_ram_bytes=profile.ram_bytes if profile else None,
        reserve_bytes=reserve_bytes,
        effective_budget_bytes=effective_budget_bytes,
        planned_arena_head_bytes=planned_arena_head_bytes,
        remaining_bytes=remaining,
        utilization_ratio=ratio,
        utilization_percent=ratio * 100 if ratio is not None else None,
        status=status,
    )


def evaluate_direct_budget(planned_arena_head_bytes: int, budget_bytes: int) -> ArenaHeadBudgetResult:
    return _evaluate(planned_arena_head_bytes, budget_bytes, source="direct", profile=None, reserve_bytes=0)


def evaluate_profile_budget(
    planned_arena_head_bytes: int,
    profile: MCUProfile,
    reserve_bytes: int = 0,
) -> ArenaHeadBudgetResult:
    if reserve_bytes > profile.ram_bytes:
        raise ValueError(
            f"Reserve ({reserve_bytes} bytes) exceeds profile RAM ({profile.ram_bytes} bytes)"
        )
    return _evaluate(
        planned_arena_head_bytes,
        profile.ram_bytes - reserve_bytes,
        source="profile",
        profile=profile,
        reserve_bytes=reserve_bytes,
    )


def render_profile_listing() -> str:
    lines = [
        f"{profile.profile_id}\t{profile.display_name}\t{profile.ram_bytes} bytes"
        for profile in MCU_PROFILES
    ]
    return "\n".join((*lines, PROFILE_DISCLAIMER, PROFILE_USAGE_HINT))
