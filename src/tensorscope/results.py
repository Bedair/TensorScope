from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MemoryScope = Literal["arena_head", "arena_tail", "arena_total"]
Confidence = Literal[
    "exact",
    "estimated",
    "bounded",
    "unsupported",
    "not_estimated",
]
ValidationState = Literal[
    "exact_match",
    "not_validated",
    "mismatch",
    "unavailable",
]
MemorySource = Literal["static_analysis", "tflm_oracle"]


@dataclass(frozen=True)
class MemoryFigure:
    """A memory value together with its scope and evidence."""

    bytes: int | None
    scope: MemoryScope
    confidence: Confidence
    source: MemorySource | None
    validation_state: ValidationState
    validated_tflm_revision: str | None = None

    def __post_init__(self) -> None:
        if self.bytes is not None and self.bytes < 0:
            raise ValueError("Memory bytes must be non-negative")

        if self.confidence == "not_estimated" and self.bytes is not None:
            raise ValueError("A not-estimated figure cannot have a value")

        if self.source == "":
            raise ValueError("Memory source must be a named source or None")

        if self.validation_state == "exact_match":
            if not self.validated_tflm_revision:
                raise ValueError("Validated figures require a TFLM revision")
        elif self.validated_tflm_revision is not None:
            raise ValueError(
                "Only exact-match figures may name a validated TFLM revision"
            )

    @property
    def validated(self) -> bool:
        return self.validation_state == "exact_match"

    def to_dict(self) -> dict[str, object]:
        return {
            "bytes": self.bytes,
            "scope": self.scope,
            "confidence": self.confidence,
            "source": self.source,
            "validation_state": self.validation_state,
            "validated": self.validated,
            "validated_tflm_revision": self.validated_tflm_revision,
        }
