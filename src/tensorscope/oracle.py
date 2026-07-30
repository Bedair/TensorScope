from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ORACLE_EXECUTABLE = (
    REPOSITORY_ROOT
    / "tools"
    / "tflm_oracle"
    / "build"
    / "tflm_oracle"
)


class TFLMOracleError(RuntimeError):
    """Raised when the host-side TFLM oracle fails."""


@dataclass(frozen=True)
class AllocationCategory:
    """One allocation category reported by RecordingMicroAllocator."""

    name: str
    used_bytes: int
    requested_bytes: int
    allocation_count: int | None
    object_description: str

    def __post_init__(self) -> None:
        if self.used_bytes < 0:
            raise TFLMOracleError(
                "Category used bytes must be non-negative"
            )

        if self.requested_bytes < 0:
            raise TFLMOracleError(
                "Category requested bytes must be non-negative"
            )

        if (
            self.allocation_count is not None
            and self.allocation_count < 0
        ):
            raise TFLMOracleError(
                "Category allocation count must be non-negative"
            )


@dataclass(frozen=True)
class TFLMOracleResult:
    """Parsed result from one TFLM oracle invocation."""

    model_path: Path
    model_size: int
    schema_version: int
    subgraph_count: int
    operator_code_count: int
    arena_capacity: int
    arena_used: int
    arena_head: int
    arena_tail: int
    categories: tuple[AllocationCategory, ...]
    raw_output: str

    def __post_init__(self) -> None:
        numeric_fields = {
            "model_size": self.model_size,
            "schema_version": self.schema_version,
            "subgraph_count": self.subgraph_count,
            "operator_code_count": self.operator_code_count,
            "arena_capacity": self.arena_capacity,
            "arena_used": self.arena_used,
            "arena_head": self.arena_head,
            "arena_tail": self.arena_tail,
        }

        for name, value in numeric_fields.items():
            if value < 0:
                raise TFLMOracleError(
                    f"{name} must be non-negative: {value}"
                )

        if self.arena_used != self.arena_head + self.arena_tail:
            raise TFLMOracleError(
                "Arena total does not equal head plus tail: "
                f"{self.arena_used} != "
                f"{self.arena_head} + {self.arena_tail}"
            )

    def category(
        self,
        name: str,
    ) -> AllocationCategory:
        for category in self.categories:
            if category.name == name:
                return category

        raise TFLMOracleError(
            f"Allocation category was not reported: {name}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_path": str(self.model_path),
            "model_size": self.model_size,
            "schema_version": self.schema_version,
            "subgraph_count": self.subgraph_count,
            "operator_code_count": self.operator_code_count,
            "arena_capacity": self.arena_capacity,
            "arena_used": self.arena_used,
            "arena_head": self.arena_head,
            "arena_tail": self.arena_tail,
            "categories": [
                asdict(category)
                for category in self.categories
            ],
        }


_SUMMARY_PATTERN = re.compile(
    r"TENSOR_SCOPE_ORACLE_BEGIN\s*"
    r"model_path=(?P<model_path>[^\r\n]+)\s*"
    r"model_size=(?P<model_size>\d+)\s*"
    r"schema_version=(?P<schema_version>\d+)\s*"
    r"subgraph_count=(?P<subgraph_count>\d+)\s*"
    r"operator_code_count=(?P<operator_code_count>\d+)\s*"
    r"arena_capacity=(?P<arena_capacity>\d+)\s*"
    r"arena_used=(?P<arena_used>\d+)\s*"
    r"TENSOR_SCOPE_ORACLE_END",
    re.MULTILINE,
)

_TOTAL_PATTERN = re.compile(
    r"\[RecordingMicroAllocator\]\s+"
    r"Arena allocation total\s+"
    r"(?P<value>\d+)\s+bytes"
)

_HEAD_PATTERN = re.compile(
    r"\[RecordingMicroAllocator\]\s+"
    r"Arena allocation head\s+"
    r"(?P<value>\d+)\s+bytes"
)

_TAIL_PATTERN = re.compile(
    r"\[RecordingMicroAllocator\]\s+"
    r"Arena allocation tail\s+"
    r"(?P<value>\d+)\s+bytes"
)

_CATEGORY_PATTERN = re.compile(
    r"\[RecordingMicroAllocator\]\s+"
    r"'(?P<name>[^']+)'\s+used\s+"
    r"(?P<used>\d+)\s+bytes"
    r"(?:\s+with alignment overhead)?\s*"
    r"\(requested\s+"
    r"(?P<requested>\d+)\s+bytes"
    r"(?:\s+for\s+"
    r"(?P<count>\d+)\s+"
    r"(?P<description>[^)]*))?"
    r"\)"
)


def _required_match(
    pattern: re.Pattern[str],
    output: str,
    description: str,
) -> re.Match[str]:
    match = pattern.search(output)

    if match is None:
        raise TFLMOracleError(
            f"Unable to parse {description} from oracle output"
        )

    return match


def parse_tflm_oracle_output(
    output: str,
) -> TFLMOracleResult:
    """Parse output produced by the C++ host oracle."""

    summary = _required_match(
        _SUMMARY_PATTERN,
        output,
        "oracle summary",
    )

    total = _required_match(
        _TOTAL_PATTERN,
        output,
        "arena total",
    )

    head = _required_match(
        _HEAD_PATTERN,
        output,
        "arena head",
    )

    tail = _required_match(
        _TAIL_PATTERN,
        output,
        "arena tail",
    )

    categories: list[AllocationCategory] = []

    for match in _CATEGORY_PATTERN.finditer(output):
        count_text = match.group("count")

        categories.append(
            AllocationCategory(
                name=match.group("name"),
                used_bytes=int(
                    match.group("used")
                ),
                requested_bytes=int(
                    match.group("requested")
                ),
                allocation_count=(
                    int(count_text)
                    if count_text is not None
                    else None
                ),
                object_description=(
                    match.group("description") or ""
                ).strip(),
            )
        )

    arena_used = int(
        summary.group("arena_used")
    )

    recorded_total = int(
        total.group("value")
    )

    if arena_used != recorded_total:
        raise TFLMOracleError(
            "Interpreter arena usage differs from the "
            "RecordingMicroAllocator total: "
            f"{arena_used} != {recorded_total}"
        )

    return TFLMOracleResult(
        model_path=Path(
            summary.group("model_path")
        ),
        model_size=int(
            summary.group("model_size")
        ),
        schema_version=int(
            summary.group("schema_version")
        ),
        subgraph_count=int(
            summary.group("subgraph_count")
        ),
        operator_code_count=int(
            summary.group("operator_code_count")
        ),
        arena_capacity=int(
            summary.group("arena_capacity")
        ),
        arena_used=arena_used,
        arena_head=int(
            head.group("value")
        ),
        arena_tail=int(
            tail.group("value")
        ),
        categories=tuple(categories),
        raw_output=output,
    )


def run_tflm_oracle(
    model_path: str | Path,
    *,
    executable: str | Path = DEFAULT_ORACLE_EXECUTABLE,
) -> TFLMOracleResult:
    """Run the compiled TFLM oracle for one model."""

    model = Path(
        model_path
    ).expanduser().resolve()

    oracle = Path(
        executable
    ).expanduser().resolve()

    if not model.is_file():
        raise FileNotFoundError(
            f"TFLite model does not exist: {model}"
        )

    if not oracle.is_file():
        raise FileNotFoundError(
            "TFLM oracle executable does not exist: "
            f"{oracle}\n"
            "Build it with:\n"
            "  make -C tools/tflm_oracle"
        )

    completed = subprocess.run(
        [
            str(oracle),
            str(model),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    combined_output = "\n".join(
        part
        for part in (
            completed.stdout,
            completed.stderr,
        )
        if part
    )

    if completed.returncode != 0:
        raise TFLMOracleError(
            "TFLM oracle failed with exit code "
            f"{completed.returncode}:\n"
            f"{combined_output}"
        )

    return parse_tflm_oracle_output(
        combined_output
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the host-side TensorFlow Lite Micro "
            "allocation oracle."
        )
    )

    parser.add_argument(
        "model",
        type=Path,
        help="Path to a .tflite model",
    )

    parser.add_argument(
        "--oracle",
        type=Path,
        default=DEFAULT_ORACLE_EXECUTABLE,
        help="Path to the compiled oracle executable",
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON result",
    )

    return parser


def main() -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args()

    try:
        result = run_tflm_oracle(
            arguments.model,
            executable=arguments.oracle,
        )
    except (
        FileNotFoundError,
        TFLMOracleError,
    ) as error:
        print(
            f"Error: {error}",
            file=sys.stderr,
        )

        return 1

    print(
        json.dumps(
            result.to_dict(),
            indent=2 if arguments.pretty else None,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())