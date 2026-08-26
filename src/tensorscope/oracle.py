from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


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

    def __init__(self, message: str, *, category: str | None = None) -> None:
        super().__init__(message)
        self.category = category


UNREGISTERED_OPERATOR = "unregistered_operator"
STRUCTURALLY_UNSUPPORTED = "structurally_unsupported"

_UNREGISTERED_OPERATOR_TEXT = "unsupported operator at index"
_UNREGISTERED_KERNEL_TEXT = "didn't find op for builtin opcode"
_PREPARE_FAILURE_TEXT = "failed to prepare with status"
_INVOKE_FAILURE_TEXT = "failed to invoke with status"


def classify_oracle_incompatibility(
    exit_code: int,
    output: str,
) -> str | None:
    """Classify a non-zero oracle exit as a known operator-support failure.

    Two distinct problems surface as oracle failures and need different
    follow-up:

    ``unregistered_operator`` -- the oracle's own resolver has not
    registered an operator this model uses. This happens either before
    the interpreter runs at all (the oracle's pre-flight allowlist in
    ``ValidateRegisteredOperators`` rejects the builtin opcode, exit code
    6, message "unsupported operator at index") or at allocation time
    (TFLM's runtime resolver cannot find a kernel for an opcode the
    allowlist did recognize, TFLM's own "Didn't find op for builtin
    opcode" message). TFLM may already implement the operator elsewhere
    in its kernel set; this is a coverage gap in
    ``tools/tflm_oracle/main.cc``, not proof the model cannot run on
    TFLM.

    ``structurally_unsupported`` -- TFLM found a kernel for the operator
    and started preparing or invoking it, but that kernel refused this
    model's specific configuration (for example hybrid int8/float32
    quantization, which TFLM's kernel implementations explicitly
    reject). TFLM's own engine reports this generically as a node
    "failed to prepare with status" or "failed to invoke with status".
    The model cannot run on stock TFLM regardless of oracle
    registration; it was likely exported for a different runtime or
    toolchain.

    Returns ``None`` when the failure does not match either signature
    (for example a malformed model, a missing oracle executable, or the
    tensor arena being too small) -- that is a real failure, just not
    one of these two operator-support categories.
    """

    lowered = output.lower()
    if exit_code == 6 or _UNREGISTERED_OPERATOR_TEXT in lowered:
        return UNREGISTERED_OPERATOR
    if _UNREGISTERED_KERNEL_TEXT in lowered:
        return UNREGISTERED_OPERATOR
    if _PREPARE_FAILURE_TEXT in lowered or _INVOKE_FAILURE_TEXT in lowered:
        return STRUCTURALLY_UNSUPPORTED
    return None


@dataclass(frozen=True)
class OracleArenaObservation:
    """Memory observed during one pinned host-side TFLM allocator run."""

    capacity_bytes: int | None
    used_bytes: int | None
    head_bytes: int | None
    tail_bytes: int | None
    temporary_bytes: int | None
    remaining_bytes: int | None
    alignment_bytes: int | None
    tflm_revision: str | None
    source: Literal["tflm_oracle"] = "tflm_oracle"
    observation_scope: Literal["host_allocator_run"] = "host_allocator_run"

    def __post_init__(self) -> None:
        values = {
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self.used_bytes,
            "head_bytes": self.head_bytes,
            "tail_bytes": self.tail_bytes,
            "temporary_bytes": self.temporary_bytes,
            "remaining_bytes": self.remaining_bytes,
            "alignment_bytes": self.alignment_bytes,
        }
        for name, value in values.items():
            if value is not None and value < 0:
                raise TFLMOracleError(f"{name} must be non-negative: {value}")
        if self.capacity_bytes is not None and self.used_bytes is not None:
            if self.used_bytes > self.capacity_bytes:
                raise TFLMOracleError("used_bytes must not exceed capacity_bytes")
        if None not in (self.capacity_bytes, self.used_bytes, self.remaining_bytes):
            assert self.capacity_bytes is not None
            assert self.used_bytes is not None
            expected = self.capacity_bytes - self.used_bytes
            if self.remaining_bytes != expected:
                raise TFLMOracleError(
                    f"remaining_bytes is inconsistent: expected {expected}, got {self.remaining_bytes}"
                )
        if None not in (self.used_bytes, self.head_bytes, self.tail_bytes):
            assert self.head_bytes is not None
            assert self.tail_bytes is not None
            expected_used = self.head_bytes + self.tail_bytes
            if self.used_bytes != expected_used:
                raise TFLMOracleError(
                    f"used_bytes does not equal head_bytes plus tail_bytes: {self.used_bytes} != {expected_used}"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "observation_scope": self.observation_scope,
            "tflm_revision": self.tflm_revision,
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self.used_bytes,
            "head_bytes": self.head_bytes,
            "tail_bytes": self.tail_bytes,
            "temporary_bytes": self.temporary_bytes,
            "remaining_bytes": self.remaining_bytes,
            "alignment_bytes": self.alignment_bytes,
        }


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
    arena_remaining: int | None = None
    arena_temporary: int | None = None
    allocator_alignment: int | None = None
    tflm_revision: str | None = None

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

        self.observation

    @property
    def observation(self) -> OracleArenaObservation:
        return OracleArenaObservation(
            capacity_bytes=self.arena_capacity,
            used_bytes=self.arena_used,
            head_bytes=self.arena_head,
            tail_bytes=self.arena_tail,
            temporary_bytes=self.arena_temporary,
            remaining_bytes=self.arena_remaining,
            alignment_bytes=self.allocator_alignment,
            tflm_revision=self.tflm_revision,
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
            "arena_remaining": self.arena_remaining,
            "arena_temporary": self.arena_temporary,
            "allocator_alignment": self.allocator_alignment,
            "tflm_revision": self.tflm_revision,
            "categories": [
                asdict(category)
                for category in self.categories
            ],
        }


_STRUCTURED_BEGIN = "TENSOR_SCOPE_ORACLE_BEGIN"
_STRUCTURED_END = "TENSOR_SCOPE_ORACLE_END"
_REQUIRED_STRUCTURED_FIELDS = (
    "model_path", "model_size", "schema_version", "subgraph_count",
    "operator_code_count", "arena_capacity", "arena_used",
)
_NUMERIC_STRUCTURED_FIELDS = {
    "model_size", "schema_version", "subgraph_count", "operator_code_count",
    "arena_capacity", "arena_used", "arena_head_bytes", "arena_tail_bytes",
    "arena_temporary_bytes", "arena_remaining_bytes", "allocator_alignment_bytes",
}

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


def _parse_structured_block(output: str) -> dict[str, str]:
    begin_count = output.count(_STRUCTURED_BEGIN)
    end_count = output.count(_STRUCTURED_END)
    if begin_count != 1 or end_count != 1:
        raise TFLMOracleError(
            "Oracle output must contain exactly one structured oracle block"
        )
    body = output.split(_STRUCTURED_BEGIN, 1)[1].split(_STRUCTURED_END, 1)[0]
    fields: dict[str, str] = {}
    for line in body.splitlines():
        rendered = line.strip()
        if not rendered:
            continue
        if "=" not in rendered:
            raise TFLMOracleError(f"Malformed structured oracle line: {rendered!r}")
        key, value = rendered.split("=", 1)
        if key in fields:
            raise TFLMOracleError(f"Duplicate structured oracle field: {key}")
        fields[key] = value
    for key in _REQUIRED_STRUCTURED_FIELDS:
        if key not in fields:
            raise TFLMOracleError(f"Missing required structured oracle field: {key}")
    for key in _NUMERIC_STRUCTURED_FIELDS.intersection(fields):
        value = fields[key]
        if key == "arena_temporary_bytes" and value == "unavailable":
            continue
        if not value.isdecimal():
            raise TFLMOracleError(
                f"Malformed integer for structured oracle field {key}: {value!r}"
            )
    return fields


def parse_tflm_oracle_output(
    output: str,
) -> TFLMOracleResult:
    """Parse output produced by the C++ host oracle."""

    summary = _parse_structured_block(output)

    structured_head = summary.get("arena_head_bytes")
    structured_tail = summary.get("arena_tail_bytes")
    legacy_output = structured_head is None or structured_tail is None
    total = _required_match(
        _TOTAL_PATTERN, output, "arena total"
    ) if legacy_output else None
    head = None if structured_head is not None else _required_match(
        _HEAD_PATTERN, output, "arena head"
    )
    tail = None if structured_tail is not None else _required_match(
        _TAIL_PATTERN, output, "arena tail"
    )

    categories: list[AllocationCategory] = []

    # Categories are parsed only for compatibility with legacy oracle output.
    # Current structured output deliberately does not infer categories from
    # human-readable diagnostics.
    for match in _CATEGORY_PATTERN.finditer(output) if legacy_output else ():
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
        summary["arena_used"]
    )

    recorded_total = int(total.group("value")) if total is not None else arena_used

    if total is not None and arena_used != recorded_total:
        raise TFLMOracleError(
            "Interpreter arena usage differs from the "
            "RecordingMicroAllocator total: "
            f"{arena_used} != {recorded_total}"
        )

    return TFLMOracleResult(
        model_path=Path(
            summary["model_path"]
        ),
        model_size=int(
            summary["model_size"]
        ),
        schema_version=int(
            summary["schema_version"]
        ),
        subgraph_count=int(
            summary["subgraph_count"]
        ),
        operator_code_count=int(
            summary["operator_code_count"]
        ),
        arena_capacity=int(
            summary["arena_capacity"]
        ),
        arena_used=arena_used,
        arena_head=int(structured_head if structured_head is not None else head.group("value")),
        arena_tail=int(structured_tail if structured_tail is not None else tail.group("value")),
        categories=tuple(categories),
        raw_output=output,
        arena_remaining=(
            int(summary["arena_remaining_bytes"])
            if "arena_remaining_bytes" in summary else None
        ),
        arena_temporary=(
            None if summary.get("arena_temporary_bytes") in (None, "unavailable")
            else int(summary["arena_temporary_bytes"])
        ),
        allocator_alignment=(
            int(summary["allocator_alignment_bytes"])
            if "allocator_alignment_bytes" in summary else None
        ),
        tflm_revision=summary.get("tflm_revision"),
    )


def oracle_is_runnable(executable: str | Path) -> bool:
    """True only if ``executable`` both exists and can actually be launched
    as a subprocess on this platform.

    The compiled oracle binary is a Linux ELF executable committed to the
    repository, so ``Path.is_file()`` alone is not a safe gate for
    oracle-dependent tests: the file exists (and passes ``is_file()``) on
    any platform that checked it out from git, including native Windows,
    where trying to exec it raises ``OSError`` instead. This runs a cheap,
    argument-less probe invocation and treats an ``OSError`` from the
    subprocess launch itself (not a non-zero exit, which still proves the
    OS could run it) as "not runnable here".
    """

    path = Path(executable)
    if not path.is_file():
        return False
    try:
        subprocess.run([str(path)], capture_output=True, timeout=10)
    except OSError:
        return False
    return True


def run_tflm_oracle(
    model_path: str | Path,
    *,
    executable: str | Path | None = None,
) -> TFLMOracleResult:
    """Run the compiled TFLM oracle for one model."""

    model = Path(
        model_path
    ).expanduser().resolve()

    oracle = Path(
        executable
        if executable is not None
        else os.environ.get("TENSORSCOPE_TFLM_ORACLE", DEFAULT_ORACLE_EXECUTABLE)
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

    try:
        completed = subprocess.run(
            [
                str(oracle),
                str(model),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        # The committed oracle binary is a Linux ELF executable. It can
        # exist as a file (and pass the is_file() check above) on a
        # platform that cannot run it -- most commonly native Windows,
        # where launching it raises OSError (WinError 193, "%1 is not a
        # valid Win32 application") rather than anything is_file() catches.
        raise FileNotFoundError(
            "TFLM oracle executable exists but cannot run on this platform: "
            f"{oracle}\n"
            f"({error})\n"
            "The committed oracle binary is Linux-only. Run "
            "`tensorscope validate` from WSL or another Linux host, or "
            "build a native oracle for this platform and point "
            "TENSORSCOPE_TFLM_ORACLE at it. From a source checkout on "
            "Linux/WSL:\n"
            "  make -C tools/tflm_oracle"
        ) from error

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
            f"{combined_output}",
            category=classify_oracle_incompatibility(
                completed.returncode, combined_output
            ),
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
