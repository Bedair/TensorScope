from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tensorscope.graph import (
    calculate_graph_memory_plan,
    convert_tflite_model,
)
from tensorscope.oracle import (
    TFLMOracleResult,
    run_tflm_oracle,
)
from tensorscope.tflite.model_loader import (
    load_tflite_model,
)


@dataclass(frozen=True)
class OracleValidationResult:
    """Comparison of TensorScope and TFLM head-memory results."""

    model_path: Path
    tensorscope_head: int
    tflm_head: int
    head_delta: int
    exact_match: bool
    oracle: TFLMOracleResult

    def __post_init__(self) -> None:
        if self.tensorscope_head < 0:
            raise ValueError(
                "TensorScope head must be non-negative"
            )

        if self.tflm_head < 0:
            raise ValueError(
                "TFLM head must be non-negative"
            )

        expected_delta = (
            self.tflm_head
            - self.tensorscope_head
        )

        if self.head_delta != expected_delta:
            raise ValueError(
                "Head delta is inconsistent: "
                f"expected {expected_delta}, "
                f"got {self.head_delta}"
            )

        if self.exact_match != (
            self.tensorscope_head == self.tflm_head
        ):
            raise ValueError(
                "Exact-match flag is inconsistent"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_path": str(self.model_path),
            "tensorscope_head": self.tensorscope_head,
            "tflm_head": self.tflm_head,
            "head_delta": self.head_delta,
            "exact_match": self.exact_match,
            "tflm_arena_used": self.oracle.arena_used,
            "tflm_arena_tail": self.oracle.arena_tail,
            "oracle_arena_observation": self.oracle.observation.to_dict(),
        }


def validate_model_against_tflm(
    model_path: str | Path,
) -> OracleValidationResult:
    """Compare TensorScope's planned head with the TFLM oracle."""

    path = Path(
        model_path
    ).expanduser().resolve()

    graph = convert_tflite_model(
        load_tflite_model(path)
    )

    memory_plan = calculate_graph_memory_plan(
        graph
    )

    oracle = run_tflm_oracle(path)

    tensorscope_head = (
        memory_plan.maximum_memory_size
    )

    tflm_head = oracle.arena_head

    return OracleValidationResult(
        model_path=path,
        tensorscope_head=tensorscope_head,
        tflm_head=tflm_head,
        head_delta=(
            tflm_head
            - tensorscope_head
        ),
        exact_match=(
            tensorscope_head == tflm_head
        ),
        oracle=oracle,
    )
