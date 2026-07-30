from __future__ import annotations

from pathlib import Path

import pytest

from tensorscope.oracle import (
    DEFAULT_ORACLE_EXECUTABLE,
)
from tensorscope.oracle_validation import (
    validate_model_against_tflm,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

CORPUS_ROOT = (
    REPOSITORY_ROOT
    / "tests"
    / "model_corpus"
    / "models"
)


pytestmark = pytest.mark.skipif(
    not DEFAULT_ORACLE_EXECUTABLE.is_file(),
    reason=(
        "TFLM oracle is not built; run "
        "'make -C tools/tflm_oracle'"
    ),
)


@pytest.mark.parametrize(
    "model_name",
    [
        "hello_world_float.tflite",
        "hello_world_int8.tflite",
        "simple_add_model.tflite",
        "conv0.tflite",
        "micro_speech_quantized.tflite",
    ],
)
def test_oracle_head_is_not_smaller_than_static_plan(
    model_name: str,
) -> None:
    result = validate_model_against_tflm(
        CORPUS_ROOT / model_name
    )

    assert result.tensorscope_head >= 0
    assert result.tflm_head >= result.tensorscope_head
    assert result.head_delta >= 0


def test_hello_world_float_is_exact_match() -> None:
    result = validate_model_against_tflm(
        CORPUS_ROOT / "hello_world_float.tflite"
    )

    assert result.tensorscope_head == 128
    assert result.tflm_head == 128
    assert result.head_delta == 0
    assert result.exact_match