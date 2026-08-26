from __future__ import annotations

from pathlib import Path
import json

import pytest

from tensorscope.oracle import (
    DEFAULT_ORACLE_EXECUTABLE,
    oracle_is_runnable,
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
    not oracle_is_runnable(DEFAULT_ORACLE_EXECUTABLE),
    reason=(
        "TFLM oracle is not available on this platform (not built, or the "
        "committed Linux binary cannot run here); run "
        "'make -C tools/tflm_oracle' from Linux/WSL, or set "
        "TENSORSCOPE_TFLM_ORACLE to a working binary"
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
        "operator_chain_float.tflite",
        "quantize_dequantize_int8.tflite",
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


@pytest.mark.parametrize(
    ("model_name", "expected_head"),
    [
        ("operator_chain_float.tflite", 128),
        ("quantize_dequantize_int8.tflite", 48),
        ("pad0.tflite", 2608),
        ("strided_slice0.tflite", 9296),
        ("sub0.tflite", 6272),
        ("leaky_relu22.tflite", 10240),
        ("residual_add_float.tflite", 192),
    ],
)
def test_operator_coverage_models_are_exact_matches(
    model_name: str, expected_head: int,
) -> None:
    result = validate_model_against_tflm(CORPUS_ROOT / model_name)
    assert result.tensorscope_head == expected_head
    assert result.tflm_head == expected_head
    assert result.head_delta == 0
    assert result.exact_match


def test_residual_block_demonstrates_skip_connection_reuse_blocking() -> None:
    """The skip connection is the point of this fixture: the graph input is
    read again by the final ADD after an unrelated CONV_2D runs in between,
    so it must stay live across that operator and cannot hand its slot to
    anything -- the exact scenario explain.py's ReuseBlocker documents."""

    result = validate_model_against_tflm(CORPUS_ROOT / "residual_add_float.tflite")

    assert result.exact_match
    assert result.tensorscope_head == 192


def test_validation_matrix_is_deterministic_and_exact() -> None:
    matrix_path = REPOSITORY_ROOT / "tests" / "model_corpus" / "validation_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert matrix["scope"] == "planned_arena_head"
    assert [row["filename"] for row in matrix["models"]] == sorted(
        row["filename"] for row in matrix["models"]
    )
    for row in matrix["models"]:
        result = validate_model_against_tflm(CORPUS_ROOT / row["filename"])
        assert result.tensorscope_head == row["tensorscope_head"]
        assert result.tflm_head == row["tflm_head"]
        assert result.head_delta == row["delta_bytes"] == 0
        assert row["validation_state"] == "exact_match"


def test_oracle_observation_matrix_is_deterministic_and_accounted() -> None:
    matrix_path = REPOSITORY_ROOT / "tests" / "model_corpus" / "oracle_observation_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert matrix["source"] == "tflm_oracle"
    assert matrix["observation_scope"] == "host_allocator_run"
    assert [row["filename"] for row in matrix["models"]] == sorted(
        row["filename"] for row in matrix["models"]
    )
    for row in matrix["models"]:
        result = validate_model_against_tflm(CORPUS_ROOT / row["filename"])
        observation = result.oracle.observation
        assert result.exact_match
        assert observation.head_bytes == row["observed_head_bytes"]
        assert observation.tail_bytes == row["observed_tail_bytes"]
        assert observation.used_bytes == row["observed_used_bytes"]
        assert observation.capacity_bytes == row["capacity_bytes"]
        assert observation.remaining_bytes == row["remaining_bytes"]
        assert observation.used_bytes <= observation.capacity_bytes
        assert observation.remaining_bytes == observation.capacity_bytes - observation.used_bytes
        assert observation.tflm_revision == row["tflm_revision"] == matrix["tflm_revision"]
