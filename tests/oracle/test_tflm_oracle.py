from __future__ import annotations

from pathlib import Path

import pytest

from tensorscope.oracle import (
    TFLMOracleError,
    parse_tflm_oracle_output,
    run_tflm_oracle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

ORACLE_EXECUTABLE = (
    REPOSITORY_ROOT
    / "tools"
    / "tflm_oracle"
    / "build"
    / "tflm_oracle"
)

CORPUS_ROOT = (
    REPOSITORY_ROOT
    / "tests"
    / "model_corpus"
    / "models"
)


SAMPLE_OUTPUT = """
TENSOR_SCOPE_ORACLE_BEGIN
model_path=model.tflite
model_size=3164
schema_version=3
subgraph_count=1
operator_code_count=1
arena_capacity=2097152
arena_used=1424
TENSOR_SCOPE_ORACLE_END
[RecordingMicroAllocator] Arena allocation total 1424 bytes
[RecordingMicroAllocator] Arena allocation head 128 bytes
[RecordingMicroAllocator] Arena allocation tail 1296 bytes
[RecordingMicroAllocator] 'TfLiteEvalTensor data' used 240 bytes with alignment overhead (requested 240 bytes for 10 allocations)
[RecordingMicroAllocator] 'Persistent TfLiteTensor data' used 128 bytes with alignment overhead (requested 128 bytes for 2 tensors)
[RecordingMicroAllocator] 'Persistent buffer data' used 232 bytes with alignment overhead (requested 208 bytes for 5 allocations)
[RecordingMicroAllocator] 'NodeAndRegistration struct' used 192 bytes with alignment overhead (requested 192 bytes for 3 NodeAndRegistration structs)
"""


def test_parse_tflm_oracle_output() -> None:
    result = parse_tflm_oracle_output(
        SAMPLE_OUTPUT
    )

    assert result.model_path == Path(
        "model.tflite"
    )
    assert result.model_size == 3164
    assert result.schema_version == 3
    assert result.subgraph_count == 1
    assert result.operator_code_count == 1
    assert result.arena_capacity == 2097152
    assert result.arena_used == 1424
    assert result.arena_head == 128
    assert result.arena_tail == 1296

    categories = {
        category.name: category
        for category in result.categories
    }

    assert (
        categories[
            "TfLiteEvalTensor data"
        ].used_bytes
        == 240
    )

    assert (
        categories[
            "Persistent buffer data"
        ].requested_bytes
        == 208
    )

    assert (
        categories[
            "NodeAndRegistration struct"
        ].allocation_count
        == 3
    )


def test_category_lookup() -> None:
    result = parse_tflm_oracle_output(
        SAMPLE_OUTPUT
    )

    category = result.category(
        "Persistent TfLiteTensor data"
    )

    assert category.used_bytes == 128
    assert category.allocation_count == 2


def test_missing_category_is_rejected() -> None:
    result = parse_tflm_oracle_output(
        SAMPLE_OUTPUT
    )

    with pytest.raises(
        TFLMOracleError,
        match="was not reported",
    ):
        result.category("missing")


def test_inconsistent_recorded_total_is_rejected() -> None:
    invalid_output = SAMPLE_OUTPUT.replace(
        "Arena allocation total 1424 bytes",
        "Arena allocation total 1400 bytes",
    )

    with pytest.raises(
        TFLMOracleError,
        match="differs",
    ):
        parse_tflm_oracle_output(
            invalid_output
        )


def test_inconsistent_head_and_tail_are_rejected() -> None:
    invalid_output = SAMPLE_OUTPUT.replace(
        "Arena allocation tail 1296 bytes",
        "Arena allocation tail 1200 bytes",
    )

    with pytest.raises(
        TFLMOracleError,
        match="head plus tail",
    ):
        parse_tflm_oracle_output(
            invalid_output
        )


@pytest.mark.skipif(
    not ORACLE_EXECUTABLE.is_file(),
    reason=(
        "TFLM oracle is not built; run "
        "'make -C tools/tflm_oracle'"
    ),
)
@pytest.mark.parametrize(
    (
        "model_name",
        "expected_head",
        "expected_tail",
        "expected_total",
    ),
    [
        (
            "hello_world_float.tflite",
            128,
            1296,
            1424,
        ),
        (
            "hello_world_int8.tflite",
            32,
            1360,
            1392,
        ),
        (
            "simple_add_model.tflite",
            49152,
            992,
            50144,
        ),
        (
            "conv0.tflite",
            10432,
            1008,
            11440,
        ),
        (
            "micro_speech_quantized.tflite",
            5968,
            1552,
            7520,
        ),
        ("operator_chain_float.tflite", 128, 2704, 2832),
        ("quantize_dequantize_int8.tflite", 48, 880, 928),
    ],
)
def test_oracle_matches_recorded_corpus_results(
    model_name: str,
    expected_head: int,
    expected_tail: int,
    expected_total: int,
) -> None:
    result = run_tflm_oracle(
        CORPUS_ROOT / model_name,
        executable=ORACLE_EXECUTABLE,
    )

    assert result.model_path.name == model_name
    assert result.schema_version == 3
    assert result.subgraph_count == 1
    assert result.arena_head == expected_head
    assert result.arena_tail == expected_tail
    assert result.arena_used == expected_total
