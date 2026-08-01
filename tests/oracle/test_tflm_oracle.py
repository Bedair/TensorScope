from __future__ import annotations

from pathlib import Path

import pytest

from tensorscope.oracle import (
    OracleArenaObservation,
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

COMPLETE_OUTPUT = SAMPLE_OUTPUT.replace(
    "arena_used=1424\n",
    "arena_used=1424\narena_head_bytes=128\narena_tail_bytes=1296\n"
    "arena_temporary_bytes=unavailable\narena_remaining_bytes=2095728\n"
    "allocator_alignment_bytes=16\n"
    "tflm_revision=b89fb3e06e59d2f6af67e758242243da599bfedf\n"
    "future_field=ignored\n",
)


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
    assert categories["Persistent buffer data"].requested_bytes == 208
    assert categories["NodeAndRegistration struct"].allocation_count == 3


def test_complete_structured_observation_ignores_unknown_keys() -> None:
    result = parse_tflm_oracle_output(COMPLETE_OUTPUT)
    assert result.observation.to_dict() == {
        "source": "tflm_oracle",
        "observation_scope": "host_allocator_run",
        "tflm_revision": "b89fb3e06e59d2f6af67e758242243da599bfedf",
        "capacity_bytes": 2097152,
        "used_bytes": 1424,
        "head_bytes": 128,
        "tail_bytes": 1296,
        "temporary_bytes": None,
        "remaining_bytes": 2095728,
        "alignment_bytes": 16,
    }
    assert list(result.observation.to_dict()) == [
        "source", "observation_scope", "tflm_revision", "capacity_bytes",
        "used_bytes", "head_bytes", "tail_bytes", "temporary_bytes",
        "remaining_bytes", "alignment_bytes",
    ]


def test_observation_model_is_immutable_and_supports_optional_fields() -> None:
    observation = OracleArenaObservation(None, None, None, None, None, None, None, None)
    assert observation.used_bytes is None
    with pytest.raises((AttributeError, TypeError)):
        observation.used_bytes = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"used_bytes": -1}, "used_bytes must be non-negative"),
        ({"capacity_bytes": 1, "used_bytes": 2}, "must not exceed"),
        ({"capacity_bytes": 10, "used_bytes": 4, "remaining_bytes": 5}, "remaining_bytes is inconsistent"),
        ({"used_bytes": 5, "head_bytes": 2, "tail_bytes": 2}, "does not equal"),
    ],
)
def test_observation_rejects_proven_invariant_failures(kwargs: dict[str, int], message: str) -> None:
    values: dict[str, object] = {
        "capacity_bytes": None, "used_bytes": None, "head_bytes": None,
        "tail_bytes": None, "temporary_bytes": None, "remaining_bytes": None,
        "alignment_bytes": None, "tflm_revision": None,
    }
    values.update(kwargs)
    with pytest.raises(TFLMOracleError, match=message):
        OracleArenaObservation(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (COMPLETE_OUTPUT.replace("arena_used=1424", "arena_used=nope"), "arena_used"),
        (COMPLETE_OUTPUT.replace("arena_head_bytes=128", "arena_head_bytes=-1"), "arena_head_bytes"),
        (COMPLETE_OUTPUT.replace("model_size=3164\n", ""), "Missing required.*model_size"),
        (COMPLETE_OUTPUT.replace("arena_used=1424\n", "arena_used=1424\narena_used=1424\n"), "Duplicate.*arena_used"),
    ],
)
def test_structured_parser_rejects_bad_fields(output: str, message: str) -> None:
    with pytest.raises(TFLMOracleError, match=message):
        parse_tflm_oracle_output(output)


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
    assert result.arena_remaining == result.arena_capacity - expected_total
    assert result.allocator_alignment == 16
    assert result.tflm_revision == "b89fb3e06e59d2f6af67e758242243da599bfedf"
    assert "arena_head_bytes=" in result.raw_output
    assert "[RecordingMicroAllocator] Arena allocation total" in result.raw_output
