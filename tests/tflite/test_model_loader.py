from __future__ import annotations

from pathlib import Path

import pytest

from tensorscope.tflite.model_loader import (
    TFLiteModelError,
    load_tflite_model,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

CORPUS_ROOT = (
    REPOSITORY_ROOT
    / "tests"
    / "model_corpus"
    / "models"
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
def test_loads_corpus_models(
    model_name: str,
) -> None:
    loaded = load_tflite_model(
        CORPUS_ROOT / model_name
    )

    assert loaded.path.name == model_name
    assert loaded.schema_version > 0
    assert loaded.subgraph_count >= 1
    assert loaded.operator_code_count >= 1
    assert loaded.buffer_count >= 1


def test_hello_world_float_metadata() -> None:
    loaded = load_tflite_model(
        CORPUS_ROOT / "hello_world_float.tflite"
    )

    assert loaded.schema_version == 3
    assert loaded.subgraph_count == 1


def test_missing_model_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_tflite_model(
            CORPUS_ROOT / "missing.tflite"
        )


def test_too_small_file_is_rejected(
    tmp_path: Path,
) -> None:
    invalid_path = tmp_path / "small.tflite"
    invalid_path.write_bytes(b"abc")

    with pytest.raises(
        TFLiteModelError,
        match="too small",
    ):
        load_tflite_model(invalid_path)


def test_wrong_identifier_is_rejected(
    tmp_path: Path,
) -> None:
    invalid_path = tmp_path / "invalid.tflite"
    invalid_path.write_bytes(
        b"\x00\x00\x00\x00NOPEinvalid"
    )

    with pytest.raises(
        TFLiteModelError,
        match="Invalid TFLite file identifier",
    ):
        load_tflite_model(invalid_path)
