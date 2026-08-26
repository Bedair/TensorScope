from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPOSITORY_ROOT / "tests" / "model_corpus"
MODELS_ROOT = CORPUS_ROOT / "models"
MANIFEST_PATH = CORPUS_ROOT / "manifest.json"


REQUIRED_MODELS = {
    "hello_world_float.tflite",
    "hello_world_int8.tflite",
    "simple_add_model.tflite",
    "conv0.tflite",
    "micro_speech_quantized.tflite",
    "operator_chain_float.tflite",
    "quantize_dequantize_int8.tflite",
    "pad0.tflite",
    "strided_slice0.tflite",
    "sub0.tflite",
    "leaky_relu22.tflite",
    "residual_add_float.tflite",
}


def load_manifest() -> dict[str, Any]:
    return json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_manifest_exists() -> None:
    assert MANIFEST_PATH.is_file()


def test_manifest_schema_version() -> None:
    manifest = load_manifest()

    assert manifest["schema_version"] == 1


def test_manifest_contains_twelve_models() -> None:
    manifest = load_manifest()

    assert manifest["model_count"] == 12
    assert len(manifest["models"]) == 12


def test_required_models_exist() -> None:
    model_names = {
        path.name
        for path in MODELS_ROOT.glob("*.tflite")
    }

    assert model_names == REQUIRED_MODELS


def test_all_manifest_models_exist() -> None:
    manifest = load_manifest()

    for model in manifest["models"]:
        path = REPOSITORY_ROOT / model["path"]

        assert path.is_file(), f"Missing model: {path}"
        assert path.stat().st_size > 0
        assert path.suffix == ".tflite"


def test_manifest_sizes_match_files() -> None:
    manifest = load_manifest()

    for model in manifest["models"]:
        path = REPOSITORY_ROOT / model["path"]

        assert path.stat().st_size == model["size_bytes"]


def test_manifest_hashes_match_files() -> None:
    manifest = load_manifest()

    for model in manifest["models"]:
        path = REPOSITORY_ROOT / model["path"]

        assert calculate_sha256(path) == model["sha256"]


def test_every_model_has_provenance() -> None:
    manifest = load_manifest()

    for model in manifest["models"]:
        assert model["source"]
        assert model["source"] != "unknown"
        assert model["description"]
        assert model["category"]
