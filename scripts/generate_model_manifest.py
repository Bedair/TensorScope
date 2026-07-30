#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPOSITORY_ROOT / "tests" / "model_corpus"
MODELS_ROOT = CORPUS_ROOT / "models"
OUTPUT_PATH = CORPUS_ROOT / "manifest.json"


MODEL_METADATA: dict[str, dict[str, Any]] = {
    "hello_world_float.tflite": {
        "description": (
            "Floating-point Hello World sine approximation model."
        ),
        "source": (
            "third_party/tflite-micro/tensorflow/lite/micro/"
            "examples/hello_world/models/hello_world_float.tflite"
        ),
        "category": "baseline",
        "quantization": "float32",
        "expected_operators": [
            "FULLY_CONNECTED",
        ],
        "expected_use": [
            "model parsing",
            "float tensor handling",
            "tensor-size calculation",
            "lifetime analysis",
            "memory-planner validation",
        ],
    },
    "hello_world_int8.tflite": {
        "description": (
            "Integer-quantized Hello World sine approximation model."
        ),
        "source": (
            "third_party/tflite-micro/tensorflow/lite/micro/"
            "examples/hello_world/models/hello_world_int8.tflite"
        ),
        "category": "baseline",
        "quantization": "int8",
        "expected_operators": [
            "FULLY_CONNECTED",
        ],
        "expected_use": [
            "quantization metadata parsing",
            "int8 tensor handling",
            "tensor-size calculation",
            "memory-planner validation",
        ],
    },
    "simple_add_model.tflite": {
        "description": (
            "Minimal addition model from the TFLM memory footprint example."
        ),
        "source": (
            "third_party/tflite-micro/tensorflow/lite/micro/"
            "examples/memory_footprint/models/simple_add_model.tflite"
        ),
        "category": "single-operator",
        "quantization": "unknown",
        "expected_operators": [
            "ADD",
        ],
        "expected_use": [
            "minimal parser validation",
            "single-operator graph validation",
            "tensor lifetime validation",
        ],
    },
    "conv0.tflite": {
        "description": (
            "Small convolution integration-test model from TFLM."
        ),
        "source": (
            "third_party/tflite-micro/tensorflow/lite/micro/"
            "integration_tests/seanet/conv/conv0.tflite"
        ),
        "category": "single-operator",
        "quantization": "unknown",
        "expected_operators": [
            "CONV_2D",
        ],
        "expected_use": [
            "Conv2D parser validation",
            "activation tensor sizing",
            "scratch-buffer analysis",
            "memory-planner validation",
        ],
    },
    "micro_speech_quantized.tflite": {
        "description": (
            "Quantized keyword-spotting model from the TFLM "
            "micro_speech example."
        ),
        "source": (
            "third_party/tflite-micro/tensorflow/lite/micro/"
            "examples/micro_speech/models/micro_speech_quantized.tflite"
        ),
        "category": "representative-application",
        "quantization": "int8",
        "expected_operators": [],
        "expected_use": [
            "multi-operator graph parsing",
            "realistic tensor lifetime analysis",
            "memory-planner validation",
            "operator coverage testing",
        ],
    },
}


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def build_manifest_entry(path: Path) -> dict[str, Any]:
    metadata = MODEL_METADATA.get(
        path.name,
        {
            "description": "Model awaiting manual classification.",
            "source": "unknown",
            "category": "unclassified",
            "quantization": "unknown",
            "expected_operators": [],
            "expected_use": [],
        },
    )

    return {
        "name": path.name,
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": calculate_sha256(path),
        **metadata,
    }


def main() -> int:
    if not MODELS_ROOT.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {MODELS_ROOT}"
        )

    model_paths = sorted(MODELS_ROOT.glob("*.tflite"))

    if not model_paths:
        raise RuntimeError(
            f"No .tflite models found under {MODELS_ROOT}"
        )

    unknown_models = [
        path.name
        for path in model_paths
        if path.name not in MODEL_METADATA
    ]

    if unknown_models:
        unknown_list = ", ".join(unknown_models)
        raise RuntimeError(
            "Metadata is missing for these models: "
            f"{unknown_list}"
        )

    manifest = {
        "schema_version": 1,
        "description": "TensorScope MVP model validation corpus.",
        "model_count": len(model_paths),
        "models": [
            build_manifest_entry(path)
            for path in model_paths
        ],
    }

    OUTPUT_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Wrote {len(model_paths)} model entries "
        f"to {OUTPUT_PATH}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())