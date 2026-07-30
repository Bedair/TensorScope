#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPOSITORY_ROOT / "tests" / "model_corpus"
MODELS_ROOT = CORPUS_ROOT / "models"
MANIFEST_PATH = CORPUS_ROOT / "manifest.json"


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if data.get("schema_version") != 1:
        raise ValueError(
            "Unsupported corpus manifest schema version."
        )

    models = data.get("models")

    if not isinstance(models, list):
        raise ValueError(
            "Manifest field 'models' must be a list."
        )

    return data


def validate_model(entry: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    name = entry.get("name", "<unknown>")
    relative_path = entry.get("path")

    if not isinstance(relative_path, str):
        return [f"{name}: missing or invalid path"]

    path = REPOSITORY_ROOT / relative_path

    if not path.is_file():
        return [
            f"{name}: file does not exist: {relative_path}"
        ]

    if path.suffix.lower() != ".tflite":
        errors.append(
            f"{name}: expected a .tflite file"
        )

    expected_size = entry.get("size_bytes")
    actual_size = path.stat().st_size

    if actual_size != expected_size:
        errors.append(
            f"{name}: size mismatch; "
            f"expected {expected_size}, got {actual_size}"
        )

    expected_hash = entry.get("sha256")
    actual_hash = calculate_sha256(path)

    if actual_hash != expected_hash:
        errors.append(
            f"{name}: SHA-256 mismatch; "
            f"expected {expected_hash}, got {actual_hash}"
        )

    source = entry.get("source")

    if not isinstance(source, str) or not source:
        errors.append(
            f"{name}: missing source information"
        )

    category = entry.get("category")

    if not isinstance(category, str) or not category:
        errors.append(
            f"{name}: missing category"
        )

    return errors


def validate_no_untracked_models(
    manifest: dict[str, Any],
) -> list[str]:
    manifest_names = {
        entry["name"]
        for entry in manifest["models"]
        if isinstance(entry.get("name"), str)
    }

    filesystem_names = {
        path.name
        for path in MODELS_ROOT.glob("*.tflite")
    }

    errors: list[str] = []

    missing_from_manifest = filesystem_names - manifest_names

    for name in sorted(missing_from_manifest):
        errors.append(
            f"{name}: present in corpus but missing from manifest"
        )

    missing_from_filesystem = manifest_names - filesystem_names

    for name in sorted(missing_from_filesystem):
        errors.append(
            f"{name}: present in manifest but missing from corpus"
        )

    return errors


def main() -> int:
    manifest = load_manifest()
    errors: list[str] = []

    declared_count = manifest.get("model_count")
    actual_count = len(manifest["models"])

    if declared_count != actual_count:
        errors.append(
            "Manifest model_count mismatch; "
            f"declared {declared_count}, actual {actual_count}"
        )

    for entry in manifest["models"]:
        errors.extend(validate_model(entry))

    errors.extend(validate_no_untracked_models(manifest))

    if errors:
        print("Model corpus validation failed:")

        for error in errors:
            print(f"- {error}")

        return 1

    print(
        "Model corpus validation passed: "
        f"{actual_count} models"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())