#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

TFLM_ROOT = (
    REPOSITORY_ROOT
    / "third_party"
    / "tflite-micro"
)

SOURCE_BINDING = (
    TFLM_ROOT
    / "tensorflow"
    / "lite"
    / "python"
    / "schema_py_generated.py"
)

OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "src"
    / "tensorscope"
    / "tflite"
    / "schema"
)

OUTPUT_BINDING = (
    OUTPUT_DIRECTORY
    / "schema_generated.py"
)

OUTPUT_INIT = (
    OUTPUT_DIRECTORY
    / "__init__.py"
)

OUTPUT_README = (
    OUTPUT_DIRECTORY
    / "README.md"
)

OUTPUT_METADATA = (
    OUTPUT_DIRECTORY
    / "SOURCE.txt"
)


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def validate_source() -> None:
    if not SOURCE_BINDING.is_file():
        raise FileNotFoundError(
            "Pinned TFLM Python schema binding was not found:\n"
            f"  {SOURCE_BINDING}"
        )

    source_text = SOURCE_BINDING.read_text(
        encoding="utf-8",
        errors="replace",
    )

    required_symbols = [
        "class Model",
        "class SubGraph",
        "class Tensor",
        "class Operator",
        "class OperatorCode",
        "class Buffer",
    ]

    missing_symbols = [
        symbol
        for symbol in required_symbols
        if symbol not in source_text
    ]

    if missing_symbols:
        formatted = "\n".join(
            f"- {symbol}"
            for symbol in missing_symbols
        )

        raise RuntimeError(
            "Pinned schema binding is missing required symbols:\n"
            f"{formatted}"
        )


def create_output_directory() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def copy_binding() -> None:
    shutil.copyfile(
        SOURCE_BINDING,
        OUTPUT_BINDING,
    )


def write_init_file() -> None:
    OUTPUT_INIT.write_text(
        '"""Vendored TFLite FlatBuffer schema binding."""\n\n'
        "from .schema_generated import *  # noqa: F401,F403\n",
        encoding="utf-8",
    )


def write_readme() -> None:
    OUTPUT_README.write_text(
        "# TFLite Schema Binding\n\n"
        "This directory contains the generated Python FlatBuffer "
        "binding copied from the pinned TensorFlow Lite Micro "
        "submodule.\n\n"
        "The generated file is not edited manually.\n\n"
        "Synchronize it with:\n\n"
        "```bash\n"
        "python3 scripts/generate_tflite_bindings.py\n"
        "```\n\n"
        "The source file is:\n\n"
        "```text\n"
        "third_party/tflite-micro/tensorflow/lite/python/"
        "schema_py_generated.py\n"
        "```\n",
        encoding="utf-8",
    )


def write_metadata() -> None:
    source_hash = calculate_sha256(
        SOURCE_BINDING
    )

    relative_source = SOURCE_BINDING.relative_to(
        REPOSITORY_ROOT
    )

    OUTPUT_METADATA.write_text(
        f"source={relative_source}\n"
        f"sha256={source_hash}\n",
        encoding="utf-8",
    )


def verify_copy() -> None:
    source_hash = calculate_sha256(
        SOURCE_BINDING
    )

    output_hash = calculate_sha256(
        OUTPUT_BINDING
    )

    if source_hash != output_hash:
        raise RuntimeError(
            "Copied schema binding does not match "
            "the pinned TFLM source."
        )


def main() -> int:
    try:
        validate_source()
        create_output_directory()
        copy_binding()
        write_init_file()
        write_readme()
        write_metadata()
        verify_copy()

        print("TFLite schema binding synchronized successfully.")
        print(f"Source: {SOURCE_BINDING}")
        print(f"Output: {OUTPUT_BINDING}")
        print(
            "SHA-256: "
            f"{calculate_sha256(OUTPUT_BINDING)}"
        )

        return 0

    except (
        FileNotFoundError,
        RuntimeError,
        OSError,
    ) as error:
        print(
            f"Error: {error}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())