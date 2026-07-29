#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TFLM_DIR="$ROOT_DIR/third_party/tflite-micro"
MAKEFILE="tensorflow/lite/micro/tools/make/Makefile"

if [[ ! -d "$TFLM_DIR" ]]; then
    echo "Error: TFLM submodule not found."
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

echo "TFLM commit:"
git -C "$TFLM_DIR" rev-parse HEAD

echo "Building and testing TFLM for x86-64 host..."

make \
    -C "$TFLM_DIR" \
    -f "$MAKEFILE" \
    -j"$(nproc)" \
    test

echo "TFLM host build and tests completed successfully."
