# TFLM Host Oracle

The TFLM host oracle runs the pinned TensorFlow Lite Micro allocator
against a `.tflite` model and reports the actual arena allocations.

The oracle is used to validate TensorScope's static memory analysis.

## Build

From the repository root:

```bash
make -C tools/tflm_oracle