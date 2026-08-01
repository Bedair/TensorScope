# Validated TFLM operator coverage

TensorScope's checked-in corpus validates the **planned arena head only** against
TensorFlow Lite Micro revision
`b89fb3e06e59d2f6af67e758242243da599bfedf`. “Validated” means that the
TensorScope greedy plan and the pinned TFLM recording allocator produced the
same arena-head byte count for the particular model and tensor configuration.
It is not a universal operator-support claim and says nothing about scratch,
persistent arena tail, complete arena total, or device fit.

Validated on the checked-in corpus: ADD, MUL, RESHAPE, SOFTMAX, CONV_2D,
DEPTHWISE_CONV_2D, MAX_POOL_2D, AVERAGE_POOL_2D, FULLY_CONNECTED, RELU, RELU6,
LOGISTIC, QUANTIZE, and DEQUANTIZE. Configurations include float32, int8, int16,
int32, and int64 tensors; per-tensor int8 quantization; constants; and both
single-operator and linear multi-operator graphs.

## Deterministic validation matrix

| Model | Operators | Tensor types | TensorScope head | TFLM head | Delta | Arena-head validation |
|---|---|---|---:|---:|---:|---|
| `conv0.tflite` | CONV_2D | INT8, INT16, INT64 | 10432 | 10432 | 0 | exact_match |
| `hello_world_float.tflite` | FULLY_CONNECTED | FLOAT32 | 128 | 128 | 0 | exact_match |
| `hello_world_int8.tflite` | FULLY_CONNECTED | INT8, INT32 | 32 | 32 | 0 | exact_match |
| `micro_speech_quantized.tflite` | DEPTHWISE_CONV_2D, FULLY_CONNECTED, RESHAPE, SOFTMAX | INT8, INT32 | 5968 | 5968 | 0 | exact_match |
| `operator_chain_float.tflite` | ADD, MUL, RESHAPE, SOFTMAX, CONV_2D, DEPTHWISE_CONV_2D, MAX_POOL_2D, AVERAGE_POOL_2D, FULLY_CONNECTED, RELU, RELU6, LOGISTIC | FLOAT32, INT32 | 128 | 128 | 0 | exact_match |
| `quantize_dequantize_int8.tflite` | QUANTIZE, DEQUANTIZE | FLOAT32, INT8 | 48 | 48 | 0 | exact_match |
| `simple_add_model.tflite` | ADD | INT8 | 49152 | 49152 | 0 | exact_match |

The generated models are produced deterministically with:

```console
PYTHONPATH=src python3 scripts/generate_operator_coverage_models.py
```

To add another validation model, prefer a small model already in the pinned
TFLM tree. Otherwise extend that generator, document provenance, operator and
tensor types, quantization, and unique coverage in `manifest.json`; explicitly
register only the needed kernel in the oracle; record an exact zero-delta row;
then run `make -C tools/tflm_oracle`, `pytest -q`, and
`python3 scripts/validate_model_corpus.py`.

Known unsupported cases include dynamic or negative shapes, variable-width
STRING/RESOURCE/VARIANT storage, variable tensors in arena-head planning,
control-flow/cross-subgraph lifetime semantics, operator scratch and persistent
buffers, custom kernels, and builtins not explicitly registered by the oracle.
Synthetic tests cover fan-out, branch merge, skip/late consumption, multiple
consumers, graph outputs, constants, deterministic ordering, overlap, and reuse;
those graph patterns are not yet independently represented by an oracle corpus
model.
