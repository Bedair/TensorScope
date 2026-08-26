# Validated TFLM operator coverage

TensorScope's checked-in corpus validates the **planned arena head only** against
TensorFlow Lite Micro revision
`b89fb3e06e59d2f6af67e758242243da599bfedf`. “Validated” means that the
TensorScope greedy plan and the pinned TFLM recording allocator produced the
same arena-head byte count for the particular model and tensor configuration.
It is not a universal operator-support claim and says nothing about scratch,
persistent arena tail, complete arena total, or device fit.

Validated on the checked-in corpus: ADD, MUL, SUB, RESHAPE, SOFTMAX, CONV_2D,
DEPTHWISE_CONV_2D, MAX_POOL_2D, AVERAGE_POOL_2D, FULLY_CONNECTED, RELU, RELU6,
LOGISTIC, LEAKY_RELU, QUANTIZE, DEQUANTIZE, PAD, and STRIDED_SLICE.
Configurations include float32, int8, int16, int32, and int64 tensors;
per-tensor int8 quantization; constants; single-operator and linear
multi-operator graphs; and a skip-connection graph (a tensor consumed by two
non-adjacent operators, blocking reuse across the branch).

TRANSPOSE_CONV is registered in the oracle's resolver (TFLM genuinely
implements it) but is deliberately **not** claimed validated: the smallest
vendored fixture (`seanet/transpose_conv/transpose_conv4.tflite`) mismatches
the oracle by 41,472 bytes, consistent with TFLM's TransposeConv kernel
requesting a scratch buffer that TensorScope's static planner does not model
-- see "operator scratch and persistent buffers" below. No fixture for this
operator is included in the corpus.

## Deterministic validation matrix

| Model | Operators | Tensor types | TensorScope head | TFLM head | Delta | Arena-head validation |
|---|---|---|---:|---:|---:|---|
| `conv0.tflite` | CONV_2D | INT8, INT16, INT64 | 10432 | 10432 | 0 | exact_match |
| `hello_world_float.tflite` | FULLY_CONNECTED | FLOAT32 | 128 | 128 | 0 | exact_match |
| `hello_world_int8.tflite` | FULLY_CONNECTED | INT8, INT32 | 32 | 32 | 0 | exact_match |
| `leaky_relu22.tflite` | LEAKY_RELU | INT16 | 10240 | 10240 | 0 | exact_match |
| `micro_speech_quantized.tflite` | DEPTHWISE_CONV_2D, FULLY_CONNECTED, RESHAPE, SOFTMAX | INT8, INT32 | 5968 | 5968 | 0 | exact_match |
| `operator_chain_float.tflite` | ADD, MUL, RESHAPE, SOFTMAX, CONV_2D, DEPTHWISE_CONV_2D, MAX_POOL_2D, AVERAGE_POOL_2D, FULLY_CONNECTED, RELU, RELU6, LOGISTIC | FLOAT32, INT32 | 128 | 128 | 0 | exact_match |
| `pad0.tflite` | PAD | INT16 | 2608 | 2608 | 0 | exact_match |
| `quantize_dequantize_int8.tflite` | QUANTIZE, DEQUANTIZE | FLOAT32, INT8 | 48 | 48 | 0 | exact_match |
| `residual_add_float.tflite` | CONV_2D, ADD | FLOAT32 | 192 | 192 | 0 | exact_match |
| `simple_add_model.tflite` | ADD | INT8 | 49152 | 49152 | 0 | exact_match |
| `strided_slice0.tflite` | STRIDED_SLICE | INT16 | 9296 | 9296 | 0 | exact_match |
| `sub0.tflite` | SUB | INT16 | 6272 | 6272 | 0 | exact_match |

`residual_add_float.tflite` is the corpus's skip-connection fixture: `input`
feeds the first `CONV_2D` and is read again, unmodified, by the final `ADD`
after a second `CONV_2D` runs in between that never touches it. `input`
therefore stays live from scope 0 through scope 3 and blocks reuse for the
whole graph -- the exact pattern `explain.py`'s `ReuseBlocker` documents.
`tensorscope analyze --details` on this model shows one safe reuse pair
(the output reuses the first conv's freed slot) and four reuse blockers,
including the skip-connected input.

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
buffers (see TRANSPOSE_CONV above for a concrete, measured example), custom
kernels, and builtins not explicitly registered by the oracle.
Synthetic tests cover fan-out, branch merge, multiple consumers, graph
outputs, constants, deterministic ordering, overlap, and reuse. Skip
connections (a tensor consumed by two non-adjacent operators) are now also
represented by an oracle corpus model, `residual_add_float.tflite`; other
graph patterns in that list are not yet independently represented by one.

## When `validate` cannot reach the oracle

`tensorscope validate` fails with exit code 3 (`validation_unavailable`) for
several unrelated reasons -- a missing oracle executable, a malformed model,
or the oracle process itself failing. When the failure is caused by the model
using an operator the oracle cannot run, the JSON/text error output carries
one of two more specific `error_type` values instead, distinguished by
`tensorscope.oracle.classify_oracle_incompatibility` from the oracle's exit
code and TFLM's own diagnostic text:

- **`unregistered_operator`** -- the oracle's `MicroMutableOpResolver` in
  `tools/tflm_oracle/main.cc` has not registered an operator this model uses.
  This is a coverage gap in the oracle build, not proof TFLM cannot run the
  model: TFLM may already ship a kernel for it. Two distinct oracle exit
  paths land here: the oracle's own pre-flight allowlist rejecting the
  builtin opcode before the interpreter runs at all (`ValidateRegisteredOperators`,
  exit code 6, "unsupported operator at index ..."), or TFLM's runtime
  resolver failing to find a kernel for an opcode the allowlist did
  recognize ("Didn't find op for builtin opcode ..." -- the exact bug fixed
  for `MAX_POOL_2D`/`QUANTIZE` above). **Next step:** check whether
  `third_party/tflite-micro/tensorflow/lite/micro/kernels/` actually
  implements the operator, and if so, register it in `main.cc` and file a
  coverage gap.

  Real example already in this repository: `UNIDIRECTIONAL_SEQUENCE_LSTM`
  genuinely has a TFLM kernel (`kernels/unidirectional_sequence_lstm.cc`) but
  is not in the oracle's allowlist, so running the oracle against the
  vendored fixture
  `third_party/tflite-micro/tensorflow/lite/micro/examples/mnist_lstm/trained_lstm_int8.tflite`
  reliably reproduces this category (exit code 6, `builtin opcode 44`).
  PAD, STRIDED_SLICE, SUB, and LEAKY_RELU used to serve as this example; a
  later corpus expansion registered all four (see the validation matrix
  above), so they no longer demonstrate a coverage gap.

- **`structurally_unsupported`** -- TFLM found and started preparing (or
  invoking) a kernel for the operator, but that kernel refused this model's
  specific configuration. TFLM's engine reports this generically as a node
  "failed to prepare with status ..." or "failed to invoke with status ...".
  The clearest known instance is hybrid int8-weight/float32-activation
  quantization, which `conv_common.cc` rejects outright ("Hybrid models are
  not supported on TFLite Micro"). Registering the operator differently in
  the oracle will not help here: the model cannot run on stock TFLM
  regardless of registration. **Next step:** check whether the model was
  exported for a different runtime or toolchain (for example a vendor NPU
  compiler that accepts hybrid quantization) rather than filing a coverage
  gap.

  No bundled fixture currently exercises this path: TFLM's own examples and
  this project's synthetic generator deliberately avoid quantization schemes
  TFLM rejects, and building one requires a real TensorFlow conversion
  pipeline this repository does not carry. The classification is instead
  covered by unit tests against TFLM's real, previously observed diagnostic
  text (see `tests/oracle/test_tflm_oracle.py`). If a cheap real fixture
  becomes available, add it there and update this note.

A failure that matches neither signature (for example the tensor arena being
too small) keeps the generic `validation_unavailable` category -- that is a
real failure, just not one of these two operator-support cases.
