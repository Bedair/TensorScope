# Oracle arena observations

TensorScope uses three deliberately separate terms:

- **Static calculation** computes the planned arena head from the model graph.
- **Oracle observation** records memory used by the pinned host-side TFLM
  allocator during `AllocateTensors()`.
- **Arena-head validation** compares only the static planned head with the
  oracle-observed head.

The `analyze` command remains static: `arena_tail` and `arena_total` are not
estimated. The `validate` command additionally reports an
`oracle_arena_observation` with capacity, observed used/head/tail, remaining
bytes, allocator alignment, and the pinned TFLM revision
`b89fb3e06e59d2f6af67e758242243da599bfedf`.

For this pinned allocator, direct APIs prove:

```text
observed used = observed non-persistent/head + observed persistent/tail
remaining = oracle arena capacity - observed used
```

Temporary-only allocation usage is reported as unavailable. The pinned API
combines current temporary usage with non-persistent usage and does not expose a
reliable separate value after allocation. Structured allocation categories are
also deferred: the pinned recording categories explicitly omit scratch tracking
and disable OpData tracking, so presenting them as a complete breakdown would be
misleading. Human-readable `RecordingMicroAllocator` diagnostics remain available.

Reproduce the observation:

```console
make -C tools/tflm_oracle
tools/tflm_oracle/build/tflm_oracle tests/model_corpus/models/hello_world_float.tflite
PYTHONPATH=src python3 -m tensorscope validate tests/model_corpus/models/hello_world_float.tflite
PYTHONPATH=src python3 -m tensorscope validate tests/model_corpus/models/hello_world_float.tflite --json
```

Arena tail and complete arena usage are oracle observations, not static
TensorScope estimates. They describe one pinned host allocator run; they do not
guarantee complete arena requirements on another TFLM revision or target, nor
complete MCU or firmware fit. Firmware, stack, heap, DMA, RTOS, application
memory, and target-specific behavior remain outside this observation.
