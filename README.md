# TensorScope

TensorScope is a standalone command-line tool for deterministic TensorFlow Lite
Micro planned arena-head analysis. It reports tensor lifetimes and reuse,
evidence-based guidance, model comparisons, CI policies, baselines, batch
reports, SARIF, GNU ld map checks, and deployment planning artifacts.

```console
python -m pip install .
tensorscope analyze model.tflite
tensorscope compare baseline.tflite candidate.tflite --fail-on-regression
tensorscope check model.tflite --policy tensorscope.yaml
```

`validate` requires the separately built pinned oracle. From a source checkout,
run `make -C tools/tflm_oracle`. An installed CLI can use the same binary by
setting `TENSORSCOPE_TFLM_ORACLE=/absolute/path/to/tflm_oracle`.

See `tensorscope --help` and the documents under `docs/` for complete workflows.
TensorScope calculates planned arena head only. Static arena tail and complete
arena total remain unavailable. Oracle values are pinned host observations, MCU
profiles are planning presets, and no report establishes complete firmware fit.

Multi-subgraph peaks are not combined without proven execution semantics.
Scratch metrics are unavailable through the pinned TFLM recording API. No
verified target profiles are bundled without authoritative target provenance.

TensorScope does not provide a supported public Python library API.
