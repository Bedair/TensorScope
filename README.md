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

See `tensorscope --help` and the documents under `docs/` for complete workflows.
TensorScope calculates planned arena head only. Static arena tail and complete
arena total remain unavailable. Oracle values are pinned host observations, MCU
profiles are planning presets, and no report establishes complete firmware fit.

TensorScope does not provide a supported public Python library API.
