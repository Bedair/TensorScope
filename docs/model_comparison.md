# Comparing two models

TensorScope compares a baseline and candidate using static planned arena-head
analysis:

```console
PYTHONPATH=src python3 -m tensorscope compare baseline.tflite candidate.tflite
PYTHONPATH=src python3 -m tensorscope compare baseline.tflite candidate.tflite --json
PYTHONPATH=src python3 -m tensorscope compare baseline.tflite candidate.tflite --html comparison.html
```

`--arena-head-budget SIZE` or `--mcu-profile PROFILE [--reserve SIZE]` applies
the same effective arena-head budget to both models. `--details` shows every
tensor change. In CI, `--fail-on-regression` writes the requested output first
and then returns exit code 7 when the deterministic regression rule fires.

## Tensor matching

Runtime tensors are matched one-to-one without using IDs or allocation offsets:

1. Nonempty normalized name plus exact type and shape (`exact`).
2. Unique nonempty normalized name (`high`).
3. Unique type, shape, graph-input/output roles, producer operator name, and
   ordered consumer operator names (`medium`).
4. Otherwise the tensors remain unmatched and are reported as added/removed.

Ambiguous groups are never paired. Matching indicates structural evidence, not
semantic model equivalence.

## Metrics and status

The report compares planned head, peak occupied/live bytes, runtime and constant
tensor counts, operator count, logical/aligned runtime sums, alignment overhead,
safe reuse pairs, and reuse blockers. Each metric includes baseline, candidate,
signed delta, direction, and a percentage when the baseline is nonzero.

A candidate is a regression when its planned head grows by both at least 256
bytes and at least 5%, when a fitting/exact budget becomes exceeded, or when it
introduces a critical budget-pressure finding. A decrease of at least 256 bytes
and 5% without budget regression is `improved`. Identical comparisons are
`unchanged`; other meaningful tradeoffs are `mixed`.

Budget comparisons cover planned arena head only. They do not establish complete
MCU or firmware fit. Comparison does not prove model equivalence, compare static
arena tail or complete arena total, validate accuracy, or guarantee that an
observed tensor change caused the arena-head delta.
