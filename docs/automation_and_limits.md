# Automation commands, schemas, and audited limits

TensorScope is a standalone CLI; its Python modules are internal implementation
details and are not a supported library API.

## Commands

- `analyze MODEL [--json | --html PATH | --sarif PATH]`
- `validate MODEL [--json]`
- `compare BASELINE CANDIDATE [--json | --html PATH] [--fail-on-regression]`
- `check MODEL --policy POLICY [--json PATH] [--sarif PATH]`
- `baseline create MODEL --output PATH`
- `baseline check MODEL --baseline PATH [--json PATH]`
- `batch PATH... --output-dir DIR [--recursive] [--fail-fast] [--sarif]`
- `firmware-check MODEL --map-file MAP --arena-symbol NAME [--arena-size N]`
- `deploy-report MODEL --output-dir DIR [--margin-percent N]`
- `list-profiles`

Stable schema versions are 1 for policy, policy results, baselines, baseline
checks, batch aggregates, firmware checks, deployment manifests, and model
comparison. SARIF uses 2.1.0.

Exit codes: 0 success; 2 invalid input; 3 oracle unavailable; 4 oracle head
mismatch; 5 report write failure; 6 analyze budget exceeded when requested; 7
comparison regression; 8 policy failure; 9 baseline drift; 10 batch errors; 11
incomplete/failed firmware check. Failure outputs are written before these
dedicated nonzero returns.

Policies reject unknown keys. Supported rules cover operator allow/deny lists,
tensor-count and planned-head maxima, maximum risk/high findings, forbidden
guidance categories, growth thresholds, and deterministic comparison regression.
Baseline manifests omit timestamps and record SHA-256, assumptions, static
metrics, reuse, guidance, operators, and an optional budget.

Batch inputs are de-duplicated and sorted by resolved path. Each successful
model receives JSON and HTML; aggregates are stable JSON and CSV, with optional
SARIF. Errors continue by default.

## Audited limitations and deferrals

- **Scratch:** inspected `recording_micro_allocator.h` and its implementation.
  The pinned API explicitly leaves scratch tracking as a TODO and exposes no
  robust request total, peak, or operator attribution. Scratch is unavailable;
  allocator prose is not parsed. A future pinned API must expose typed records.
- **Multi-subgraph/control flow:** the loader enumerates subgraphs, but planning,
  guidance, and reports are primary-subgraph scoped. IF/WHILE/CALL_ONCE execution
  relationships are not modeled, so peaks are never combined. Future work needs
  validated control-flow lifetime semantics and corpus fixtures.
- **Dynamic/variable tensors:** IR retains `is_variable`, concrete shape, and
  shape signature. Exact planning currently fails through existing tensor-size
  diagnostics when concrete dimensions are unavailable. Persistent mutable-state
  allocation is not statically estimated.
- **Additional operators:** no new operator is claimed validated in this phase.
  Reliable coverage requires deterministic generated models plus exact pinned
  oracle matches; parsing alone is insufficient.
- **Verified targets:** generic presets remain separate. No authoritative target
  facts were bundled. A verified-profile catalog is deferred until provenance
  (target identity, source title/revision/URL) can be supplied and reviewed.
- **GNU ld maps:** firmware-check supports a documented minimal MEMORY-region and
  symbol-line subset. Arena size requires `--arena-size`; without it status is
  `incomplete`. Stack/heap inputs are reported assumptions, not proof of fit.
- **Operator attribution/timeline/graph:** analysis JSON now exposes represented
  input/output bytes, live bytes, occupied extent, retained outputs, blockers,
  pressure, and peak flags per operator. HTML renders deterministic accessible
  timeline and graph SVGs with a text fallback and an 80-node limit. These are
  live-set views and are explicitly non-additive; scratch stays unavailable.
- **Quantization comparison/matching extensions:** comparison now reports matched
  element-type and quantization metadata changes, logical/aligned deltas, model
  file-size change, and required accuracy/calibration warnings. Matching includes
  quantization metadata and producer/ordered-consumer neighborhoods. Constant
  content hashes and explicit ambiguity-group serialization remain deferred;
  adding them safely requires retained constant bytes and a schema extension.
- **Performance:** `scripts/benchmark_tensorscope.py` provides an opt-in end-to-end
  phase benchmark without fragile test thresholds. Per-phase regression gates are
  deferred until representative large models and stable CI hardware are available.

All outputs concern planned arena head unless explicitly labeled host oracle or
firmware-map observations. They do not establish semantic equivalence, target
behavior, model accuracy, calibration correctness, or complete MCU/firmware fit.
