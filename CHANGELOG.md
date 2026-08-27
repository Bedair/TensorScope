# Changelog

## [Unreleased]

Nothing pending yet.

## [0.4.0] - 2026-08-27

Compute-cost (MAC/FLOP) analysis: a new, honestly-scoped static metric,
additive to everything 0.3.0 shipped.

### Added

- New `compute_cost.py`: per-operator arithmetic-volume classification for
  every operator this project supports, verified against real corpus
  models and the vendored TFLM kernel source (not assumed):
  - **MAC-bearing** (CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED,
    TRANSPOSE_CONV): a real multiply-accumulate count, derived from
    weight-tensor shape.
  - **Zero-compute** (RESHAPE, STRIDED_SLICE, PAD): reported as a genuine
    0, not omitted -- "data movement only, no arithmetic."
  - **Elementwise** (ADD, SUB, MUL, RELU, RELU6, LEAKY_RELU, LOGISTIC,
    SOFTMAX, QUANTIZE, DEQUANTIZE): real per-element work, reported as a
    separate output-element count -- never summed into the MAC total,
    since the two aren't commensurate units of cost.
  - **Unavailable** (MAX_POOL_2D, AVERAGE_POOL_2D): honestly reported as
    unavailable rather than guessed -- their op count depends on kernel
    size, which lives only in `Pool2DOptions` (`builtin_options`), which
    this project does not parse for any operator today.
- `analyze`'s compact default and `--details` both gain a `Compute`
  section (MAC total in compact; a full per-operator breakdown in
  `--details`); a new JSON `compute_cost` key; and a new guidance finding
  ("Compute is concentrated in N operator(s)..."), gated to never fire
  when total MACs are 0.
- Every appearance of a MAC/elementwise figure is paired with an inline
  caveat -- *"Compute cost (MACs) — not a latency or timing estimate."* —
  from one shared function (`render_compute_cost_caveat()`), the same
  single-source-of-truth pattern already used for the budget verdict,
  adopted here from the start rather than after wording drifted across
  render surfaces the way the budget label did twice before.

### Not built (investigated and intentionally out of scope)

- Wall-clock latency estimation: confirmed out of statically-answerable
  scope, and harder than arena tail -- cycle counts depend on target core,
  clock frequency, memory wait-states, and reference-vs-optimized kernel
  selection, none of which exist in a `.tflite` file or are portable from
  a host-architecture oracle run.
- `Pool2DOptions` parsing for a real pooling op count: legitimate, well-
  defined follow-up work, deliberately not included here.

Same standard as prior releases: no formula shipped without verification
against a real corpus model or (for TRANSPOSE_CONV, which has no corpus
fixture) the real vendored fixture directly.

## [0.3.0] - 2026-08-27

Stage 1 of the compact-view feature: flash capacity for real `--target`
profiles, and a redesigned default `analyze` text output.

Built, merged to `main`, and tested, but never tagged as a release or
uploaded to PyPI before 0.4.0 shipped on top of it — noted here so anyone
comparing this changelog against PyPI's published release history
(`0.2.0`, `0.2.1`, `0.4.0`) understands the gap instead of wondering if
something's missing or broken. No republish of 0.3.0 is possible or
needed; its changes are part of 0.4.0.

### Added

- `total_flash_bytes` for three of the four real per-vendor `--target`
  profiles, sourced from the same primary datasheets already cited for
  SRAM: STM32U585 (2,097,152 bytes), nRF52840 (1,048,576 bytes), and
  CY8C624ABZI-S2D44 (2,097,152 bytes). ESP32-S3's is honestly `null`: the
  bare die has no embedded flash (external SPI only, chosen per
  module/board), and its dev-kit alias itself ships in multiple
  flash-size SKUs (8 MB or 32 MB) with no single default — picking one
  would have meant guessing.
- `analyze`'s compact default now shows a `Flash (model)` line for a real
  `--target` run: the model's constant-tensor byte total (weights/constants
  that end up in flash) against the target's flash capacity, or an honest
  "unavailable" note when that capacity isn't a single well-defined figure
  (ESP32-S3 today). Not shown for `--mcu-profile` (generic classes have no
  flash concept) or when no target/profile was given at all.
- `render_budget_source_label()`'s status-word counterpart
  (`BUDGET_STATUS_LABELS`) is now a single shared public mapping in
  `memory_budget.py`, used by the HTML renderer and this new compact text
  view alike, instead of letting a third copy of the same 3-entry mapping
  accumulate.

### Fixed

- nRF52840's SRAM citation's `revision`/`page` were `null` (not visible on
  the originally-fetched HTML page); now `v1.11`/page 21, sourced from the
  same sentence in the official PDF edition that supplied the new flash
  figure.

### Changed

- **Breaking change to `analyze`'s default text output shape.** The
  default is now a compact summary: model and target/profile identity, a
  `Memory` section (flash usage when a real `--target` was given, the RAM
  arena-head budget verdict), and an arena-tail-unavailable note. The full
  breakdown this used to show unconditionally — tensor tables, packing
  detail, full memory guidance, operator-level pressure — now requires
  `--details`, reusing the same flag that already truncated guidance
  findings rather than adding a second one. `--json` output is entirely
  unaffected by this change. Anyone scripting against the old default text
  output needs to add `--details`, or switch to `--json` if they aren't
  already.

## [0.2.1] - 2026-08-27

### Fixed

- `--target` budget checks in text-mode CLI output (e.g.
  `tensorscope analyze model.tflite --target esp32-s3`, no `--html`)
  previously showed `Budget source: Generic MCU planning profile` even when
  a real, cited MCU/dev-kit target was used — the verdict line right below
  it was correct (it named the target and cited its datasheet), but the
  source label directly contradicted it. This was the same mislabeling
  bug fixed for the HTML report in 0.2.0, present independently in the
  plain-text renderer because the two renderers kept separate copies of
  the same labeling logic.
- Fixed at the root: text, HTML, and JSON output now share one budget
  source labeling function (`render_budget_source_label()` in
  `memory_budget.py`) instead of each render path keeping its own copy,
  so this class of drift can't recur between them. A regression test now
  checks all three output modes agree, for every kind of budget check
  (`--target`, `--mcu-profile`, `--arena-head-budget`).

## [0.2.0] - 2026-08-26

Everything below shipped on `main` since the v0.1.0 tag.

### Added

- Oracle-side operator-coverage registrations for `MAX_POOL_2D`, `QUANTIZE`,
  `PAD`, `STRIDED_SLICE`, `SUB`, and `LEAKY_RELU`, closing validation gaps
  where TFLM itself supported an operator but the oracle's resolver hadn't
  registered it.
- Classification of oracle validation failures as `unregistered_operator`
  (a coverage gap in the oracle's resolver, not proof TFLM can't run the
  model) versus `structurally_unsupported` (TFLM itself refuses the model's
  configuration, e.g. hybrid quantization) — surfaced in `validate`'s JSON
  and text error output with a one-line next-step explanation for each.
- Safe reuse hand-off markers and reuse-blocker badges on the HTML report's
  arena-placement chart, sharing one explanation with the prose
  reuse-blocker list so the chart and the text can't drift apart.
- Model corpus expanded from 7 to 12 models: four real operator-coverage
  fixtures vendored from `third_party/tflite-micro` (PAD, STRIDED_SLICE,
  SUB, LEAKY_RELU) and one hand-built fixture (`residual_add_float`)
  exercising the documented skip-connection reuse-blocking pattern.
- Real, data-driven per-vendor MCU/dev-kit target profiles (`--target`,
  `list-targets`), each backed by a cited primary-source datasheet:
  STM32U585 (STMicroelectronics), nRF52840 (Nordic, including the Arduino
  Nano 33 BLE family), ESP32-S3 (Espressif), and CY8C624ABZI-S2D44 /
  CY8CKIT-062S2-AI (Infineon PSoC 6) — alongside the existing generic
  `--mcu-profile` size-class presets.
- Arena-head budget verdicts (FITS / EXACT FIT / EXCEEDS BUDGET) now state
  their head-only scope, and for `--target` their datasheet citation,
  inline in the verdict text itself — in text, JSON, and HTML output alike.
- `COMMERCIAL.md` documenting that non-AGPL commercial licensing is
  available on request.
- A shared `oracle_is_runnable()` check and clearer error text for the case
  where the committed (Linux-only) oracle binary exists but cannot execute
  on the current platform.

### Changed

- Relicensed from Apache-2.0 to AGPL-3.0-or-later.
- `list-profiles` and the budget verdict now point at how to actually use
  them (`--mcu-profile`/`--target`/`--arena-head-budget`) instead of just
  enumerating values with no path to acting on them.
- Redesigned the self-contained HTML report: a two-tier verdict banner
  (bold headline plus a citation/caveat line) and a plain-language purpose
  statement now sit above the fold; the top disclaimer leads with what's
  proven (exact, deterministic static analysis) before what's out of
  scope; the three overlapping tensor tables (peak / largest / all)
  collapsed into one table with visible PEAK/LARGEST badges; secondary
  views (operator pressure, execution timeline, graph, compact offset
  view, execution scopes, safe-reuse detail) moved behind native
  `<details>` disclosures; the limitations list trimmed from 12
  near-duplicate bullets to 7.
- README rewritten to match the current CLI, the real target catalog, the
  confidence/scope model, and verified install/platform behavior, in place
  of examples and caveats that predated all of the above.
- First-run error messages for malformed `.tflite` files (wrong file
  identifier, unparseable FlatBuffer, no subgraphs, too-small file) now
  state what's wrong in plain terms first and suggest a next step, instead
  of leading with a raw byte-string comparison or no guidance at all.

### Fixed

- Budget verdict's "Budget source" field was mislabeled "Generic MCU
  planning profile" even when `--target` supplied a real, cited datasheet.
- A long model path in the HTML report's metadata table could visually
  overlap the adjacent field at wide viewports.
- Oracle-dependent tests used `Path.is_file()` to decide whether to skip on
  an unsupported platform, which is `True` on native Windows for the
  committed Linux oracle binary even though it cannot execute there —
  around two dozen tests failed on Windows instead of skipping cleanly.
  Tests now probe actual runnability; `validate`'s own error message
  similarly now explains a Linux-only oracle binary being present but
  unusable, instead of surfacing a raw platform `OSError`.
- The pinned oracle binary in this repository predated an operator-table
  expansion in its own source and needed rebuilding to match.

## 0.1.0

- Initial standalone CLI for exact static planned arena-head analysis on the
  validated operator corpus.
- Pinned host-side TFLM oracle validation and observed allocator breakdowns.
- Explainable text, deterministic JSON, ASCII packing, self-contained HTML,
  memory guidance, model comparison, MCU planning budgets, and regression checks.
- CI policies, deterministic baselines, batch reports, and SARIF 2.1.0 output.
- Limited GNU ld map checks and deterministic deployment-planning artifacts.
- Static arena tail and complete total are not estimated. Scratch metrics are
  unavailable in the pinned recording API, multi-subgraph peaks are not combined,
  and no report establishes complete MCU or firmware memory fit.
