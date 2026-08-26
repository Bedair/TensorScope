# Changelog

## [Unreleased]

Everything below has shipped on `main` since the v0.1.0 tag but has not yet
been cut into a numbered release.

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
