# TensorScope

TensorScope is a standalone command-line tool for deterministic, confidence-aware
memory analysis of TensorFlow Lite Micro models. It statically computes the
**planned arena head** (the exact, deterministic byte count TFLM's own greedy
allocator would plan for a model's runtime tensors), explains it tensor by
tensor, checks it against real MCU RAM budgets, and — when you build the
included host-side oracle — validates it against a pinned TFLM interpreter.

## Install

```console
python -m pip install .
```

No C++ toolchain is required for this step. `analyze`, `compare`, `check`,
`baseline`, `batch`, budget/target checks, and all report output (text, JSON,
HTML, SARIF) are pure Python and have zero dependency on the compiled oracle
binary.

Only `validate` needs the separately built, pinned TFLM oracle — see
[Validate and the TFLM oracle](#validate-and-the-tflm-oracle) below.

## Quick start

```console
tensorscope analyze model.tflite
tensorscope analyze model.tflite --target esp32-s3
tensorscope analyze model.tflite --mcu-profile cortex-m4-256k --html report.html
tensorscope compare baseline.tflite candidate.tflite --fail-on-regression
tensorscope check model.tflite --policy tensorscope.yaml
tensorscope validate model.tflite   # requires the built oracle, see below
```

Run `tensorscope --help` for the full subcommand list (`analyze`, `validate`,
`compare`, `check`, `baseline`, `batch`, `firmware-check`, `deploy-report`,
`list-profiles`, `list-targets`), or `tensorscope <subcommand> --help` for a
subcommand's flags.

## What "arena head" means, and what TensorScope does not claim

TensorScope statically and exactly computes the **planned arena head**: the
memory TFLM's greedy planner would allocate for runtime tensors, derived
directly from tensor lifetimes and sizes in the model file. This number is
not an estimate — it's deterministic, and (when you run `validate`) checked
byte-for-byte against a real pinned TFLM interpreter run.

What it is **not**: the **arena tail** (scratch/persistent buffers some
kernels request at prepare time) and the **complete arena total** are not
statically estimated by TensorScope at all. Firmware stack, general heap,
DMA buffers, RTOS memory, and other application memory are entirely outside
its scope. No `analyze` report — with or without a `--target`/`--mcu-profile`
budget check — proves complete MCU or firmware memory fit on its own; it
tells you whether the model's runtime tensor memory fits a given RAM budget,
which is one necessary input to that larger question, not the whole answer.

Every report states this scope inline, not just in a separate limitations
section — the FITS/EXACT FIT/EXCEEDS BUDGET verdict itself says "head
only" and points at `validate` for an oracle-observed tail.

## Checking a model against real hardware

Two ways to check planned arena head against a RAM budget:

- `--mcu-profile <id>`: five generic size classes (`cortex-m0-32k` through
  `cortex-m7-1m`) — planning presets, not a specific chip's specification.
  `tensorscope list-profiles` lists all five.
- `--target <name>`: real vendor MCU parts and dev-kit boards, each backed
  by a cited primary-source datasheet. Resolves case-insensitively by
  either the chip's part number or a known dev-kit board name.
  `tensorscope list-targets` lists all of them; as of this writing:

  | Part | Vendor | RAM | Dev-kit / board aliases |
  |---|---|---|---|
  | STM32U585 | STMicroelectronics | 804,864 bytes | NUCLEO-U575ZI-Q |
  | nRF52840 | Nordic Semiconductor | 262,144 bytes | nRF52840-DK, Arduino Nano 33 BLE, Arduino Nano 33 BLE Sense, Adafruit Feather nRF52840 Sense |
  | ESP32-S3 | Espressif Systems | 524,288 bytes | ESP32-S3-DevKitC-1 |
  | CY8C624ABZI-S2D44 | Infineon Technologies | 1,048,576 bytes | CY8CKIT-062S2-AI |

  Each RAM figure is sourced from that vendor's actual datasheet or
  reference manual, not a marketing page — see each profile's `source`
  field (`src/tensorscope/profiles/mcu/*.json`) for the exact citation, and
  `docs/mcu_memory_budgets.md` for how these are resolved and validated.

Example verdict (`analyze model.tflite --target esp32-s3`):

```
Arena-head budget result: FITS (head only — 32 / 524,288 bytes on ESP32-S3,
per Espressif Systems datasheet; arena tail is not estimated here — run
`tensorscope validate` for an oracle-observed tail)
```

The `--html` report renders this same verdict as a two-tier banner (a bold
headline plus a citation/caveat line), followed by the full tensor-by-tensor
explanation: an arena-placement chart, per-tensor tables with peak/largest
highlights, safe-reuse and reuse-blocker detail, memory optimization
guidance, and a limitations section — all in one self-contained HTML file
with zero external assets, zero JavaScript.

## Validate and the TFLM oracle

`validate` cross-checks the planned arena head against a real, pinned TFLM
interpreter run on the host. It's the strongest confidence signal TensorScope
can give (`exact_match`/`mismatch`, not an estimate), but it requires a
separately built binary:

```console
make -C tools/tflm_oracle
```

This needs a C++ toolchain and the pinned `third_party/tflite-micro`
submodule (`git submodule update --init`). An installed CLI (not built from
a source checkout) can point at a binary built elsewhere with:

```console
export TENSORSCOPE_TFLM_ORACLE=/absolute/path/to/tflm_oracle
```

**The committed oracle binary is Linux-only.** Building and running it
requires Linux or WSL; the project has not confirmed a native-Windows build
of the C++ oracle. If you try to run `validate` without a working oracle,
TensorScope tells you clearly what's missing and how to fix it — it does
not fail with a raw stack trace.

### Platform support

| | Linux | WSL | native Windows | macOS |
|---|---|---|---|---|
| `analyze`, `compare`, `check`, `baseline`, `batch`, budget/target checks, all report formats | ✓ | ✓ | ✓ | ✓ (untested, pure Python) |
| `validate` (needs the built oracle) | ✓ | ✓ | not currently supported | untested |

This is honest about the current state, not a promise: the core analysis
path is pure Python with no platform-specific code, and the full test suite
(`pytest`, oracle tests skip automatically where the oracle can't run) has
been verified passing on both a native Windows Python environment and WSL.
`validate` specifically has only been built and run on Linux/WSL.

## Confidence model

Every figure TensorScope reports carries an explicit `confidence` and
`source`:

- `exact` / `static_analysis` — the planned arena head, always. Deterministic,
  not an estimate.
- `not_estimated` — arena tail and complete arena total, until you run
  `validate`, at which point tail becomes an oracle observation (a real
  measurement from one pinned host allocator run — still not a static
  TensorScope estimate).
- `exact_match` / `mismatch` — the validation state once `validate` has run,
  comparing the static plan against the oracle's actual allocation.

Nothing is silently rounded, guessed, or upgraded in confidence without a
real check backing it.

## Documentation

See `tensorscope --help` and the documents under `docs/` for complete
workflows: MCU memory budgets and targets, model comparison, memory
guidance, oracle arena observations, validated operator coverage, and CI
automation/limits. `CHANGELOG.md` tracks what's shipped release to release.

TensorScope does not provide a supported public Python library API — use the
CLI.

## License

TensorScope is licensed under the [GNU Affero General Public License v3.0
or later](LICENSE) (AGPL-3.0-or-later). See [COMMERCIAL.md](COMMERCIAL.md)
for commercial (non-AGPL) licensing.
