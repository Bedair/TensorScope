# MCU arena-head budget checks

TensorScope can compare the statically planned arena head with a direct byte
budget, a generic MCU planning profile, or a real per-vendor MCU/dev-kit
target:

```console
python -m tensorscope analyze model.tflite --arena-head-budget 64KiB
python -m tensorscope analyze model.tflite --mcu-profile cortex-m4-256k
python -m tensorscope analyze model.tflite --mcu-profile cortex-m4-256k --reserve 64KiB
```

Accepted sizes are non-negative whole numbers in plain bytes or with `B`,
`KiB`, or `MiB` (case-insensitive). Units are binary: 1 KiB is 1,024 bytes and
1 MiB is 1,048,576 bytes. Decimal `KB`/`MB` suffixes and fractions are rejected;
values are never rounded.

List profiles without supplying a model:

```console
python -m tensorscope analyze --list-mcu-profiles
```

The catalog contains `cortex-m0-32k`, `cortex-m4-128k`, `cortex-m4-256k`,
`cortex-m7-512k`, and `cortex-m7-1m`. These are generic planning presets, not
specifications for every MCU using the named processor core. Cortex architecture
does not define a fixed RAM capacity.

`--reserve` subtracts an exact byte count from profile RAM. A reserve equal to
profile RAM is valid and produces a zero-byte effective budget. For a zero-byte
budget, zero planned bytes is an exact fit and any nonzero plan exceeds; utilization
is reported as null/not defined to avoid division by zero.

## Real per-vendor targets

`--mcu-profile` is deliberately generic -- five Cortex-class RAM tiers, no
part names. `--target <name>` is the separate, additive alternative: it
resolves against real MCU part numbers and dev-kit board names, each sourced
from a vendor datasheet or reference manual, not a rounded class estimate.

```console
python -m tensorscope analyze model.tflite --target STM32U585
python -m tensorscope analyze model.tflite --target NUCLEO-U575ZI-Q
python -m tensorscope analyze model.tflite --target "Arduino Nano 33 BLE Sense" --reserve 64KiB
python -m tensorscope analyze --list-targets
```

Matching is case-insensitive and exact against a profile's MCU part number,
internal id, or any of its dev-kit aliases (a dev-kit name is a pure alias
for its MCU's data, not a separate figure) -- never partial, prefix, or
fuzzy. An unrecognized name is a hard error listing every known part and
alias; it is never guessed.

`--target` shares the exact same evaluation, `--reserve` handling,
`--fail-on-budget-exceeded` behavior, and text/JSON/HTML rendering as
`--mcu-profile` -- it is mutually exclusive with `--mcu-profile` and
`--arena-head-budget`, not a replacement for either.

Profiles are data, not code: each one is a JSON file under
`src/tensorscope/profiles/mcu/`, validated at load time (a malformed file or
a name that collides with another profile's id/part/alias is a hard error,
never silently skipped or silently resolved). Adding a part means adding a
file, not editing Python. Every field, including the `total_sram_bytes`
figure's exact datasheet citation (title, revision, section, page, URL --
`null` where a fetch genuinely couldn't confirm one rather than a guess), is
in the JSON and inspectable directly.

JSON and `--html PATH` include the same structured evaluation. By default an
exceeded budget is informational and exits successfully. Add
`--fail-on-budget-exceeded` for exit code 6 when the result is `exceeds`; `fits`
and `exact_fit` still exit 0, and an HTML report is written before code 6 is
returned.

This check answers only whether the planned arena head fits the selected
arena-head budget. It is not a complete MCU or firmware memory-fit conclusion.
It does not cover arena tail, scratch or persistent buffers, complete arena total,
firmware, stack, heap, DMA, RTOS, or application memory.

## Catalog scope decisions

Parts investigated and deliberately excluded from the `--target` catalog,
recorded here so they aren't silently rediscovered and re-investigated later:

- **Analog Devices MAX78000 / MAX78002** -- excluded entirely, not added with
  a lower-confidence figure. Both parts pair a normal Cortex-M4 with a
  dedicated CNN hardware accelerator that has its own separate weight/data
  SRAM, entirely outside the Cortex-M4's memory map and outside TFLM's
  interpreter and allocator. A model deployed to the CNN accelerator is
  converted by ADI's own tooling into weight-loading and instruction-sequence
  form, not run as a normal TFLM graph against an arena -- so this project's
  `total_sram_bytes`/`total_flash_bytes` figures would only ever describe a
  non-accelerated, CPU-only build that ignores the chip's actual intended use.
  Shipping a profile next to real MCU targets risked implying an equivalence
  that doesn't exist.
