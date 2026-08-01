# MCU arena-head budget checks

TensorScope can compare the statically planned arena head with either a direct
byte budget or a generic MCU planning profile:

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

JSON and `--html PATH` include the same structured evaluation. By default an
exceeded budget is informational and exits successfully. Add
`--fail-on-budget-exceeded` for exit code 6 when the result is `exceeds`; `fits`
and `exact_fit` still exit 0, and an HTML report is written before code 6 is
returned.

This check answers only whether the planned arena head fits the selected
arena-head budget. It is not a complete MCU or firmware memory-fit conclusion.
It does not cover arena tail, scratch or persistent buffers, complete arena total,
firmware, stack, heap, DMA, RTOS, or application memory.
