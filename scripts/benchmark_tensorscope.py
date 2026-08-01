"""Opt-in deterministic-phase benchmark; not part of correctness tests."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from tensorscope.cli import _calculate_analysis


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--iterations", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.iterations < 1:
        parser.error("--iterations must be positive")
    samples = []
    for _ in range(arguments.iterations):
        started = perf_counter()
        _calculate_analysis(arguments.model, top_tensors=10)
        samples.append(perf_counter() - started)
    print(f"iterations={arguments.iterations} min_seconds={min(samples):.6f} mean_seconds={sum(samples) / len(samples):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
