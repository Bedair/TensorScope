from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from tensorscope import __version__
from tensorscope.graph import calculate_graph_memory_plan, convert_tflite_model
from tensorscope.oracle import TFLMOracleError
from tensorscope.oracle_validation import validate_model_against_tflm
from tensorscope.results import MemoryFigure
from tensorscope.tflite.model_loader import TFLiteModelError, load_tflite_model


TFLM_REVISION = "b89fb3e06e59d2f6af67e758242243da599bfedf"

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_VALIDATION_UNAVAILABLE = 3
EXIT_VALIDATION_MISMATCH = 4


class ValidationUnavailableError(RuntimeError):
    """Raised when a valid model cannot be checked by the oracle."""


def _unknown_figures() -> tuple[MemoryFigure, MemoryFigure]:
    return (
        MemoryFigure(
            bytes=None,
            scope="arena_tail",
            confidence="not_estimated",
            source=None,
            validation_state="not_validated",
        ),
        MemoryFigure(
            bytes=None,
            scope="arena_total",
            confidence="not_estimated",
            source=None,
            validation_state="not_validated",
        ),
    )


def analyze_model(model_path: str | Path) -> dict[str, object]:
    path = Path(model_path).expanduser().resolve()
    graph = convert_tflite_model(load_tflite_model(path))
    plan = calculate_graph_memory_plan(graph)
    tail, total = _unknown_figures()
    head = MemoryFigure(
        bytes=plan.maximum_memory_size,
        scope="arena_head",
        confidence="exact",
        source="static_analysis",
        validation_state="not_validated",
    )
    return {
        "model_path": str(path),
        "command": "analyze",
        "arena_head": head.to_dict(),
        "arena_tail": tail.to_dict(),
        "arena_total": total.to_dict(),
    }


def validate_model(model_path: str | Path) -> tuple[dict[str, object], bool]:
    path = Path(model_path).expanduser().resolve()
    # Load first so a missing or malformed model remains an input error.  Any
    # FileNotFoundError after this point refers to the external oracle.
    load_tflite_model(path)
    try:
        validation = validate_model_against_tflm(path)
    except FileNotFoundError as error:
        raise ValidationUnavailableError(str(error)) from error
    tail, total = _unknown_figures()
    state = "exact_match" if validation.exact_match else "mismatch"
    head = MemoryFigure(
        bytes=validation.tensorscope_head,
        scope="arena_head",
        confidence="exact" if validation.exact_match else "estimated",
        source="static_analysis",
        validation_state=state,
        validated_tflm_revision=(TFLM_REVISION if validation.exact_match else None),
    )
    result: dict[str, object] = {
        "model_path": str(validation.model_path),
        "command": "validate",
        "arena_head": head.to_dict(),
        "arena_tail": tail.to_dict(),
        "arena_total": total.to_dict(),
        "validation": {
            "scope": "arena_head",
            "state": state,
            "tensorscope_bytes": validation.tensorscope_head,
            "tflm_oracle_bytes": validation.tflm_head,
            "delta_bytes": validation.head_delta,
            "tflm_revision": TFLM_REVISION,
        },
    }
    return result, validation.exact_match


def _format_figure(label: str, figure: dict[str, object]) -> str:
    value = figure["bytes"]
    rendered_value = "not estimated" if value is None else f"{value:,} bytes"
    metadata = [str(figure["confidence"])]
    if figure["source"] is not None:
        metadata.append(str(figure["source"]))
    metadata.append(str(figure["validation_state"]))
    return f"{label}: {rendered_value} [{'; '.join(metadata)}]"


def _render_text(result: dict[str, object]) -> str:
    lines = [f"Model: {result['model_path']}"]
    validation = result.get("validation")
    if isinstance(validation, dict):
        if validation["state"] == "exact_match":
            lines.append("Arena-head validation: EXACT MATCH")
        else:
            lines.append("Arena-head validation: MISMATCH")
        lines.extend(
            [
                "Validation scope: arena head only.",
                f"TensorScope arena head: {validation['tensorscope_bytes']:,} bytes",
                f"TFLM oracle arena head: {validation['tflm_oracle_bytes']:,} bytes",
                f"Arena head delta: {validation['delta_bytes']:+,} bytes",
                f"Validated TFLM revision: {validation['tflm_revision']}",
            ]
        )
    else:
        lines.append(_format_figure("Arena head", result["arena_head"]))

    lines.extend(
        [
            _format_figure("Arena tail", result["arena_tail"]),
            "Arena tail is not yet statically estimated.",
            _format_figure("Complete arena total", result["arena_total"]),
            "Complete arena total is not yet statically estimated.",
        ]
    )
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tensorscope")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("analyze", "statically analyze arena-head memory"),
        ("validate", "validate arena-head memory with the TFLM oracle"),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("model", type=Path, help="path to a .tflite model")
        command_parser.add_argument(
            "--json", action="store_true", help="emit stable machine-readable JSON"
        )
    return parser


def _print_error(
    message: str,
    *,
    as_json: bool,
    error_type: str,
    exit_code: int,
) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "confidence": "unsupported",
                    "error": message,
                    "error_type": error_type,
                    "exit_code": exit_code,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
    else:
        print(f"Error ({error_type}): {message}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        if arguments.command == "analyze":
            result = analyze_model(arguments.model)
            exit_code = EXIT_SUCCESS
        else:
            result, exact_match = validate_model(arguments.model)
            exit_code = EXIT_SUCCESS if exact_match else EXIT_VALIDATION_MISMATCH
    except (FileNotFoundError, TFLiteModelError, ValueError) as error:
        _print_error(
            str(error),
            as_json=arguments.json,
            error_type="unsupported_input",
            exit_code=EXIT_INPUT_ERROR,
        )
        return EXIT_INPUT_ERROR
    except (TFLMOracleError, ValidationUnavailableError, OSError) as error:
        _print_error(
            str(error),
            as_json=arguments.json,
            error_type="validation_unavailable",
            exit_code=EXIT_VALIDATION_UNAVAILABLE,
        )
        return EXIT_VALIDATION_UNAVAILABLE

    if arguments.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print(_render_text(result))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
