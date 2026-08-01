from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from tensorscope import __version__
from tensorscope.explain import MemoryExplanation, explain_primary_subgraph_memory
from tensorscope.graph import (
    calculate_graph_lifetimes,
    calculate_graph_memory_plan,
    convert_tflite_model,
)
from tensorscope.html_report import (
    HTMLReportError,
    render_html_report,
    write_html_report,
)
from tensorscope.memory_budget import (
    ArenaHeadBudgetResult,
    evaluate_direct_budget,
    evaluate_profile_budget,
    get_mcu_profile,
    parse_size,
    render_profile_listing,
)
from tensorscope.oracle import TFLMOracleError
from tensorscope.oracle_validation import validate_model_against_tflm
from tensorscope.results import MemoryFigure
from tensorscope.text_report import render_memory_explanation
from tensorscope.tflite.model_loader import TFLiteModelError, load_tflite_model


TFLM_REVISION = "b89fb3e06e59d2f6af67e758242243da599bfedf"

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_VALIDATION_UNAVAILABLE = 3
EXIT_VALIDATION_MISMATCH = 4
EXIT_REPORT_ERROR = 5
EXIT_BUDGET_EXCEEDED = 6


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


def _calculate_analysis(
    model_path: str | Path,
    *,
    top_tensors: int,
) -> tuple[dict[str, object], MemoryExplanation]:
    path = Path(model_path).expanduser().resolve()
    graph = convert_tflite_model(load_tflite_model(path))
    lifetimes = calculate_graph_lifetimes(graph)
    plan = calculate_graph_memory_plan(graph, lifetimes)
    explanation = explain_primary_subgraph_memory(
        graph,
        lifetimes=lifetimes,
        memory_plan=plan,
        largest_limit=top_tensors,
    )
    tail, total = _unknown_figures()
    head = MemoryFigure(
        bytes=plan.maximum_memory_size,
        scope="arena_head",
        confidence="exact",
        source="static_analysis",
        validation_state="not_validated",
    )
    result = {
        "model_path": str(path),
        "model": {
            "filename": path.name,
            "path": str(path),
            "schema_version": graph.schema_version,
        },
        "command": "analyze",
        "arena_head": head.to_dict(),
        "arena_tail": tail.to_dict(),
        "arena_total": total.to_dict(),
        "analysis": explanation.to_dict(),
    }
    return result, explanation


def analyze_model(
    model_path: str | Path,
    *,
    top_tensors: int = 10,
) -> dict[str, object]:
    result, _ = _calculate_analysis(model_path, top_tensors=top_tensors)
    return result


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


def _render_text(
    result: dict[str, object],
    *,
    explanation: MemoryExplanation | None = None,
    details: bool = False,
    include_ascii: bool = True,
    budget: ArenaHeadBudgetResult | None = None,
) -> str:
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
    if explanation is not None:
        lines.append(
            render_memory_explanation(
                explanation,
                details=details,
                include_ascii=include_ascii,
            )
        )
    if budget is not None:
        lines.extend(["", _render_budget_text(budget)])
    return "\n".join(lines)


def _render_budget_text(budget: ArenaHeadBudgetResult) -> str:
    source = "Direct arena-head budget" if budget.source == "direct" else "Generic MCU planning profile"
    lines = ["Arena-head budget check", f"Budget source: {source}"]
    if budget.profile_name is not None:
        lines.extend(
            [
                f"Profile: {budget.profile_name} ({budget.profile_id})",
                f"Profile RAM: {budget.profile_ram_bytes:,} bytes",
            ]
        )
    lines.extend(
        [
            f"Reserved RAM: {budget.reserve_bytes:,} bytes",
            f"Effective arena-head budget: {budget.effective_budget_bytes:,} bytes",
            f"Planned arena head: {budget.planned_arena_head_bytes:,} bytes",
        ]
    )
    if budget.remaining_bytes >= 0:
        lines.append(f"Remaining budget: {budget.remaining_bytes:,} bytes")
    else:
        lines.append(f"Exceeded by: {-budget.remaining_bytes:,} bytes")
    utilization = (
        "not defined for a zero-byte budget"
        if budget.utilization_percent is None
        else f"{budget.utilization_percent:.2f}%"
    )
    status = {"fits": "FITS", "exact_fit": "EXACT FIT", "exceeds": "EXCEEDS BUDGET"}[budget.status]
    lines.extend(
        [
            f"Utilization: {utilization}",
            f"Arena-head budget result: {status}",
            "This check covers planned arena head only.",
            "This is not a complete MCU or firmware memory-fit conclusion.",
        ]
    )
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tensorscope")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser(
        "analyze", help="statically analyze arena-head memory"
    )
    analyze_parser.add_argument("model", type=Path, nargs="?", help="path to a .tflite model")
    output_group = analyze_parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--json", action="store_true", help="emit stable machine-readable JSON"
    )
    output_group.add_argument(
        "--html",
        type=Path,
        metavar="PATH",
        help="write a self-contained HTML analysis report",
    )
    analyze_parser.add_argument(
        "--details",
        action="store_true",
        help="show every allocation and conservative reuse blockers",
    )
    budget_group = analyze_parser.add_mutually_exclusive_group()
    budget_group.add_argument("--arena-head-budget", metavar="SIZE", help="arena-head byte budget")
    budget_group.add_argument("--mcu-profile", metavar="PROFILE", help="generic MCU planning profile")
    analyze_parser.add_argument("--reserve", metavar="SIZE", help="RAM reserved from the selected profile")
    analyze_parser.add_argument("--list-mcu-profiles", action="store_true", help="list generic MCU planning profiles and exit")
    analyze_parser.add_argument("--fail-on-budget-exceeded", action="store_true", help="return a dedicated failure code when the planned arena head exceeds the budget")
    analyze_parser.add_argument(
        "--top-tensors",
        type=int,
        default=10,
        metavar="N",
        help="number of largest tensors to report (default: 10)",
    )
    analyze_parser.add_argument(
        "--ascii",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the ASCII arena-head packing view (default: enabled)",
    )
    validate_parser = subparsers.add_parser(
        "validate", help="validate arena-head memory with the TFLM oracle"
    )
    validate_parser.add_argument("model", type=Path, help="path to a .tflite model")
    validate_parser.add_argument(
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
    if arguments.command == "analyze" and arguments.list_mcu_profiles:
        print(render_profile_listing())
        return EXIT_SUCCESS
    explanation: MemoryExplanation | None = None
    budget: ArenaHeadBudgetResult | None = None
    try:
        if arguments.command == "analyze":
            if arguments.model is None:
                raise ValueError("A model path is required unless --list-mcu-profiles is used")
            if arguments.reserve is not None and arguments.mcu_profile is None:
                raise ValueError("--reserve may be used only with --mcu-profile")
            if arguments.fail_on_budget_exceeded and arguments.arena_head_budget is None and arguments.mcu_profile is None:
                raise ValueError("--fail-on-budget-exceeded requires --arena-head-budget or --mcu-profile")
            result, explanation = _calculate_analysis(
                arguments.model,
                top_tensors=arguments.top_tensors,
            )
            planned = result["arena_head"]["bytes"]
            assert isinstance(planned, int)
            if arguments.arena_head_budget is not None:
                budget = evaluate_direct_budget(planned, parse_size(arguments.arena_head_budget))
            elif arguments.mcu_profile is not None:
                reserve = parse_size(arguments.reserve) if arguments.reserve is not None else 0
                budget = evaluate_profile_budget(planned, get_mcu_profile(arguments.mcu_profile), reserve)
            if budget is not None:
                result["arena_head_budget"] = budget.to_dict()
            exit_code = EXIT_SUCCESS
        else:
            result, exact_match = validate_model(arguments.model)
            exit_code = EXIT_SUCCESS if exact_match else EXIT_VALIDATION_MISMATCH
        if arguments.command == "analyze" and arguments.html is not None:
            assert explanation is not None
            html = render_html_report(
                result,
                explanation,
                tool_version=__version__,
                budget=budget,
            )
            report_path = write_html_report(arguments.html, html)
        else:
            report_path = None
    except (FileNotFoundError, TFLiteModelError, ValueError) as error:
        _print_error(
            str(error),
            as_json=arguments.json,
            error_type="unsupported_input",
            exit_code=EXIT_INPUT_ERROR,
        )
        return EXIT_INPUT_ERROR
    except HTMLReportError as error:
        _print_error(
            str(error),
            as_json=False,
            error_type="report_write_error",
            exit_code=EXIT_REPORT_ERROR,
        )
        return EXIT_REPORT_ERROR
    except (TFLMOracleError, ValidationUnavailableError, OSError) as error:
        _print_error(
            str(error),
            as_json=arguments.json,
            error_type="validation_unavailable",
            exit_code=EXIT_VALIDATION_UNAVAILABLE,
        )
        return EXIT_VALIDATION_UNAVAILABLE

    if report_path is not None:
        print(f"HTML report written: {report_path}")
    elif arguments.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print(
            _render_text(
                result,
                explanation=explanation,
                details=getattr(arguments, "details", False),
                include_ascii=getattr(arguments, "ascii", False),
                budget=budget,
            )
        )
    if budget is not None and arguments.fail_on_budget_exceeded and budget.status == "exceeds":
        return EXIT_BUDGET_EXCEEDED
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
