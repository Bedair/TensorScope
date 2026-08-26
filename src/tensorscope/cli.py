from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from tensorscope import __version__
from tensorscope.analysis_views import build_analysis_views, build_model_diagnostics
from tensorscope.automation import (
    BATCH_SCHEMA_VERSION,
    aggregate_csv,
    analysis_sarif,
    atomic_write_json,
    atomic_write_text,
    check_baseline,
    create_baseline_manifest,
    deployment_artifacts,
    evaluate_policy,
    load_policy,
    parse_gnu_map,
    resolve_models,
    sarif_document,
)
from tensorscope.comparison import ComparisonInput, ModelComparison, compare_models, render_comparison_text
from tensorscope.comparison_report import render_comparison_html
from tensorscope.explain import MemoryExplanation, explain_primary_subgraph_memory
from tensorscope.graph import (
    GraphModel,
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
    render_budget_verdict,
    render_profile_listing,
)
from tensorscope.target_profiles import (
    as_mcu_profile,
    render_target_listing,
    render_target_verdict_clause,
    resolve_target,
)
from tensorscope.oracle import (
    STRUCTURALLY_UNSUPPORTED,
    UNREGISTERED_OPERATOR,
    TFLMOracleError,
)
from tensorscope.oracle_validation import validate_model_against_tflm
from tensorscope.recommendations import (
    MemoryRiskAssessment,
    assess_memory_risk,
    render_memory_guidance,
)
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
EXIT_COMPARISON_REGRESSION = 7
EXIT_POLICY_FAILURE = 8
EXIT_BASELINE_DRIFT = 9
EXIT_BATCH_FAILURE = 10
EXIT_FIRMWARE_CHECK_FAILURE = 11


class ValidationUnavailableError(RuntimeError):
    """Raised when a valid model cannot be checked by the oracle."""


_ORACLE_INCOMPATIBILITY_EXPLANATIONS = {
    UNREGISTERED_OPERATOR: (
        "TensorScope's TFLM oracle has not registered an operator this model "
        "uses. TFLM may already implement it elsewhere in its kernel set -- "
        "file a coverage gap and consider registering it in "
        "tools/tflm_oracle/main.cc."
    ),
    STRUCTURALLY_UNSUPPORTED: (
        "TFLM found this operator but refuses to run it for this model's "
        "configuration (for example hybrid int8/float32 quantization). The "
        "model cannot run on stock TFLM regardless of oracle registration -- "
        "check whether it was exported for a different runtime or toolchain."
    ),
}


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
) -> tuple[dict[str, object], MemoryExplanation, GraphModel]:
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
        "analysis_views": build_analysis_views(graph, explanation),
        "model_diagnostics": build_model_diagnostics(graph),
    }
    return result, explanation, graph


def analyze_model(
    model_path: str | Path,
    *,
    top_tensors: int = 10,
) -> dict[str, object]:
    result, explanation, graph = _calculate_analysis(model_path, top_tensors=top_tensors)
    result["memory_guidance"] = assess_memory_risk(graph, explanation).to_dict()
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
        "oracle_arena_observation": validation.oracle.observation.to_dict(),
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
    guidance: MemoryRiskAssessment | None = None,
    target_clause: str | None = None,
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
    observation = result.get("oracle_arena_observation")
    if isinstance(observation, dict):
        def observed_bytes(key: str) -> str:
            value = observation.get(key)
            return "not available" if value is None else f"{value:,} bytes"

        lines.extend(
            [
                "",
                "TFLM oracle memory observation",
                f"Observation source: {observation['source']}",
                f"Allocator capacity: {observed_bytes('capacity_bytes')}",
                f"Observed arena used: {observed_bytes('used_bytes')}",
                f"Observed arena head: {observed_bytes('head_bytes')}",
                f"Observed arena tail: {observed_bytes('tail_bytes')}",
                f"Temporary bytes: {observed_bytes('temporary_bytes')}",
                f"Allocator remaining: {observed_bytes('remaining_bytes')}",
                f"Allocator alignment: {observed_bytes('alignment_bytes')}",
                f"Observed TFLM revision: {observation.get('tflm_revision') or 'not available'}",
                "Arena tail and complete arena usage are oracle observations, not static TensorScope estimates.",
                "These values describe the pinned host-side TFLM allocator run and are not a complete MCU or firmware memory-fit guarantee.",
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
        lines.extend(["", _render_budget_text(budget, target_clause=target_clause)])
    if guidance is not None:
        lines.extend(["", render_memory_guidance(guidance, details=details)])
    views = result.get("analysis_views")
    if isinstance(views, dict) and explanation is not None:
        attribution = views["operator_attribution"]
        operators = attribution["operators"] if details else attribution["operators"][:5]
        lines.extend(["", "Operator-level arena-head pressure (non-additive)"])
        for item in operators:
            lines.append(
                f"  Operator {item['operator_id']} {item['operator_name']}: "
                f"live {item['live_aligned_bytes_at_scope']:,} bytes; "
                f"occupied extent {item['occupied_extent_bytes_at_scope']:,} bytes; "
                f"pressure {item['pressure']}"
            )
        lines.append("These represented live-set values are not independently additive contributions to planned arena head.")
    return "\n".join(lines)


def _render_budget_text(budget: ArenaHeadBudgetResult, *, target_clause: str | None = None) -> str:
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
    lines.extend(
        [
            f"Utilization: {utilization}",
            f"Arena-head budget result: {render_budget_verdict(budget, target_clause=target_clause)}",
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
    output_group.add_argument("--sarif", type=Path, metavar="PATH", help="write SARIF 2.1.0 findings")
    analyze_parser.add_argument(
        "--details",
        action="store_true",
        help="show every allocation and conservative reuse blockers",
    )
    budget_group = analyze_parser.add_mutually_exclusive_group()
    budget_group.add_argument("--arena-head-budget", metavar="SIZE", help="arena-head byte budget")
    budget_group.add_argument("--mcu-profile", metavar="PROFILE", help="generic MCU planning profile")
    budget_group.add_argument("--target", metavar="NAME", help="real MCU part or dev-kit board name (case-insensitive), sourced from a vendor datasheet")
    analyze_parser.add_argument("--reserve", metavar="SIZE", help="RAM reserved from the selected profile or target")
    analyze_parser.add_argument("--list-mcu-profiles", action="store_true", help="list generic MCU planning profiles and exit (use --mcu-profile <id> to check a model against one)")
    analyze_parser.add_argument("--list-targets", action="store_true", help="list real per-vendor MCU/dev-kit targets and exit (use --target <name> to check a model against one)")
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
    compare_parser = subparsers.add_parser(
        "compare", help="compare static arena-head memory for two models"
    )
    compare_parser.add_argument("baseline", type=Path, help="baseline .tflite model")
    compare_parser.add_argument("candidate", type=Path, help="candidate .tflite model")
    compare_output = compare_parser.add_mutually_exclusive_group()
    compare_output.add_argument("--json", action="store_true", help="emit stable machine-readable JSON")
    compare_output.add_argument("--html", type=Path, metavar="PATH", help="write a self-contained comparison HTML report")
    compare_parser.add_argument("--details", action="store_true", help="show all tensor changes")
    compare_budget = compare_parser.add_mutually_exclusive_group()
    compare_budget.add_argument("--arena-head-budget", metavar="SIZE", help="shared arena-head byte budget")
    compare_budget.add_argument("--mcu-profile", metavar="PROFILE", help="shared generic MCU planning profile")
    compare_parser.add_argument("--reserve", metavar="SIZE", help="RAM reserved from the selected profile")
    compare_parser.add_argument("--fail-on-regression", action="store_true", help="return code 7 after output when candidate is a regression")
    check_parser = subparsers.add_parser("check", help="evaluate a strict CI policy")
    check_parser.add_argument("model", type=Path)
    check_parser.add_argument("--policy", type=Path, required=True)
    check_parser.add_argument("--json", type=Path, metavar="PATH")
    check_parser.add_argument("--sarif", type=Path, metavar="PATH")
    baseline_parser = subparsers.add_parser("baseline", help="create or check deterministic baselines")
    baseline_sub = baseline_parser.add_subparsers(dest="baseline_command", required=True)
    baseline_create = baseline_sub.add_parser("create")
    baseline_create.add_argument("model", type=Path)
    baseline_create.add_argument("--output", type=Path, required=True)
    baseline_check_parser = baseline_sub.add_parser("check")
    baseline_check_parser.add_argument("model", type=Path)
    baseline_check_parser.add_argument("--baseline", type=Path, required=True)
    baseline_check_parser.add_argument("--json", type=Path, metavar="PATH")
    batch_parser = subparsers.add_parser("batch", help="analyze model files and directories")
    batch_parser.add_argument("paths", type=Path, nargs="+")
    batch_parser.add_argument("--output-dir", type=Path, required=True)
    batch_parser.add_argument("--recursive", action="store_true")
    batch_parser.add_argument("--fail-fast", action="store_true")
    batch_budget = batch_parser.add_mutually_exclusive_group()
    batch_budget.add_argument("--arena-head-budget", metavar="SIZE")
    batch_budget.add_argument("--mcu-profile", metavar="PROFILE")
    batch_parser.add_argument("--reserve", metavar="SIZE")
    batch_parser.add_argument("--sarif", action="store_true", help="write aggregate SARIF")
    firmware_parser = subparsers.add_parser("firmware-check", help="check planned head against a GNU ld map arena")
    firmware_parser.add_argument("model", type=Path)
    firmware_parser.add_argument("--map-file", type=Path, required=True)
    firmware_parser.add_argument("--arena-symbol", required=True)
    firmware_parser.add_argument("--arena-size", type=int)
    firmware_parser.add_argument("--ram-region")
    firmware_parser.add_argument("--stack-reserve", type=int, default=0)
    firmware_parser.add_argument("--heap-reserve", type=int, default=0)
    firmware_parser.add_argument("--json", type=Path, metavar="PATH")
    deploy_parser = subparsers.add_parser("deploy-report", help="generate deterministic deployment artifacts")
    deploy_parser.add_argument("model", type=Path)
    deploy_parser.add_argument("--output-dir", type=Path, required=True)
    deploy_parser.add_argument("--margin-percent", type=int, default=10)
    subparsers.add_parser("list-profiles", help="list generic planning profiles (use analyze --mcu-profile <id> to check a model against one)")
    subparsers.add_parser("list-targets", help="list real per-vendor MCU/dev-kit targets (use analyze --target <name> to check a model against one)")
    return parser


def _comparison_budget(arguments: argparse.Namespace, planned: int) -> ArenaHeadBudgetResult | None:
    if arguments.arena_head_budget is not None:
        return evaluate_direct_budget(planned, parse_size(arguments.arena_head_budget))
    if arguments.mcu_profile is not None:
        reserve = parse_size(arguments.reserve) if arguments.reserve is not None else 0
        return evaluate_profile_budget(planned, get_mcu_profile(arguments.mcu_profile), reserve)
    return None


def _calculate_comparison(arguments: argparse.Namespace) -> ModelComparison:
    if arguments.reserve is not None and arguments.mcu_profile is None:
        raise ValueError("--reserve may be used only with --mcu-profile")
    baseline_result, baseline_explanation, baseline_graph = _calculate_analysis(arguments.baseline, top_tensors=10)
    candidate_result, candidate_explanation, candidate_graph = _calculate_analysis(arguments.candidate, top_tensors=10)
    baseline_head = baseline_result["arena_head"]["bytes"]
    candidate_head = candidate_result["arena_head"]["bytes"]
    assert isinstance(baseline_head, int) and isinstance(candidate_head, int)
    baseline_budget = _comparison_budget(arguments, baseline_head)
    candidate_budget = _comparison_budget(arguments, candidate_head)
    baseline_guidance = assess_memory_risk(baseline_graph, baseline_explanation, budget=baseline_budget)
    candidate_guidance = assess_memory_risk(candidate_graph, candidate_explanation, budget=candidate_budget)
    return compare_models(
        ComparisonInput(str(Path(arguments.baseline).expanduser().resolve()), baseline_graph, baseline_explanation, baseline_guidance, baseline_budget),
        ComparisonInput(str(Path(arguments.candidate).expanduser().resolve()), candidate_graph, candidate_explanation, candidate_guidance, candidate_budget),
    )


def _print_error(
    message: str,
    *,
    as_json: bool,
    error_type: str,
    exit_code: int,
    explanation: str | None = None,
) -> None:
    if as_json:
        payload: dict[str, object] = {
            "confidence": "unsupported",
            "error": message,
            "error_type": error_type,
            "exit_code": exit_code,
        }
        if explanation is not None:
            payload["explanation"] = explanation
        print(
            json.dumps(payload, sort_keys=True),
            file=sys.stderr,
        )
    else:
        rendered = f"Error ({error_type}): {message}"
        if explanation is not None:
            rendered += f"\n{explanation}"
        print(rendered, file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    if arguments.command == "analyze" and arguments.list_mcu_profiles:
        print(render_profile_listing())
        return EXIT_SUCCESS
    if arguments.command == "list-profiles":
        print(render_profile_listing())
        return EXIT_SUCCESS
    if arguments.command == "analyze" and arguments.list_targets:
        print(render_target_listing())
        return EXIT_SUCCESS
    if arguments.command == "list-targets":
        print(render_target_listing())
        return EXIT_SUCCESS
    explanation: MemoryExplanation | None = None
    budget: ArenaHeadBudgetResult | None = None
    target_clause: str | None = None
    guidance: MemoryRiskAssessment | None = None
    graph: GraphModel | None = None
    comparison: ModelComparison | None = None
    try:
        if arguments.command == "baseline":
            analysis = analyze_model(arguments.model)
            if arguments.baseline_command == "create":
                manifest = create_baseline_manifest(arguments.model, analysis, tool_version=__version__)
                destination = atomic_write_json(arguments.output, manifest)
                print(f"Baseline manifest written: {destination}")
                return EXIT_SUCCESS
            manifest = json.loads(arguments.baseline.read_text(encoding="utf-8"))
            baseline_result = check_baseline(manifest, analysis, arguments.model)
            if arguments.json is not None:
                destination = atomic_write_json(arguments.json, baseline_result)
                print(f"Baseline check written: {destination}")
            else:
                print(json.dumps(baseline_result, sort_keys=True, separators=(",", ":")))
            return EXIT_BASELINE_DRIFT if baseline_result["status"] == "failed" else EXIT_SUCCESS
        if arguments.command == "check":
            analysis = analyze_model(arguments.model)
            policy = load_policy(arguments.policy)
            baseline_result = None
            policy_comparison = None
            if policy.get("baseline_manifest") is not None:
                baseline_path = (arguments.policy.parent / str(policy["baseline_manifest"])).resolve()
                baseline_manifest = json.loads(baseline_path.read_text(encoding="utf-8"))
                baseline_result = check_baseline(baseline_manifest, analysis, arguments.model)
                baseline_head = baseline_manifest["metrics"]["planned_arena_head_bytes"]
                candidate_head = analysis["analysis"]["summary"]["planned_arena_head_bytes"]
                delta = candidate_head - baseline_head
                percent = delta * 100 / baseline_head if baseline_head else None
                policy_comparison = {
                    "metrics": {"planned_arena_head_bytes": {"delta": delta, "percent_delta": percent}},
                    "regression": {"is_regression": delta >= 256 and percent is not None and percent >= 5},
                }
            policy_result = evaluate_policy(policy, analysis, comparison=policy_comparison, baseline_result=baseline_result)
            if arguments.json is not None:
                atomic_write_json(arguments.json, policy_result)
            if arguments.sarif is not None:
                findings = [
                    {"rule_id": item["rule_id"], "message": item["message"], "level": "error", "properties": {"actual": item["actual"], "limit": item["limit"]}}
                    for item in policy_result["failures"]
                ]
                atomic_write_json(arguments.sarif, sarif_document(arguments.model, findings))
            print(json.dumps(policy_result, sort_keys=True, separators=(",", ":")))
            return EXIT_POLICY_FAILURE if policy_result["status"] == "failed" else EXIT_SUCCESS
        if arguments.command == "batch":
            if arguments.reserve is not None and arguments.mcu_profile is None:
                raise ValueError("--reserve may be used only with --mcu-profile")
            models = resolve_models(arguments.paths, recursive=arguments.recursive)
            output_dir = arguments.output_dir.expanduser().resolve()
            rows: list[dict[str, object]] = []
            sarif_runs: list[dict[str, object]] = []
            for model in models:
                try:
                    model_result, model_explanation, model_graph = _calculate_analysis(model, top_tensors=10)
                    planned = model_result["arena_head"]["bytes"]
                    assert isinstance(planned, int)
                    model_budget = _comparison_budget(arguments, planned)
                    model_guidance = assess_memory_risk(model_graph, model_explanation, budget=model_budget)
                    if model_budget is not None:
                        model_result["arena_head_budget"] = model_budget.to_dict()
                    model_result["memory_guidance"] = model_guidance.to_dict()
                    stem = model.name
                    atomic_write_json(output_dir / f"{stem}.json", model_result)
                    html = render_html_report(model_result, model_explanation, tool_version=__version__, budget=model_budget, guidance=model_guidance)
                    write_html_report(output_dir / f"{stem}.html", html)
                    rows.append({"model": str(model), "status": "ok", "planned_arena_head_bytes": planned,
                                 "overall_risk": model_guidance.overall_risk,
                                 "budget_status": model_budget.status if model_budget else None, "error": None})
                    if arguments.sarif:
                        sarif_runs.extend(analysis_sarif(model, model_result)["runs"])
                except Exception as error:
                    rows.append({"model": str(model), "status": "error", "planned_arena_head_bytes": None,
                                 "overall_risk": None, "budget_status": None, "error": str(error)})
                    if arguments.fail_fast:
                        break
            aggregate = {"batch_schema_version": BATCH_SCHEMA_VERSION, "model_count": len(rows),
                         "success_count": sum(item["status"] == "ok" for item in rows),
                         "error_count": sum(item["status"] == "error" for item in rows), "models": rows}
            atomic_write_json(output_dir / "aggregate.json", aggregate)
            atomic_write_text(output_dir / "aggregate.csv", aggregate_csv(rows))
            if arguments.sarif:
                atomic_write_json(output_dir / "aggregate.sarif.json", {"$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0", "runs": sarif_runs})
            print(f"Batch analyzed {aggregate['success_count']} model(s); {aggregate['error_count']} error(s). Output: {output_dir}")
            return EXIT_BATCH_FAILURE if aggregate["error_count"] else EXIT_SUCCESS
        if arguments.command == "firmware-check":
            if arguments.arena_size is not None and arguments.arena_size < 0:
                raise ValueError("--arena-size must be non-negative")
            if arguments.stack_reserve < 0 or arguments.heap_reserve < 0:
                raise ValueError("stack and heap reserves must be non-negative")
            analysis = analyze_model(arguments.model)
            map_result = parse_gnu_map(arguments.map_file.read_text(encoding="utf-8", errors="replace"), arguments.arena_symbol, arguments.ram_region)
            planned = analysis["analysis"]["summary"]["planned_arena_head_bytes"]
            arena_size = arguments.arena_size
            status = "incomplete" if arena_size is None else "fits" if planned <= arena_size else "exceeds"
            firmware_result = {"firmware_check_schema_version": 1, "status": status, "scope": "planned_arena_head_in_reserved_arena",
                               "planned_arena_head_bytes": planned, "arena_size_bytes": arena_size,
                               "stack_reserve_bytes": arguments.stack_reserve, "heap_reserve_bytes": arguments.heap_reserve,
                               "map": map_result,
                               "limitations": ["GNU ld map subset only", "complete MCU or firmware fit is not established"]}
            if arguments.json is not None:
                atomic_write_json(arguments.json, firmware_result)
            print(json.dumps(firmware_result, sort_keys=True, separators=(",", ":")))
            return EXIT_FIRMWARE_CHECK_FAILURE if status != "fits" else EXIT_SUCCESS
        if arguments.command == "deploy-report":
            analysis = analyze_model(arguments.model)
            artifacts = deployment_artifacts(arguments.model, analysis, margin_percent=arguments.margin_percent)
            output_dir = arguments.output_dir.expanduser().resolve()
            for name, content in artifacts.items():
                atomic_write_text(output_dir / name, content)
            atomic_write_json(output_dir / "analysis.json", analysis)
            print(f"Deployment artifacts written: {output_dir}")
            return EXIT_SUCCESS
        if arguments.command == "analyze":
            if arguments.model is None:
                raise ValueError("A model path is required unless --list-mcu-profiles or --list-targets is used")
            if arguments.reserve is not None and arguments.mcu_profile is None and arguments.target is None:
                raise ValueError("--reserve may be used only with --mcu-profile or --target")
            if (
                arguments.fail_on_budget_exceeded
                and arguments.arena_head_budget is None
                and arguments.mcu_profile is None
                and arguments.target is None
            ):
                raise ValueError("--fail-on-budget-exceeded requires --arena-head-budget, --mcu-profile, or --target")
            result, explanation, graph = _calculate_analysis(
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
            elif arguments.target is not None:
                reserve = parse_size(arguments.reserve) if arguments.reserve is not None else 0
                resolved_target = resolve_target(arguments.target)
                budget = evaluate_profile_budget(planned, as_mcu_profile(resolved_target), reserve)
                target_clause = render_target_verdict_clause(resolved_target)
            if budget is not None:
                result["arena_head_budget"] = budget.to_dict()
                if target_clause is not None:
                    result["arena_head_budget"]["verdict"] = render_budget_verdict(
                        budget, target_clause=target_clause
                    )
            guidance = assess_memory_risk(graph, explanation, budget=budget)
            result["memory_guidance"] = guidance.to_dict()
            exit_code = EXIT_SUCCESS
        elif arguments.command == "validate":
            result, exact_match = validate_model(arguments.model)
            exit_code = EXIT_SUCCESS if exact_match else EXIT_VALIDATION_MISMATCH
        else:
            comparison = _calculate_comparison(arguments)
            result = comparison.to_dict()
            exit_code = EXIT_SUCCESS
        if arguments.command == "analyze" and arguments.html is not None:
            assert explanation is not None
            html = render_html_report(
                result,
                explanation,
                tool_version=__version__,
                budget=budget,
                guidance=guidance,
                target_clause=target_clause,
            )
            report_path = write_html_report(arguments.html, html)
        else:
            report_path = None
        if arguments.command == "analyze" and arguments.sarif is not None:
            report_path = atomic_write_json(arguments.sarif, analysis_sarif(arguments.model, result))
        if arguments.command == "compare" and arguments.html is not None:
            assert comparison is not None
            report_path = write_html_report(
                arguments.html,
                render_comparison_html(comparison, tool_version=__version__),
            )
    except (FileNotFoundError, TFLiteModelError, ValueError) as error:
        _print_error(
            str(error),
            as_json=getattr(arguments, "json", False),
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
        category = getattr(error, "category", None)
        _print_error(
            str(error),
            as_json=getattr(arguments, "json", False),
            error_type=category or "validation_unavailable",
            exit_code=EXIT_VALIDATION_UNAVAILABLE,
            explanation=_ORACLE_INCOMPATIBILITY_EXPLANATIONS.get(category),
        )
        return EXIT_VALIDATION_UNAVAILABLE

    if report_path is not None:
        label = "SARIF report" if arguments.command == "analyze" and arguments.sarif is not None else "HTML report"
        print(f"{label} written: {report_path}")
    elif getattr(arguments, "json", False):
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    elif arguments.command == "compare":
        assert comparison is not None
        print(render_comparison_text(comparison, details=arguments.details))
    else:
        print(
            _render_text(
                result,
                explanation=explanation,
                details=getattr(arguments, "details", False),
                include_ascii=getattr(arguments, "ascii", False),
                budget=budget,
                guidance=guidance,
                target_clause=target_clause,
            )
        )
    if arguments.command == "analyze" and budget is not None and arguments.fail_on_budget_exceeded and budget.status == "exceeds":
        return EXIT_BUDGET_EXCEEDED
    if arguments.command == "compare" and arguments.fail_on_regression and comparison is not None and comparison.regression.is_regression:
        return EXIT_COMPARISON_REGRESSION
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
