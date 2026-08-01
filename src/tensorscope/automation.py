from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable

import yaml


BASELINE_SCHEMA_VERSION = 1
POLICY_SCHEMA_VERSION = 1
BATCH_SCHEMA_VERSION = 1
DEPLOYMENT_SCHEMA_VERSION = 1
SARIF_VERSION = "2.1.0"


def atomic_write_text(path: str | Path, content: str) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return destination


def atomic_write_json(path: str | Path, value: object) -> Path:
    return atomic_write_text(path, json.dumps(value, sort_keys=True, indent=2) + "\n")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def operator_names(analysis: dict[str, object]) -> list[str]:
    scopes = analysis["analysis"]["scopes"]  # type: ignore[index]
    return [item["operator_name"] for item in scopes if item["operator_name"] is not None]


def create_baseline_manifest(
    model_path: str | Path,
    analysis: dict[str, object],
    *,
    tool_version: str,
) -> dict[str, object]:
    path = Path(model_path).expanduser().resolve()
    detail = analysis["analysis"]
    summary = detail["summary"]
    guidance = analysis["memory_guidance"]
    return {
        "baseline_schema_version": BASELINE_SCHEMA_VERSION,
        "tensorscope_version": tool_version,
        "model": {"display_name": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size},
        "assumptions": {"scope": "primary_subgraph_arena_head", "alignment_bytes": summary["arena_alignment_bytes"]},
        "metrics": {
            "planned_arena_head_bytes": summary["planned_arena_head_bytes"],
            "peak_occupied_extent_bytes": detail["peak"]["occupied_extent_bytes"],
            "peak_live_aligned_bytes": detail["peak"]["live_aligned_bytes"],
            "runtime_tensor_count": summary["runtime_tensor_count"],
            "constant_tensor_count": summary["constant_tensor_count"],
            "operator_count": summary["operator_count"],
            "logical_runtime_tensor_bytes": summary["logical_runtime_tensor_bytes"],
            "aligned_runtime_tensor_bytes": summary["aligned_runtime_tensor_bytes"],
            "alignment_overhead_bytes": summary["alignment_overhead_bytes"],
            "safe_reuse_pair_count": len(detail["reuse"]),
            "reuse_blocker_count": len(detail["reuse_blockers"]),
        },
        "guidance": {
            "overall_risk": guidance["overall_risk"],
            "summary": guidance["summary"],
            "categories": sorted({item["category"] for item in guidance["findings"]}),
        },
        "operators": {"names": operator_names(analysis), "unique_names": sorted(set(operator_names(analysis)))},
        "budget": analysis.get("arena_head_budget"),
    }


def check_baseline(manifest: dict[str, object], current: dict[str, object], model_path: str | Path) -> dict[str, object]:
    if manifest.get("baseline_schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValueError("Unsupported baseline_schema_version")
    reasons: list[str] = []
    expected_hash = manifest["model"]["sha256"]  # type: ignore[index]
    actual_hash = sha256_file(model_path)
    if actual_hash != expected_hash:
        reasons.append("model SHA-256 changed")
    current_manifest = create_baseline_manifest(model_path, current, tool_version=str(manifest.get("tensorscope_version", "unknown")))
    for name, expected in manifest["metrics"].items():  # type: ignore[union-attr]
        actual = current_manifest["metrics"][name]  # type: ignore[index]
        if actual != expected:
            reasons.append(f"metric {name} changed from {expected} to {actual}")
    if manifest["guidance"]["categories"] != current_manifest["guidance"]["categories"]:  # type: ignore[index]
        reasons.append("guidance categories changed")
    if manifest.get("budget") != current_manifest.get("budget"):
        reasons.append("budget result changed")
    return {
        "baseline_check_schema_version": 1, "status": "failed" if reasons else "passed",
        "model": str(Path(model_path).resolve()), "expected_sha256": expected_hash,
        "actual_sha256": actual_hash, "reasons": reasons,
    }


_POLICY_KEYS = {
    "schema_version", "allowed_operators", "forbidden_operators",
    "maximum_runtime_tensor_count", "maximum_constant_tensor_count",
    "maximum_arena_head_bytes", "maximum_risk", "maximum_high_critical_findings",
    "forbidden_guidance_categories", "baseline_manifest",
    "maximum_growth_bytes", "maximum_growth_percent", "fail_on_regression",
}


def load_policy(path: str | Path) -> dict[str, object]:
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Unable to load policy: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("Policy must be a mapping")
    unknown = sorted(set(value) - _POLICY_KEYS)
    if unknown:
        raise ValueError(f"Unknown policy keys: {', '.join(unknown)}")
    if value.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("Policy schema_version must be 1")
    for key in ("maximum_runtime_tensor_count", "maximum_constant_tensor_count", "maximum_arena_head_bytes", "maximum_high_critical_findings", "maximum_growth_bytes"):
        if key in value and (not isinstance(value[key], int) or value[key] < 0):
            raise ValueError(f"Policy {key} must be a non-negative integer")
    return value


def evaluate_policy(
    policy: dict[str, object], analysis: dict[str, object], *,
    comparison: dict[str, object] | None = None,
    baseline_result: dict[str, object] | None = None,
) -> dict[str, object]:
    summary = analysis["analysis"]["summary"]  # type: ignore[index]
    guidance = analysis["memory_guidance"]
    operators = set(operator_names(analysis))
    failures: list[dict[str, object]] = []

    def fail(rule_id: str, message: str, actual: object, limit: object) -> None:
        failures.append({"rule_id": rule_id, "message": message, "actual": actual, "limit": limit})

    allowed = set(policy.get("allowed_operators", []))
    forbidden = set(policy.get("forbidden_operators", []))
    if allowed and operators - allowed:
        fail("TS-POLICY-OPERATORS-ALLOWED", "model contains operators outside the allowlist", sorted(operators - allowed), sorted(allowed))
    if operators & forbidden:
        fail("TS-POLICY-OPERATORS-FORBIDDEN", "model contains forbidden operators", sorted(operators & forbidden), sorted(forbidden))
    metric_rules = (
        ("maximum_runtime_tensor_count", "runtime_tensor_count", "TS-POLICY-RUNTIME-TENSORS"),
        ("maximum_constant_tensor_count", "constant_tensor_count", "TS-POLICY-CONSTANT-TENSORS"),
        ("maximum_arena_head_bytes", "planned_arena_head_bytes", "TS-POLICY-ARENA-HEAD"),
    )
    for policy_key, metric_key, rule_id in metric_rules:
        if policy_key in policy and summary[metric_key] > policy[policy_key]:
            fail(rule_id, f"{metric_key} exceeds policy", summary[metric_key], policy[policy_key])
    risk_rank = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    maximum_risk = policy.get("maximum_risk")
    if maximum_risk is not None:
        if maximum_risk not in risk_rank:
            raise ValueError("Policy maximum_risk is invalid")
        if risk_rank[guidance["overall_risk"]] > risk_rank[maximum_risk]:
            fail("TS-POLICY-RISK", "overall risk exceeds policy", guidance["overall_risk"], maximum_risk)
    maximum_high = policy.get("maximum_high_critical_findings")
    actual_high = guidance["summary"]["high_or_critical_count"]
    if maximum_high is not None and actual_high > maximum_high:
        fail("TS-POLICY-HIGH-FINDINGS", "high/critical finding count exceeds policy", actual_high, maximum_high)
    categories = {item["category"] for item in guidance["findings"]}
    forbidden_categories = set(policy.get("forbidden_guidance_categories", []))
    if categories & forbidden_categories:
        fail("TS-POLICY-GUIDANCE", "forbidden guidance categories are present", sorted(categories & forbidden_categories), sorted(forbidden_categories))
    if comparison is not None:
        head = comparison["metrics"]["planned_arena_head_bytes"]
        for key, observed, rule in (
            ("maximum_growth_bytes", head["delta"], "TS-POLICY-GROWTH-BYTES"),
            ("maximum_growth_percent", head["percent_delta"], "TS-POLICY-GROWTH-PERCENT"),
        ):
            if key in policy and observed is not None and observed > policy[key]:
                fail(rule, f"{key} exceeded", observed, policy[key])
        if policy.get("fail_on_regression") and comparison["regression"]["is_regression"]:
            fail("TS-POLICY-REGRESSION", "candidate is a deterministic regression", True, False)
    if baseline_result is not None and baseline_result["status"] == "failed":
        fail("TS-POLICY-BASELINE-DRIFT", "model differs from the selected baseline manifest",
             baseline_result["reasons"], [])
    failures.sort(key=lambda item: item["rule_id"])
    return {"policy_result_schema_version": 1, "status": "failed" if failures else "passed", "failure_count": len(failures), "failures": failures}


def sarif_document(model_path: str | Path, findings: list[dict[str, object]]) -> dict[str, object]:
    ordered = sorted(findings, key=lambda item: (str(item["rule_id"]), str(item.get("message", ""))))
    rules = []
    seen: set[str] = set()
    results = []
    for finding in ordered:
        rule_id = str(finding["rule_id"])
        if rule_id not in seen:
            rules.append({"id": rule_id, "name": rule_id, "shortDescription": {"text": str(finding.get("title", rule_id))}})
            seen.add(rule_id)
        results.append({
            "ruleId": rule_id, "level": finding.get("level", "warning"),
            "message": {"text": str(finding.get("message", rule_id))},
            "locations": [{"physicalLocation": {"artifactLocation": {"uri": Path(model_path).resolve().as_uri()}}}],
            "properties": finding.get("properties", {}),
        })
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": SARIF_VERSION,
        "runs": [{"tool": {"driver": {"name": "TensorScope", "rules": rules}}, "results": results}],
    }


def analysis_sarif(model_path: str | Path, analysis: dict[str, object]) -> dict[str, object]:
    severity = {"critical": "error", "high": "error", "medium": "warning", "low": "note", "info": "note"}
    findings = [
        {"rule_id": f"TS-GUIDANCE-{item['category'].upper().replace('_', '-')}", "title": item["title"],
         "message": item["explanation"], "level": severity[item["severity"]],
         "properties": {"finding_id": item["finding_id"], "tensor_ids": item["affected_tensor_ids"], "operator_ids": item["affected_operator_ids"]}}
        for item in analysis["memory_guidance"]["findings"] if item["severity"] in {"high", "critical"}
    ]
    budget = analysis.get("arena_head_budget")
    if budget and budget["status"] == "exceeds":
        findings.append({"rule_id": "TS-BUDGET-EXCEEDED", "message": "planned arena head exceeds the selected budget", "level": "error", "properties": budget})
    return sarif_document(model_path, findings)


def resolve_models(paths: list[Path], *, recursive: bool) -> tuple[Path, ...]:
    resolved: set[Path] = set()
    for value in paths:
        path = value.expanduser().resolve()
        if path.is_file():
            resolved.add(path)
        elif path.is_dir():
            iterator = path.rglob("*.tflite") if recursive else path.glob("*.tflite")
            resolved.update(item.resolve() for item in iterator if item.is_file())
        else:
            raise ValueError(f"Batch input does not exist: {path}")
    return tuple(sorted(resolved, key=lambda item: str(item)))


def aggregate_csv(rows: list[dict[str, object]]) -> str:
    stream = io.StringIO(newline="")
    fields = ["model", "status", "planned_arena_head_bytes", "overall_risk", "budget_status", "error"]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: row.get(key) for key in fields} for row in rows)
    return stream.getvalue()


_MAP_REGION = re.compile(r"^\s*([A-Za-z_][\w.]*)\s+0x([0-9A-Fa-f]+)\s+0x([0-9A-Fa-f]+)\s+[xrw-]+\s*$")
_MAP_SYMBOL = re.compile(r"^\s*0x([0-9A-Fa-f]+)\s+([A-Za-z_]\w*)\s*$")


def parse_gnu_map(text: str, arena_symbol: str, ram_region: str | None = None) -> dict[str, object]:
    regions: dict[str, tuple[int, int]] = {}
    symbols: dict[str, list[int]] = defaultdict(list)
    for line in text.splitlines():
        region = _MAP_REGION.match(line)
        if region:
            regions[region.group(1)] = (int(region.group(2), 16), int(region.group(3), 16))
        symbol = _MAP_SYMBOL.match(line)
        if symbol:
            symbols[symbol.group(2)].append(int(symbol.group(1), 16))
    if arena_symbol not in symbols:
        raise ValueError(f"Arena symbol not found in GNU ld map: {arena_symbol}")
    if len(symbols[arena_symbol]) != 1:
        raise ValueError(f"Arena symbol is duplicated in GNU ld map: {arena_symbol}")
    selected = ram_region or ("RAM" if "RAM" in regions else None)
    if selected is None or selected not in regions:
        raise ValueError("RAM region was not found; specify --ram-region")
    origin, length = regions[selected]
    address = symbols[arena_symbol][0]
    if not origin <= address < origin + length:
        raise ValueError("Arena symbol address is outside the selected RAM region")
    return {"map_schema_version": 1, "arena_symbol": arena_symbol, "arena_address": address,
            "ram_region": selected, "ram_origin": origin, "ram_length": length}


def deployment_artifacts(
    model_path: str | Path, analysis: dict[str, object], *, margin_percent: int,
) -> dict[str, str]:
    if margin_percent < 0:
        raise ValueError("Margin percent must be non-negative")
    head = analysis["analysis"]["summary"]["planned_arena_head_bytes"]  # type: ignore[index]
    suggested = (head * (100 + margin_percent) + 99) // 100
    digest = sha256_file(model_path)
    schema = analysis["model"]["schema_version"]  # type: ignore[index]
    header = f"""#pragma once
#include <cstddef>
// Planned arena head only; not a complete tensor arena or MCU RAM requirement.
constexpr std::size_t kTensorScopePlannedArenaHeadBytes = {head};
constexpr std::size_t kTensorScopeSuggestedArenaHeadWithMarginBytes = {suggested};
constexpr int kTensorScopeModelSchemaVersion = {schema};
constexpr char kTensorScopeModelSha256[] = "{digest}";
"""
    operators = sorted(set(operator_names(analysis)))
    resolver_lines = "\n".join(f"  // resolver.Add{name.title().replace('_', '')}();" for name in operators)
    resolver = f"""// Review against the pinned TFLM API before enabling.
// Generated operator inventory; comments avoid claiming compile-time resolver compatibility.
void RegisterTensorScopeModelOperators() {{
{resolver_lines}
}}
"""
    manifest = json.dumps({"deployment_schema_version": DEPLOYMENT_SCHEMA_VERSION, "model_sha256": digest,
                           "planned_arena_head_bytes": head, "suggested_arena_head_with_margin_bytes": suggested,
                           "margin_percent": margin_percent, "scope": "arena_head"}, sort_keys=True, indent=2) + "\n"
    return {"tensorscope_deployment.h": header, "tensorscope_operators.cc": resolver, "deployment.json": manifest}
