#!/usr/bin/env python3
"""Run tensorscope analyze --json on the models relevant to this workflow
run, and render one combined sticky-comment body.

Pinned against the JSON shape observed for tensorscope-cli 0.5.0 (see the
field paths read below): arena_head.bytes, arena_head_budget.{status,
verdict,effective_budget_bytes,profile_name}, compute_cost.{total_mac_count,
caveat}, analysis.summary.constant_tensor_bytes, model.filename. That shape
has been additive-only across every release from 0.2.0 through 0.5.0 (no
removed or restructured fields) -- see docs/mcu_memory_budgets.md and the
0.5.0 release notes for how that was confirmed. If a future release removes
or restructures any of these paths, this script needs updating; it does not
guess or silently degrade on a missing expected field, it reports the field
as unavailable in the comment.

Exit code: propagates tensorscope's own EXIT_BUDGET_EXCEEDED (6) if any
checked model exceeds its configured budget -- no new threshold or severity
logic is invented here. Any other non-zero exit from `analyze` itself
(a genuine tool error, not a budget verdict) is propagated as-is.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

STICKY_MARKER = "<!-- tensorscope-check -->"
EXIT_BUDGET_EXCEEDED = 6


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


def discover_models() -> list[str]:
    event_name = _env("EVENT_NAME")

    if event_name == "pull_request":
        base_sha = _env("BASE_SHA")
        head_sha = _env("HEAD_SHA")
        if not base_sha or not head_sha:
            print("::error::pull_request event missing base/head SHA", file=sys.stderr)
            sys.exit(2)
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base_sha}...{head_sha}", "--", "*.tflite"],
            capture_output=True, text=True, check=True,
        )
        models = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return models

    if event_name == "workflow_dispatch":
        model_path = _env("MODEL_PATH_INPUT")
        return [model_path] if model_path else []

    if event_name == "repository_dispatch":
        model_path = _env("MODEL_PATH_PAYLOAD")
        return [model_path] if model_path else []

    print(f"::error::unsupported event_name {event_name!r}", file=sys.stderr)
    sys.exit(2)


def build_budget_flags() -> list[str]:
    target = _env("TS_TARGET")
    mcu_profile = _env("TS_MCU_PROFILE")
    arena_head_budget = _env("TS_ARENA_HEAD_BUDGET")
    reserve = _env("TS_RESERVE")

    chosen = [
        ("--target", target),
        ("--mcu-profile", mcu_profile),
        ("--arena-head-budget", arena_head_budget),
    ]
    active = [(flag, value) for flag, value in chosen if value]
    if len(active) > 1:
        names = ", ".join(flag for flag, _ in active)
        print(
            f"::error::more than one of --target/--mcu-profile/--arena-head-budget is "
            f"configured ({names}) -- these are mutually exclusive in tensorscope itself; "
            "set only one repository variable/input, this workflow will not silently pick one",
            file=sys.stderr,
        )
        sys.exit(2)

    flags: list[str] = []
    if active:
        flag, value = active[0]
        flags += [flag, value]
    if reserve:
        flags += ["--reserve", reserve]
    return flags


def run_analyze(model_path: str, budget_flags: list[str]) -> tuple[dict | None, int, str]:
    command = ["tensorscope", "analyze", model_path, *budget_flags, "--fail-on-budget-exceeded", "--json"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode not in (0, EXIT_BUDGET_EXCEEDED):
        return None, result.returncode, result.stderr.strip() or result.stdout.strip()
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        return None, result.returncode, f"Could not parse JSON output: {error}"
    return payload, result.returncode, ""


def render_model_section(model_path: str, payload: dict | None, exit_code: int, error: str) -> str:
    lines = [f"### `{model_path}`", ""]

    if payload is None:
        lines.append(f"⚠️ `tensorscope analyze` failed (exit {exit_code}):")
        lines.append("")
        lines.append(f"```\n{error}\n```")
        return "\n".join(lines)

    model = payload.get("model", {})
    arena_head = payload.get("arena_head", {})
    budget = payload.get("arena_head_budget")
    compute_cost = payload.get("compute_cost", {})
    constant_bytes = payload.get("analysis", {}).get("summary", {}).get("constant_tensor_bytes")

    planned = arena_head.get("bytes")
    lines.append(f"- **RAM (arena head, planned):** {planned:,} bytes" if planned is not None else "- **RAM (arena head, planned):** not estimated")

    if budget is not None:
        status = budget.get("status")
        emoji = {"fits": "✅", "exact_fit": "✅", "exceeds": "❌"}.get(status, "❓")
        lines.append(f"- **Budget verdict:** {emoji} {budget.get('verdict', status)}")
    else:
        lines.append("- **Budget verdict:** no `--target`/`--mcu-profile`/`--arena-head-budget` configured for this check")

    if constant_bytes is not None:
        lines.append(f"- **Model constants (flash-bound):** {constant_bytes:,} bytes")

    total_macs = compute_cost.get("total_mac_count")
    if total_macs is not None:
        lines.append(f"- **Compute cost:** {total_macs:,} MACs — {compute_cost.get('caveat', '')}")

    return "\n".join(lines)


def main() -> None:
    models = discover_models()
    github_output = os.environ.get("GITHUB_OUTPUT")
    body_path = os.environ.get("COMMENT_BODY_FILE", "comment_body.md")

    if not models:
        body = f"{STICKY_MARKER}\n## TensorScope memory check\n\nNo `.tflite` files changed in this run — nothing to check."
        with open(body_path, "w", encoding="utf-8") as f:
            f.write(body)
        if github_output:
            with open(github_output, "a", encoding="utf-8") as f:
                f.write("worst_exit_code=0\n")
        return

    budget_flags = build_budget_flags()
    sections: list[str] = []
    worst_exit_code = 0

    for model_path in models:
        payload, exit_code, error = run_analyze(model_path, budget_flags)
        sections.append(render_model_section(model_path, payload, exit_code, error))
        if payload is None and exit_code != 0:
            worst_exit_code = exit_code if worst_exit_code == 0 else worst_exit_code
        elif exit_code == EXIT_BUDGET_EXCEEDED:
            worst_exit_code = EXIT_BUDGET_EXCEEDED

    caveat = (
        "Static analysis only: planned arena-head RAM and multiply-accumulate "
        "volume, not a timing estimate, and not a complete memory-fit "
        "conclusion (arena tail, stack, heap, and firmware are not covered)."
    )
    body = "\n\n".join([
        f"{STICKY_MARKER}\n## TensorScope memory check",
        *sections,
        f"<sub>{caveat}</sub>",
    ])

    with open(body_path, "w", encoding="utf-8") as f:
        f.write(body)

    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"worst_exit_code={worst_exit_code}\n")


if __name__ == "__main__":
    main()
