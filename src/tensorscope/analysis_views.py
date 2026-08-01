from __future__ import annotations

from tensorscope.explain import MemoryExplanation
from tensorscope.graph import GraphModel


def build_analysis_views(graph: GraphModel, explanation: MemoryExplanation, *, limit: int = 80) -> dict[str, object]:
    allocations = {item.tensor_id: item for item in explanation.allocations}
    scopes = {item.scope: item for item in explanation.scopes}
    peak_scopes = set(explanation.peak.tied_scopes)
    blockers: dict[int, list[int]] = {}
    for blocker in explanation.reuse_blockers:
        for operator_id in ([blocker.last_consumer_operator_id] if blocker.last_consumer_operator_id is not None else []):
            blockers.setdefault(operator_id, []).append(blocker.tensor_id)
    operators = []
    for operator in graph.primary_subgraph.operators:
        scope = scopes[operator.id + 1]
        input_bytes = sum(allocations[item].aligned_bytes for item in operator.inputs if item in allocations)
        output_tensors = [allocations[item] for item in operator.outputs if item in allocations]
        output_bytes = sum(item.aligned_bytes for item in output_tensors)
        retained = sum(item.aligned_bytes for item in output_tensors if item.last_used_scope > operator.id + 1)
        pressure = "high" if scope.live_aligned_bytes * 4 >= explanation.summary.planned_arena_head_bytes * 3 else "medium" if scope.live_aligned_bytes * 2 >= explanation.summary.planned_arena_head_bytes else "low"
        operators.append({
            "operator_id": operator.id, "operator_name": operator.name,
            "represented_input_aligned_bytes": input_bytes,
            "represented_output_aligned_bytes": output_bytes,
            "live_aligned_bytes_at_scope": scope.live_aligned_bytes,
            "occupied_extent_bytes_at_scope": scope.occupied_extent_bytes,
            "retained_output_aligned_bytes": retained,
            "retained_output_lifetimes": [[item.tensor_id, item.first_used_scope, item.last_used_scope] for item in output_tensors if item.last_used_scope > operator.id + 1],
            "selected_peak": operator.id == explanation.peak.operator_id,
            "tied_peak": operator.id + 1 in peak_scopes,
            "blocker_associated_tensor_ids": sorted(blockers.get(operator.id, [])),
            "scratch_observation": {"availability": "unavailable", "source": "pinned_tflm_api_audit"},
            "pressure": pressure,
        })
    operators.sort(key=lambda item: (-item["live_aligned_bytes_at_scope"], item["operator_id"]))
    timeline = [
        {"scope": item.scope, "operator_id": item.operator_id, "operator_name": item.operator_name,
         "live_aligned_bytes": item.live_aligned_bytes, "occupied_extent_bytes": item.occupied_extent_bytes,
         "selected_peak": item.scope == explanation.peak.scope, "tied_peak": item.scope in peak_scopes}
        for item in explanation.scopes[: limit + 1]
    ]
    graph_nodes = [
        {"operator_id": item.id, "operator_name": item.name,
         "input_tensor_ids": list(item.inputs), "output_tensor_ids": list(item.outputs)}
        for item in graph.primary_subgraph.operators[:limit]
    ]
    tensors = [
        {"tensor_id": item.tensor_id, "name": item.name, "data_type": item.data_type,
         "shape": list(item.shape), "aligned_bytes": item.aligned_bytes,
         "is_graph_input": item.is_graph_input, "is_graph_output": item.is_graph_output,
         "live_at_selected_peak": item.tensor_id in explanation.peak.live_tensor_ids}
        for item in explanation.allocations[:limit]
    ]
    return {
        "schema_version": 1, "scope": "primary_subgraph_arena_head",
        "operator_attribution": {"non_additive": True, "operators": operators},
        "execution_timeline": {"truncated": len(explanation.scopes) > limit + 1, "scopes": timeline},
        "graph_view": {"truncated": len(graph.primary_subgraph.operators) > limit, "operators": graph_nodes, "runtime_tensors": tensors},
        "limitations": ["operator values are represented live-set metrics and are not independently additive contributions", "scratch observation is unavailable"],
    }


def build_model_diagnostics(graph: GraphModel) -> dict[str, object]:
    subgraphs = []
    variables = []
    unknown_shapes = []
    control_flow = []
    for subgraph in graph.subgraphs:
        for tensor in subgraph.tensors:
            if tensor.is_variable:
                variables.append({"subgraph_id": subgraph.id, "tensor_id": tensor.id, "name": tensor.name})
            if any(item < 0 for item in tensor.shape_signature) or any(item < 0 for item in tensor.shape):
                unknown_shapes.append({"subgraph_id": subgraph.id, "tensor_id": tensor.id, "shape": list(tensor.shape), "shape_signature": list(tensor.shape_signature)})
        controls = [item for item in subgraph.operators if item.name in {"IF", "WHILE", "CALL_ONCE"}]
        control_flow.extend({"subgraph_id": subgraph.id, "operator_id": item.id, "operator_name": item.name} for item in controls)
        subgraphs.append({"subgraph_id": subgraph.id, "name": subgraph.name, "tensor_count": len(subgraph.tensors),
                          "operator_count": len(subgraph.operators), "input_tensor_ids": list(subgraph.inputs),
                          "output_tensor_ids": list(subgraph.outputs), "control_flow_operator_count": len(controls)})
    partial = len(graph.subgraphs) > 1 or bool(control_flow or variables or unknown_shapes)
    return {
        "schema_version": 1, "subgraph_count": len(graph.subgraphs), "subgraphs": subgraphs,
        "overall_combination_status": "not_combined" if len(graph.subgraphs) > 1 else "single_subgraph",
        "analysis_confidence": "partial" if partial else "exact_for_supported_static_semantics",
        "control_flow_operators": control_flow, "variable_tensors": variables,
        "unknown_or_dynamic_shapes": unknown_shapes,
        "persistent_buffer_estimate": None,
        "diagnostics": (["subgraph peaks are not added; execution relationships are not modeled"] if len(graph.subgraphs) > 1 else [])
                       + (["mutable variable allocation semantics are not statically estimated"] if variables else [])
                       + (["dynamic shapes require concrete allocation semantics"] if unknown_shapes else []),
    }
