"""Graph serialization and replay executor.

Provides:
- ``export_graph_json`` / ``import_graph_json`` for JSON round-trip
- ``replay_graph`` for rebuilding a model from a recorded graph

Usage::

    from simplecadapi.serializer import export_graph_json, import_graph_json, replay_graph

    # Serialize
    json_str = export_graph_json(session.graph)

    # Deserialize
    graph = import_graph_json(json_str)

    # Rebuild
    solids = replay_graph(graph)
"""

from __future__ import annotations

import math

from typing import Any, Dict, List, Optional, Sequence, Tuple, cast
import warnings

from .errors import raise_harness_error

from .core import AnyShape, Edge, Face, Solid
from .field import deserialize_scalar_field
from .graph import attach_graph_node, suspend_graph_recording
from .ql import selector_from_dict
from .topology import (
    OperationGraph,
    semantic_delta_to_dict,
    topo_delta_to_dict,
    topo_ref_from_dict,
)
from . import operations as ops
from .selection_signature import match_selected_subshape_items, selected_subshape_items


PUBLIC_API_COVERAGE: Dict[str, Dict[str, str]] = {
    # Core geometry ops that are recorded and replayable
    "make_point_rvertex": {"status": "replayable", "op": "make_point_rvertex"},
    "make_line_redge": {"status": "replayable", "op": "make_line_redge"},
    "make_segment_redge": {"status": "expanded_macro", "op": "make_line_redge"},
    "make_segment_rwire": {
        "status": "macro",
        "reason": "Composite convenience API that should lower into make_line_redge + make_wire_from_edges_rwire.",
    },
    "make_circle_redge": {"status": "replayable", "op": "make_circle_redge"},
    "make_circle_rwire": {
        "status": "macro",
        "reason": "Composite convenience API that should lower into make_circle_redge + make_wire_from_edges_rwire.",
    },
    "make_circle_rface": {
        "status": "macro",
        "reason": "Composite convenience API that should lower into edge/wire/face low-level operations.",
    },
    "make_rectangle_rwire": {
        "status": "macro",
        "reason": "Composite convenience API that should lower into make_line_redge + make_wire_from_edges_rwire.",
    },
    "make_rectangle_rface": {
        "status": "macro",
        "reason": "Composite convenience API that should lower into low-level line/wire/face operations.",
    },
    "make_face_from_wire_rface": {"status": "replayable", "op": "make_face_from_wire"},
    "make_wire_from_edges_rwire": {
        "status": "replayable",
        "op": "make_wire_from_edges_rwire",
    },
    "make_box_rsolid": {
        "status": "macro",
        "reason": "Composite convenience API that should lower into low-level sketch + make_extrude_rsolid operations.",
    },
    "make_cylinder_rsolid": {
        "status": "macro",
        "reason": "Composite convenience API that should lower into low-level circle/face + make_extrude_rsolid operations.",
    },
    "make_cone_rsolid": {
        "status": "macro",
        "reason": "Composite convenience API that should lower into low-level profile + make_revolve_rsolid operations.",
    },
    "make_sphere_rsolid": {
        "status": "macro",
        "reason": "Composite convenience API that should lower into low-level profile + make_revolve_rsolid operations.",
    },
    "make_three_point_arc_redge": {
        "status": "replayable",
        "op": "make_three_point_arc_redge",
    },
    "make_three_point_arc_rwire": {
        "status": "macro",
        "reason": "Composite convenience API that should lower into make_three_point_arc_redge + make_wire_from_edges_rwire.",
    },
    "make_angle_arc_redge": {"status": "replayable", "op": "make_angle_arc_redge"},
    "make_angle_arc_rwire": {
        "status": "macro",
        "reason": "Composite convenience API that should lower into make_angle_arc_redge + make_wire_from_edges_rwire.",
    },
    "make_spline_redge": {"status": "replayable", "op": "make_spline_redge"},
    "make_spline_rwire": {
        "status": "macro",
        "reason": "Composite convenience API that should lower into make_spline_redge + make_wire_from_edges_rwire when open.",
    },
    "make_polyline_rwire": {
        "status": "macro",
        "reason": "Composite convenience API that should lower into make_line_redge + make_wire_from_edges_rwire.",
    },
    "make_helix_redge": {"status": "replayable", "op": "make_helix_redge"},
    "make_helix_rwire": {
        "status": "macro",
        "reason": "Composite convenience API that should lower into make_helix_redge + make_wire_from_edges_rwire.",
    },
    "translate_shape": {"status": "replayable", "op": "make_translate_rshape"},
    "rotate_shape": {"status": "replayable", "op": "make_rotate_rshape"},
    "mirror_shape": {"status": "replayable", "op": "make_mirror_rshape"},
    "extrude_rsolid": {"status": "replayable", "op": "make_extrude_rsolid"},
    "revolve_rsolid": {"status": "replayable", "op": "make_revolve_rsolid"},
    "loft_rsolid": {"status": "replayable", "op": "make_loft_rsolid"},
    "sweep_rsolid": {"status": "replayable", "op": "make_sweep_rsolid"},
    "helical_sweep_rsolid": {
        "status": "expanded_macro",
        "op": "make_sweep_rsolid",
        "reason": "Recorded as make_helix_wire + sweep macro instead of a dedicated core IR node.",
    },
    "union_rsolid": {"status": "replayable", "op": "make_union_rsolid"},
    "cut_rsolidlist": {"status": "replayable", "op": "make_cut_rsolidlist"},
    "intersect_rsolidlist": {"status": "replayable", "op": "make_intersect_rsolidlist"},
    "fillet_rsolid": {"status": "replayable", "op": "make_fillet_rsolid"},
    "chamfer_rsolid": {"status": "replayable", "op": "make_chamfer_rsolid"},
    "shell_rsolid": {"status": "replayable", "op": "make_shell_rsolid"},
    "linear_pattern_rsolidlist": {
        "status": "macro",
        "reason": "Pattern convenience API that should lower into repeated make_translate_rshape nodes.",
    },
    "radial_pattern_rsolidlist": {
        "status": "macro",
        "reason": "Pattern convenience API that should lower into repeated make_rotate_rshape nodes.",
    },
    # Explicit gaps / separate systems
    "make_field_surface_rsolid": {
        "status": "replayable",
        "op": "make_field_surface_rsolid",
        "reason": "Replay is supported for ScalarField trees; arbitrary Python callable fields remain opaque and non-replayable.",
    },
    "make_n_hole_flange_rsolid": {
        "status": "macro",
        "reason": "Expanded evolve macro is not serialized as a stable user-level node yet.",
    },
    "make_naca_propeller_blade_rsolid": {
        "status": "macro",
        "reason": "Expanded evolve macro is not serialized as a stable user-level node yet.",
    },
    "make_threaded_rod_rsolid": {
        "status": "macro",
        "reason": "Expanded evolve macro is not serialized as a stable user-level node yet.",
    },
    "Assembly": {
        "status": "separate_system",
        "reason": "Assembly constraints require a dedicated pose/constraint graph.",
    },
    "make_assembly_rassembly": {
        "status": "separate_system",
        "reason": "Assembly constraints require a dedicated pose/constraint graph.",
    },
}


CANONICAL_CORE_OP_SET: Tuple[str, ...] = (
    "make_point_rvertex",
    "make_line_redge",
    "make_circle_redge",
    "make_three_point_arc_redge",
    "make_angle_arc_redge",
    "make_spline_redge",
    "make_helix_redge",
    "make_wire_from_edges_rwire",
    "make_face_from_wire_rface",
    "make_field_surface_rsolid",
    "make_extrude_rsolid",
    "make_revolve_rsolid",
    "make_loft_rsolid",
    "make_sweep_rsolid",
    "make_translate_rshape",
    "make_rotate_rshape",
    "make_mirror_rshape",
    "make_cut_rsolidlist",
    "make_union_rsolid",
    "make_intersect_rsolidlist",
    "make_fillet_rsolid",
    "make_chamfer_rsolid",
    "make_shell_rsolid",
)

SELECTION_REF_SCHEMA: Dict[str, Any] = {
    "selection_param": "selected_subshapes",
    "edge_param": "selected_edges",
    "face_param": "selected_faces",
    "edge_index_param": "selected_edge_indices",
    "face_index_param": "selected_face_indices",
    "required_topo_ref_fields": [
        "graph_id",
        "node_id",
        "output_slot",
        "kind",
        "topo_id",
    ],
    "required_item_fields": [
        "item_id",
        "kind",
        "source",
        "geometry_signature",
        "original_selection_method",
        "order",
    ],
    "optional_fields": ["selector_hint", "selection_query_debug"],
    "replay_resolution_order": [
        "geometry_signature",
        "explicit_topo_refs",
        "legacy_index_fallback",
        "legacy_selection_query",
        "legacy_selector_hint",
    ],
}


def _canonical_contract_payload() -> Dict[str, Any]:
    return {
        "contract_version": "2.0-final-state",
        "graph_roles": {
            "graph": "canonical_low_level_graph",
            "leaf_ids": "explicit_result_set",
        },
        "replay_policy": {
            "preferred_graph": "graph",
        },
        "core_op_set": list(CANONICAL_CORE_OP_SET),
        "selection_ref_schema": {
            "selection_param": SELECTION_REF_SCHEMA["selection_param"],
            "edge_param": SELECTION_REF_SCHEMA["edge_param"],
            "face_param": SELECTION_REF_SCHEMA["face_param"],
            "edge_index_param": SELECTION_REF_SCHEMA["edge_index_param"],
            "face_index_param": SELECTION_REF_SCHEMA["face_index_param"],
            "required_topo_ref_fields": list(
                SELECTION_REF_SCHEMA["required_topo_ref_fields"]
            ),
            "optional_fields": list(SELECTION_REF_SCHEMA["optional_fields"]),
            "required_item_fields": list(SELECTION_REF_SCHEMA["required_item_fields"]),
            "replay_resolution_order": list(
                SELECTION_REF_SCHEMA["replay_resolution_order"]
            ),
        },
    }


def _assert_graph_is_canonical(graph: OperationGraph) -> None:
    invalid_ops = sorted(
        {node.op for node in graph.nodes if node.op not in CANONICAL_CORE_OP_SET}
    )
    if invalid_ops:
        raise ValueError(
            "graph contains non-canonical operations: " + ", ".join(invalid_ops)
        )


def _as_vec3_tuple(value: Any) -> Tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Expected a 3D vector-like value")
    return (float(value[0]), float(value[1]), float(value[2]))


def _replay_make_field_surface(params: Dict[str, Any]) -> Solid:
    mode = str(params.get("field_serialization_mode", "opaque_callable"))
    if mode != "scalar_field":
        raise ValueError(
            "Cannot replay make_field_surface_rsolid from graph JSON when the original field was an opaque Python callable. "
            "Use a ScalarField tree for replayable field surfaces."
        )

    field_tree = params.get("field_tree")
    if not isinstance(field_tree, dict):
        raise ValueError("Serialized field_tree is missing or invalid")

    bounds_payload = params.get("bounds") or {}
    bounds = (
        _as_vec3_tuple(bounds_payload.get("min", (0.0, 0.0, 0.0))),
        _as_vec3_tuple(bounds_payload.get("max", (0.0, 0.0, 0.0))),
    )
    resolution = tuple(int(v) for v in params.get("resolution", (24, 24, 24)))
    iso = float(params.get("iso", 0.0))
    cap_bounds = bool(params.get("cap_bounds", True))

    field = deserialize_scalar_field(field_tree)
    return ops.make_field_surface_rsolid(
        field,
        bounds=bounds,
        resolution=cast(Any, resolution),
        iso=iso,
        cap_bounds=cap_bounds,
    )


# ---------------------------------------------------------------------------
# Export / Import
# ---------------------------------------------------------------------------


def export_graph_json(graph: OperationGraph, indent: int = 2) -> str:
    """Export an OperationGraph to a JSON string.

    Args:
        graph: The graph to export.
        indent: JSON indentation level.

    Returns:
        JSON string representation.
    """
    return graph.to_json(indent=indent)


def export_session_json(session: "GraphSession", indent: int = 2) -> str:
    """Export a graph session including its expression graph."""

    import json

    return json.dumps(
        {
            "graph": session.graph.to_dict(),
            "expression_graph": session.expression_graph.to_dict(),
            "frame_graph": session.frame_graph.to_dict(),
        },
        indent=indent,
    )


def import_graph_json(json_str: str) -> OperationGraph:
    """Import an OperationGraph from a JSON string.

    Args:
        json_str: JSON string to parse.

    Returns:
        Reconstructed OperationGraph.
    """
    import json

    try:
        payload = json.loads(json_str)
        schema_version = str(payload.get("schema_version", "1.0"))
        if not schema_version.startswith("1."):
            raise ValueError(
                f"Unsupported graph schema_version '{schema_version}'. Expected 1.x."
            )
        return OperationGraph.from_dict(payload)
    except Exception as e:
        raise_harness_error(
            operation="import_graph_json",
            what_happened="Failed to import the graph JSON payload.",
            possible_causes=[
                "The input string is not valid JSON.",
                "The payload does not follow the expected graph schema.",
                "The graph schema_version is unsupported.",
            ],
            how_to_fix=[
                "Pass a valid JSON string produced by export_graph_json().",
                "Make sure the payload includes a 1.x graph schema_version.",
                "If you edited the payload manually, validate the nodes and edges structure before retrying.",
            ],
            error=e,
        )


def import_session_json(json_str: str) -> Dict[str, Any]:
    """Import session payload containing graph and expression graph."""

    import json

    from .expr import ExpressionGraph
    from .frame import FrameGraph

    try:
        payload = json.loads(json_str)
        graph_payload = payload.get("graph")
        if not isinstance(graph_payload, dict):
            raise ValueError("Session payload is missing 'graph'")

        expr_payload = payload.get("expression_graph")
        if expr_payload is None:
            expr_graph = ExpressionGraph()
        elif isinstance(expr_payload, dict):
            expr_graph = ExpressionGraph.from_dict(expr_payload)
        else:
            raise ValueError("Session payload 'expression_graph' must be an object")

        frame_payload = payload.get("frame_graph")
        if frame_payload is None:
            frame_graph = FrameGraph()
        elif isinstance(frame_payload, dict):
            frame_graph = FrameGraph.from_dict(frame_payload)
        else:
            raise ValueError("Session payload 'frame_graph' must be an object")

        return {
            "graph": OperationGraph.from_dict(graph_payload),
            "expression_graph": expr_graph,
            "frame_graph": frame_graph,
        }
    except Exception as e:
        raise_harness_error(
            operation="import_session_json",
            what_happened="Failed to import the session JSON payload.",
            possible_causes=[
                "The input string is not valid JSON.",
                "The session payload is missing the required 'graph' object.",
                "The expression_graph or frame_graph fields use the wrong JSON type.",
            ],
            how_to_fix=[
                "Pass a valid JSON string produced by export_session_json().",
                "Make sure 'graph' is present and is a JSON object.",
                "Use JSON objects for 'expression_graph' and 'frame_graph', not strings or arrays.",
            ],
            error=e,
        )


def export_model_json(
    session: "GraphSession",
    indent: int = 2,
    assembly: Any | None = None,
) -> str:
    """Export the canonical 2.0 model seed JSON.

    Current Phase 1 scope uses the active session as the container of:
    - operation graph
    - expression graph
    - capabilities/schema metadata
    """

    import json

    try:
        geometry_registry: List[Dict[str, Any]] = []
        semantic_entity_registry: List[Dict[str, Any]] = []
        sketch_profile_registry: List[Dict[str, Any]] = []
        semantic_delta_log: List[Dict[str, Any]] = []
        topology_delta_log: List[Dict[str, Any]] = []
        assembly_registry: List[Dict[str, Any]] = []
        constraint_registry: List[Dict[str, Any]] = []

        for node in session.graph.topological_order():
            if node.semantic_delta is not None:
                semantic_delta_log.append(
                    {
                        "node_id": node.node_id,
                        "op": node.op,
                        "delta": semantic_delta_to_dict(node.semantic_delta),
                    }
                )
                for ref in node.semantic_delta.created:
                    geometry_registry.append(
                        {
                            "graph_id": ref.graph_id,
                            "node_id": ref.node_id,
                            "entity_type": ref.entity_type,
                            "entity_id": ref.entity_id,
                            "source_op": node.op,
                        }
                    )
                    semantic_entity_registry.append(
                        {
                            "graph_id": ref.graph_id,
                            "node_id": ref.node_id,
                            "entity_type": ref.entity_type,
                            "entity_id": ref.entity_id,
                            "source_op": node.op,
                        }
                    )
            else:
                for slot in range(node.output_count):
                    geometry_registry.append(
                        {
                            "graph_id": session.graph.graph_id,
                            "node_id": node.node_id,
                            "entity_type": "ShapeOutput",
                            "entity_id": f"{node.op}:{slot}",
                            "source_op": node.op,
                        }
                    )

            if node.topo_delta is not None:
                topology_delta_log.append(
                    {
                        "node_id": node.node_id,
                        "op": node.op,
                        "delta": topo_delta_to_dict(node.topo_delta),
                    }
                )

            if node.op in {
                "make_point",
                "make_line",
                "make_segment_wire",
                "make_circle_edge",
                "make_circle_wire",
                "make_circle_face",
                "make_rectangle_wire",
                "make_rectangle_face",
                "make_three_point_arc",
                "make_three_point_arc_wire",
                "make_angle_arc",
                "make_angle_arc_wire",
                "make_spline",
                "make_spline_wire",
                "make_polyline_wire",
                "make_helix",
                "make_helix_wire",
                "make_wire_from_edges",
                "make_face_from_wire",
                "make_point_rvertex",
                "make_line_redge",
                "make_circle_redge",
                "make_three_point_arc_redge",
                "make_angle_arc_redge",
                "make_spline_redge",
                "make_helix_redge",
                "make_wire_from_edges_rwire",
                "make_face_from_wire_rface",
            }:
                sketch_profile_registry.append(
                    {
                        "graph_id": session.graph.graph_id,
                        "node_id": node.node_id,
                        "op": node.op,
                        "params": dict(node.params),
                    }
                )

        assembly_payload: Any | None = None
        assembly_frame_nodes: List[Dict[str, Any]] = []
        if assembly is not None:
            if hasattr(assembly, "to_dict"):
                assembly_payload = assembly.to_dict()
            else:
                raise ValueError("assembly must provide a to_dict() method")

        if isinstance(assembly_payload, dict):
            parts = assembly_payload.get("parts", [])
            if isinstance(parts, list):
                assembly_frame_nodes.append(
                    {
                        "frame_id": f"assembly:{assembly_payload.get('name', 'assembly')}",
                        "origin": (0.0, 0.0, 0.0),
                        "x_axis": (1.0, 0.0, 0.0),
                        "y_axis": (0.0, 1.0, 0.0),
                        "z_axis": (0.0, 0.0, 1.0),
                        "parent_frame_id": None,
                        "metadata": {
                            "assembly": assembly_payload.get("name", "assembly")
                        },
                    }
                )
                assembly_registry.append(
                    {
                        "name": assembly_payload.get("name", "assembly"),
                        "part_count": len(parts),
                    }
                )
                for part in parts:
                    if isinstance(part, dict):
                        local_transform = part.get("local_transform") or [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                        tx = float(local_transform[0][3])
                        ty = float(local_transform[1][3])
                        tz = float(local_transform[2][3])
                        assembly_frame_nodes.append(
                            {
                                "frame_id": f"assembly:{assembly_payload.get('name', 'assembly')}:part:{part.get('name')}",
                                "origin": (tx, ty, tz),
                                "x_axis": (1.0, 0.0, 0.0),
                                "y_axis": (0.0, 1.0, 0.0),
                                "z_axis": (0.0, 0.0, 1.0),
                                "parent_frame_id": (
                                    f"assembly:{assembly_payload.get('name', 'assembly')}:part:{part.get('parent')}"
                                    if part.get("parent")
                                    else f"assembly:{assembly_payload.get('name', 'assembly')}"
                                ),
                                "metadata": {
                                    "assembly": assembly_payload.get(
                                        "name", "assembly"
                                    ),
                                    "part": part.get("name"),
                                },
                            }
                        )
                        assembly_registry.append(
                            {
                                "assembly": assembly_payload.get("name", "assembly"),
                                "part": part.get("name"),
                                "parent": part.get("parent"),
                            }
                        )
            constraints = assembly_payload.get("constraint_param_exprs", [])
            if isinstance(constraints, list):
                for idx, constraint in enumerate(constraints):
                    if isinstance(constraint, dict):
                        constraint_registry.append(
                            {
                                "assembly": assembly_payload.get("name", "assembly"),
                                "constraint_index": idx,
                                **constraint,
                            }
                        )
                        semantic_entity_registry.append(
                            {
                                "graph_id": session.graph.graph_id,
                                "node_id": f"assembly-constraint:{idx}",
                                "entity_type": "AssemblyConstraint",
                                "entity_id": f"constraint:{idx}",
                                "source_op": constraint.get("type", "constraint"),
                            }
                        )

        frame_graph_payload = session.frame_graph.to_dict()
        if assembly_frame_nodes:
            frame_graph_payload = dict(frame_graph_payload)
            frame_graph_payload["nodes"] = (
                list(frame_graph_payload.get("nodes", [])) + assembly_frame_nodes
            )

        _assert_graph_is_canonical(session.graph)
        leaf_ids = [node.node_id for node in session.graph.leaf_nodes()]

        payload: Dict[str, Any] = {
            "schema_version": "2.0-draft",
            "canonical_contract": _canonical_contract_payload(),
            "graph": session.graph.to_dict(),
            "leaf_ids": leaf_ids,
            "expression_graph": session.expression_graph.to_dict(),
            "frame_graph": frame_graph_payload,
            "geometry_registry": geometry_registry,
            "semantic_entity_registry": semantic_entity_registry,
            "sketch_profile_registry": sketch_profile_registry,
            "assembly_registry": assembly_registry,
            "constraint_registry": constraint_registry,
            "semantic_delta_log": semantic_delta_log,
            "topology_delta_log": topology_delta_log,
        }
        if assembly_payload is not None:
            payload["assembly"] = assembly_payload

        return json.dumps(payload, indent=indent)
    except Exception as e:
        raise_harness_error(
            operation="export_model_json",
            what_happened="Failed to export the canonical model JSON payload.",
            possible_causes=[
                "The session contains non-serializable graph, expression, or frame data.",
                "The optional assembly object does not expose a compatible to_dict() method.",
                "The graph contains non-canonical operations instead of the strict low-level op set.",
            ],
            how_to_fix=[
                "Pass a valid GraphSession object built by SimpleCADAPI.",
                "If you pass assembly=..., make sure it provides a to_dict() method returning JSON-compatible data.",
                "Make sure composite builtins only emit strict low-level graph nodes before exporting model JSON.",
            ],
            error=e,
        )


def import_model_json(json_str: str) -> Dict[str, Any]:
    """Import canonical 2.0 model seed JSON."""

    import json

    try:
        payload = json.loads(json_str)
        schema_version = str(payload.get("schema_version", ""))
        if not schema_version.startswith("2.0"):
            raise ValueError(
                "Unsupported model schema_version; expected 2.0 draft payload"
            )

        session_payload = import_session_json(
            json.dumps(
                {
                    "graph": payload.get("graph", {}),
                    "expression_graph": payload.get("expression_graph", {}),
                    "frame_graph": payload.get("frame_graph", {}),
                }
            )
        )
        session_payload["geometry_registry"] = list(
            payload.get("geometry_registry", [])
        )
        session_payload["canonical_contract"] = dict(
            payload.get("canonical_contract", _canonical_contract_payload())
        )
        session_payload["semantic_entity_registry"] = list(
            payload.get("semantic_entity_registry", [])
        )
        session_payload["sketch_profile_registry"] = list(
            payload.get("sketch_profile_registry", [])
        )
        session_payload["assembly_registry"] = list(
            payload.get("assembly_registry", [])
        )
        session_payload["constraint_registry"] = list(
            payload.get("constraint_registry", [])
        )
        session_payload["semantic_delta_log"] = list(
            payload.get("semantic_delta_log", [])
        )
        session_payload["topology_delta_log"] = list(
            payload.get("topology_delta_log", [])
        )
        session_payload["leaf_ids"] = [str(v) for v in payload.get("leaf_ids", [])]
        if "assembly" in payload:
            session_payload["assembly"] = payload["assembly"]
        return session_payload
    except Exception as e:
        raise_harness_error(
            operation="import_model_json",
            what_happened="Failed to import the canonical model JSON payload.",
            possible_causes=[
                "The input string is not valid JSON.",
                "The payload does not use the expected 2.0 model schema_version.",
                "One or more nested graph payloads are malformed.",
            ],
            how_to_fix=[
                "Pass a valid JSON string produced by export_model_json().",
                "Make sure schema_version starts with 2.0.",
                "If you edited the payload manually, validate graph, expression_graph, and frame_graph fields before retrying.",
            ],
            error=e,
        )


def replay_model_json(json_str: str) -> List[AnyShape]:
    """Replay a model payload using its canonical low-level graph."""

    try:
        payload = import_model_json(json_str)
        graph = payload.get("graph")
        if not isinstance(graph, OperationGraph):
            raise ValueError("Model payload does not contain a replayable graph")

        explicit_leaf_ids = payload.get("leaf_ids")
        return _execute_graph(graph, cast(Optional[Sequence[str]], explicit_leaf_ids))
    except Exception as e:
        raise_harness_error(
            operation="replay_model_json",
            what_happened="Failed to replay the model JSON payload.",
            possible_causes=[
                "The model payload is malformed or missing a replayable graph.",
                "The graph contains an unsupported or invalid node payload.",
                "One of the replayed operations failed due to invalid parameters or missing references.",
            ],
            how_to_fix=[
                "Start from export_model_json() output instead of hand-written payloads when possible.",
                "Make sure the model includes a valid canonical low-level graph section.",
                "If replay fails on a specific operation, inspect that node's params and compare them to the operation signature and help() output.",
            ],
            error=e,
        )


# ---------------------------------------------------------------------------
# Replay executor
# ---------------------------------------------------------------------------

# Registry mapping op names to factory functions.
# Each factory takes (params_dict) -> shape or list of shapes.
_OP_REGISTRY: Dict[str, Any] = {
    "make_point_rvertex": lambda p: ops.make_point_rvertex(
        p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)
    ),
    "make_line_redge": lambda p: ops.make_line_redge(
        tuple(p.get("start", (0, 0, 0))), tuple(p.get("end", (0, 0, 0)))
    ),
    "make_field_surface_rsolid": _replay_make_field_surface,
    "make_circle_redge": lambda p: ops.make_circle_redge(
        tuple(p.get("center", (0, 0, 0))),
        p.get("radius", 1.0),
        tuple(p.get("normal", (0, 0, 1))),
    ),
    "make_three_point_arc_redge": lambda p: ops.make_three_point_arc_redge(
        tuple(p.get("start", (0, 0, 0))),
        tuple(p.get("middle", (0, 0, 0))),
        tuple(p.get("end", (0, 0, 0))),
    ),
    "make_angle_arc_redge": lambda p: ops.make_angle_arc_redge(
        tuple(p.get("center", (0, 0, 0))),
        p.get("radius", 1.0),
        p.get("start_angle", 0.0),
        p.get("end_angle", 1.0),
        tuple(p.get("normal", (0, 0, 1))),
    ),
    "make_spline_redge": lambda p: ops.make_spline_redge(
        p.get("points", []), tangents=p.get("tangents")
    ),
    "make_helix_redge": lambda p: ops.make_helix_redge(
        p.get("pitch", 1.0),
        p.get("height", 1.0),
        p.get("radius", 1.0),
        center=tuple(p.get("center", (0, 0, 0))),
        dir=tuple(p.get("dir", (0, 0, 1))),
    ),
    "make_cut_rsolidlist": lambda p: None,  # handled specially below
    "make_union_rsolid": lambda p: None,  # handled specially below
    "make_intersect_rsolidlist": lambda p: None,  # handled specially below
}


def _normalize_output(result: Any) -> List[AnyShape]:
    if result is None:
        return []
    if isinstance(result, list):
        return result
    return [result]


def _shape_topo_ref_dict(shape: AnyShape) -> Dict[str, Any]:
    topo_ref = shape.get_metadata("topo_ref")
    return topo_ref if isinstance(topo_ref, dict) else {}


def _distance3(
    a: Optional[Tuple[float, float, float]], b: Optional[Tuple[float, float, float]]
) -> float:
    if a is None or b is None:
        return 1e6
    return math.dist(a, b)


def _tuple3_from_any(value: Any) -> Optional[Tuple[float, float, float]]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return None


def _edge_hint_score(edge: Edge, hint: Dict[str, Any]) -> float:
    score = 0.0
    if "length" in hint:
        score += abs(float(edge.get_length()) - float(hint["length"])) * 10.0

    start: Optional[Tuple[float, float, float]] = None
    end: Optional[Tuple[float, float, float]] = None
    try:
        start = cast(
            Tuple[float, float, float],
            tuple(float(v) for v in edge.get_start_vertex().get_coordinates()),
        )
        end = cast(
            Tuple[float, float, float],
            tuple(float(v) for v in edge.get_end_vertex().get_coordinates()),
        )
    except Exception:
        pass

    hint_start = hint.get("start")
    hint_end = hint.get("end")
    hint_start_tuple = _tuple3_from_any(hint_start)
    hint_end_tuple = _tuple3_from_any(hint_end)
    if (
        start is not None
        and end is not None
        and hint_start_tuple is not None
        and hint_end_tuple is not None
    ):
        direct = _distance3(start, hint_start_tuple) + _distance3(end, hint_end_tuple)
        reverse = _distance3(start, hint_end_tuple) + _distance3(end, hint_start_tuple)
        score += min(direct, reverse)
    elif hint.get("center") is not None:
        center = edge.get_center()
        center_tuple = (float(center.x), float(center.y), float(center.z))
        score += _distance3(center_tuple, _tuple3_from_any(hint["center"]))

    if "tags" in hint:
        hint_tags = set(hint["tags"])
        common = len(hint_tags & set(edge.get_tags()))
        score -= common * 0.1
    return score


def _face_hint_score(face: Face, hint: Dict[str, Any]) -> float:
    score = 0.0
    if "area" in hint:
        score += abs(float(face.get_area()) - float(hint["area"]))

    center = face.get_center()
    center_tuple = (float(center.x), float(center.y), float(center.z))
    hint_center = hint.get("center")
    hint_center_tuple = _tuple3_from_any(hint_center)
    if hint_center_tuple is not None:
        score += _distance3(center_tuple, hint_center_tuple) * 10.0

    hint_normal = hint.get("normal")
    hint_normal_tuple = _tuple3_from_any(hint_normal)
    if hint_normal_tuple is not None:
        normal = face.get_normal_at()
        normal_tuple = (float(normal.x), float(normal.y), float(normal.z))
        score += _distance3(normal_tuple, hint_normal_tuple) * 5.0

    if "tags" in hint:
        hint_tags = set(hint["tags"])
        common = len(hint_tags & set(face.get_tags()))
        score -= common * 0.1
    return score


def _resolve_edges_from_selector_hints(
    solid: Solid, refs: Sequence[Dict[str, Any]]
) -> List[Edge]:
    edges = solid.get_edges()
    remaining = list(edges)
    resolved: List[Edge] = []
    for ref_dict in refs:
        hint = ref_dict.get("selector_hint")
        if not isinstance(hint, dict) or not remaining:
            continue
        best = min(
            remaining,
            key=lambda edge: _edge_hint_score(edge, cast(Dict[str, Any], hint)),
        )
        resolved.append(best)
        remaining.remove(best)
    return resolved


def _resolve_faces_from_selector_hints(
    solid: Solid, refs: Sequence[Dict[str, Any]]
) -> List[Face]:
    faces = solid.get_faces()
    remaining = list(faces)
    resolved: List[Face] = []
    for ref_dict in refs:
        hint = ref_dict.get("selector_hint")
        if not isinstance(hint, dict) or not remaining:
            continue
        best = min(
            remaining,
            key=lambda face: _face_hint_score(face, cast(Dict[str, Any], hint)),
        )
        resolved.append(best)
        remaining.remove(best)
    return resolved


def _resolve_edges_from_refs(
    solid: Solid, refs: Sequence[Dict[str, Any]]
) -> List[Edge]:
    if not refs:
        return []
    edge_map = {
        _shape_topo_ref_dict(edge).get("topo_id"): edge
        for edge in solid.get_edges()
        if _shape_topo_ref_dict(edge)
    }
    resolved: List[Edge] = []
    for ref_dict in refs:
        ref = topo_ref_from_dict(ref_dict)
        edge = edge_map.get(ref.topo_id)
        if edge is not None:
            resolved.append(edge)
    return resolved


def _resolve_faces_from_refs(
    solid: Solid, refs: Sequence[Dict[str, Any]]
) -> List[Face]:
    if not refs:
        return []
    face_map = {
        _shape_topo_ref_dict(face).get("topo_id"): face
        for face in solid.get_faces()
        if _shape_topo_ref_dict(face)
    }
    resolved: List[Face] = []
    for ref_dict in refs:
        ref = topo_ref_from_dict(ref_dict)
        face = face_map.get(ref.topo_id)
        if face is not None:
            resolved.append(face)
    return resolved


def _resolve_edges_from_indices(solid: Solid, indices: Sequence[int]) -> List[Edge]:
    edges = solid.get_edges()
    return [edges[idx] for idx in indices if 0 <= idx < len(edges)]


def _resolve_faces_from_indices(solid: Solid, indices: Sequence[int]) -> List[Face]:
    faces = solid.get_faces()
    return [faces[idx] for idx in indices if 0 <= idx < len(faces)]


def _warn_legacy_index_fallback(op_name: str) -> None:
    warnings.warn(
        f"{op_name} replay is using legacy selected_*_indices fallback because the graph has no selected_subshapes geometry signatures. "
        "New graphs should store geometry signatures and should not rely on SimpleCAD/OCP index order.",
        RuntimeWarning,
        stacklevel=3,
    )


def _resolve_edges_from_selected_subshapes(
    solid: Solid, params: Dict[str, Any], *, role: str
) -> List[Edge]:
    items = selected_subshape_items(params, kind="edge", role=role)
    if not items:
        items = selected_subshape_items(params, kind="edge")
    if not items:
        return []
    return match_selected_subshape_items(solid.get_edges(), items, kind="edge")


def _resolve_faces_from_selected_subshapes(
    solid: Solid, params: Dict[str, Any], *, role: str
) -> List[Face]:
    items = selected_subshape_items(params, kind="face", role=role)
    if not items:
        items = selected_subshape_items(params, kind="face")
    if not items:
        return []
    return match_selected_subshape_items(solid.get_faces(), items, kind="face")


def _resolve_single_selected_source_shape(
    owner: Solid,
    params: Dict[str, Any],
    *,
    edge_role: str,
    face_role: str,
) -> AnyShape:
    edge_items = selected_subshape_items(params, kind="edge", role=edge_role)
    if edge_items:
        edges = match_selected_subshape_items(owner.get_edges(), edge_items, kind="edge")
        if str(params.get("selected_source_kind", "")).lower() == "wire":
            return ops.make_wire_from_edges_rwire(edges)
        if edges:
            return edges[0]
    face_items = selected_subshape_items(params, kind="face", role=face_role)
    if face_items:
        faces = match_selected_subshape_items(owner.get_faces(), face_items, kind="face")
        if faces:
            return faces[0]
    return owner


def _resolve_edges_from_selection_payload(
    solid: Solid, payload: Dict[str, Any]
) -> List[Edge]:
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    return match_selected_subshape_items(solid.get_edges(), items, kind="edge")


def _resolve_edges_from_source_items(
    outputs: Dict[str, List[AnyShape]],
    payload: Dict[str, Any],
) -> List[Edge]:
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    edges: List[Edge] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        source_node_id = source.get("node_id") if isinstance(source, dict) else None
        if source_node_id is None:
            return []
        owner_outputs = outputs.get(str(source_node_id), [])
        if not owner_outputs:
            return []
        edges.extend(
            match_selected_subshape_items(
                cast(Solid, owner_outputs[0]).get_edges(),
                [item],
                kind="edge",
            )
        )
    return edges


def _execute_graph(
    graph: OperationGraph, leaf_node_ids: Optional[Sequence[str]] = None
) -> List[AnyShape]:
    if graph.node_count == 0:
        return []

    topo_order = graph.topological_order()

    # Store per-node outputs
    outputs: Dict[str, List[AnyShape]] = {}

    def _store_outputs(node, result: Any) -> None:
        result_list = _normalize_output(result)
        for idx, output in enumerate(result_list):
            attach_graph_node(
                output,
                node,
                output_slot=idx,
                graph_id=graph.graph_id,
            )
        outputs[node.node_id] = result_list

    with suspend_graph_recording():
        for node in topo_order:
            op_name = node.op
            params = node.params

            if op_name == "make_cut_rsolidlist":
                if len(node.inputs) < 2:
                    continue
                body_list = outputs.get(node.inputs[0].node_id, [])
                tool_lists = [outputs.get(inp.node_id, []) for inp in node.inputs[1:]]
                all_tools = [cast(Solid, s) for lst in tool_lists for s in lst]
                if body_list and all_tools:
                    result = ops.cut_rsolidlist(cast(Solid, body_list[0]), all_tools)
                    _store_outputs(node, result)
                continue

            if op_name == "make_union_rsolid":
                all_solids: List[Solid] = []
                for inp in node.inputs:
                    all_solids.extend(cast(List[Solid], outputs.get(inp.node_id, [])))
                if len(all_solids) >= 2:
                    result = ops.union_rsolid(all_solids)
                    _store_outputs(node, result)
                elif all_solids:
                    _store_outputs(node, all_solids[0])
                continue

            if op_name == "make_intersect_rsolidlist":
                if len(node.inputs) < 2:
                    continue
                all_solids: List[Solid] = []
                for inp in node.inputs:
                    all_solids.extend(cast(List[Solid], outputs.get(inp.node_id, [])))
                if len(all_solids) >= 2:
                    result = ops.intersect_rsolidlist(all_solids[0], all_solids[1:])
                    _store_outputs(node, result)
                continue

            if op_name == "make_face_from_wire_rface":
                wire_outputs: List[AnyShape] = []
                boundary_items = selected_subshape_items(
                    params, kind="edge", role="face_boundary_edge"
                )
                if boundary_items and node.inputs:
                    owner_outputs = outputs.get(node.inputs[0].node_id, [])
                    if owner_outputs:
                        edges = match_selected_subshape_items(
                            cast(Solid, owner_outputs[0]).get_edges(),
                            boundary_items,
                            kind="edge",
                        )
                        wire_outputs.append(ops.make_wire_from_edges_rwire(edges))
                elif node.inputs:
                    wire_outputs = outputs.get(node.inputs[0].node_id, [])
                if wire_outputs:
                    result = ops.make_face_from_wire_rface(
                        cast(Any, wire_outputs[0]),
                        normal=cast(Any, tuple(params.get("normal", (0, 0, 1)))),
                    )
                    _store_outputs(node, result)
                continue

            if op_name == "make_wire_from_edges_rwire":
                edge_outputs: List[AnyShape] = []
                edge_items = selected_subshape_items(params, kind="edge", role="wire_edge")
                if edge_items:
                    edge_outputs.extend(
                        _resolve_edges_from_source_items(
                            outputs,
                            cast(Dict[str, Any], params.get("selected_subshapes", {})),
                        )
                    )
                else:
                    for inp in node.inputs:
                        edge_outputs.extend(outputs.get(inp.node_id, []))
                if edge_outputs:
                    result = ops.make_wire_from_edges_rwire(cast(Any, edge_outputs))
                    _store_outputs(node, result)
                continue

            if op_name == "make_translate_rshape":
                input_outputs = (
                    outputs.get(node.inputs[0].node_id, []) if node.inputs else []
                )
                if input_outputs:
                    shape = _resolve_single_selected_source_shape(
                        cast(Solid, input_outputs[0]),
                        params,
                        edge_role="translate_source_edge",
                        face_role="translate_source_face",
                    )
                    result = ops.translate_shape(
                        cast(AnyShape, shape),
                        cast(Any, tuple(params.get("vector", (0, 0, 0)))),
                    )
                    _store_outputs(node, result)
                continue

            if op_name == "make_rotate_rshape":
                input_outputs = (
                    outputs.get(node.inputs[0].node_id, []) if node.inputs else []
                )
                if input_outputs:
                    shape = _resolve_single_selected_source_shape(
                        cast(Solid, input_outputs[0]),
                        params,
                        edge_role="rotate_source_edge",
                        face_role="rotate_source_face",
                    )
                    result = ops.rotate_shape(
                        cast(AnyShape, shape),
                        params.get("angle", 0.0),
                        axis=cast(Any, tuple(params.get("axis", (0, 0, 1)))),
                        origin=cast(Any, tuple(params.get("origin", (0, 0, 0)))),
                    )
                    _store_outputs(node, result)
                continue

            if op_name == "make_extrude_rsolid":
                profile_outputs = (
                    outputs.get(node.inputs[0].node_id, []) if node.inputs else []
                )
                if profile_outputs:
                    profile = profile_outputs[0]
                    face_items = selected_subshape_items(
                        params, kind="face", role="extrude_profile_face"
                    )
                    if face_items:
                        matched_faces = match_selected_subshape_items(
                            cast(Solid, profile).get_faces(),
                            face_items,
                            kind="face",
                        )
                        profile = matched_faces[0]
                    result = ops.extrude_rsolid(
                        cast(Any, profile),
                        cast(Any, tuple(params.get("direction", (0, 0, 1)))),
                        params.get("distance", 1.0),
                    )
                    _store_outputs(node, result)
                continue

            if op_name == "make_revolve_rsolid":
                profile_outputs = (
                    outputs.get(node.inputs[0].node_id, []) if node.inputs else []
                )
                if profile_outputs:
                    profile = profile_outputs[0]
                    face_items = selected_subshape_items(
                        params, kind="face", role="revolve_profile_face"
                    )
                    if face_items:
                        matched_faces = match_selected_subshape_items(
                            cast(Solid, profile).get_faces(),
                            face_items,
                            kind="face",
                        )
                        profile = matched_faces[0]
                    result = ops.revolve_rsolid(
                        cast(Any, profile),
                        axis=cast(Any, tuple(params.get("axis", (0, 0, 1)))),
                        angle=params.get("angle", 360),
                        origin=cast(Any, tuple(params.get("origin", (0, 0, 0)))),
                    )
                    _store_outputs(node, result)
                continue

            if op_name == "make_loft_rsolid":
                profile_outputs: List[AnyShape] = []
                selected_profiles = params.get("selected_profile_subshapes")
                if isinstance(selected_profiles, list) and node.inputs:
                    for inp, selection_payload in zip(node.inputs, selected_profiles):
                        owner_outputs = outputs.get(inp.node_id, [])
                        if owner_outputs and isinstance(selection_payload, dict):
                            edges = _resolve_edges_from_selection_payload(
                                cast(Solid, owner_outputs[0]), selection_payload
                            )
                            if edges:
                                profile_outputs.append(
                                    ops.make_wire_from_edges_rwire(edges)
                                )
                else:
                    for inp in node.inputs:
                        profile_outputs.extend(outputs.get(inp.node_id, []))
                if profile_outputs:
                    result = ops.loft_rsolid(
                        cast(Any, profile_outputs), ruled=params.get("ruled", False)
                    )
                    _store_outputs(node, result)
                continue

            if op_name == "make_sweep_rsolid":
                if len(node.inputs) >= 2:
                    profile_outputs = outputs.get(node.inputs[0].node_id, [])
                    path_outputs = outputs.get(node.inputs[1].node_id, [])
                    path_payload = params.get("selected_path_subshapes")
                    if isinstance(path_payload, dict) and path_outputs:
                        edges = _resolve_edges_from_selection_payload(
                            cast(Solid, path_outputs[0]), path_payload
                        )
                        path_outputs = [
                            ops.make_wire_from_edges_rwire(edges)
                        ] if edges else []
                    if profile_outputs and path_outputs:
                        profile = profile_outputs[0]
                        face_items = selected_subshape_items(
                            params, kind="face", role="sweep_profile_face"
                        )
                        if face_items:
                            matched_faces = match_selected_subshape_items(
                                cast(Solid, profile).get_faces(),
                                face_items,
                                kind="face",
                            )
                            profile = matched_faces[0]
                        result = ops.sweep_rsolid(
                            cast(Any, profile),
                            cast(Any, path_outputs[0]),
                            is_frenet=bool(params.get("is_frenet", False)),
                        )
                        _store_outputs(node, result)
                continue

            if op_name == "make_mirror_rshape":
                input_outputs = (
                    outputs.get(node.inputs[0].node_id, []) if node.inputs else []
                )
                if input_outputs:
                    shape = _resolve_single_selected_source_shape(
                        cast(Solid, input_outputs[0]),
                        params,
                        edge_role="mirror_source_edge",
                        face_role="mirror_source_face",
                    )
                    result = ops.mirror_shape(
                        cast(Any, shape),
                        cast(Any, tuple(params.get("plane_origin", (0, 0, 0)))),
                        cast(Any, tuple(params.get("plane_normal", (0, 0, 1)))),
                    )
                    _store_outputs(node, result)
                continue

            if op_name == "make_fillet_rsolid":
                input_outputs = (
                    outputs.get(node.inputs[0].node_id, []) if node.inputs else []
                )
                if input_outputs:
                    solid = cast(Solid, input_outputs[0])
                    edges = _resolve_edges_from_selected_subshapes(
                        solid, params, role="fillet_edge"
                    )
                    if not edges:
                        selected_edges = cast(
                            Sequence[Dict[str, Any]], params.get("selected_edges", [])
                        )
                        edges = _resolve_edges_from_refs(solid, selected_edges)
                        if len(edges) != len(selected_edges):
                            _warn_legacy_index_fallback(op_name)
                            edges = _resolve_edges_from_indices(
                                solid,
                                cast(
                                    Sequence[int],
                                    params.get("selected_edge_indices", []),
                                ),
                            )
                        selection_query = params.get("selection_query")
                        if len(edges) != len(selected_edges) and isinstance(
                            selection_query, dict
                        ):
                            edges = cast(
                                List[Edge],
                                selector_from_dict(selection_query).resolve(solid),
                            )
                        if len(edges) != len(selected_edges):
                            edges = _resolve_edges_from_selector_hints(
                                solid, selected_edges
                            )
                    result = ops.fillet_rsolid(
                        solid,
                        edges,
                        params.get("radius", 0.0),
                    )
                    _store_outputs(node, result)
                continue

            if op_name == "make_chamfer_rsolid":
                input_outputs = (
                    outputs.get(node.inputs[0].node_id, []) if node.inputs else []
                )
                if input_outputs:
                    solid = cast(Solid, input_outputs[0])
                    edges = _resolve_edges_from_selected_subshapes(
                        solid, params, role="chamfer_edge"
                    )
                    if not edges:
                        selected_edges = cast(
                            Sequence[Dict[str, Any]], params.get("selected_edges", [])
                        )
                        edges = _resolve_edges_from_refs(solid, selected_edges)
                        if len(edges) != len(selected_edges):
                            _warn_legacy_index_fallback(op_name)
                            edges = _resolve_edges_from_indices(
                                solid,
                                cast(
                                    Sequence[int],
                                    params.get("selected_edge_indices", []),
                                ),
                            )
                        selection_query = params.get("selection_query")
                        if len(edges) != len(selected_edges) and isinstance(
                            selection_query, dict
                        ):
                            edges = cast(
                                List[Edge],
                                selector_from_dict(selection_query).resolve(solid),
                            )
                        if len(edges) != len(selected_edges):
                            edges = _resolve_edges_from_selector_hints(
                                solid, selected_edges
                            )
                    result = ops.chamfer_rsolid(
                        solid,
                        edges,
                        params.get("distance", 0.0),
                    )
                    _store_outputs(node, result)
                continue

            if op_name == "make_shell_rsolid":
                input_outputs = (
                    outputs.get(node.inputs[0].node_id, []) if node.inputs else []
                )
                if input_outputs:
                    solid = cast(Solid, input_outputs[0])
                    faces = _resolve_faces_from_selected_subshapes(
                        solid, params, role="shell_remove_face"
                    )
                    if not faces:
                        selected_faces = cast(
                            Sequence[Dict[str, Any]], params.get("selected_faces", [])
                        )
                        faces = _resolve_faces_from_refs(solid, selected_faces)
                        if len(faces) != len(selected_faces):
                            _warn_legacy_index_fallback(op_name)
                            faces = _resolve_faces_from_indices(
                                solid,
                                cast(
                                    Sequence[int],
                                    params.get("selected_face_indices", []),
                                ),
                            )
                        selection_query = params.get("selection_query")
                        if len(faces) != len(selected_faces) and isinstance(
                            selection_query, dict
                        ):
                            faces = cast(
                                List[Face],
                                selector_from_dict(selection_query).resolve(solid),
                            )
                        if len(faces) != len(selected_faces):
                            faces = _resolve_faces_from_selector_hints(
                                solid, selected_faces
                            )
                    result = ops.shell_rsolid(
                        solid,
                        faces,
                        params.get("thickness", 0.0),
                    )
                    _store_outputs(node, result)
                continue

            # Primitive or simple operation
            factory = _OP_REGISTRY.get(op_name)
            if factory:
                try:
                    result = factory(params)
                    _store_outputs(node, result)
                except ValueError:
                    raise
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to replay node {node.node_id} ({op_name})"
                    ) from exc
                continue

            raise ValueError(f"Unsupported operation for replay: {op_name}")

    if leaf_node_ids:
        selected: List[AnyShape] = []
        for node_id in leaf_node_ids:
            selected.extend(outputs.get(str(node_id), []))
        return selected

    leaves = graph.leaf_nodes()
    result: List[AnyShape] = []
    for leaf in leaves:
        result.extend(outputs.get(leaf.node_id, []))
    return result

def replay_graph(graph: OperationGraph) -> List[AnyShape]:
    """Replay an OperationGraph to rebuild the model.

    Executes nodes in topological order. Primitives are created from their
    parameters; boolean operations consume upstream outputs.

    Args:
        graph: The graph to replay.

    Returns:
        List of leaf-node outputs. These may be solids, faces, wires, edges,
        or vertices depending on the workflow.
    """

    return _execute_graph(graph)
