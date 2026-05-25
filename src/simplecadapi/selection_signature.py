"""Geometry signatures for runtime-selected subshapes.

The signatures in this module are intentionally descriptive rather than
topological.  They capture the face/edge that the SimpleCAD runtime actually
selected so another kernel can re-identify the corresponding subshape without
trusting SimpleCAD/OCP traversal indices.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, cast

from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.GeomAbs import (
    GeomAbs_BSplineCurve,
    GeomAbs_BSplineSurface,
    GeomAbs_BezierCurve,
    GeomAbs_BezierSurface,
    GeomAbs_Circle,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Line,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCP.gp import gp_Pnt, gp_Vec

from .core import AnyShape, Edge, Face
from .kernel.ocp_properties import bounding_box


_POINT_ABS_TOL = 1e-5
_REL_TOL = 1e-6


TShape = TypeVar("TShape", Edge, Face)


def _tuple3(value: Any) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        return (float(value.x), float(value.y), float(value.z))
    if hasattr(value, "X") and hasattr(value, "Y") and hasattr(value, "Z"):
        return (float(value.X()), float(value.Y()), float(value.Z()))
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return None


def _point_payload(value: Any) -> Optional[List[float]]:
    vec = _tuple3(value)
    if vec is None:
        return None
    return [float(vec[0]), float(vec[1]), float(vec[2])]


def _normalized(value: Any) -> Optional[List[float]]:
    vec = _tuple3(value)
    if vec is None:
        return None
    length = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    if length <= 1e-12:
        return None
    return [vec[0] / length, vec[1] / length, vec[2] / length]


def _bbox_payload(shape: Any) -> Dict[str, List[float]]:
    bb = bounding_box(shape)
    return {
        "min": [float(bb.xmin), float(bb.ymin), float(bb.zmin)],
        "max": [float(bb.xmax), float(bb.ymax), float(bb.zmax)],
        "size": [float(bb.xlen), float(bb.ylen), float(bb.zlen)],
    }


def _edge_curve_type(edge: Edge) -> str:
    try:
        adaptor = BRepAdaptor_Curve(edge.wrapped)
        curve_type = adaptor.GetType()
        if curve_type == GeomAbs_Line:
            return "line"
        if curve_type == GeomAbs_Circle:
            span = abs(float(adaptor.LastParameter()) - float(adaptor.FirstParameter()))
            return "circle" if abs(span - (2.0 * math.pi)) <= 1e-5 else "arc"
        if curve_type == GeomAbs_BSplineCurve:
            return "bspline"
        if curve_type == GeomAbs_BezierCurve:
            return "bezier"
        return str(curve_type).replace("GeomAbs_CurveType.GeomAbs_", "").lower()
    except Exception:
        return "unknown"


def _face_surface_type(face: Face) -> str:
    try:
        adaptor = BRepAdaptor_Surface(face.wrapped)
        surface_type = adaptor.GetType()
        mapping = {
            GeomAbs_Plane: "plane",
            GeomAbs_Cylinder: "cylinder",
            GeomAbs_Cone: "cone",
            GeomAbs_Sphere: "sphere",
            GeomAbs_Torus: "torus",
            GeomAbs_BSplineSurface: "bspline",
            GeomAbs_BezierSurface: "bezier",
        }
        return mapping.get(
            surface_type,
            str(surface_type).replace("GeomAbs_SurfaceType.GeomAbs_", "").lower(),
        )
    except Exception:
        return "unknown"


def _edge_endpoints(edge: Edge) -> Optional[List[List[float]]]:
    try:
        return [
            [float(v) for v in edge.get_start_vertex().get_coordinates()],
            [float(v) for v in edge.get_end_vertex().get_coordinates()],
        ]
    except Exception:
        pass

    try:
        adaptor = BRepAdaptor_Curve(edge.wrapped)
        first = float(adaptor.FirstParameter())
        last = float(adaptor.LastParameter())
        return [
            cast(List[float], _point_payload(adaptor.Value(first))),
            cast(List[float], _point_payload(adaptor.Value(last))),
        ]
    except Exception:
        return None


def _edge_midpoint(edge: Edge) -> Optional[List[float]]:
    try:
        adaptor = BRepAdaptor_Curve(edge.wrapped)
        mid = 0.5 * (float(adaptor.FirstParameter()) + float(adaptor.LastParameter()))
        return _point_payload(adaptor.Value(mid))
    except Exception:
        return _point_payload(edge.get_center())


def _edge_tangent(edge: Edge) -> Optional[List[float]]:
    try:
        adaptor = BRepAdaptor_Curve(edge.wrapped)
        mid = 0.5 * (float(adaptor.FirstParameter()) + float(adaptor.LastParameter()))
        point = gp_Pnt()
        tangent = gp_Vec()
        adaptor.D1(mid, point, tangent)
        return _normalized(tangent)
    except Exception:
        pass

    endpoints = _edge_endpoints(edge)
    if endpoints and len(endpoints) == 2:
        start, end = endpoints
        return _normalized(
            (
                float(end[0]) - float(start[0]),
                float(end[1]) - float(start[1]),
                float(end[2]) - float(start[2]),
            )
        )
    return None


def _edge_radius(edge: Edge) -> Optional[float]:
    try:
        adaptor = BRepAdaptor_Curve(edge.wrapped)
        if adaptor.GetType() == GeomAbs_Circle:
            return float(adaptor.Circle().Radius())
    except Exception:
        pass
    return None


def _face_radius(face: Face) -> Optional[float]:
    try:
        adaptor = BRepAdaptor_Surface(face.wrapped)
        surface_type = adaptor.GetType()
        if surface_type == GeomAbs_Cylinder:
            return float(adaptor.Cylinder().Radius())
        if surface_type == GeomAbs_Cone:
            return float(adaptor.Cone().RefRadius())
        if surface_type == GeomAbs_Sphere:
            return float(adaptor.Sphere().Radius())
        if surface_type == GeomAbs_Torus:
            return float(adaptor.Torus().MajorRadius())
    except Exception:
        pass
    return None


def extract_geometry_signature(shape: AnyShape) -> Dict[str, Any]:
    """Return a JSON-compatible geometry signature for a selected face or edge."""

    if isinstance(shape, Edge):
        signature: Dict[str, Any] = {
            "kind": "edge",
            "curve_type": _edge_curve_type(shape),
            "length": float(shape.get_length()),
            "bbox": _bbox_payload(shape.wrapped),
            "tags": sorted(shape.get_tags()),
        }
        center = _point_payload(shape.get_center())
        midpoint = _edge_midpoint(shape)
        tangent = _edge_tangent(shape)
        radius = _edge_radius(shape)
        endpoints = _edge_endpoints(shape)
        if center is not None:
            signature["center"] = center
        if midpoint is not None:
            signature["midpoint"] = midpoint
        if tangent is not None:
            signature["tangent"] = tangent
            signature["direction"] = tangent
        if radius is not None:
            signature["radius"] = radius
        if endpoints is not None:
            signature["endpoints"] = endpoints
        return signature

    if isinstance(shape, Face):
        signature = {
            "kind": "face",
            "surface_type": _face_surface_type(shape),
            "area": float(shape.get_area()),
            "bbox": _bbox_payload(shape.wrapped),
            "tags": sorted(shape.get_tags()),
        }
        center = _point_payload(shape.get_center())
        if center is not None:
            signature["center"] = center
        try:
            normal = _normalized(shape.get_normal_at())
            if normal is not None:
                signature["normal"] = normal
        except Exception:
            pass
        radius = _face_radius(shape)
        if radius is not None:
            signature["radius"] = radius
        return signature

    raise TypeError(
        f"Geometry signatures are supported for Edge and Face selections, got {type(shape).__name__}"
    )


def _bbox_scale(signature: Dict[str, Any]) -> float:
    bbox = signature.get("bbox")
    if isinstance(bbox, dict):
        size = bbox.get("size")
        if isinstance(size, (list, tuple)) and len(size) == 3:
            try:
                return max(
                    math.sqrt(sum(float(v) * float(v) for v in size)),
                    _POINT_ABS_TOL,
                )
            except Exception:
                pass
    for key in ("length", "radius"):
        if key in signature:
            try:
                return max(abs(float(signature[key])), _POINT_ABS_TOL)
            except Exception:
                pass
    if "area" in signature:
        try:
            return max(math.sqrt(abs(float(signature["area"]))), _POINT_ABS_TOL)
        except Exception:
            pass
    return 1.0


def _point_tol(target: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    return max(_POINT_ABS_TOL, _bbox_scale(target) * _REL_TOL, _bbox_scale(candidate) * _REL_TOL)


def _scalar_close(a: Any, b: Any) -> bool:
    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return False
    return abs(af - bf) <= max(1e-7, abs(af) * _REL_TOL, abs(bf) * _REL_TOL)


def _distance(a: Any, b: Any) -> Optional[float]:
    av = _tuple3(a)
    bv = _tuple3(b)
    if av is None or bv is None:
        return None
    return math.dist(av, bv)


def _vec_close(a: Any, b: Any, tol: float) -> bool:
    dist = _distance(a, b)
    return dist is not None and dist <= tol


def _direction_close(a: Any, b: Any) -> bool:
    av = _normalized(a)
    bv = _normalized(b)
    if av is None or bv is None:
        return False
    dot = abs(av[0] * bv[0] + av[1] * bv[1] + av[2] * bv[2])
    return dot >= 1.0 - 1e-5


def _bbox_close(a: Any, b: Any, tol: float) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    for key in ("min", "max"):
        if not _vec_close(a.get(key), b.get(key), tol):
            return False
    return True


def _endpoints_close(a: Any, b: Any, tol: float) -> bool:
    if not (
        isinstance(a, (list, tuple))
        and isinstance(b, (list, tuple))
        and len(a) == 2
        and len(b) == 2
    ):
        return False
    direct = _vec_close(a[0], b[0], tol) and _vec_close(a[1], b[1], tol)
    reverse = _vec_close(a[0], b[1], tol) and _vec_close(a[1], b[0], tol)
    return direct or reverse


def signatures_match(
    candidate: Dict[str, Any],
    target: Dict[str, Any],
    *,
    kind: str,
) -> bool:
    """Return True when a candidate signature matches a stored target signature."""

    target_kind = str(target.get("kind", kind)).lower()
    candidate_kind = str(candidate.get("kind", kind)).lower()
    if target_kind != candidate_kind or target_kind != str(kind).lower():
        return False

    compared = 0
    tol = _point_tol(target, candidate)

    type_key = "curve_type" if kind == "edge" else "surface_type"
    if target.get(type_key) and candidate.get(type_key):
        compared += 1
        if str(target[type_key]).lower() != str(candidate[type_key]).lower():
            return False

    for key in ("length", "area", "radius"):
        if key in target and key in candidate:
            compared += 1
            if not _scalar_close(target[key], candidate[key]):
                return False

    for key in ("center", "midpoint"):
        if key in target and key in candidate:
            compared += 1
            if not _vec_close(target[key], candidate[key], tol):
                return False

    for key in ("normal", "tangent", "direction"):
        if key in target and key in candidate:
            compared += 1
            if not _direction_close(target[key], candidate[key]):
                return False

    if "bbox" in target and "bbox" in candidate:
        compared += 1
        if not _bbox_close(target["bbox"], candidate["bbox"], tol):
            return False

    if "endpoints" in target and "endpoints" in candidate:
        compared += 1
        if not _endpoints_close(target["endpoints"], candidate["endpoints"], tol):
            return False

    return compared > 0


def signature_summary(signature: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact signature summary suitable for diagnostics."""

    keys = (
        "kind",
        "surface_type",
        "curve_type",
        "center",
        "midpoint",
        "normal",
        "length",
        "area",
        "radius",
        "bbox",
    )
    return {key: signature[key] for key in keys if key in signature}


def _item_signature(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    signature = item.get("geometry_signature")
    return signature if isinstance(signature, dict) else None


def _selection_error(
    *,
    reason: str,
    item: Dict[str, Any],
    kind: str,
    candidate_count: int,
    match_count: int,
) -> ValueError:
    signature = _item_signature(item) or {}
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    selected_id = item.get("item_id") or item.get("selected_subshape_id") or "<unknown>"
    method = item.get("original_selection_method") or item.get("selection_method")
    return ValueError(
        "selected_subshape geometry match failed: "
        f"{reason}; source node={source.get('node_id')}; "
        f"selected_subshape id={selected_id}; kind={kind}; "
        f"original selection method={method}; "
        f"signature={signature_summary(signature)}; "
        f"matches={match_count}; candidates={candidate_count}"
    )


def match_selected_subshape_items(
    candidates: Sequence[TShape],
    items: Sequence[Dict[str, Any]],
    *,
    kind: str,
    allow_duplicate_matches: bool = False,
) -> List[TShape]:
    """Match ordered selected_subshape items against a candidate list."""

    candidate_signatures = [extract_geometry_signature(candidate) for candidate in candidates]
    matched: List[TShape] = []
    used_indices: set[int] = set()
    normalized_kind = str(kind).lower()

    for item in items:
        signature = _item_signature(item)
        if signature is None:
            raise _selection_error(
                reason="missing geometry_signature",
                item=item,
                kind=normalized_kind,
                candidate_count=len(candidates),
                match_count=0,
            )

        matches = [
            idx
            for idx, candidate_signature in enumerate(candidate_signatures)
            if signatures_match(candidate_signature, signature, kind=normalized_kind)
        ]

        if not matches:
            raise _selection_error(
                reason="no candidate matched the stored geometry signature",
                item=item,
                kind=normalized_kind,
                candidate_count=len(candidates),
                match_count=0,
            )
        if len(matches) > 1:
            raise _selection_error(
                reason="ambiguous geometry signature matched multiple candidates",
                item=item,
                kind=normalized_kind,
                candidate_count=len(candidates),
                match_count=len(matches),
            )

        idx = matches[0]
        if idx in used_indices and not allow_duplicate_matches:
            raise _selection_error(
                reason="duplicate selection items matched the same candidate",
                item=item,
                kind=normalized_kind,
                candidate_count=len(candidates),
                match_count=1,
            )
        used_indices.add(idx)
        matched.append(candidates[idx])

    return matched


def selected_subshape_items(
    params: Dict[str, Any],
    *,
    kind: Optional[str] = None,
    role: Optional[str] = None,
) -> List[Dict[str, Any]]:
    payload = params.get("selected_subshapes")
    if not isinstance(payload, dict):
        return []
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    result: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if kind is not None and str(item.get("kind", "")).lower() != kind.lower():
            continue
        if role is not None and str(item.get("role", "")).lower() != role.lower():
            continue
        result.append(item)
    return result

