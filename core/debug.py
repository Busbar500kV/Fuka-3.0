# core/debug.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover
    go = None


def build_attractor_interest3d(
    attr_history: Optional[List[Dict[str, Any]]],
    stack_shape_T_H_W: Tuple[int, int, int],
    radius: int = 3,
) -> np.ndarray:
    """
    interest3d[t, y, x] = 1 inside an L2 disk around each attractor at time t.
    attr_history: [{"t": int, "items":[{"pos": (y,x), ...}, ...]}, ...]
    """
    T, H, W = map(int, stack_shape_T_H_W)
    interest3d = np.zeros((T, H, W), dtype=np.uint8)
    if not attr_history:
        return interest3d

    R = max(0, int(radius))
    if R == 0:
        for frame in attr_history:
            t = int(frame.get("t", -1))
            if not (0 <= t < T):
                continue
            for it in frame.get("items", []):
                pos = it.get("pos")
                if not pos:
                    continue
                y, x = int(pos[0]), int(pos[1])
                if 0 <= y < H and 0 <= x < W:
                    interest3d[t, y, x] = 1
        return interest3d

    # precompute disk kernel
    yy, xx = np.ogrid[-R:R+1, -R:R+1]
    disk = (yy * yy + xx * xx) <= (R * R)

    for frame in attr_history:
        t = int(frame.get("t", -1))
        if not (0 <= t < T):
            continue
        for it in frame.get("items", []):
            pos = it.get("pos")
            if not pos:
                continue
            yc, xc = int(pos[0]), int(pos[1])
            if not (0 <= yc < H and 0 <= xc < W):
                continue

            y0 = max(0, yc - R); y1 = min(H, yc + R + 1)
            x0 = max(0, xc - R); x1 = min(W, xc + R + 1)

            ky0 = R - (yc - y0)
            kx0 = R - (xc - x0)
            ky1 = ky0 + (y1 - y0)
            kx1 = kx0 + (x1 - x0)

            # OR in the cropped disk
            interest3d[t, y0:y1, x0:x1] |= disk[ky0:ky1, kx0:kx1]
    return interest3d


def partition_edges_by_attractors_endpoints(
    edges_t: np.ndarray,
    edges_x0: np.ndarray,
    edges_y0: np.ndarray,
    edges_x1: np.ndarray,
    edges_y1: np.ndarray,
    interest3d: np.ndarray,
) -> Dict[str, Any]:
    """
    Classify edges by midpoints against interest3d.
    Returns mask_near + counts + integer midpoints (with validity mask).
    """
    tm = np.asarray(edges_t, dtype=int)
    x0 = np.asarray(edges_x0, dtype=float)
    y0 = np.asarray(edges_y0, dtype=float)
    x1 = np.asarray(edges_x1, dtype=float)
    y1 = np.asarray(edges_y1, dtype=float)

    xm = np.rint(0.5 * (x0 + x1)).astype(int)
    ym = np.rint(0.5 * (y0 + y1)).astype(int)

    T, H, W = interest3d.shape
    valid = (tm >= 0) & (tm < T) & (ym >= 0) & (ym < H) & (xm >= 0) & (xm < W)

    mask_near = np.zeros(tm.shape, dtype=bool)
    if valid.any():
        mask_near[valid] = interest3d[tm[valid], ym[valid], xm[valid]] != 0

    near = int(mask_near[valid].sum())
    total = int(valid.sum())
    far = int(total - near)
    pct = (100.0 * near / total) if total > 0 else 0.0

    return {
        "mask_near": mask_near,
        "counts": {"near": near, "far": far, "total": total, "pct_near": pct},
        "mids": {"t": tm, "x": xm, "y": ym, "valid": valid},
    }


def partition_edges_by_attractors_polylines(
    conn_x: List[Optional[float]],
    conn_y: List[Optional[float]],
    conn_z: List[Optional[float]],
    interest3d: np.ndarray,
) -> Dict[str, Any]:
    """
    conn_x/conn_y/conn_z: lists of coordinates with None separators per edge.
    Assumes 2 points per edge segment.
    """
    T, H, W = interest3d.shape
    N = len(conn_x)
    i = 0
    near = far = total = 0
    x_near: List[Optional[float]] = []
    y_near: List[Optional[float]] = []
    z_near: List[Optional[float]] = []

    while i < N:
        if conn_x[i] is None:
            i += 1
            continue
        j = i
        while j < N and conn_x[j] is not None:
            j += 1
        # segment [i..j-1]
        if j - i >= 2:
            x0, x1 = conn_x[i], conn_x[i + 1]
            y0, y1 = conn_y[i], conn_y[i + 1]
            z0 = conn_z[i]
            if None not in (x0, x1, y0, y1, z0):
                xm = int(round(0.5 * (x0 + x1)))
                ym = int(round(0.5 * (y0 + y1)))
                tm = int(round(z0))
                if 0 <= tm < T and 0 <= ym < H and 0 <= xm < W:
                    total += 1
                    if interest3d[tm, ym, xm] != 0:
                        near += 1
                        x_near.extend([x0, x1, None])
                        y_near.extend([y0, y1, None])
                        z_near.extend([z0, z0, None])
                    else:
                        far += 1
        i = j + 1  # skip segment and its trailing None

    pct = (100.0 * near / total) if total > 0 else 0.0
    return {
        "counts": {"near": near, "far": far, "total": total, "pct_near": pct},
        "near_poly": {"x": x_near, "y": y_near, "z": z_near},
    }


def add_near_trace(
    fig,
    near_poly: Dict[str, List[Optional[float]]],
    name: str = "Connections (near attractors)",
    color: str = "#00FFAA",
    width: int = 4,
):
    """Add a separate 3D line trace for the 'near' subset (safe no-op if Plotly missing)."""
    if go is None or fig is None:
        return
    fig.add_trace(go.Scatter3d(
        x=near_poly.get("x", []),
        y=near_poly.get("y", []),
        z=near_poly.get("z", []),
        mode="lines",
        line=dict(width=width, color=color),
        name=name,
        showlegend=True,
    ))


def format_counts(c: Dict[str, float]) -> str:
    """Return 'near/total near (pct%)'."""
    total = int(c.get("total", 0))
    near = int(c.get("near", 0))
    pct = (100.0 * near / total) if total > 0 else 0.0
    return f"{near}/{total} near ({pct:.1f}%)"