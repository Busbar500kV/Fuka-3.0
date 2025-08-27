# core/debug.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

try:
    # Optional: only needed if you use add_near_trace
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover
    go = None


# --------------------------------------------------------------------
# 1) Build a 3-D "interest" mask from the UI's attr_history
#    interest3d[t, y, x] = 1 inside a radius-disk around attractors
# --------------------------------------------------------------------
def build_attractor_interest3d(
    attr_history: Optional[List[Dict[str, Any]]],
    stack_shape_T_H_W: Tuple[int, int, int],
    radius: int = 3,
) -> np.ndarray:
    """
    Parameters
    ----------
    attr_history : list like [{"t": int, "items":[{"pos": (y,x), ...}, ...]}, ...]
    stack_shape_T_H_W : shape tuple of (T, H, W) (use E_stack.shape)
    radius : L2 disk radius in (y, x) plane per time-slice

    Returns
    -------
    interest3d : np.ndarray uint8 of shape (T, H, W)
    """
    T, H, W = map(int, stack_shape_T_H_W)
    interest3d = np.zeros((T, H, W), dtype=np.uint8)
    if not attr_history:
        return interest3d

    R = max(0, int(radius))
    if R == 0:
        # mark exact points only
        for frame in attr_history:
            t = int(frame.get("t", 0))
            if 0 <= t < T:
                for it in frame.get("items", []):
                    y, x = it.get("pos", (None, None))
                    if y is None or x is None:
                        continue
                    yi, xi = int(y), int(x)
                    if 0 <= yi < H and 0 <= xi < W:
                        interest3d[t, yi, xi] = 1
        return interest3d

    # disk footprint (L2) as small kernel we can splat
    yy, xx = np.ogrid[-R:R+1, -R:R+1]
    disk = (yy*yy + xx*xx) <= (R*R)

    for frame in attr_history:
        t = int(frame.get("t", 0))
        if not (0 <= t < T):
            continue
        for it in frame.get("items", []):
            y, x = it.get("pos", (None, None))
            if y is None or x is None:
                continue
            yc = int(y); xc = int(x)
            if not (0 <= yc < H and 0 <= xc < W):
                continue
            y0 = max(0, yc - R); y1 = min(H, yc + R + 1)
            x0 = max(0, xc - R); x1 = min(W, xc + R + 1)

            ky0 = R - (yc - y0)  # slice disk to match crop
            kx0 = R - (xc - x0)
            ky1 = ky0 + (y1 - y0)
            kx1 = kx0 + (x1 - x0)

            interest3d[t, y0:y1, x0:x1] |= disk[ky0:ky1, kx0:kx1]
    return interest3d


# --------------------------------------------------------------------
# 2) Partition edges (ENDPOINTS format) into near/far sets
# --------------------------------------------------------------------
def partition_edges_by_attractors_endpoints(
    edges_t: np.ndarray,
    edges_x0: np.ndarray,
    edges_y0: np.ndarray,
    edges_x1: np.ndarray,
    edges_y1: np.ndarray,
    interest3d: np.ndarray,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    edges_* : arrays of the same length (per edge)
    interest3d : mask from build_attractor_interest3d

    Returns
    -------
    dict with:
      mask_near: bool array (len = #edges)
      counts: {"near": int, "far": int, "total": int, "pct_near": float}
      mids: {"t": int[], "x": int[], "y": int[]} midpoint integer coords
    """
    edges_t = np.asarray(edges_t, dtype=int)
    x0 = np.asarray(edges_x0, dtype=float)
    y0 = np.asarray(edges_y0, dtype=float)
    x1 = np.asarray(edges_x1, dtype=float)
    y1 = np.asarray(edges_y1, dtype=float)

    tm = edges_t
    xm = np.rint(0.5 * (x0 + x1)).astype(int)
    ym = np.rint(0.5 * (y0 + y1)).astype(int)

    T, H, W = interest3d.shape
    valid = (tm >= 0) & (tm < T) & (ym >= 0) & (ym < H) & (xm >= 0) & (xm < W)

    mask_near = np.zeros_like(valid, dtype=bool)
    if valid.any():
        mask_near[valid] = interest3d[tm[valid], ym[valid], xm[valid]] > 0

    near = int(mask_near[valid].sum())
    total = int(valid.sum())
    far = int(total - near)
    pct = (100.0 * near / total) if total > 0 else 0.0

    return {
        "mask_near": mask_near,
        "counts": {"near": near, "far": far, "total": total, "pct_near": pct},
        "mids": {"t": tm, "x": xm, "y": ym, "valid": valid},
    }


# --------------------------------------------------------------------
# 3) Partition edges (PLOTLY POLYLINE format) into near/far sets
#    conn_x/conn_y/conn_z are sequences with None separators
#    Returns a compact near-subset you can plot as a separate trace.
# --------------------------------------------------------------------
def partition_edges_by_attractors_polylines(
    conn_x: List[Optional[float]],
    conn_y: List[Optional[float]],
    conn_z: List[Optional[float]],
    interest3d: np.ndarray,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    conn_x/conn_y/conn_z : lists with None separators between edges
                           (two points per edge expected)
    interest3d : mask from build_attractor_interest3d

    Returns
    -------
    dict with:
      counts: {"near": int, "far": int, "total": int, "pct_near": float}
      near_poly: {"x": list, "y": list, "z": list}  # points to plot (no Nones)
    """
    T, H, W = interest3d.shape
    N = len(conn_x)
    i = 0
    near = far = total = 0
    x_near: List[float] = []
    y_near: List[float] = []
    z_near: List[float] = []

    while i < N:
        # collect segment until None
        if conn_x[i] is None:
            i += 1
            continue
        j = i
        while j < N and conn_x[j] is not None:
            j += 1
        # segment [i..j-1] is one edge; assume 2 points
        if j - i >= 2:
            x0, x1 = conn_x[i], conn_x[i + 1]
            y0, y1 = conn_y[i], conn_y[i + 1]
            z0 = conn_z[i]  # time is constant along an edge polyline
            if None not in (x0, x1, y0, y1, z0):
                xm = int(round(0.5 * (x0 + x1)))
                ym = int(round(0.5 * (y0 + y1)))
                tm = int(round(z0))
                if 0 <= tm < T and 0 <= ym < H and 0 <= xm < W:
                    total += 1
                    if interest3d[tm, ym, xm] > 0:
                        near += 1
                        # store both points for a solid segment
                        x_near.extend([x0, x1, None])
                        y_near.extend([y0, y1, None])
                        z_near.extend([z0, z0, None])
                    else:
                        far += 1
        # jump over this segment and the None (if present)
        i = j + 1

    pct = (100.0 * near / total) if total > 0 else 0.0
    return {
        "counts": {"near": near, "far": far, "total": total, "pct_near": pct},
        "near_poly": {"x": x_near, "y": y_near, "z": z_near},
    }


# --------------------------------------------------------------------
# 4) Optional Plotly helper to add the highlight trace
# --------------------------------------------------------------------
def add_near_trace(
    fig,
    near_poly: Dict[str, List[Optional[float]]],
    name: str = "Connections (near attractors)",
    color: str = "#00FFAA",
    width: int = 4,
):
    """
    Adds a separate line trace for the 'near' subset.
    Safe no-op if Plotly isn't available.
    """
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


# --------------------------------------------------------------------
# 5) Tiny utility to format a summary string
# --------------------------------------------------------------------
def format_counts(c: Dict[str, float]) -> str:
    return f"{c.get('near',0)}/{c.get('total',0)} near ({c.get('pct_near',0.0):.1f}%)"


# --------------------------------------------------------------------
# Example usage (in your app.py or app_helpers after you have a figure):
# --------------------------------------------------------------------
# from core import debug as dbg
#
# interest = dbg.build_attractor_interest3d(
#     attr_history=attr_history,
#     stack_shape_T_H_W=E_stack.shape,   # (T, H, W)
#     radius=4,
# )
#
# # If you have PLOTLY POLYLINE arrays with None separators:
# stats = dbg.partition_edges_by_attractors_polylines(conn_x, conn_y, conn_z, interest)
# st.info("Near-attractor connections: " + dbg.format_counts(stats["counts"]))
# dbg.add_near_trace(fig, stats["near_poly"])
#
# # If you have ENDPOINT arrays per edge instead:
# parts = dbg.partition_edges_by_attractors_endpoints(edges_t, x0, y0, x1, y1, interest)
# st.info("Near-attractor connections: " + dbg.format_counts(parts["counts"]))
# # You can also use parts["mask_near"] to plot a filtered subset yourself.