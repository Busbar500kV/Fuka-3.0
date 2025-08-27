# app_helpers.py
from __future__ import annotations
import json, os
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# -------------------- Config & validation --------------------

def load_defaults_strict(path: str = "defaults.json") -> Dict[str, Any]:
    if not os.path.exists(path):
        st.error(
            "defaults.json not found in project root. "
            "Create it (copy from README/template) and remove any comments."
        )
        st.stop()
    try:
        with open(path, "r") as f:
            return json.load(f)  # strict JSON
    except Exception as e:
        st.error(
            "Failed to parse defaults.json as strict JSON.\n\n"
            "Tip: remove any // or /* */ comments.\n\n"
            f"Parser error: {e}"
        )
        st.stop()

def find_missing(cfg: Dict[str, Any]) -> List[str]:
    req_top = ["seed", "frames", "space", "k_flux", "k_motor", "diffuse", "decay", "band", "bc", "env"]
    missing: List[str] = [k for k in req_top if k not in cfg]
    if "env" not in cfg or not isinstance(cfg["env"], dict):
        missing.extend(["env.length", "env.height", "env.frames", "env.noise_sigma"])
    else:
        env = cfg["env"]
        for k in ["length", "height", "frames", "noise_sigma"]:
            if k not in env:
                missing.append(f"env.{k}")
    return missing

# -------------------- Session keys --------------------

def ensure_session_keys():
    for base in ("combo2d_count", "energy_count", "stats_count", "combo3d_count", "conn3d_count", "run_id"):
        if base not in st.session_state:
            st.session_state[base] = 0

def new_key(base: str) -> str:
    st.session_state[base + "_count"] += 1
    return f"{base}_{st.session_state['run_id']}_{st.session_state[base + '_count']}"

# -------------------- Numeric UI helpers --------------------

def _num_step(v: float) -> float:
    v = abs(float(v)) if v != 0 else 1.0
    mag = 10 ** int(np.floor(np.log10(v)))
    return max(mag * 0.01, 10 ** (int(np.floor(np.log10(v))) - 2))

def _float_slider_bounds(label: str, val: float) -> Tuple[float, float, float]:
    name = label.lower()
    step = round(_num_step(val), 6)
    if step < 1e-6:  # never pass a zero step to Streamlit
        step = 1e-6
    if 0.0 <= val <= 1.0 or any(s in name for s in [
        "sigma","noise","decay","diffuse","k_","thr","threshold","opacity","gamma","floor"
    ]):
        lo, hi = 0.0, 1.0
        if val > 1.0:
            hi = max(1.0, float(val) * 10.0)
        return lo, hi, step
    if val < 0:
        m = abs(val)
        lo, hi = -max(1.0, m * 10.0), max(1.0, m * 10.0)
        return lo, hi, step
    hi = max(1.0, float(val) * 10.0)
    lo = 0.0
    return lo, hi, step

def _int_slider_bounds(label: str, val: int) -> Tuple[int, int, int]:
    name = label.lower()
    if "seed" in name:
        return 0, 10_000_000, 1
    if any(k in name for k in ["frame", "space", "length", "len", "height", "width", "band", "center", "gate"]):
        base = max(1, int(val))
        hi = max(base * 10, base + 10)
        lo = 0
        step = max(1, base // 10)
        return lo, hi, step
    if val >= 0:
        lo, hi = 0, max(10, val * 10)
    else:
        m = abs(val)
        lo, hi = -max(10, m * 10), max(10, m * 10)
    return lo, hi, 1

# -------------------- Sidebar renderers --------------------

def render_scalar(label: str, value: Any, path: str):
    key = f"w:{path}"
    if isinstance(value, bool):
        return st.checkbox(label, value=bool(value), key=key)
    if isinstance(value, int) and not isinstance(value, bool):
        lo, hi, step = _int_slider_bounds(label, int(value))
        v0 = int(np.clip(int(value), lo, hi))
        return st.slider(label, min_value=int(lo), max_value=int(hi), value=v0, step=int(step), key=key)
    if isinstance(value, float):
        lo, hi, step = _float_slider_bounds(label, float(value))
        v0 = float(np.clip(float(value), lo, hi))
        if hi == lo:
            hi = lo + max(step, 1e-6)
        return st.slider(label, min_value=float(lo), max_value=float(hi), value=v0, step=float(max(step, 1e-6)), key=key)
    if isinstance(value, str):
        return st.text_input(label, value=value, key=key)
    return st.text_area(label, value=json.dumps(value, indent=2), key=key)

def render_list(label: str, value: list, path: str):
    key = f"w:{path}"
    txt = st.text_area(f"{label} (JSON)", value=json.dumps(value, indent=2), height=220, key=key)
    try:
        return json.loads(txt)
    except Exception as e:
        st.warning(f"{label}: JSON parse error — using previous value. ({e})")
        return value

def render_object(label: str, obj: Dict[str, Any], path: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        child_path = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            with st.expander(k, expanded=True):
                out[k] = render_object(k, v, path=child_path)
        elif isinstance(v, list):
            out[k] = render_list(k, v, path=child_path)
        else:
            out[k] = render_scalar(k, v, path=child_path)
    return out

# -------------------- Plot helpers --------------------

def _norm(A: np.ndarray) -> np.ndarray:
    m = float(np.nanmin(A)); M = float(np.nanmax(A))
    if not np.isfinite(m) or not np.isfinite(M) or M - m < 1e-12:
        return np.zeros_like(A)
    return (A - m) / (M - m + 1e-12)

def _apply_floor_gamma(Z: np.ndarray, floor: float, gamma: float) -> np.ndarray:
    Z = np.clip(Z, 0.0, 1.0)
    if gamma != 1.0: Z = Z ** float(gamma)
    if floor > 0.0: Z = np.where(Z >= floor, Z, np.nan)
    return Z

def _resample_rows(M: np.ndarray, new_len: int) -> np.ndarray:
    T, X = M.shape
    if X == new_len: return M
    x_src = np.linspace(0.0, 1.0, X)
    x_tgt = np.linspace(0.0, 1.0, new_len)
    out = np.zeros((T, new_len), dtype=float)
    for t in range(T):
        out[t] = np.interp(x_tgt, x_src, M[t])
    return out

def draw_combined_heatmap(ph, E_stack: np.ndarray, S_stack: np.ndarray, y_row: int,
                          heat_floor: float, heat_gamma: float,
                          env_opacity: float, sub_opacity: float,
                          new_key_fn) -> None:
    E = E_stack; S = S_stack
    if E.ndim == 3: E = E[:, y_row, :]
    if S.ndim == 3: S = S[:, y_row, :]
    if S.shape[1] != E.shape[1]:
        S_res = _resample_rows(S, E.shape[1])
    else:
        S_res = S
    En = _apply_floor_gamma(_norm(E), heat_floor, heat_gamma)
    Sn = _apply_floor_gamma(_norm(S_res), heat_floor, heat_gamma)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=En, coloraxis="coloraxis", zsmooth=False, name="Env", opacity=env_opacity))
    fig.add_trace(go.Heatmap(z=Sn, coloraxis="coloraxis2", zsmooth=False, name="Substrate", opacity=sub_opacity))
    fig.update_layout(
        title="Env + Substrate (combined, zoomable)",
        xaxis_title="x (space)", yaxis_title="t (time)",
        coloraxis=dict(colorscale="Viridis", colorbar=dict(title="Env")),
        coloraxis2=dict(colorscale="Inferno", colorbar=dict(title="Substrate", x=1.08)),
        height=620, template="plotly_dark",
    )
    ph.plotly_chart(fig, use_container_width=True, theme=None, key=new_key_fn("combo2d"))

def assert_equal_grids(E_stack: np.ndarray, S_stack: np.ndarray) -> Tuple[int,int,int]:
    """
    Ensure env and substrate share the exact (T,H,W) grid.
    Returns (T,H,W) on success; stops the app otherwise.
    """
    if E_stack.shape != S_stack.shape:
        st.error(
            f"Grid mismatch: Env {E_stack.shape} vs Substrate {S_stack.shape}. "
            "Physics and plotting must use the same (T,H,W)."
        )
        st.stop()
    T, H, W = S_stack.shape
    return T, H, W

def draw_energy_timeseries(ph, t, e_cell, e_env, e_flux, new_key_fn, title="Energy vs time"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=e_cell, name="E_cell"))
    fig.add_trace(go.Scatter(x=t, y=e_env,  name="E_env"))
    fig.add_trace(go.Scatter(x=t, y=e_flux, name="E_flux"))
    fig.update_layout(xaxis_title="t (frames)", yaxis_title="energy", title=title, height=380, template="plotly_dark")
    ph.plotly_chart(fig, use_container_width=True, theme=None, key=new_key_fn("energy"))

def draw_stats_timeseries(ph, t, entropy=None, variance=None, total_mass=None, new_key_fn=None, title="Telemetry"):
    has_any = False
    fig = go.Figure()
    if entropy is not None and len(entropy) > 0:
        fig.add_trace(go.Scatter(x=t, y=entropy, name="entropy")); has_any = True
    if variance is not None and len(variance) > 0:
        fig.add_trace(go.Scatter(x=t, y=variance, name="variance")); has_any = True
    if total_mass is not None and len(total_mass) > 0:
        fig.add_trace(go.Scatter(x=t, y=total_mass, name="total_mass")); has_any = True
    if not has_any: return
    fig.update_layout(xaxis_title="t (frames)", yaxis_title="value", title=title, height=360, template="plotly_dark")
    ph.plotly_chart(fig, use_container_width=True, theme=None, key=new_key_fn("stats"))

# -------------------- 3‑D helpers & connections --------------------

def _norm_local(A):
    m, M = float(np.nanmin(A)), float(np.nanmax(A))
    if not np.isfinite(m) or not np.isfinite(M) or M - m < 1e-12:
        return np.zeros_like(A)
    return (A - m) / (M - m + 1e-12)

def _draw_3d_env_points(fig: go.Figure, E_stack: np.ndarray, thr: float, portion: float = 0.35, cap_pts: int = 200000):
    E = E_stack
    if E.ndim == 2:  # (T,X) -> (T,1,X)
        E = E[:, None, :]
    En = _norm_local(E)
    tE, yE, xE = np.where(En >= thr)
    n = len(xE)
    if n > 0:
        keep = int(max(1, portion * min(n, cap_pts)))
        idx = np.random.choice(n, size=keep, replace=False)
        xE, yE, tE = xE[idx], yE[idx], tE[idx]
    fig.add_trace(go.Scatter3d(
        x=xE, y=yE, z=tE, mode="markers",
        marker=dict(size=2, opacity=0.55),
        name="Env"
    ))

def draw_3d_points_legacy(
    ph,
    E_stack: np.ndarray,
    S_stack: np.ndarray,
    *,
    thr_points: float,
    max_points_total: int,
    uirevision: str = "points3d"
):
    """
    Legacy sparse 3‑D point cloud: Env dots + Substrate dots.
    Keeps camera/zoom stable with uirevision + stable key.
    """
    fig = go.Figure()

    # Env points (reuse helper)
    _draw_3d_env_points(fig, E_stack, thr=float(thr_points), portion=0.25)

    # Substrate points
    Sn_full = S_stack if S_stack.ndim == 3 else S_stack[:, None, :]
    Sn = _norm_local(Sn_full)
    tS, yS, xS = np.where(Sn >= float(thr_points))
    nS = len(xS)
    if nS > 0:
        keep = int(min(nS, max_points_total // 2))
        idx = np.random.choice(nS, size=keep, replace=False)
        xS, yS, tS = xS[idx], yS[idx], tS[idx]
    fig.add_trace(go.Scatter3d(
        x=xS, y=yS, z=tS, mode="markers",
        marker=dict(size=2, opacity=0.8),
        name="Substrate"
    ))

    fig.update_layout(
        title="Sparse 3‑D energy (points)",
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="t",
            aspectmode="data",
            dragmode="orbit",
        ),
        height=540,
        template="plotly_dark",
        showlegend=True,
        uirevision=uirevision,   # <-- prevents resets between reruns/tool changes
        margin=dict(l=0, r=0, t=40, b=0),
    )

    ph.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        key="points3d_plot",      # <-- stable key so Streamlit doesn't remount
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "doubleClick": "false",
        },
    )

def _time_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    am = a.mean(axis=0); bm = b.mean(axis=0)
    az = a - am; bz = b - bm
    num = (az * bz).mean(axis=0)
    den = a.std(axis=0) * b.std(axis=0) + 1e-12
    return num / den

def _select_edges_windowed(S_stack: np.ndarray,
                           t_end: int,
                           base_win: int,
                           eval_win: int,
                           corr_thr: float,
                           dvar_thr: float,
                           energy_q: float):
    T, H, W = S_stack.shape
    t0_eval = t_end - eval_win + 1
    t0_base = t0_eval - base_win
    if t0_base < 0:
        return np.array([]), np.array([]), np.array([])
    base = S_stack[t0_base:t0_eval]
    eval_ = S_stack[t0_eval:t_end+1]

    var_base = np.var(base, axis=0)
    var_eval = np.var(eval_, axis=0)
    dvar = var_base - var_eval
    mean_abs = np.mean(np.abs(eval_), axis=0)
    floor = float(np.quantile(mean_abs, energy_q))

    a = eval_[:, :, :-1].reshape(eval_win, -1)
    b = eval_[:, :,  1:].reshape(eval_win, -1)
    corr_h = _time_corr(a, b)
    dvar_h = np.minimum(dvar[:, :-1], dvar[:, 1:]).reshape(-1)
    ener_h = np.minimum(mean_abs[:, :-1], mean_abs[:, 1:]).reshape(-1)
    mask_h = (corr_h >= corr_thr) & (dvar_h >= dvar_thr) & (ener_h >= floor)
    y_h, x_h = np.where(mask_h.reshape(H, W-1))

    a = eval_[:, :-1, :].reshape(eval_win, -1)
    b = eval_[:,  1:, :].reshape(eval_win, -1)
    corr_v = _time_corr(a, b)
    dvar_v = np.minimum(dvar[:-1, :], dvar[1:, :]).reshape(-1)
    ener_v = np.minimum(mean_abs[:-1, :], mean_abs[1:, :]).reshape(-1)
    mask_v = (corr_v >= corr_thr) & (dvar_v >= dvar_thr) & (ener_v >= floor)
    y_v, x_v = np.where(mask_v.reshape(H-1, W))

    z = t_end
    xs, ys, zs = [], [], []
    for yy, xx in zip(y_h, x_h):
        xs += [xx, xx+1, None]; ys += [yy, yy,   None]; zs += [z,  z,  None]
    for yy, xx in zip(y_v, x_v):
        xs += [xx,   xx, None]; ys += [yy, yy+1, None]; zs += [z,  z,  None]
    return np.array(xs), np.array(ys), np.array(zs)


def draw_3d_connections_over_time(
    ph,
    E_stack: np.ndarray,
    S_stack: np.ndarray,
    thr_env: float,
    base_win: int,
    eval_win: int,
    corr_thr: float,
    dvar_thr: float,
    energy_q: float,
    stride_t: int,
    max_edges_total: int,
    attr_overlay: bool,
    attr_scale: float,
    attr_alpha: float,
    get_attractor_items_fn,   # kept for compatibility (unused when attr_history provided)
    new_key_fn,               # kept for compatibility
    *,
    camera: dict | None = None,
    uirevision: str = "conn3d",
    attr_history: list | None = None,   # per-frame attractor snapshots
):
    """
    Draw Env dots + time-layered substrate connections in 3D.

    If `attr_history` is provided as [{"t": int, "items":[{id, pos(y,x), amp}]}],
    we (a) draw those attractors and (b) restrict connection segments to the same
    time range (+/- a small padding) so proximity checks are meaningful.

    Returns (optional debug):
        {"conn_x": [...], "conn_y": [...], "conn_z": [...]}
    """
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure()

    # -------- Env points (sparse) --------
    _draw_3d_env_points(fig, E_stack, thr=thr_env, portion=0.30, cap_pts=150000)

    # -------- Determine time range if we have attr_history --------
    T, H, W = S_stack.shape
    t_lo, t_hi = 0, T - 1
    if attr_overlay and isinstance(attr_history, list) and len(attr_history) > 0:
        ts = [int(fr.get("t", 0)) for fr in attr_history if isinstance(fr, dict)]
        if ts:
            pad = max(2, int(max(1, stride_t)))  # small padding around the visible attractor window
            t_lo = max(0, min(ts) - pad)
            t_hi = min(T - 1, max(ts) + pad)

    # -------- Substrate connections over time (restricted to [t_lo, t_hi]) --------
    xs_all, ys_all, zs_all = [], [], []
    budget = int(max_edges_total)

    # windows end at t (so they cover [t-(base_win+eval_win-1) ... t])
    t_start = max(base_win + eval_win - 1, t_lo)
    for t in range(t_start, t_hi + 1, max(1, int(stride_t))):
        xs, ys, zs = _select_edges_windowed(
            S_stack, t_end=t,
            base_win=int(base_win), eval_win=int(eval_win),
            corr_thr=float(corr_thr), dvar_thr=float(dvar_thr), energy_q=float(energy_q)
        )
        if xs.size == 0:
            continue

        # budget guard (keep full triplets [x0,x1,None])
        if (len(xs_all) + len(xs)) > budget:
            keep = max(0, budget - len(xs_all))
            if keep <= 0:
                break
            triplets = len(xs) // 3
            if triplets > 0:
                idx = np.random.choice(triplets, size=max(1, keep // 3), replace=False)
                xs_t, ys_t, zs_t = [], [], []
                for k in idx:
                    i = 3 * k
                    xs_t += [xs[i], xs[i+1], None]
                    ys_t += [ys[i], ys[i+1], None]
                    zs_t += [zs[i], zs[i+1], None]
                xs = np.array(xs_t, dtype=object)
                ys = np.array(ys_t, dtype=object)
                zs = np.array(zs_t, dtype=object)

        xs_all.append(xs); ys_all.append(ys); zs_all.append(zs)

    conn_x, conn_y, conn_z = [], [], []
    if xs_all:
        xs_all = np.concatenate(xs_all)
        ys_all = np.concatenate(ys_all)
        zs_all = np.concatenate(zs_all)

        conn_x = xs_all.tolist()
        conn_y = ys_all.tolist()
        conn_z = zs_all.tolist()

        fig.add_trace(go.Scatter3d(
            x=xs_all, y=ys_all, z=zs_all,
            mode="lines",
            line=dict(width=2),
            name="Substrate connections"
        ))

    # -------- Attractors over time --------
    if attr_overlay:
        if attr_history and isinstance(attr_history, list):
            # points + tracks
            Xp, Yp, Zp, Sz = [], [], [], []
            from collections import defaultdict
            tracks = defaultdict(list)  # id -> [(t,y,x)]

            for frame in attr_history:
                t = int(frame.get("t", 0))
                # skip attractors outside our plotting time window (shouldn't happen with the trim above, but safe)
                if t < t_lo or t > t_hi:
                    continue
                for it in frame.get("items", []):
                    y, x = int(it.get("pos", (0, 0))[0]), int(it.get("pos", (0, 0))[1])
                    amp  = float(it.get("amp", 0.0))
                    Xp.append(x); Yp.append(y); Zp.append(t); Sz.append(2.0 + 6.0 * amp)
                    tracks[int(it.get("id", -1))].append((t, y, x))

            if Xp:
                fig.add_trace(go.Scatter3d(
                    x=Xp, y=Yp, z=Zp,
                    mode="markers",
                    marker=dict(size=Sz, opacity=max(0.15, float(attr_alpha)*0.75)),
                    name="Attractors (pts over time)"
                ))

            Xt, Yt, Zt = [], [], []
            for _id, seq in tracks.items():
                seq.sort(key=lambda p: p[0])
                for i in range(len(seq)-1):
                    t0, y0, x0 = seq[i]
                    t1, y1, x1 = seq[i+1]
                    # only connect if the segment stays inside the window
                    if (t0 < t_lo or t0 > t_hi) or (t1 < t_lo or t1 > t_hi):
                        continue
                    Xt += [x0, x1, None]
                    Yt += [y0, y1, None]
                    Zt += [t0, t1, None]
            if Xt:
                fig.add_trace(go.Scatter3d(
                    x=Xt, y=Yt, z=Zt,
                    mode="lines",
                    line=dict(width=4),
                    opacity=float(attr_alpha),
                    name="Attractor tracks"
                ))
        else:
            # Back-compat: single final snapshot
            try:
                items = get_attractor_items_fn()
                z0 = min(max(T - 1, t_lo), t_hi)
                Xs, Ys, Zs = [], [], []
                for it in items:
                    pos = it.get("pos", (0, 0))
                    y0, x0 = int(pos[0]), int(pos[1])
                    theta = float(it.get("theta", 0.0))
                    r_par = float(it.get("r_par", 1.0))
                    amp   = float(it.get("amp", 0.5))
                    L = max(0.2, r_par) * max(0.2, amp) * float(attr_scale)
                    dx = L * np.cos(theta); dy = L * np.sin(theta)
                    Xs += [x0 - dx, x0 + dx, None]
                    Ys += [y0 - dy, y0 + dy, None]
                    Zs += [z0,      z0,      None]
                if Xs:
                    fig.add_trace(go.Scatter3d(
                        x=np.array(Xs, dtype=object), y=np.array(Ys, dtype=object), z=np.array(Zs, dtype=object),
                        mode="lines",
                        line=dict(width=4),
                        opacity=float(attr_alpha),
                        name="Attractors (final)"
                    ))
            except Exception as e:
                st.info(f"Attractor overlay unavailable ({e}). Continue without it.")

    # -------- Layout --------
    fig.update_layout(
        uirevision=uirevision,
        scene=dict(
            camera=(camera or {}),
            aspectmode="data",
            xaxis_title="x", yaxis_title="y", zaxis_title="t",
        ),
        template="plotly_dark",
        showlegend=True,
        height=640,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    ph.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        key="conn3d_plot",
        config=dict(
            scrollZoom=True,
            displaylogo=False,
            doubleClick="false",
            modeBarButtonsToRemove=[
                "resetCameraDefault3d", "resetCameraLastSave3d",
                "autoScale", "toImage"
            ],
        ),
    )

    return {"conn_x": conn_x, "conn_y": conn_y, "conn_z": conn_z}
