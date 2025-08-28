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
    if "run_id" not in st.session_state or not st.session_state["run_id"]:
        # stable-ish ID per app lifetime
        st.session_state["run_id"] = str(np.random.randint(1_000_000_000))

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
    if step < 1e-6:
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

# -------------------- Plot helpers (2-D) --------------------

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

# -------------------- 3-D helpers & connections --------------------

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
    return np.array(xs, dtype=object), np.array(ys, dtype=object), np.array(zs, dtype=object)

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
    attr_history: list | None = None,   # per-frame attractor snapshots (full history)
    debug_show_midpoints: bool = True,
):
    """
    Draw Env dots + time-layered substrate connections in 3D.

    If `attr_history` is provided as [{"t": int, "items":[{id, pos(y,x), amp, ...}]}],
    we draw those attractors across the whole run (no trimming) and color by encoding.
    Returns a dict with polyline arrays for optional debugging.
    """
    fig = go.Figure()

    # -------- Env points (sparse) --------
    _draw_3d_env_points(fig, E_stack, thr=float(thr_env), portion=0.30, cap_pts=150000)

    # -------- Time span (all frames by default) --------
    T, H, W = S_stack.shape
    t_lo, t_hi = 0, T - 1

    # -------- Substrate connections over time (no overlay dependency) --------
    xs_all, ys_all, zs_all = [], [], []
    budget = int(max_edges_total)

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
    total_segments = 0
    near_xy = 0
    near_yx = 0
    dlist_xy = []
    dlist_yx = []

    if xs_all:
        xs_all = np.concatenate(xs_all)
        ys_all = np.concatenate(ys_all)
        zs_all = np.concatenate(zs_all)

        conn_x = xs_all.tolist()
        conn_y = ys_all.tolist()
        conn_z = zs_all.tolist()

        # --- collect midpoints and basic near stats against attractors (if provided) ---
        attr_pts_by_t = {}
        if attr_overlay and isinstance(attr_history, list):
            for frame in attr_history:
                t = int(frame.get("t", 0))
                pts = attr_pts_by_t.setdefault(t, [])
                for it in frame.get("items", []):
                    y, x = int(it.get("pos", (0, 0))[0]), int(it.get("pos", (0, 0))[1])
                    pts.append((float(x), float(y)))  # store as (x,y)

        mid_x, mid_y, mid_z = [], [], []
        radius_px = 3.0
        r2 = radius_px * radius_px
        n = len(xs_all)
        i = 0
        while i + 2 < n:
            x0, x1, xN = xs_all[i], xs_all[i+1], xs_all[i+2]
            y0, y1, yN = ys_all[i], ys_all[i+1], ys_all[i+2]
            z0, z1, zN = zs_all[i], zs_all[i+1], zs_all[i+2]
            i += 3

            if x0 is None or x1 is None or xN is not None:  continue
            if y0 is None or y1 is None or yN is not None:  continue
            if z0 is None or z1 is None or zN is not None:  continue
            if z0 != z1:                                    continue

            total_segments += 1
            t = int(z0)
            pts_xy = attr_pts_by_t.get(t, [])

            xm = 0.5 * (float(x0) + float(x1))
            ym = 0.5 * (float(y0) + float(y1))

            if debug_show_midpoints:
                mid_x.append(xm); mid_y.append(ym); mid_z.append(t)

            if pts_xy:
                d2_xy = min((xm - ax)**2 + (ym - ay)**2 for (ax, ay) in pts_xy)
                dlist_xy.append(d2_xy**0.5)
                if d2_xy <= r2:
                    near_xy += 1

                # swap mapping check
                d2_yx = min((ym - ax)**2 + (xm - ay)**2 for (ax, ay) in pts_xy)
                dlist_yx.append(d2_yx**0.5)
                if d2_yx <= r2:
                    near_yx += 1

        # plot the connections
        fig.add_trace(go.Scatter3d(
            x=xs_all, y=ys_all, z=zs_all,
            mode="lines",
            line=dict(width=2),
            name="Substrate connections"
        ))

        if debug_show_midpoints and mid_x:
            fig.add_trace(go.Scatter3d(
                x=np.array(mid_x), y=np.array(mid_y), z=np.array(mid_z),
                mode="markers",
                marker=dict(size=2, opacity=0.35),
                name="Conn midpoints"
            ))

    # -------- Attractors over time (no trimming) --------
    if attr_overlay:
        enc_thr = 0.5  # threshold to separate Structured vs Noisy categories (by encoding)

        if attr_history and isinstance(attr_history, list):
            # points across the entire run
            Xp, Yp, Zp, Sz = [], [], [], []
            Enc, Sig, Ids, Amp = [], [], [], []
            from collections import defaultdict
            tracks = defaultdict(list)  # id -> [(t,y,x)]

            for frame in attr_history:
                t = int(frame.get("t", 0))
                for it in frame.get("items", []):
                    y, x = int(it.get("pos", (0, 0))[0]), int(it.get("pos", (0, 0))[1])
                    amp   = float(it.get("amp", 0.0))
                    enc   = float(it.get("encoding", it.get("enc", 0.0)))
                    sig   = float(it.get("sigma_signal", it.get("sigma", np.nan)))
                    _id   = int(it.get("id", -1))

                    Xp.append(x); Yp.append(y); Zp.append(t)
                    Sz.append(2.0 + 6.0 * max(0.0, amp))
                    Enc.append(float(np.clip(enc, 0.0, 1.0)))
                    Sig.append(sig)
                    Ids.append(_id); Amp.append(amp)

                    tracks[_id].append((t, y, x))

            if Xp:
                # continuous color by encoding
                fig.add_trace(go.Scatter3d(
                    x=Xp, y=Yp, z=Zp,
                    mode="markers",
                    marker=dict(
                        size=Sz,
                        opacity=max(0.15, float(attr_alpha)*0.75),
                        color=Enc, colorscale="Viridis", cmin=0.0, cmax=1.0,
                        colorbar=dict(title="encoding", x=1.16, y=0.5, len=0.70, thickness=12)
                    ),
                    hovertemplate=(
                        "t=%{z}<br>x=%{x}, y=%{y}"
                        "<br>id=%{customdata[0]} amp=%{customdata[1]:.3f}"
                        "<br>enc=%{customdata[2]:.2f} σ_sig=%{customdata[3]:.2f}<extra></extra>"
                    ),
                    customdata=np.column_stack([Ids, Amp, Enc, Sig]),
                    name="Attractors (pts over time)"
                ))

                # legend categories: Structured vs Noisy by encoding threshold
                cat_struct_x, cat_struct_y, cat_struct_z = [], [], []
                cat_noisy_x,  cat_noisy_y,  cat_noisy_z  = [], [], []
                for x, y, z, enc in zip(Xp, Yp, Zp, Enc):
                    if enc >= enc_thr:
                        cat_struct_x.append(x); cat_struct_y.append(y); cat_struct_z.append(z)
                    else:
                        cat_noisy_x.append(x);  cat_noisy_y.append(y);  cat_noisy_z.append(z)

                if cat_struct_x:
                    fig.add_trace(go.Scatter3d(
                        x=cat_struct_x, y=cat_struct_y, z=cat_struct_z,
                        mode="markers",
                        marker=dict(size=3, opacity=0.35),
                        name=f"Structured (enc ≥ {enc_thr:.2f})",
                        showlegend=True
                    ))
                if cat_noisy_x:
                    fig.add_trace(go.Scatter3d(
                        x=cat_noisy_x, y=cat_noisy_y, z=cat_noisy_z,
                        mode="markers",
                        marker=dict(size=3, opacity=0.20),
                        name=f"Noisy (enc < {enc_thr:.2f})",
                        showlegend=True
                    ))

            # --- draw short tracks ONLY when an attractor persists across adjacent frames ---
            Xt, Yt, Zt = [], [], []
            consecutive_gap = 1  # connect only if frames are consecutive (<= 1 apart)

            for _id, seq in tracks.items():
                # seq contains tuples: (t, y, x)
                seq.sort(key=lambda p: p[0])
                for i in range(len(seq) - 1):
                    t0, y0, x0 = seq[i]
                    t1, y1, x1 = seq[i + 1]

                    # keep only segments inside our visible time range
                    if (t0 < t_lo or t0 > t_hi) or (t1 < t_lo or t1 > t_hi):
                        continue

                    # *** Option B rule: only connect if the same id appears in adjacent frames ***
                    if (t1 - t0) <= consecutive_gap:
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
            # fallback: single-snapshot API (kept for compatibility)
            try:
                items = get_attractor_items_fn()
                z0 = T - 1
                Xs, Ys, Zs = [], [], []
                Cc = []  # color by encoding if available
                for it in items:
                    pos = it.get("pos", (0, 0))
                    y0, x0 = int(pos[0]), int(pos[1])
                    theta = float(it.get("theta", 0.0))
                    r_par = float(it.get("r_par", 1.0))
                    amp   = float(it.get("amp", 0.5))
                    enc   = float(it.get("encoding", 0.0))
                    L = max(0.2, r_par) * max(0.2, amp) * float(attr_scale)
                    dx = L * np.cos(theta); dy = L * np.sin(theta)
                    Xs += [x0 - dx, x0 + dx, None]
                    Ys += [y0 - dy, y0 + dy, None]
                    Zs += [z0,      z0,      None]
                    Cc += [enc, enc, None]

                if Xs:
                    fig.add_trace(go.Scatter3d(
                        x=np.array(Xs, dtype=object), y=np.array(Ys, dtype=object), z=np.array(Zs, dtype=object),
                        mode="lines",
                        line=dict(width=4),
                        opacity=float(attr_alpha),
                        marker=dict(color=Cc, colorscale="Viridis", cmin=0.0, cmax=1.0),
                        name="Attractors (final)"
                    ))
            except Exception as e:
                st.info(f"Attractor overlay unavailable ({e}). Continue without it.)")

    # -------- Title & diagnostics --------
    def _stats(vals):
        if not vals:
            return {"min": float("inf"), "median": float("inf")}
        a = np.asarray(vals, dtype=float)
        return {"min": float(np.min(a)), "median": float(np.median(a))}

    stats_xy = _stats(dlist_xy)
    stats_yx = _stats(dlist_yx)

    title_txt = "3-D connections"
    if total_segments > 0:
        r_xy = 100.0 * near_xy / float(total_segments)
        r_yx = 100.0 * near_yx / float(total_segments)
        title_txt += (f" — near@XY: {near_xy}/{total_segments} ({r_xy:.1f}%), "
                      f"near@YX: {near_yx}/{total_segments} ({r_yx:.1f}%)")

    fig.update_layout(
        title=title_txt,
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

    # quick caption with distance diagnostics
    try:
        st.caption(
            f"Distance stats — mapping XY: min {stats_xy['min']:.2f}, med {stats_xy['median']:.2f} | "
            f"mapping YX: min {stats_yx['min']:.2f}, med {stats_yx['median']:.2f}"
        )
    except Exception:
        pass

    return {
        "conn_x": conn_x,
        "conn_y": conn_y,
        "conn_z": conn_z,
        "near_xy": int(near_xy),
        "near_yx": int(near_yx),
        "total_segments": int(total_segments),
        "dist_xy": {"min": float(stats_xy["min"]), "median": float(stats_xy["median"])},
        "dist_yx": {"min": float(stats_yx["min"]), "median": float(stats_yx["median"])},
    }