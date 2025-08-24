from __future__ import annotations
import json, os
from copy import deepcopy
from typing import Any, Dict, Tuple, List

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from core import physics
from core.engine import Engine

# ---------- Page ----------
st.set_page_config(page_title="Fuka 3.0 — Free‑Energy Simulation", layout="wide")

# ---------- Strict defaults loading (no silent fallback) ----------
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

# ---------- Minimal schema validation ----------
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

cfg_default = load_defaults_strict()
missing_keys = find_missing(cfg_default)
if missing_keys:
    st.error(
        "defaults.json is missing required keys for the simulation. "
        "Please add the following keys and rerun:\n\n"
        + "\n".join(f"• {k}" for k in missing_keys)
    )
    st.stop()

# ---------- session keys for plot uniqueness ----------
for base in ("combo2d_count", "energy_count", "stats_count", "combo3d_count", "conn3d_count", "run_id"):
    if base not in st.session_state:
        st.session_state[base] = 0

def new_key(base: str) -> str:
    st.session_state[base + "_count"] += 1
    return f"{base}_{st.session_state['run_id']}_{st.session_state[base + '_count']}"

# ---------- Numeric UI helpers ----------
def _num_step(v: float) -> float:
    v = abs(float(v)) if v != 0 else 1.0
    mag = 10 ** int(np.floor(np.log10(v)))
    return max(mag * 0.01, 10 ** (int(np.floor(np.log10(v))) - 2))

def _float_slider_bounds(label: str, val: float) -> Tuple[float, float, float]:
    name = label.lower()
    step = round(_num_step(val), 6)
    if 0.0 <= val <= 1.0 or any(s in name for s in ["sigma", "noise", "decay", "diffuse", "k_", "thr", "threshold", "opacity", "gamma", "floor"]):
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

# ---------- Dynamic sidebar renderers ----------
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
        return st.slider(label, min_value=float(lo), max_value=float(hi), value=v0, step=float(step), key=key)
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

# ---------- Sidebar: render defaults.json exactly ----------
with st.sidebar:
    st.header("Configuration (from defaults.json)")
    user_cfg = render_object("", deepcopy(cfg_default))

# Optional UI knobs (present only if in defaults.json)
had_live  = "live"  in user_cfg
had_chunk = "chunk" in user_cfg
had_thr3d = "thr3d" in user_cfg
had_max3d = "max3d" in user_cfg
had_vis   = "vis"   in user_cfg

live  = bool(user_cfg.pop("live",  True))   if had_live  else None
chunk = int(user_cfg.pop("chunk",  150))    if had_chunk else None
thr3d = float(user_cfg.pop("thr3d", 0.75))  if had_thr3d else None
max3d = int(user_cfg.pop("max3d",  40000))  if had_max3d else None
vis   = user_cfg.pop("vis", {})             if had_vis   else None

# Visual knobs for 2‑D heatmap
if vis is None:
    heat_floor, heat_gamma, env_opacity, sub_opacity = 0.10, 1.0, 1.0, 0.85
else:
    heat_floor  = float(vis.get("heat_floor", 0.10))
    heat_gamma  = float(vis.get("heat_gamma", 1.0))
    env_opacity = float(vis.get("env_opacity", 1.0))
    sub_opacity = float(vis.get("sub_opacity", 0.85))

# ---------- New: 3‑D connections UI ----------
with st.sidebar.expander("3‑D Connections (substrate encoding)", expanded=True):
    conn_enable = st.checkbox("Show connections", value=True, key="conn:enable")
    conn_thr_env = st.slider("Env point threshold (norm)", 0.0, 1.0, float(thr3d if thr3d is not None else 0.1), 0.01, key="conn:thr_env")
    conn_base_win = st.slider("Baseline window (frames)", 8, 256, 64, 1, key="conn:base")
    conn_eval_win = st.slider("Evaluation window (frames)", 8, 256, 64, 1, key="conn:eval")
    conn_corr_thr = st.slider("Neighbour time-corr threshold", 0.0, 1.0, 0.6, 0.01, key="conn:corr")
    conn_dvar_thr = st.slider("Variance drop Δ threshold", 0.0, 1e-2, 1e-4, 1e-5, key="conn:dvar")
    conn_energy_q = st.slider("Energy floor (quantile)", 0.0, 1.0, 0.70, 0.01, key="conn:q")
    conn_stride_t = st.slider("Time stride for connections", 1, 20, 4, 1, key="conn:stride")
    conn_max_edges = st.slider("Max edges plotted", 1000, 100000, int(max3d if max3d is not None else 40000), 1000, key="conn:max")

# ---------- Layout placeholders ----------
st.title("Simulation")
combo2d_ph = st.empty()
energy_ph  = st.empty()
stats_ph   = st.empty()
points3d_ph  = st.empty()   # old point cloud (Env/Substrate)
conn3d_ph    = st.empty()   # new connections view

# ---------- Plot helpers ----------
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

def draw_combined_heatmap(ph, E_stack: np.ndarray, S_stack: np.ndarray, title="Env + Substrate (combined, zoomable)"):
    E = E_stack; S = S_stack
    if E.ndim == 3: E = E[:, E.shape[1] // 2, :]
    if S.ndim == 3: S = S[:, S.shape[1] // 2, :]
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
        title=title, xaxis_title="x (space)", yaxis_title="t (time)",
        coloraxis=dict(colorscale="Viridis", colorbar=dict(title="Env")),
        coloraxis2=dict(colorscale="Inferno", colorbar=dict(title="Substrate", x=1.08)),
        height=620, template="plotly_dark",
    )
    ph.plotly_chart(fig, use_container_width=True, theme=None, key=new_key("combo2d"))

def draw_energy_timeseries(ph, t, e_cell, e_env, e_flux, title="Energy vs time"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=e_cell, name="E_cell"))
    fig.add_trace(go.Scatter(x=t, y=e_env,  name="E_env"))
    fig.add_trace(go.Scatter(x=t, y=e_flux, name="E_flux"))
    fig.update_layout(xaxis_title="t (frames)", yaxis_title="energy", title=title, height=380, template="plotly_dark")
    ph.plotly_chart(fig, use_container_width=True, theme=None, key=new_key("energy"))

def draw_stats_timeseries(ph, t, entropy=None, variance=None, total_mass=None, title="Telemetry (entropy / variance / mass)"):
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
    ph.plotly_chart(fig, use_container_width=True, theme=None, key=new_key("stats"))

# ----- 3‑D: env dots -----
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

# ----- 3‑D: connections (per‑frame layered segments) -----
def _time_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a,b shape: (T,N)
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
    """
    At time t_end, compare variance in [t_end-eval_win+1, t_end] to baseline
    variance in the window immediately before: [t_end-eval_win-base_win+1, ..., t_end-eval_win].
    Return arrays of line segments (xs, ys, zs) at z=t_end for edges that satisfy:
      - neighbour time-corr >= corr_thr over the eval window
      - variance dropped by >= dvar_thr at both endpoints
      - mean |S| over eval window above energy quantile floor
    """
    T, H, W = S_stack.shape
    t0_eval = t_end - eval_win + 1
    t0_base = t0_eval - base_win
    if t0_base < 0:
        return np.array([]), np.array([]), np.array([])
    base = S_stack[t0_base:t0_eval]  # (base_win,H,W)
    eval_ = S_stack[t0_eval:t_end+1] # (eval_win,H,W)

    var_base = np.var(base, axis=0)
    var_eval = np.var(eval_, axis=0)
    dvar = var_base - var_eval
    mean_abs = np.mean(np.abs(eval_), axis=0)
    floor = float(np.quantile(mean_abs, energy_q))

    # horizontal edges
    a = eval_[:, :, :-1].reshape(eval_win, -1)
    b = eval_[:, :,  1:].reshape(eval_win, -1)
    corr_h = _time_corr(a, b)
    dvar_h = np.minimum(dvar[:, :-1], dvar[:, 1:]).reshape(-1)
    ener_h = np.minimum(mean_abs[:, :-1], mean_abs[:, 1:]).reshape(-1)
    mask_h = (corr_h >= corr_thr) & (dvar_h >= dvar_thr) & (ener_h >= floor)
    y_h, x_h = np.where(mask_h.reshape(H, W-1))

    # vertical edges
    a = eval_[:, :-1, :].reshape(eval_win, -1)
    b = eval_[:,  1:, :].reshape(eval_win, -1)
    corr_v = _time_corr(a, b)
    dvar_v = np.minimum(dvar[:-1, :], dvar[1:, :]).reshape(-1)
    ener_v = np.minimum(mean_abs[:-1, :], mean_abs[1:, :]).reshape(-1)
    mask_v = (corr_v >= corr_thr) & (dvar_v >= dvar_thr) & (ener_v >= floor)
    y_v, x_v = np.where(mask_v.reshape(H-1, W))

    z = t_end
    xs, ys, zs = [], [], []
    # horizontal segments
    for yy, xx in zip(y_h, x_h):
        xs += [xx, xx+1, None]; ys += [yy, yy,   None]; zs += [z,  z,  None]
    # vertical segments
    for yy, xx in zip(y_v, x_v):
        xs += [xx,   xx, None]; ys += [yy, yy+1, None]; zs += [z,  z,  None]

    return np.array(xs), np.array(ys), np.array(zs)

def draw_3d_connections_over_time(ph, E_stack: np.ndarray, S_stack: np.ndarray,
                                  thr_env: float,
                                  base_win: int,
                                  eval_win: int,
                                  corr_thr: float,
                                  dvar_thr: float,
                                  energy_q: float,
                                  stride_t: int,
                                  max_edges_total: int):
    """
    Env: dots. Substrate: time‑layered connections; each frame contributes segments at z=t.
    """
    fig = go.Figure()

    # Env dots (single pass)
    _draw_3d_env_points(fig, E_stack, thr=thr_env, portion=0.30, cap_pts=150000)

    # Build connection segments across time with stride
    T, H, W = S_stack.shape
    xs_all, ys_all, zs_all = [], [], []
    budget = max_edges_total
    for t in range(base_win + eval_win - 1, T, max(1, stride_t)):
        xs, ys, zs = _select_edges_windowed(
            S_stack, t_end=t,
            base_win=base_win, eval_win=eval_win,
            corr_thr=corr_thr, dvar_thr=dvar_thr, energy_q=energy_q
        )
        if xs.size == 0:
            continue
        if len(xs_all) + len(xs) > budget:
            # trim this frame's segments to fit budget
            keep = max(0, budget - len(xs_all))
            if keep <= 0:
                break
            idx = np.random.choice(len(xs)//3, size=keep//3, replace=False)  # triplets + None-separators
            # rebuild trimmed arrays
            xs_t = []; ys_t = []; zs_t = []
            # triplets grouped as [x0, x1, None], etc.
            for k in idx:
                i = 3*k
                xs_t += [xs[i], xs[i+1], None]
                ys_t += [ys[i], ys[i+1], None]
                zs_t += [zs[i], zs[i+1], None]
            xs, ys, zs = np.array(xs_t), np.array(ys_t), np.array(zs_t)
        xs_all.append(xs); ys_all.append(ys); zs_all.append(zs)

    if xs_all:
        xs_all = np.concatenate(xs_all)
        ys_all = np.concatenate(ys_all)
        zs_all = np.concatenate(zs_all)
        fig.add_trace(go.Scatter3d(
            x=xs_all, y=ys_all, z=zs_all, mode="lines",
            line=dict(width=2),
            name="Substrate connections"
        ))

    fig.update_layout(
        title="3‑D: Env dots + Substrate connections over time",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="t"),
        height=640,
        template="plotly_dark",
        showlegend=True,
    )
    ph.plotly_chart(fig, use_container_width=True, theme=None, key=new_key("conn3d"))

# ---------- Run ----------
if st.button("Run / Rerun", use_container_width=True):
    st.session_state["run_id"] += 1
    st.session_state["combo2d_count"] = 0
    st.session_state["energy_count"]  = 0
    st.session_state["stats_count"]   = 0
    st.session_state["combo3d_count"] = 0
    st.session_state["conn3d_count"]  = 0

    engine = Engine(user_cfg)   # Engine normalizes dicts itself

    # Histories
    T = int(engine.cfg.get("frames", 2000))
    env_frames: List[np.ndarray] = []
    sub_frames: List[np.ndarray] = []
    t_series: List[int] = []
    e_cell_series: List[float] = []
    e_env_series: List[float] = []
    e_flux_series: List[float] = []
    entropy_series: List[float] = []
    variance_series: List[float] = []
    total_mass_series: List[float] = []

    prog = st.progress(0, text="Running simulation…")
    status = st.empty()

    ok = True
    try:
        for step_idx in range(T):
            S, flux, E = engine.step()

            env_frames.append(E.copy())
            sub_frames.append(S.copy())
            t_series.append(step_idx)

            rows = physics.get_fuka3_metrics()
            if rows:
                m0 = rows[0]
                e_cell = float(m0["free_energy_total"] + m0["bound_energy_total"])
                ent = float(m0["entropy_mean"])
            else:
                e_cell = 0.0
                ent = 0.0

            e_env = float(np.sum(np.abs(E)))
            varS = float(np.var(S))
            massS = float(np.sum(np.abs(S)))

            e_cell_series.append(e_cell)
            e_env_series.append(e_env)
            e_flux_series.append(float(flux))
            entropy_series.append(ent)
            variance_series.append(varS)
            total_mass_series.append(massS)

            # progress UI
            if T > 0:
                pct = int(100 * (step_idx + 1) / T)
                prog.progress(pct, text=f"Running simulation… {pct}%")
                if (step_idx % 50) == 0:
                    status.write(f"Frame {step_idx+1}/{T}  |  ⌀|S|={np.mean(np.abs(S)):.4f}  flux={flux:.5f}")
    except Exception as e:
        ok = False
        prog.empty()
        st.exception(e)

    if ok:
        prog.progress(100, text="Simulation complete")

        # Stack and draw
        try:
            E_stack = np.stack(env_frames, axis=0)  # (T, H_env, W_env)
            S_stack = np.stack(sub_frames, axis=0)  # (T, H, W)
        except Exception as e:
            st.exception(e)
            st.stop()

        # 2‑D overlays
        draw_combined_heatmap(combo2d_ph, E_stack, S_stack)

        # Basic energy & stats
        draw_energy_timeseries(energy_ph, t_series, e_cell_series, e_env_series, e_flux_series)
        draw_stats_timeseries(stats_ph, t_series, entropy_series, variance_series, total_mass_series)

        # Old 3‑D point cloud (optional; keeps parity)
        if (thr3d is not None) and (max3d is not None):
            fig_pts = go.Figure()
            # env
            _draw_3d_env_points(fig_pts, E_stack, thr=float(thr3d), portion=0.25)
            # substrate points
            Sn = _norm_local(S_stack if S_stack.ndim == 3 else S_stack[:, None, :])
            tS, yS, xS = np.where(Sn >= float(thr3d))
            nS = len(xS)
            if nS > 0:
                keep = int(min(nS, max3d // 2))
                idx = np.random.choice(nS, size=keep, replace=False)
                xS, yS, tS = xS[idx], yS[idx], tS[idx]
            fig_pts.add_trace(go.Scatter3d(x=xS, y=yS, z=tS, mode="markers", marker=dict(size=2, opacity=0.8), name="Substrate"))
            fig_pts.update_layout(title="Sparse 3‑D energy (points)", scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="t"),
                                  height=540, template="plotly_dark", showlegend=True)
            points3d_ph.plotly_chart(fig_pts, use_container_width=True, theme=None, key=new_key("combo3d"))
        else:
            st.warning("3‑D points view disabled: add 'thr3d' and 'max3d' to defaults.json to enable.")

        # New 3‑D connections (per‑frame layers)
        if conn_enable and (max3d is not None):
            draw_3d_connections_over_time(
                conn3d_ph,
                E_stack, S_stack,
                thr_env=float(conn_thr_env),
                base_win=int(conn_base_win),
                eval_win=int(conn_eval_win),
                corr_thr=float(conn_corr_thr),
                dvar_thr=float(conn_dvar_thr),
                energy_q=float(conn_energy_q),
                stride_t=int(conn_stride_t),
                max_edges_total=int(conn_max_edges),
            )
        else:
            st.info("Connections view is off. Enable it in the sidebar.")