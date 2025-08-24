# app.py
from __future__ import annotations
import json, os, time
from copy import deepcopy
from typing import Any, Dict, Tuple, List

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# from core.config import make_config_from_dict
from core.engine import Engine
from core.metrics import collect as collect_metrics

# ---------- Page ----------
st.set_page_config(page_title="Fuka 3.0 — Free‑Energy Simulation", layout="wide")

# ---------- Strict defaults loading (no silent fallback) ----------
def load_defaults_strict(path: str = "defaults.json") -> Dict[str, Any]:
    if not os.path.exists(path):
        st.error(
            "default.json not found in project root. "
            "Create it (copy from README/template) and remove any comments."
        )
        st.stop()
    try:
        with open(path, "r") as f:
            return json.load(f)  # strict JSON (no // or /* */ comments)
    except Exception as e:
        st.error(
            f"Failed to parse default.json as strict JSON.\n\n"
            f"Tip: remove any // or /* */ comments.\n\n"
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
        "default.json is missing required keys for the simulation. "
        "Please add the following keys and rerun:\n\n"
        + "\n".join(f"• {k}" for k in missing_keys)
    )
    st.stop()

# ---------- session keys for plot uniqueness ----------
for base in ("combo2d_count", "energy_count", "stats_count", "combo3d_count", "run_id"):
    if base not in st.session_state:
        st.session_state[base] = 0

def new_key(base: str) -> str:
    st.session_state[base + "_count"] += 1
    return f"{base}_{st.session_state['run_id']}_{st.session_state[base + '_count']}"

# ---------- Numeric UI helpers (sliders) ----------
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

# ---------- Dynamic sidebar renderers (SLIDERS for numbers) ----------
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

# ---------- Sidebar: render exactly what's in default.json ----------
with st.sidebar:
    st.header("Configuration (from default.json)")
    user_cfg = render_object("", deepcopy(cfg_default))

# Pull out optional UI knobs (present only if in default.json)
had_live  = "live"  in user_cfg
had_chunk = "chunk" in user_cfg
had_thr3d = "thr3d" in user_cfg
had_max3d = "max3d" in user_cfg
had_vis   = "vis"   in user_cfg

# We keep reading these, but we won’t attempt mid‑run plotting (Streamlit limitation).
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

# ---------- Layout placeholders ----------
st.title("Simulation")
combo2d_ph = st.empty()
energy_ph  = st.empty()
stats_ph   = st.empty()
plot3d_ph  = st.empty()

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
    if E.ndim == 3: E = E[:, 0, :]
    if S.ndim == 3: S = S[:, 0, :]
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

def draw_sparse_3d(ph, E_stack: np.ndarray, S_stack: np.ndarray, thr: float, max_points: int):
    def _subsample(xx, yy, zz, vmax):
        n = len(xx)
        if n <= vmax: return xx, yy, zz
        idx = np.random.choice(n, size=vmax, replace=False)
        return xx[idx], yy[idx], zz[idx]

    def _norm_local(A):
        m, M = float(np.nanmin(A)), float(np.nanmax(A))
        if not np.isfinite(m) or not np.isfinite(M) or M - m < 1e-12:
            return np.zeros_like(A)
        return (A - m) / (M - m + 1e-12)

    E = E_stack; S = S_stack
    fig = go.Figure()
    if E.ndim == 2 and S.ndim == 2:
        En = _norm_local(E); Sn = _norm_local(S)
        tE, xE = np.where(En >= thr); tS, xS = np.where(Sn >= thr)
        zE, yE = tE, np.zeros_like(tE); zS, yS = tS, np.ones_like(tS)
        xE, yE, zE = _subsample(xE, yE, zE, max_points // 2)
        xS, yS, zS = _subsample(xS, yS, zS, max_points // 2)
        fig.add_trace(go.Scatter3d(x=xE, y=yE, z=zE, mode="markers", marker=dict(size=2, opacity=0.7), name="Env"))
        fig.add_trace(go.Scatter3d(x=xS, y=yS, z=zS, mode="markers", marker=dict(size=2, opacity=0.8), name="Substrate"))
        scene = dict(xaxis_title="x", yaxis_title="layer", zaxis_title="t")
    else:
        if E.ndim == 2: E = E[:, None, :]
        if S.ndim == 2: S = S[:, None, :]
        En = _norm_local(E); Sn = _norm_local(S)
        tE, yE, xE = np.where(En >= thr); tS, yS, xS = np.where(Sn >= thr)
        xE, yE, zE = _subsample(xE, yE, tE, max_points // 2)
        xS, yS, zS = _subsample(xS, yS, tS, max_points // 2)
        fig.add_trace(go.Scatter3d(x=xE, y=yE, z=zE, mode="markers", marker=dict(size=2, opacity=0.5), name="Env"))
        fig.add_trace(go.Scatter3d(x=xS, y=yS, z=zS, mode="markers", marker=dict(size=2, opacity=0.8), name="Substrate"))
        scene = dict(xaxis_title="x", yaxis_title="y", zaxis_title="t")
    fig.update_layout(title="Sparse 3‑D energy", scene=scene, height=640, template="plotly_dark", showlegend=True)
    ph.plotly_chart(fig, use_container_width=True, theme=None, key=new_key("combo3d"))

# ---------- Run ----------
if st.button("Run / Rerun", use_container_width=True):
    

                
    st.session_state["run_id"] += 1
    st.session_state["combo2d_count"] = 0
    st.session_state["energy_count"]  = 0
    st.session_state["stats_count"]   = 0
    st.session_state["combo3d_count"] = 0

    # ecfg = make_config_from_dict(user_cfg)
    # engine = Engine(ecfg)
    
    engine = Engine(user_cfg)   # Engine now normalizes dicts itself

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

    # Progress bar (updates are visible during execution)
    prog = st.progress(0, text="Running simulation…")
    status = st.empty()

    ok = True
    try:
        for step_idx in range(T):
            S, flux, E = engine.step()
            if step_idx == 0 or step_idx == chunk:
                st.write({"E_min": float(np.min(E)), "E_max": float(np.max(E)),
                "E_mean": float(np.mean(E)), "E_any": bool(np.any(E)),
                "E_shape": E.shape})
                
            # Append frames & stats
            env_frames.append(E.copy())
            sub_frames.append(S.copy())
            t_series.append(step_idx)

            metrics_rows = collect_metrics()
            if metrics_rows:
                m0 = metrics_rows[0]
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
            E_stack = np.stack(env_frames, axis=0)
            S_stack = np.stack(sub_frames, axis=0)
        except Exception as e:
            st.exception(e)
            st.stop()

        draw_combined_heatmap(combo2d_ph, E_stack, S_stack)
        draw_energy_timeseries(energy_ph, t_series, e_cell_series, e_env_series, e_flux_series)
        draw_stats_timeseries(stats_ph, t_series, entropy_series, variance_series, total_mass_series)
        if (thr3d is not None) and (max3d is not None):
            draw_sparse_3d(plot3d_ph, E_stack, S_stack, thr=float(thr3d), max_points=int(max3d))
        else:
            st.warning("3‑D view disabled: add 'thr3d' and 'max3d' to default.json to enable.")