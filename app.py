# app.py
from __future__ import annotations
import json
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from core import physics
from core.engine import Engine

from core import debug as dbg  # ← optional debug helpers

# optional camera controls module (safe if unused)
try:
    from ui_camera import camera_controls  # noqa: F401
except Exception:
    camera_controls = None  # not required

from app_helpers import (
    load_defaults_strict, find_missing,
    ensure_session_keys, new_key,
    render_object,
    draw_combined_heatmap, draw_energy_timeseries, draw_stats_timeseries,
    draw_3d_connections_over_time,
    assert_equal_grids,
)

# ---------------- Page ----------------
st.set_page_config(page_title="Fuka 3.0 — Free-Energy Simulation", layout="wide")

# ---------------- Config load & validate ----------------
cfg_default = load_defaults_strict()
missing_keys = find_missing(cfg_default)
if missing_keys:
    st.error(
        "defaults.json is missing required keys for the simulation. "
        "Please add the following keys and rerun:\n\n" + "\n".join(f"• {k}" for k in missing_keys)
    )
    st.stop()

# ---------------- Session keys ----------------
ensure_session_keys()
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None

# ---------------- Sidebar: render config ----------------
with st.sidebar:
    st.header("Configuration (from defaults.json)")
    user_cfg = render_object("", deepcopy(cfg_default))
    test_number = str(user_cfg.get("test_number", "")).strip()
    if test_number:
        st.subheader(f"Test: {test_number}")

# Optional UI knobs (present only if in defaults.json)
had_live  = "live"  in user_cfg
had_chunk = "chunk" in user_cfg
had_thr3d = "thr3d" in user_cfg
had_max3d = "max3d" in user_cfg
had_vis   = "vis"   in user_cfg
had_conn  = "connections" in user_cfg

live    = bool(user_cfg.pop("live",  True)) if had_live  else None
chunk   = int(user_cfg.pop("chunk",  150))  if had_chunk else None
thr3d   = float(user_cfg.pop("thr3d", 0.75)) if had_thr3d else None
max3d   = int(user_cfg.pop("max3d",  40000)) if had_max3d else None
vis     = user_cfg.pop("vis", {}) if had_vis else None
conn_cfg = user_cfg.pop("connections", {}) if had_conn else {}

# Visual knobs for 2-D heatmap overlays
if vis is None:
    heat_floor, heat_gamma, env_opacity, sub_opacity = 0.10, 1.0, 1.0, 0.85
else:
    heat_floor  = float(vis.get("heat_floor", 0.10))
    heat_gamma  = float(vis.get("heat_gamma", 1.0))
    env_opacity = float(vis.get("env_opacity", 1.0))
    sub_opacity = float(vis.get("sub_opacity", 0.85))

# ---------------- Connections UI ----------------
with st.sidebar.expander("3-D Connections (substrate encoding)", expanded=True):
    conn_enable = st.checkbox("Show connections", value=bool(conn_cfg.get("enable", True)), key="conn:enable")
    conn_thr_env = st.slider("Env point threshold (norm)", 0.0, 1.0,
                             float(conn_cfg.get("thr_env", thr3d if thr3d is not None else 0.1)), 0.01, key="conn:thr_env")
    conn_base_win = st.slider("Baseline window (frames)", 8, 256, int(conn_cfg.get("base_window", 64)), 1, key="conn:base")
    conn_eval_win = st.slider("Evaluation window (frames)", 8, 256, int(conn_cfg.get("eval_window", 64)), 1, key="conn:eval")
    conn_corr_thr = st.slider("Neighbour time-corr threshold", 0.0, 1.0, float(conn_cfg.get("corr_threshold", 0.6)), 0.01, key="conn:corr")
    conn_dvar_thr = st.slider("Variance drop Δ threshold", 0.0, 1e-2, float(conn_cfg.get("dvar_threshold", 1e-4)), 1e-5, key="conn:dvar")
    conn_energy_q = st.slider("Energy floor (quantile)", 0.0, 1.0, float(conn_cfg.get("energy_quantile", 0.70)), 0.01, key="conn:q")
    conn_stride_t = st.slider("Time stride for connections", 1, 20, int(conn_cfg.get("stride_t", 4)), 1, key="conn:stride")
    conn_max_edges = st.slider("Max edges plotted", 1000, 100000,
                               int(conn_cfg.get("max_edges", max3d if max3d is not None else 40000)), 1000, key="conn:max")
                            
    st.markdown("---")
    dbg_enable = st.checkbox("Debug: measure connections near attractors", value=True, key="conn:dbg")
    dbg_radius = st.slider("Debug radius (grid cells)", 1, 8, 4, 1, key="conn:dbgR")

# ---------------- Attractors overlay UI ----------------
with st.sidebar.expander("Attractors overlay (3-D)", expanded=False):
    # default CHECKED now
    attr_enable = st.checkbox("Show attractors (needs physics.get_attractors_snapshot)", value=True, key="attr:enable")
    attr_scale  = st.slider("Glyph length scale", 0.1, 5.0, 1.0, 0.1, key="attr:scale")
    attr_alpha  = st.slider("Opacity", 0.1, 1.0, 0.8, 0.05, key="attr:alpha")
    # NEW: cap frames to keep overlay light enough for the browser
    attr_max_frames = st.slider("Max frames in overlay", 5, 200, 40, 1, key="attr:max_frames")

# ---------------- Heatmap slice control ----------------
with st.sidebar.expander("2-D heatmap slice", expanded=False):
    hm_slice_y = st.slider(
        "Y row (for 2-D time × X view)",
        0,
        max(0, int(cfg_default["env"]["height"]) - 1),
        int(cfg_default["env"]["height"]) // 2,
        1,
    )

# ---------------- Layout placeholders ----------------
st.title("Simulation")
combo2d_ph = st.empty()
energy_ph  = st.empty()
stats_ph   = st.empty()
points3d_ph  = st.empty()   # legacy points (initial-substrate box overlayed)
conn3d_ph    = st.empty()   # connections plot

# ---------------- Utility ----------------
def compute_init_bounds(H: int, W: int, space_side: int) -> Tuple[int, int, int, int]:
    s = int(max(1, min(space_side, H, W)))
    y0 = (H - s) // 2
    x0 = (W - s) // 2
    y1 = y0 + s - 1
    x1 = x0 + s - 1
    return y0, y1, x0, x1

def add_box_wireframe(fig: go.Figure, *, y0: int, y1: int, x0: int, x1: int, z: int = 0,
                      name: str = "Init substrate bounds", opacity: float = 0.5):
    xs = [x0, x1, x1, x0, x0]
    ys = [y0, y0, y1, y1, y0]
    zs = [z,  z,  z,  z,  z]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(width=3),
                               opacity=float(opacity), name=name))

# =====================================================================
# Run button: compute & SAVE results to st.session_state["last_run"]
# =====================================================================
if st.button("Run / Rerun", use_container_width=True):
    physics.clear_states()
    engine = Engine(user_cfg)

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
    attr_history = []   # per-frame attractors (lightweight)
    try:
        for step_idx in range(T):
            S, flux, E = engine.step()

            # Capture attractors at this frame (flatten shape list)
            snap = physics.get_attractors_snapshot()
            flat_items = []
            for entry in snap:
                for it in entry.get("items", []):
                    flat_items.append({
                        "id": int(it.get("id", -1)),
                        "pos": tuple(it.get("pos", (0, 0))),   # (y, x)
                        "amp": float(it.get("amp", 0.0)),
                    })
            attr_history.append({"t": int(step_idx), "items": flat_items})

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

            if T > 0:
                pct = int(100 * (step_idx + 1) / T)
                prog.progress(pct, text=f"Running simulation… {pct}%")
                if (step_idx % 50) == 0:
                    tn = f"[{test_number}] " if test_number else ""
                    status.write(
                        f"{tn}Frame {step_idx+1}/{T}  |  ⌀|S|={np.mean(np.abs(S)):.4f}  "
                        f"flux={flux:.5f}  H_S={ent:.6g}"
                    )
    except Exception as e:
        ok = False
        prog.empty()
        st.exception(e)

    if ok:
        prog.progress(100, text="Simulation complete")
        try:
            E_stack = np.stack(env_frames, axis=0)  # (T,H,W)
            S_stack = np.stack(sub_frames, axis=0)  # (T,H,W)
        except Exception as e:
            st.exception(e); st.stop()

        # Enforce SAME grid for plotting (no resampling)
        T_chk, H_chk, W_chk = assert_equal_grids(E_stack, S_stack)

        summary = {
            "test_number": test_number,
            "seed": int(engine.cfg.get("seed", 0)),
            "frames": int(engine.cfg.get("frames", 0)),
            "grid": [int(engine.env_H), int(engine.env_W)],
            "E_env_mean": float(np.mean(e_env_series)) if e_env_series else 0.0,
            "E_cell_final": float(e_cell_series[-1]) if e_cell_series else 0.0,
            "flux_mean": float(np.mean(e_flux_series)) if e_flux_series else 0.0,
            "entropy_mean_over_time": float(np.mean(entropy_series)) if entropy_series else 0.0,
            "variance_mean_over_time": float(np.mean(variance_series)) if variance_series else 0.0,
            "total_mass_final": float(total_mass_series[-1]) if total_mass_series else 0.0,
        }

        st.session_state["last_run"] = {
            "E_stack": E_stack,
            "S_stack": S_stack,
            "t_series": t_series,
            "e_cell_series": e_cell_series,
            "e_env_series": e_env_series,
            "e_flux_series": e_flux_series,
            "entropy_series": entropy_series,
            "variance_series": variance_series,
            "total_mass_series": total_mass_series,
            "H": H_chk, "W": W_chk,
            "space": int(cfg_default.get("space", min(H_chk, W_chk))),
            "summary": summary,
            "attr_history": attr_history,
        }

# =====================================================================
# Always render if we have a cached result
# =====================================================================
run = st.session_state["last_run"]
if run is not None:
    E_stack = run["E_stack"]; S_stack = run["S_stack"]
    H_chk = run["H"]; W_chk = run["W"]
    t_series = run["t_series"]
    e_cell_series = run["e_cell_series"]; e_env_series = run["e_env_series"]; e_flux_series = run["e_flux_series"]
    entropy_series = run["entropy_series"]; variance_series = run["variance_series"]; total_mass_series = run["total_mass_series"]
    attr_history_all = run["attr_history"]

    # --------- 2-D overlays (combined heatmaps) ---------
    y_pick = int(np.clip(hm_slice_y, 0, S_stack.shape[1]-1))
    draw_combined_heatmap(
        combo2d_ph, E_stack, S_stack, y_row=y_pick,
        heat_floor=heat_floor, heat_gamma=heat_gamma,
        env_opacity=env_opacity, sub_opacity=sub_opacity,
        new_key_fn=new_key
    )

    # --------- Energy & stats ---------
    draw_energy_timeseries(energy_ph, t_series, e_cell_series, e_env_series, e_flux_series, new_key)
    draw_stats_timeseries(stats_ph, t_series, entropy_series, variance_series, total_mass_series, new_key)

    # --------- Legacy 3-D points with INITIAL SUBSTRATE BOX ---------
    if (thr3d is not None) and (max3d is not None):
        fig_pts = go.Figure()

        def _norm_local(A):
            m, M = float(np.nanmin(A)), float(np.nanmax(A))
            if not np.isfinite(m) or not np.isfinite(M) or M - m < 1e-12:
                return np.zeros_like(A)
            return (A - m) / (M - m + 1e-12)

        # Env points
        E3 = E_stack if E_stack.ndim == 3 else E_stack[:, None, :]
        En = _norm_local(E3)
        tE, yE, xE = np.where(En >= float(thr3d))
        if len(xE) > 0:
            keep = int(max(1, 0.25 * min(len(xE), int(max3d))))
            idx = np.random.choice(len(xE), size=keep, replace=False)
            xE, yE, tE = xE[idx], yE[idx], tE[idx]
        fig_pts.add_trace(go.Scatter3d(x=xE, y=yE, z=tE, mode="markers",
                                       marker=dict(size=2, opacity=0.55), name="Env"))

        # Substrate points
        S3 = S_stack if S_stack.ndim == 3 else S_stack[:, None, :]
        Sn = _norm_local(S3)
        tS, yS, xS = np.where(Sn >= float(thr3d))
        if len(xS) > 0:
            keep = int(min(len(xS), int(max3d) // 2))
            idx = np.random.choice(len(xS), size=keep, replace=False)
            xS, yS, tS = xS[idx], yS[idx], tS[idx]
        fig_pts.add_trace(go.Scatter3d(x=xS, y=yS, z=tS, mode="markers",
                                       marker=dict(size=2, opacity=0.8), name="Substrate"))

        # Initial substrate bounds (wireframe at z=0)
        space_side = int(cfg_default.get("space", min(H_chk, W_chk)))
        y0, y1, x0, x1 = compute_init_bounds(H_chk, W_chk, space_side)
        add_box_wireframe(fig_pts, y0=y0, y1=y1, x0=x0, x1=x1, z=0, name="Init substrate (z=0)", opacity=0.6)

        fig_pts.update_layout(
            title="Sparse 3-D energy (points) — with initial substrate bounds",
            scene=dict(xaxis_title="x (grid index)", yaxis_title="y (grid index)", zaxis_title="t (frame)",
                       aspectmode="data", dragmode="orbit"),
            height=540, template="plotly_dark", showlegend=True,
            uirevision="points3d", margin=dict(l=0, r=0, t=40, b=0),
        )
        points3d_ph.plotly_chart(fig_pts, use_container_width=True, theme=None,
                                 key="points3d_plot",
                                 config={"scrollZoom": True, "displaylogo": False, "doubleClick": "false"})
    else:
        st.warning("3-D points view disabled: add 'thr3d' and 'max3d' to defaults.json to enable.")

    # --------- 3-D connections (helper) ---------
    if conn_enable and (conn_max_edges is not None):
        # Trim overlay to keep payload bounded (prevents front-end blow-ups)
        ah = attr_history_all
        if attr_enable and isinstance(ah, list) and len(ah) > 0:
            k = int(st.session_state.get("attr:max_frames", 40))
            if len(ah) > k:
                ah = ah[-k:]
        else:
            ah = None

        def _get_attr_items():
            if not hasattr(physics, "get_attractors_snapshot"):
                return []
            snap = physics.get_attractors_snapshot()
            items = []
            if isinstance(snap, list):
                for entry in snap:
                    items.extend(entry.get("items", []))
            elif isinstance(snap, dict) and "items" in snap:
                items = snap["items"]
            return items

        try:
            ret = draw_3d_connections_over_time(
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
                attr_overlay=bool(attr_enable),
                attr_scale=float(attr_scale),
                attr_alpha=float(attr_alpha),
                get_attractor_items_fn=_get_attr_items,
                new_key_fn=new_key,
                attr_history=ah,  # trimmed history
            )
            # ---------------- Debug: quantify “near attractors” connections ----------------
            if dbg_enable:
                # 1) Build a (T,H,W) interest mask from trimmed attr history
                interest = dbg.build_attractor_interest3d(
                    attr_history=ah,
                    stack_shape_T_H_W=E_stack.shape,
                    radius=int(dbg_radius),
                )
                # 2) If the helper returned polylines, partition them
                if isinstance(ret, dict) and all(k in ret for k in ("conn_x", "conn_y", "conn_z")):
                    stats = dbg.partition_edges_by_attractors_polylines(
                        ret["conn_x"], ret["conn_y"], ret["conn_z"], interest
                    )
                    st.info("Near-attractor connections: " + dbg.format_counts(stats["counts"]))
                else:
                    # Helper didn’t return polylines; silently skip (no breakage)
                    pass
            
            
        except Exception as e:
            st.warning("Attractor overlay failed; rendering connections without overlay.")
            try:
                ret = draw_3d_connections_over_time(
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
                    attr_overlay=False,
                    get_attractor_items_fn=_get_attr_items,
                    new_key_fn=new_key,
                )
                if dbg_enable:
                    interest = dbg.build_attractor_interest3d(
                        attr_history=ah,
                        stack_shape_T_H_W=E_stack.shape,
                        radius=int(dbg_radius),
                    )
                    if isinstance(ret, dict) and all(k in ret for k in ("conn_x", "conn_y", "conn_z")):
                        stats = dbg.partition_edges_by_attractors_polylines(
                            ret["conn_x"], ret["conn_y"], ret["conn_z"], interest
                        )
                        st.info("Near-attractor connections: " + dbg.format_counts(stats["counts"]))
             finally:
                 st.exception(e)
    else:
        st.info("Connections view is off. Enable it in the sidebar.")

    # --------- Run summary ---------
    st.markdown("### Run summary (copy/paste)")
    st.text_area("Summary JSON", value=json.dumps(run["summary"], indent=2), height=220)