from __future__ import annotations
import json, os
from copy import deepcopy
from typing import Any, Dict, List
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from core import physics
from core.engine import Engine

from ui_camera import camera_controls

# helpers
from app_helpers import (
    load_defaults_strict, find_missing,
    ensure_session_keys, new_key,
    render_object,
    draw_combined_heatmap, draw_energy_timeseries, draw_stats_timeseries,
    draw_3d_connections_over_time,
)

st.set_page_config(page_title="Fuka 3.0 — Free‑Energy Simulation", layout="wide")

# --- Config load & validate ---
cfg_default = load_defaults_strict()
missing_keys = find_missing(cfg_default)
if missing_keys:
    st.error(
        "defaults.json is missing required keys for the simulation. "
        "Please add the following keys and rerun:\n\n" + "\n".join(f"• {k}" for k in missing_keys)
    )
    st.stop()

# --- Session keys ---
ensure_session_keys()

# --- Sidebar: render config & light extras ---
with st.sidebar:
    st.header("Configuration (from defaults.json)")
    user_cfg = render_object("", deepcopy(cfg_default))
    test_number = str(user_cfg.get("test_number", "")).strip()
    if test_number:
        st.subheader(f"Test: {test_number}")

# Optional UI knobs
had_live  = "live"  in user_cfg
had_chunk = "chunk" in user_cfg
had_thr3d = "thr3d" in user_cfg
had_max3d = "max3d" in user_cfg
had_vis   = "vis"   in user_cfg
had_conn  = "connections" in user_cfg

live  = bool(user_cfg.pop("live",  True))   if had_live  else None
chunk = int(user_cfg.pop("chunk",  150))    if had_chunk else None
thr3d = float(user_cfg.pop("thr3d", 0.75))  if had_thr3d else None
max3d = int(user_cfg.pop("max3d",  40000))  if had_max3d else None
vis   = user_cfg.pop("vis", {})             if had_vis   else None
conn_cfg = user_cfg.pop("connections", {})  if had_conn  else {}

# Visual knobs
if vis is None:
    heat_floor, heat_gamma, env_opacity, sub_opacity = 0.10, 1.0, 1.0, 0.85
else:
    heat_floor  = float(vis.get("heat_floor", 0.10))
    heat_gamma  = float(vis.get("heat_gamma", 1.0))
    env_opacity = float(vis.get("env_opacity", 1.0))
    sub_opacity = float(vis.get("sub_opacity", 0.85))

# Connections UI
with st.sidebar.expander("3‑D Connections (substrate encoding)", expanded=True):
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

# Attractors overlay UI
with st.sidebar.expander("Attractors overlay (3‑D)", expanded=False):
    attr_enable = st.checkbox("Show attractors (needs physics.get_attractors_snapshot)", value=False, key="attr:enable")
    attr_scale  = st.slider("Glyph length scale", 0.1, 5.0, 1.0, 0.1, key="attr:scale")
    attr_alpha  = st.slider("Opacity", 0.1, 1.0, 0.8, 0.05, key="attr:alpha")

# Heatmap slice control
with st.sidebar.expander("2‑D heatmap slice", expanded=False):
    hm_slice_y = st.slider(
        "Y row (for 2‑D time × X view)",
        0,
        max(0, int(cfg_default["env"]["height"]) - 1),
        int(cfg_default["env"]["height"]) // 2,
        1,
    )

# Layout placeholders
st.title("Simulation")
combo2d_ph = st.empty()
energy_ph  = st.empty()
stats_ph   = st.empty()
points3d_ph  = st.empty()
conn3d_ph    = st.empty()

# ---------- Run ----------
if st.button("Run / Rerun", use_container_width=True):
    st.session_state["run_id"] += 1
    for base in ("combo2d_count","energy_count","stats_count","combo3d_count","conn3d_count"):
        st.session_state[base] = 0

    physics.clear_states()           # fresh run
    engine = Engine(user_cfg)        # Engine normalizes dicts

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

            if T > 0:
                pct = int(100 * (step_idx + 1) / T)
                prog.progress(pct, text=f"Running simulation… {pct}%")
                if (step_idx % 50) == 0:
                    tn = f"[{test_number}] " if test_number else ""
                    status.write(
                        f"{tn}Frame {step_idx+1}/{T}  |  ⌀|S|={np.mean(np.abs(S)):.4f}  flux={flux:.5f}  "
                        f"H_S={ent:.6g}"
                    )
    except Exception as e:
        ok = False
        prog.empty()
        st.exception(e)

    if ok:
        prog.progress(100, text="Simulation complete")
        try:
            E_stack = np.stack(env_frames, axis=0)
            S_stack = np.stack(sub_frames, axis=0)
        except Exception as e:
            st.exception(e); st.stop()

        # 2‑D overlays
        y_pick = int(np.clip(hm_slice_y, 0, S_stack.shape[1]-1))
        draw_combined_heatmap(
            combo2d_ph, E_stack, S_stack, y_row=y_pick,
            heat_floor=heat_floor, heat_gamma=heat_gamma,
            env_opacity=env_opacity, sub_opacity=sub_opacity,
            new_key_fn=new_key
        )

        # Energy & stats
        draw_energy_timeseries(energy_ph, t_series, e_cell_series, e_env_series, e_flux_series, new_key)
        draw_stats_timeseries(stats_ph, t_series, entropy_series, variance_series, total_mass_series, new_key)

        # Optional: legacy points (kept as-is)
        if (thr3d is not None) and (max3d is not None):
            # env points
            fig_pts = go.Figure()
            from app_helpers import _norm_local, _draw_3d_env_points  # reuse internal helpers
            _draw_3d_env_points(fig_pts, E_stack, thr=float(thr3d), portion=0.25)
            # substrate points
            Sn_full = S_stack if S_stack.ndim == 3 else S_stack[:, None, :]
            Sn = _norm_local(Sn_full)
            tS, yS, xS = np.where(Sn >= float(thr3d))
            nS = len(xS)
            if nS > 0:
                keep = int(min(nS, max3d // 2))
                idx = np.random.choice(nS, size=keep, replace=False)
                xS, yS, tS = xS[idx], yS[idx], tS[idx]
            fig_pts.add_trace(go.Scatter3d(x=xS, y=yS, z=tS, mode="markers", marker=dict(size=2, opacity=0.8), name="Substrate"))
            
            fig_pts.update_layout(
                title="Sparse 3‑D energy (points)",
                scene=dict(
                    xaxis_title="x", yaxis_title="y", zaxis_title="t",
                    aspectmode="data",
                    dragmode="orbit"
                ),
                height=540,
                template="plotly_dark",
                showlegend=True,
                uirevision="points3d"
            )
            
            points3d_ph.plotly_chart(
                fig_pts,
                use_container_width=True,
                theme=None,
                key="points3d_plot",            # <- stable key
                config={
                    "scrollZoom": True,
                    "displaylogo": False
                },
            )
            
        else:
            st.warning("3‑D points view disabled: add 'thr3d' and 'max3d' to defaults.json to enable.")

        # 3‑D connections + optional attractors
        if conn_enable and (conn_max_edges is not None):
            def _get_attr_items():
                # flatten per-shape snapshot into a single list
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
                attr_overlay=bool(attr_enable),
                attr_scale=float(attr_scale),
                attr_alpha=float(attr_alpha),
                get_attractor_items_fn=_get_attr_items,
                new_key_fn=new_key,
            )
        else:
            st.info("Connections view is off. Enable it in the sidebar.")

        # Copy‑paste run summary
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
        st.markdown("### Run summary (copy/paste)")
        st.text_area("Summary JSON", value=json.dumps(summary, indent=2), height=220)