# ui_camera.py
from __future__ import annotations
import streamlit as st

def camera_controls(title: str, key_prefix: str):
    """Render sliders in the sidebar and return a Plotly 3D camera dict."""
    with st.sidebar.expander(title, expanded=False):
        ex = st.slider("eye.x",    -3.0, 3.0,  1.25, 0.01, key=f"{key_prefix}:ex")
        ey = st.slider("eye.y",    -3.0, 3.0,  1.25, 0.01, key=f"{key_prefix}:ey")
        ez = st.slider("eye.z",    -3.0, 3.0,  1.25, 0.01, key=f"{key_prefix}:ez")
        cx = st.slider("center.x", -1.0, 1.0,  0.00, 0.01, key=f"{key_prefix}:cx")
        cy = st.slider("center.y", -1.0, 1.0,  0.00, 0.01, key=f"{key_prefix}:cy")
        cz = st.slider("center.z", -1.0, 1.0,  0.00, 0.01, key=f"{key_prefix}:cz")
    return dict(eye=dict(x=ex, y=ey, z=ez), center=dict(x=cx, y=cy, z=cz))