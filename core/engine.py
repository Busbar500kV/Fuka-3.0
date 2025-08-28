# core/engine.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple

from . import physics
from . import env as envmod
from .config import FieldCfg
from .metrics import collect, format_for_log  # optional logging


# ---------- Config normalization ----------
def _config_to_dict(cfg_obj: Any) -> Dict[str, Any]:
    """
    Accept a plain dict or an object with to_dict/as_dict/dict/toJSON, else
    shallow-copy __dict__. This matches your current setup (defaults.json).
    """
    if isinstance(cfg_obj, dict):
        return cfg_obj
    for attr in ("to_dict", "as_dict", "dict", "toJSON"):
        if hasattr(cfg_obj, attr) and callable(getattr(cfg_obj, attr)):
            try:
                out = getattr(cfg_obj, attr)()
                if isinstance(out, dict):
                    return out
            except Exception:
                pass
    if hasattr(cfg_obj, "__dict__") and isinstance(cfg_obj.__dict__, dict):
        out = {k: v for k, v in cfg_obj.__dict__.items() if not k.startswith("_")}
        if out:
            return out
    raise TypeError(
        "Engine expected a dict-like config. Got type "
        f"{type(cfg_obj).__name__} without a supported to_dict/as_dict."
    )


# ---------- Engine ----------
class Engine:
    """
    Full-grid mode:
    - Substrate S matches the environment grid (env.height × env.length)
    - Environment is generated once with env.build_env (periodic, torus)
    - Physics runs on the entire grid (no resampling)
    """
    def __init__(self, cfg: Any):
        # normalize config
        cfg = _config_to_dict(cfg)
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)))

        # frames: prefer top-level; fall back to env.frames; default 2000
        self.frames = int(cfg.get("frames", int(cfg.get("env", {}).get("frames", 2000))))

        # environment config (prepare FieldCfg with sane fallbacks)
        env_dict = dict(cfg.get("env", {})) if isinstance(cfg.get("env", {}), dict) else {}
        # default spatial size falls back to top-level "space"
        space_default = int(cfg.get("space", 64))
        env_dict.setdefault("length", space_default)
        env_dict.setdefault("height", space_default)
        env_dict.setdefault("frames", self.frames)
        env_dict.setdefault("noise_sigma", float(env_dict.get("noise_sigma", 0.0)))
        env_dict.setdefault("sources", list(env_dict.get("sources", [])))

        # build periodic environment timeline E_seq: (T, H, W) or (T, X)
        E_seq = envmod.build_env(FieldCfg(**env_dict), self.rng)
        # ensure 2D spatial substrate (H, W)
        if E_seq.ndim == 2:
            # (T, X) -> treat as (T, 1, X) and use a 1×X substrate
            E_seq = E_seq[:, None, :]
        self.E_seq = E_seq
        self.env_frames, self.env_H, self.env_W = self.E_seq.shape[0], self.E_seq.shape[1], self.E_seq.shape[2]

        # classic knobs
        self.space    = int(cfg.get("space", max(self.env_H, self.env_W)))  # kept for UI/back-compat
        self.k_flux   = float(cfg.get("k_flux", 0.08))
        self.k_motor  = float(cfg.get("k_motor", 0.20))
        self.diffuse  = float(cfg.get("diffuse", 0.05))
        self.decay    = float(cfg.get("decay", 0.05))
        self.band     = int(cfg.get("band", 3))
        self.bc       = str(cfg.get("bc", "reflect"))

        # physics & fuka3 blocks (pass-through to step_physics)
        self.physics_cfg = dict(cfg.get("physics", {}))
        self.fuka3_cfg   = dict(cfg.get("fuka3", {}))

        # substrate S matches the environment (FULL-GRID)
        self.S = np.zeros((self.env_H, self.env_W), dtype=float)

        # runtime counter
        self.frame_idx = 0

    # ---------- One simulation step ----------
    def step(self) -> Tuple[np.ndarray, float, np.ndarray]:
        # pick the current environment frame (periodic wrap)
        t = self.frame_idx % self.env_frames
        E = self.E_seq[t]

        # Physics on FULL GRID (S and E have the same shape)
        S_next, flux = physics.step_physics(
            self.S, E,
            self.k_flux, self.k_motor, self.diffuse, self.decay, self.rng,
            band=self.band, bc=self.bc,
            # legacy physics knobs (UI preserved)
            T=self.physics_cfg.get("T", 0.001),
            flux_limit=self.physics_cfg.get("flux_limit", 0.20),
            boundary_leak=self.physics_cfg.get("boundary_leak", 0.01),
            update_mode=self.physics_cfg.get("update_mode", "random"),
            # Fuka 3.0 block (passed verbatim)
            fuka3=self.fuka3_cfg,
        )

        self.S = S_next
        self.frame_idx += 1
        return self.S, flux, E

    # ---------- Optional: log metrics every N frames ----------
    def maybe_log_metrics(self, N: int = 10):
        if (self.frame_idx % N) == 0:
            try:
                rows = collect()
                print(format_for_log(rows))
            except Exception:
                # Never crash the sim on logging
                pass


# Convenience runner (optional; unused by app.py)
def run(cfg: Any, on_frame=None):
    eng = Engine(cfg)
    for _ in range(eng.frames):
        S, flux, E = eng.step()
        if on_frame is not None:
            on_frame(eng.frame_idx, S, E, flux)
        eng.maybe_log_metrics(10)
    return eng