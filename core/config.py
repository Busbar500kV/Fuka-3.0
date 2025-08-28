# core/engine.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import is_dataclass, asdict

from . import physics
from .metrics import collect, format_for_log  # optional logging


def _config_to_dict(cfg_obj: Any) -> Dict[str, Any]:
    """Accept a dict, a dataclass (nested ok), or an object with to_dict/as_dict/dict/toJSON."""
    if isinstance(cfg_obj, dict):
        return cfg_obj
    if is_dataclass(cfg_obj):
        try:
            return asdict(cfg_obj)
        except Exception:
            pass
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
        if "env" in out and is_dataclass(out["env"]):
            out["env"] = asdict(out["env"])
        return out
    raise TypeError(
        "Engine expected a dict-like config. Got type "
        f"{type(cfg_obj).__name__} without a supported conversion."
    )


class Engine:
    """
    Full-grid mode:
      - Substrate S is sized to the environment grid (env.height × env.length)
      - Physics runs on the entire grid (no resampling)
      - UI receives the native environment for plotting
    """
    def __init__(self, cfg: Any):
        cfg = _config_to_dict(cfg)
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)))

        # sim / UI knobs
        self.frames   = int(cfg.get("frames", 2000))
        self.k_flux   = float(cfg.get("k_flux", 0.08))
        self.k_motor  = float(cfg.get("k_motor", 0.20))
        self.diffuse  = float(cfg.get("diffuse", 0.05))
        self.decay    = float(cfg.get("decay", 0.05))
        self.band     = int(cfg.get("band", 3))
        self.bc       = str(cfg.get("bc", "reflect"))

        # pass-through blocks
        self.physics_cfg = dict(cfg.get("physics", {}))
        self.fuka3_cfg   = dict(cfg.get("fuka3", {}))

        # environment config (dict)
        self.env_cfg     = cfg.get("env", {}) if isinstance(cfg.get("env", {}), dict) else {}
        self.env_H       = int(self.env_cfg.get("height", int(cfg.get("space", 64))))
        self.env_W       = int(self.env_cfg.get("length", int(cfg.get("space", 64))))
        self.env_frames  = int(self.env_cfg.get("frames", self.frames))
        self.env_sigma   = float(self.env_cfg.get("noise_sigma", 0.0))
        self.env_sources = list(self.env_cfg.get("sources", []))

        # substrate matches environment (no resample)
        self.S = np.zeros((self.env_H, self.env_W), dtype=float)

        self.frame_idx = 0

    def _env_field(self, t: int) -> np.ndarray:
        H, W = self.env_H, self.env_W
        E = np.zeros((H, W), dtype=float)

        def add_peak_2d(amp: float, cx: float, cy: float, wx: float, wy: float):
            y, x = np.indices((H, W))
            E[:] += amp * np.exp(
                -((x - cx) ** 2) / (2.0 * wx * wx) - ((y - cy) ** 2) / (2.0 * wy * wy)
            )

        for src in self.env_sources:
            kind = src.get("kind", "moving_peak_2d")

            if kind == "moving_peak_2d":
                amp = float(src.get("amp", 1.0))
                vx  = float(src.get("speed_x", 0.0))
                vy  = float(src.get("speed_y", 0.0))
                wx  = max(1e-6, float(src.get("width_x", 6.0)))
                wy  = max(1e-6, float(src.get("width_y", 6.0)))
                sx  = float(src.get("start_x", W // 2))
                sy  = float(src.get("start_y", H // 2))
                cx  = (sx + vx * t) % W
                cy  = (sy + vy * t) % H
                add_peak_2d(amp, cx, cy, wx, wy)

            elif kind == "moving_peak":
                amp = float(src.get("amp", 0.6))
                v   = float(src.get("speed", 0.02))
                w   = max(1e-6, float(src.get("width", 5.0)))
                start = float(src.get("start", W // 2))
                y_center = src.get("y_center", "mid")
                wy  = max(1e-6, float(src.get("width_y", 18.0)))
                cx  = (start + v * t) % W
                cy  = H // 2 if y_center == "mid" else float(y_center)
                y, x = np.indices((H, W))
                E += amp * np.exp(-((x - cx) ** 2) / (2.0 * w * w)) * np.exp(
                    -((y - cy) ** 2) / (2.0 * wy * wy)
                )

        if self.env_sigma > 0.0:
            E += self.env_sigma * self.rng.standard_normal(size=E.shape)

        return E  # native H×W

    def step(self) -> Tuple[np.ndarray, float, np.ndarray]:
        E = self._env_field(self.frame_idx)

        S_next, flux = physics.step_physics(
            self.S, E,
            self.k_flux, self.k_motor, self.diffuse, self.decay, self.rng,
            band=self.band, bc=self.bc,
            T=self.physics_cfg.get("T", 0.001),
            flux_limit=self.physics_cfg.get("flux_limit", 0.20),
            boundary_leak=self.physics_cfg.get("boundary_leak", 0.01),
            update_mode=self.physics_cfg.get("update_mode", "random"),
            fuka3=self.fuka3_cfg,
        )

        self.S = S_next
        self.frame_idx += 1
        return self.S, flux, E

    def maybe_log_metrics(self, N: int = 10):
        if (self.frame_idx % N) == 0:
            try:
                rows = collect()
                print(format_for_log(rows))
            except Exception:
                pass  # never crash on logging


def run(cfg: Any, on_frame=None):
    eng = Engine(cfg)
    for _ in range(eng.frames):
        S, flux, E = eng.step()
        if on_frame is not None:
            on_frame(eng.frame_idx, S, E, flux)
        eng.maybe_log_metrics(10)
    return eng