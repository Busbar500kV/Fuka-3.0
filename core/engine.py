# core/engine.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, Optional

from . import physics
from .metrics import collect, format_for_log  # optional logging

class Engine:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)))

        # --- sim dims / UI knobs (unchanged)
        self.frames = int(cfg.get("frames", 2000))
        self.space  = int(cfg.get("space", 64))
        self.k_flux   = float(cfg.get("k_flux", 0.08))
        self.k_motor  = float(cfg.get("k_motor", 0.20))
        self.diffuse  = float(cfg.get("diffuse", 0.05))
        self.decay    = float(cfg.get("decay", 0.05))
        self.band     = int(cfg.get("band", 3))
        self.bc       = str(cfg.get("bc", "reflect"))

        # physics & fuka3 config blocks (pass through)
        self.physics_cfg = dict(cfg.get("physics", {}))
        self.fuka3_cfg   = dict(cfg.get("fuka3", {}))

        # --- substrate & environment (2D)
        # Substrate field S has shape [H, W] == [space, space]
        H = W = self.space
        self.S = np.zeros((H, W), dtype=float)

        # Environment E is supplied per frame (H, W). If your project
        # already has an env generator, plug it in here instead.
        self.env_cfg = cfg.get("env", {})
        self.env_H = int(self.env_cfg.get("height", H))
        self.env_W = int(self.env_cfg.get("length", W))
        self.env_frames = int(self.env_cfg.get("frames", self.frames))
        self.env_sigma  = float(self.env_cfg.get("noise_sigma", 0.0))
        # Prebuild a simple env to match your defaults (moving peaks).
        self.env_sources = self.env_cfg.get("sources", [])

        # runtime counters
        self.frame_idx = 0

    # ---------- Simple synthetic environment to match your JSON ----------
    def _env_field(self, t: int) -> np.ndarray:
        H, W = self.env_H, self.env_W
        E = np.zeros((H, W), dtype=float)

        def add_peak_2d(amp, cx, cy, wx, wy):
            y, x = np.indices((H, W))
            E[:] += amp * np.exp(-((x - cx) ** 2) / (2 * wx * wx) - ((y - cy) ** 2) / (2 * wy * wy))

        for src in self.env_sources:
            kind = src.get("kind", "moving_peak_2d")
            if kind == "moving_peak_2d":
                amp = float(src.get("amp", 1.0))
                vx  = float(src.get("speed_x", 0.0))
                vy  = float(src.get("speed_y", 0.0))
                wx  = float(src.get("width_x", 6.0))
                wy  = float(src.get("width_y", 6.0))
                sx  = float(src.get("start_x", W//2))
                sy  = float(src.get("start_y", H//2))
                cx  = (sx + vx * t) % W
                cy  = (sy + vy * t) % H
                add_peak_2d(amp, cx, cy, wx, wy)

            elif kind == "moving_peak":
                # 1D strip peak centered along y (like your sample)
                amp = float(src.get("amp", 0.6))
                v   = float(src.get("speed", 0.02))
                w   = float(src.get("width", 5.0))
                start = float(src.get("start", W//2))
                y_center = src.get("y_center", "mid")
                wy = float(src.get("width_y", 18.0))
                cx = (start + v * t) % W
                cy = H//2 if y_center == "mid" else float(y_center)
                # add a ridge by summing Gaussians across y
                y, x = np.indices((H, W))
                E += amp * np.exp(-((x - cx) ** 2) / (2 * w * w)) * np.exp(-((y - cy) ** 2) / (2 * wy * wy))

        if self.env_sigma > 0.0:
            E += self.env_sigma * self.rng.standard_normal(size=E.shape)
        # if your renderer expects env to match substrate size, resample if needed
        if (H, W) != self.S.shape:
            E = physics._resample_2d(E, self.S.shape)
        return E

    # ---------- One simulation step ----------
    def step(self) -> Tuple[np.ndarray, float, np.ndarray]:
        E = self._env_field(self.frame_idx)

        S_next, flux = physics.step_physics(
            self.S, E,
            self.k_flux, self.k_motor, self.diffuse, self.decay, self.rng,
            band=self.band, bc=self.bc,
            # legacy physics knobs (UI preserved)
            T=self.physics_cfg.get("T", 0.001),
            flux_limit=self.physics_cfg.get("flux_limit", 0.20),
            boundary_leak=self.physics_cfg.get("boundary_leak", 0.01),
            update_mode=self.physics_cfg.get("update_mode", "random"),
            # NEW: full Fuka 3.0 block
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
                pass  # never crash sim on logging

# Convenience runner (optional)
def run(cfg: Dict[str, Any], on_frame=None):
    eng = Engine(cfg)
    for _ in range(eng.frames):
        S, flux, E = eng.step()
        if on_frame is not None:
            on_frame(eng.frame_idx, S, E, flux)
        eng.maybe_log_metrics(10)
    return eng