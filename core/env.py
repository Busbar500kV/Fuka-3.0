# core/env.py
from __future__ import annotations
from typing import Optional
import numpy as np

from .config import FieldCfg


# ------------------------------
# helpers (periodic distances)
# ------------------------------
def _ring_dist(n: int, x: np.ndarray, x0: float) -> np.ndarray:
    """Shortest periodic distance on a ring of length n."""
    d = np.abs(x - x0)
    return np.minimum(d, n - d)


def _gauss_1d(n: int, pos: float, amp: float, width: float) -> np.ndarray:
    """Periodic 1-D Gaussian on a ring of length n."""
    x = np.arange(n, dtype=float)
    d = _ring_dist(n, x, pos)
    w = max(1e-9, float(width))
    return float(amp) * np.exp(-(d * d) / (2.0 * w * w))


def _gauss_2d(nx: int, ny: int, x0: float, y0: float, amp: float, wx: float, wy: float) -> np.ndarray:
    """
    Separable periodic Gaussian on a torus (ny, nx) centered at (x0, y0).
    Returns an array of shape (ny, nx).
    """
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)

    dx = _ring_dist(nx, xs, x0)  # (nx,)
    dy = _ring_dist(ny, ys, y0)  # (ny,)

    wx = max(1e-9, float(wx))
    wy = max(1e-9, float(wy))

    gx = np.exp(-(dx * dx) / (2.0 * wx * wx))  # (nx,)
    gy = np.exp(-(dy * dy) / (2.0 * wy * wy))  # (ny,)

    # outer product -> (ny, nx)
    return float(amp) * (gy[:, None] * gx[None, :])


# ------------------------------
# environment builder
# ------------------------------
def build_env(cfg: FieldCfg, rng: np.random.Generator) -> np.ndarray:
    """
    Build environment timeline.

    If cfg.height > 1 we produce a 2-D spatial field over time:  E[t, y, x]
    (periodic in both x and y). Otherwise we produce 1-D: E[t, x].

    Supported sources:

      1D:
        {"kind":"moving_peak", "amp":1.0, "speed":0.02, "width":4.0, "start":128}

      2D:
        {"kind":"moving_peak_2d","amp":1.0,"speed_x":0.02,"speed_y":-0.01,
         "width_x":4.0,"width_y":4.0,"start_x":X//2,"start_y":Y//2}

      Broadcast 1D onto 2D with Gaussian in y:
        {"kind":"moving_peak","amp":1.0,"speed":0.02,"width":4.0,"start":X//2,
         "y_center":"mid","width_y":6.0}

    Optional per-step small random drift:
      1D -> "jitter"; 2D -> "jitter_x"/"jitter_y".
    """
    T = int(cfg.frames)
    X = int(cfg.length)
    Y = int(getattr(cfg, "height", getattr(cfg, "H", 1)))

    sources = getattr(cfg, "sources", None) or []

    # ---------- 1-D path ----------
    if Y <= 1:
        E = np.zeros((T, X), dtype=float)

        for s in sources:
            if s.get("kind", "moving_peak") != "moving_peak":
                continue

            amp    = float(s.get("amp", 1.0))
            speed  = float(s.get("speed", 0.0)) * X  # cells/frame
            width  = float(s.get("width", 4.0))
            pos    = float(int(s.get("start", 0)) % X)
            jitter = float(s.get("jitter", 0.0))

            for t in range(T):
                E[t] += _gauss_1d(X, pos, amp, width)
                if jitter != 0.0:
                    pos += rng.normal(0.0, jitter)
                pos = (pos + speed) % X

        if getattr(cfg, "noise_sigma", 0.0) > 0.0:
            E += rng.normal(0.0, float(cfg.noise_sigma), size=E.shape)

        np.maximum(E, 0.0, out=E)
        return E

    # ---------- 2-D path: E[t, y, x] ----------
    E = np.zeros((T, Y, X), dtype=float)

    for s in sources:
        kind = s.get("kind", "moving_peak_2d")

        if kind == "moving_peak_2d":
            amp = float(s.get("amp", 1.0))
            sx  = float(s.get("speed_x", 0.0)) * X
            sy  = float(s.get("speed_y", 0.0)) * Y
            wx  = float(s.get("width_x", 4.0))
            wy  = float(s.get("width_y", 4.0))

            x   = float(int(s.get("start_x", X // 2)) % X)
            y   = float(int(s.get("start_y", Y // 2)) % Y)

            jx = float(s.get("jitter_x", 0.0))
            jy = float(s.get("jitter_y", 0.0))

            for t in range(T):
                E[t] += _gauss_2d(X, Y, x, y, amp, wx, wy)
                if jx != 0.0:
                    x += rng.normal(0.0, jx)
                if jy != 0.0:
                    y += rng.normal(0.0, jy)
                x = (x + sx) % X
                y = (y + sy) % Y

        elif kind == "moving_peak":
            # 1D along x, broadcast in y with a Gaussian profile
            amp   = float(s.get("amp", 1.0))
            speed = float(s.get("speed", 0.0)) * X
            width = float(s.get("width", 4.0))

            yc = s.get("y_center", "mid")
            if isinstance(yc, str) and yc.lower() == "mid":
                yc = Y / 2.0
            yc = float(yc)
            wy = float(s.get("width_y", width))

            x  = float(int(s.get("start", X // 2)) % X)
            jx = float(s.get("jitter", 0.0))

            for t in range(T):
                E[t] += _gauss_2d(X, Y, x, yc, amp, width, wy)
                if jx != 0.0:
                    x += rng.normal(0.0, jx)
                x = (x + speed) % X

        else:
            continue

    if getattr(cfg, "noise_sigma", 0.0) > 0.0:
        E += rng.normal(0.0, float(cfg.noise_sigma), size=E.shape)

    np.maximum(E, 0.0, out=E)
    return E