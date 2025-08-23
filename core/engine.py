# core/engine.py
from __future__ import annotations
import numpy as np

from .config import Config
from .env import build_env
from .organism import History
from .physics import step_physics, resample_row


def _resample_2d(frame: np.ndarray, new_y: int, new_x: int) -> np.ndarray:
    """
    Resample a 2D array [Y0, X0] -> [new_y, new_x] with separable linear interpolation.
    """
    Y0, X0 = frame.shape
    if (Y0, X0) == (new_y, new_x):
        return frame

    # interp along X for each row
    if X0 != new_x:
        x_src = np.linspace(0.0, 1.0, X0)
        x_tgt = np.linspace(0.0, 1.0, new_x)
        tmp = np.zeros((Y0, new_x), dtype=float)
        for y in range(Y0):
            tmp[y] = np.interp(x_tgt, x_src, frame[y])
    else:
        tmp = frame

    # interp along Y for each column
    if Y0 != new_y:
        y_src = np.linspace(0.0, 1.0, Y0)
        y_tgt = np.linspace(0.0, 1.0, new_y)
        out = np.zeros((new_y, new_x), dtype=float)
        for x in range(new_x):
            out[:, x] = np.interp(y_tgt, y_src, tmp[:, x])
        return out
    else:
        return tmp


def _entropy_from_array(A: np.ndarray) -> float:
    """
    Shannon entropy over a nonnegative normalization of A.
    We shift so min(A) -> 0, add tiny epsilon, then normalize to sum=1.
    """
    a = np.asarray(A, dtype=float)
    a = a - float(np.min(a))
    a = a + 1e-12
    s = float(np.sum(a))
    if s <= 0.0 or not np.isfinite(s):
        return 0.0
    p = a / s
    # safe entropy (ignore zeros after epsilon add)
    return float(-np.sum(p * np.log(p)))


class Engine:
    """
    Streaming‑capable engine.

    - 1D: env shape (T, X_env), substrate S shape (T, X).
    - 2D: env shape (T, Y_env, X_env), substrate S shape (T, Y, X) with Y=X=cfg.space.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Build full environment timeline
        self.env = build_env(cfg.env, self.rng)  # (T, X_env) or (T, Y_env, X_env)
        self.T = int(cfg.frames)

        # Determine dimensionality and allocate substrate
        if self.env.ndim == 2:
            # 1‑D substrate
            self.mode = "1d"
            self.X = int(cfg.space)
            self.S = np.zeros((self.T, self.X), dtype=float)
        elif self.env.ndim == 3:
            # 2‑D substrate (square: Y=X=space)
            self.mode = "2d"
            self.Y = int(cfg.space)
            self.X = int(cfg.space)
            self.S = np.zeros((self.T, self.Y, self.X), dtype=float)
        else:
            raise ValueError(f"Unsupported env dimensions: {self.env.shape}")

        self.hist = History()

        # Precompute physics kwargs from defaults.json (if any)
        self._phys_kwargs = dict(self.cfg.physics or {})

    # ---------- steps ----------
    def _step_1d(self, t: int):
        # env row -> resample to substrate width
        e_row = self.env[t]                 # (X_env,)
        e_row = resample_row(e_row, self.X) # (X,)

        prev = self.S[t-1] if t > 0 else self.S[0]
        cur, flux = step_physics(
            prev_S=prev,
            env_row=e_row,
            k_flux=self.cfg.k_flux,
            k_motor=self.cfg.k_motor,
            diffuse=self.cfg.diffuse,
            decay=self.cfg.decay,
            rng=self.rng,
            band=self.cfg.band,
            bc=self.cfg.bc,
            **self._phys_kwargs,
        )
        # cur is (X,)
        self.S[t] = cur

        # telemetry
        self.hist.t.append(t)
        self.hist.E_cell.append(float(np.mean(cur)))
        self.hist.E_env.append(float(np.mean(e_row)))
        self.hist.E_flux.append(float(flux))
        # new telemetry
        self.hist.variance.append(float(np.var(cur)))
        self.hist.total_mass.append(float(np.sum(cur)))
        self.hist.entropy.append(_entropy_from_array(cur))

    def _step_2d(self, t: int):
        # env frame (Y_env, X_env) -> resample to (Y, X)
        E_t = self.env[t]
        E_t_rs = _resample_2d(E_t, new_y=self.Y, new_x=self.X)  # (Y, X)

        prev = self.S[t-1] if t > 0 else self.S[0]              # (Y, X)
        cur, flux = step_physics(
            prev_S=prev,             # (Y, X)
            env_row=E_t_rs,          # (Y, X)
            k_flux=self.cfg.k_flux,
            k_motor=self.cfg.k_motor,
            diffuse=self.cfg.diffuse,
            decay=self.cfg.decay,
            rng=self.rng,
            band=self.cfg.band,
            bc=self.cfg.bc,
            **self._phys_kwargs,
        )

        # --- compatibility shim: if some physics impl returns (2, Y, X) stack, take channel 0
        if cur.ndim == 3 and cur.shape[0] == 2 and cur.shape[1:] == (self.Y, self.X):
            cur = cur[0]

        # Sanity: ensure we got exactly (Y, X)
        if cur.shape != (self.Y, self.X):
            raise ValueError(f"Unexpected physics output shape {cur.shape}, expected {(self.Y, self.X)}")

        self.S[t] = cur

        # telemetry
        self.hist.t.append(t)
        self.hist.E_cell.append(float(np.mean(cur)))
        self.hist.E_env.append(float(np.mean(E_t_rs)))
        self.hist.E_flux.append(float(flux))
        # new telemetry
        self.hist.variance.append(float(np.var(cur)))
        self.hist.total_mass.append(float(np.sum(cur)))
        self.hist.entropy.append(_entropy_from_array(cur))

    def step(self, t: int):
        if self.mode == "1d":
            self._step_1d(t)
        else:
            self._step_2d(t)

    # ---------- run ----------
    def run(self, progress_cb=None):
        for t in range(self.T):
            self.step(t)
            if progress_cb is not None:
                progress_cb(t)