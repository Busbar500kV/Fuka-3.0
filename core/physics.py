# core/physics.py
from __future__ import annotations
import numpy as np
from typing import Tuple

# ---------------------------
# Resampling helpers (back-compat for 1D Engine)
# ---------------------------
def resample_row(row: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a 1D row to `target_len` using linear interpolation.
    Kept for backward-compatibility with the current Engine (1-D).
    """
    row = np.asarray(row, dtype=float)
    n = row.shape[0]
    if n == target_len:
        return row
    x_src = np.linspace(0.0, 1.0, n)
    x_tgt = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_tgt, x_src, row)


def resample_plane(img: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """
    Simple separable linear resample of a 2D array to (H_tgt, W_tgt).
    Not used by the 1-D engine, but provided for 2-D substrates.
    """
    img = np.asarray(img, dtype=float)
    H, W = img.shape
    Ht, Wt = int(target_hw[0]), int(target_hw[1])
    if (H, W) == (Ht, Wt):
        return img

    # resample rows -> (H, Wt)
    x_src = np.linspace(0.0, 1.0, W)
    x_tgt = np.linspace(0.0, 1.0, Wt)
    tmp = np.empty((H, Wt), dtype=float)
    for r in range(H):
        tmp[r] = np.interp(x_tgt, x_src, img[r])

    # resample columns -> (Ht, Wt)
    y_src = np.linspace(0.0, 1.0, H)
    y_tgt = np.linspace(0.0, 1.0, Ht)
    out = np.empty((Ht, Wt), dtype=float)
    for c in range(Wt):
        out[:, c] = np.interp(y_tgt, y_src, tmp[:, c])
    return out


# ---------------------------
# Internal utilities
# ---------------------------
def _normalize(A: np.ndarray) -> np.ndarray:
    m = float(np.nanmin(A))
    M = float(np.nanmax(A))
    if not np.isfinite(m) or not np.isfinite(M) or M == m:
        return np.zeros_like(A, dtype=float)
    return (A - m) / (M - m + 1e-12)


def _edge_index_1d(n: int, bc: str) -> np.ndarray:
    """
    Return left indices for edges (i,i+1). If periodic, includes edge (n-1,0);
    otherwise only [0..n-2].
    """
    if bc == "periodic":
        return np.arange(n, dtype=int)
    else:
        if n <= 1:
            return np.array([], dtype=int)
        return np.arange(n - 1, dtype=int)


# ---------------------------
# Main physics (causal, conservative, density-dependent speed)
# ---------------------------
def step_physics(
    prev_S: np.ndarray,
    env_row: np.ndarray,
    k_flux: float,
    k_motor: float,
    diffuse: float,       # used as a global cap for per-tick conservative exchange |q|
    decay: float,
    rng: np.random.Generator,
    band: int = 0,        # kept for API compatibility; ignored here
    bc: str = "reflect",  # "periodic" | "reflect" | "absorb" | "wall"
    **kwargs,
) -> tuple[np.ndarray, float]:
    """
    Single global tick with emergent local speeds and conservative neighbor exchange.

    - Local exchange speed is energy-dependent: kappa ~ (normalized |S|)^beta.
    - Exchange is conservative (what leaves one cell enters its neighbor).
    - Environment pull is bounded per tick (flux_limit) so causality isn't broken.
    - Tiny zero-mean bath noise (T) gives natural entropy growth unless env does work.
    - 'diffuse' acts as a *cap* on conservative exchange per tick (finite local speed).
    """
    S = np.asarray(prev_S, dtype=float)
    E = np.asarray(env_row, dtype=float)
    if S.shape != E.shape:
        raise ValueError(f"step_physics: shape mismatch S{S.shape} vs E{E.shape}")

    bc = str(bc).lower()
    if bc not in ("periodic", "reflect", "absorb", "wall"):
        bc = "reflect"

    # ---- optional knobs (from defaults.json via Config.physics) ----
    alpha_speed   = float(kwargs.get("alpha_speed", 0.2))   # scales conservative exchange per tick
    beta_speed    = float(kwargs.get("beta_speed",  1.0))   # how speed grows with local energy
    flux_limit    = float(kwargs.get("flux_limit",  0.5))   # per-tick max magnitude of env pull
    T_noise       = float(kwargs.get("T",           1e-3))  # micro-bath noise std
    update_mode   = str(kwargs.get("update_mode",  "random")).lower()  # "random" | "checker" | "none"
    boundary_leak = float(kwargs.get("boundary_leak", 0.0)) if bc in ("absorb", "wall") else 0.0

    # ---- local “energy” proxy and conductivity ----
    Sn = _normalize(S)
    eps = 1e-6
    kappa = alpha_speed * np.power(eps + Sn, beta_speed)  # same shape as S

    new_S = S.copy()

    if S.ndim == 1:
        # ========== 1D conservative exchange ==========
        n = S.shape[0]
        edges = _edge_index_1d(n, bc)

        def _sweep(indices):
            for i in indices:
                j = (i + 1) % n
                if bc != "periodic" and i == n - 1:
                    continue
                k_e = min(kappa[i], kappa[j])
                q = k_e * (new_S[i] - new_S[j])
                q = float(np.clip(q, -diffuse, diffuse))
                new_S[i] -= q
                new_S[j] += q

        if update_mode == "random":
            rng.shuffle(edges)
            _sweep(edges)
        elif update_mode == "checker":
            _sweep(edges[0::2]); _sweep(edges[1::2])
        else:
            _sweep(edges)

        # boundary leak for absorb/wall
        if boundary_leak > 0.0 and n >= 2:
            new_S[0]  *= (1.0 - boundary_leak)
            new_S[-1] *= (1.0 - boundary_leak)

        # ---- env coupling (bounded) ----
        pull = k_flux * (E - new_S)
        pull = np.clip(pull, -flux_limit, flux_limit)
        new_S += pull

    elif S.ndim == 2:
        # ========== 2D conservative exchange (x then y sweeps) ==========
        Y, X = S.shape

        # X-sweep
        cols = np.arange(X if bc == "periodic" else X - 1, dtype=int)
        if update_mode == "random":
            rng.shuffle(cols)
            iters = [cols]
        elif update_mode == "checker":
            iters = [cols[0::2], cols[1::2]]
        else:
            iters = [cols]

        for idxs in iters:
            for x in idxs:
                x2 = (x + 1) % X
                if bc != "periodic" and x == X - 1:
                    continue
                k_e = np.minimum(kappa[:, x], kappa[:, x2])
                q = k_e * (new_S[:, x] - new_S[:, x2])
                q = np.clip(q, -diffuse, diffuse)
                new_S[:, x]  -= q
                new_S[:, x2] += q

        # Y-sweep
        rows = np.arange(Y if bc == "periodic" else Y - 1, dtype=int)
        if update_mode == "random":
            rng.shuffle(rows)
            iters = [rows]
        elif update_mode == "checker":
            iters = [rows[0::2], rows[1::2]]
        else:
            iters = [rows]

        for idxs in iters:
            for y in idxs:
                y2 = (y + 1) % Y
                if bc != "periodic" and y == Y - 1:
                    continue
                k_e = np.minimum(kappa[y, :], kappa[y2, :])
                q = k_e * (new_S[y, :] - new_S[y2, :])
                q = np.clip(q, -diffuse, diffuse)
                new_S[y, :]  -= q
                new_S[y2, :] += q

        # boundary leak for absorb/wall (outer ring)
        if boundary_leak > 0.0 and Y >= 2 and X >= 2:
            new_S[0, :]   *= (1.0 - boundary_leak)
            new_S[-1, :]  *= (1.0 - boundary_leak)
            new_S[:, 0]   *= (1.0 - boundary_leak)
            new_S[:, -1]  *= (1.0 - boundary_leak)

        # Env coupling (bounded)
        pull = k_flux * (E - new_S)
        pull = np.clip(pull, -flux_limit, flux_limit)
        new_S += pull

    else:
        raise ValueError("step_physics only supports 1D or 2D arrays")

    # ---- global decay ----
    new_S *= (1.0 - float(decay))

    # ---- micro-bath: small zero-mean noise ----
    if T_noise > 0.0:
        new_S += T_noise * rng.standard_normal(size=new_S.shape)

    # ---- exploratory motor kick, scaled by local normalized energy ----
    if k_motor != 0.0:
        motor_scale = np.power(1e-6 + Sn, 0.5)
        new_S += float(k_motor) * motor_scale * rng.standard_normal(size=new_S.shape)

    # Flux metric for plots (mean absolute env pull actually applied)
    flux_metric = float(np.mean(np.abs(pull)))
    return new_S, flux_metric