# core/physics.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, Any

# ============================================================
# Back-compat helpers (Engine calls these)
# ============================================================

def resample_row(row: np.ndarray, target_len: int) -> np.ndarray:
    row = np.asarray(row, dtype=float)
    n = int(row.shape[0])
    if n == target_len:
        return row
    x_src = np.linspace(0.0, 1.0, n)
    x_tgt = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_tgt, x_src, row)

def _resample_2d(img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    img = np.asarray(img, dtype=float)
    H, W = img.shape
    Ht, Wt = int(target_hw[0]), int(target_hw[1])
    if (H, W) == (Ht, Wt):
        return img
    x_src = np.linspace(0.0, 1.0, W)
    x_tgt = np.linspace(0.0, 1.0, Wt)
    tmp = np.empty((H, Wt), dtype=float)
    for r in range(H):
        tmp[r, :] = np.interp(x_tgt, x_src, img[r, :])
    y_src = np.linspace(0.0, 1.0, H)
    y_tgt = np.linspace(0.0, 1.0, Ht)
    out = np.empty((Ht, Wt), dtype=float)
    for c in range(Wt):
        out[:, c] = np.interp(y_tgt, y_src, tmp[:, c])
    return out

# ============================================================
# Utilities
# ============================================================

def _safe_get(d: Dict[str, Any], key: str, default):
    v = d.get(key, default) if isinstance(d, dict) else default
    return default if v is None else v

def _clip(a, lo, hi):
    return np.minimum(np.maximum(a, lo), hi)

def _rolling_var_1d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return np.zeros_like(x)
    w = int(win)
    pad = w // 2
    xr = np.pad(x, (pad, pad), mode="reflect")
    csum = np.cumsum(xr, dtype=float)
    mean = (csum[w:] - csum[:-w]) / float(w)
    csum2 = np.cumsum(xr * xr, dtype=float)
    mean2 = (csum2[w:] - csum2[:-w]) / float(w)
    var = mean2 - mean * mean
    return var

def _box_var2d(arr: np.ndarray) -> np.ndarray:
    """
    3x3 box variance using sum of shifted copies (no scipy).
    """
    A = arr
    H, W = A.shape
    P = np.pad(A, 1, mode="reflect")
    # sum over 3x3
    S  = (
        P[0:H,0:W] + P[0:H,1:W+1] + P[0:H,2:W+2] +
        P[1:H+1,0:W] + P[1:H+1,1:W+1] + P[1:H+1,2:W+2] +
        P[2:H+2,0:W] + P[2:H+2,1:W+1] + P[2:H+2,2:W+2]
    )
    S2 = (
        P[0:H,0:W]**2 + P[0:H,1:W+1]**2 + P[0:H,2:W+2]**2 +
        P[1:H+1,0:W]**2 + P[1:H+1,1:W+1]**2 + P[1:H+1,2:W+2]**2 +
        P[2:H+2,0:W]**2 + P[2:H+2,1:W+1]**2 + P[2:H+2,2:W+2]**2
    )
    mean  = S / 9.0
    mean2 = S2 / 9.0
    return np.maximum(0.0, mean2 - mean*mean)

def _grad2d_x(Y: np.ndarray) -> np.ndarray:
    Gx = np.zeros_like(Y)
    Gx[:,1:-1] = 0.5*(Y[:,2:] - Y[:,:-2])
    Gx[:,0]  = Y[:,1] - Y[:,0]
    Gx[:,-1] = Y[:,-1] - Y[:,-2]
    return Gx

def _grad2d_y(Y: np.ndarray) -> np.ndarray:
    Gy = np.zeros_like(Y)
    Gy[1:-1,:] = 0.5*(Y[2:,:] - Y[:-2,:])
    Gy[0,:]  = Y[1,:] - Y[0,:]
    Gy[-1,:] = Y[-1,:] - Y[-2,:]
    return Gy

def _div2d(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """
    Divergence of a vector field p = (px,py) with simple centered differences.
    """
    H, W = px.shape
    div = np.zeros((H, W), dtype=float)
    # d/dx px
    div[:,1:-1] += 0.5*(px[:,2:] - px[:,:-2])
    div[:,0]    += px[:,1] - px[:,0]
    div[:,-1]   += px[:,-1] - px[:,-2]
    # d/dy py
    div[1:-1,:] += 0.5*(py[2:,:] - py[:-2,:])
    div[0,:]    += py[1,:] - py[0,:]
    div[-1,:]   += py[-1,:] - py[-2,:]
    return div

# ============================================================
# Local state (per-shape)
# ============================================================

_STATES: Dict[Tuple[int, ...], "LocalState"] = {}

class LocalState:
    def __init__(self, shape: Tuple[int, ...], cfg_phys: Dict[str, Any], cfg_f3: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None):
        self.shape = shape
        self.ndim = len(shape)
        self.rng = rng or np.random.default_rng(0)

        # Legacy physics knobs
        self.T_base = float(_safe_get(cfg_phys, "T", 0.001))
        self.flux_limit = float(_safe_get(cfg_phys, "flux_limit", 0.2))
        self.boundary_leak = float(_safe_get(cfg_phys, "boundary_leak", 0.0))
        self.update_mode = str(_safe_get(cfg_phys, "update_mode", "random"))

        # Fuka3 (optional)
        f3 = cfg_f3 or {}
        f3E = f3.get("energy", {})
        self.source_rate = float(_safe_get(f3E, "source_rate", 1.0))
        self.transport_conductance = float(_safe_get(f3E, "transport_conductance", 0.15))
        self.use_potential_smoothing = bool(_safe_get(f3E, "use_potential_smoothing", True))
        self.work_to_dissipation_fraction = float(_safe_get(f3E, "work_to_dissipation_fraction", 0.9))
        self.step_cost_coeff = float(_safe_get(f3E, "step_cost_coeff", 0.01))
        self.alpha_energy_per_entropy = float(_safe_get(f3E, "alpha_energy_per_entropy", 1.0))
        self.refund_fraction_on_prune = float(_safe_get(f3E, "refund_fraction_on_prune", 0.0))

        f3T = f3.get("temperature", {})
        self.T0 = float(_safe_get(f3T, "T0", self.T_base))
        self.beta_bound = float(_safe_get(f3T, "beta_bound", 0.01))
        self.beta_signal = float(_safe_get(f3T, "beta_signal", 0.0))
        self.use_edge_temperature = bool(_safe_get(f3T, "use_edge_temperature", False))

        f3C = f3.get("connection", {})
        f3Ci = f3C.get("init", {})
        f3Cl = f3C.get("learning", {})
        self.eta = float(_safe_get(f3Cl, "eta", 0.05))
        self.entropy_window = int(_safe_get(f3Cl, "entropy_window", 16))
        self.entropy_kind = str(_safe_get(f3Cl, "entropy_kind", "variance"))
        self.learn_flux_limit = float(_safe_get(f3Cl, "flux_limit", self.flux_limit))

        self.A_range = tuple(_safe_get(f3Ci, "amplitude_range", (0.05, 0.15)))
        self.f_range = tuple(_safe_get(f3Ci, "frequency_range", (0.05, 0.25)))
        self.phi_range = tuple(_safe_get(f3Ci, "phase_range", (0.0, 2*np.pi)))
        self.kappa_range = tuple(_safe_get(f3Ci, "curvature_range", (0.0, 0.1)))
        self.plasticity_range = tuple(_safe_get(f3Ci, "plasticity_range", (0.01, 0.05)))

        f3D = f3.get("denoising", {})
        self.use_sinusoid = bool(_safe_get(f3D, "use_sinusoid_term", True))
        self.kappa_scale = float(_safe_get(f3D, "kappa_scale", 1.0))
        self.grad_clip = float(_safe_get(f3D, "gradient_clip", 1.0))
        self.min_entropy_drop = float(_safe_get(f3D, "min_entropy_drop", 0.0))

        rng = self.rng
        if self.ndim == 1:
            X = shape[0]
            self.F = np.full((X,), 0.5, dtype=float)
            self.B = np.zeros((X,), dtype=float)
            self.T = np.full((X,), self.T0, dtype=float)
            E = max(1, X - 1)
            self.A = rng.uniform(*self.A_range, size=E)
            self.freq = rng.uniform(*self.f_range, size=E)
            self.phi = rng.uniform(*self.phi_range, size=E)
            self.kappa = rng.uniform(*self.kappa_range, size=E)
            self.plasticity = rng.uniform(*self.plasticity_range, size=E)
            self.last_delta = np.zeros((E,), dtype=float)
        elif self.ndim == 2:
            H, W = shape
            self.F = np.full((H, W), 0.5, dtype=float)
            self.B = np.zeros((H, W), dtype=float)
            self.T = np.full((H, W), self.T0, dtype=float)
            self.A = rng.uniform(*self.A_range, size=(H, W))
            self.freq = rng.uniform(*self.f_range, size=(H, W))
            self.phi = rng.uniform(*self.phi_range, size=(H, W))
            self.kappa = rng.uniform(*self.kappa_range, size=(H, W))
            self.plasticity = rng.uniform(*self.plasticity_range, size=(H, W))
        else:
            raise ValueError(f"Unsupported substrate ndim {self.ndim}")

    # ------------ Energy & Temperature ------------
    def _inject_sources(self, E_like: np.ndarray):
        flat = E_like.ravel()
        if flat.size == 0:
            return
        k = max(1, int(0.01 * flat.size))
        idx = np.argpartition(-flat, k-1)[:k]
        if self.ndim == 1:
            self.F[idx] += self.source_rate
        else:
            H, W = E_like.shape
            ys, xs = np.unravel_index(idx, (H, W))
            self.F[ys, xs] += self.source_rate

    def _transport_free_energy_1d(self, mu: np.ndarray):
        g = self.transport_conductance
        dmu = mu[:-1] - mu[1:]
        self.F[:-1] -= g * np.clip(dmu, 0, None)
        self.F[1:]  += g * np.clip(dmu, 0, None)
        self.F[1:]  -= g * np.clip(-dmu, 0, None)
        self.F[:-1] += g * np.clip(-dmu, 0, None)

    def _transport_free_energy_2d(self, mu: np.ndarray):
        g = self.transport_conductance
        H, W = mu.shape
        # right neighbor
        d = mu[:, :-1] - mu[:, 1:]
        self.F[:, :-1] -= g * np.clip(d, 0, None)
        self.F[:, 1:]  += g * np.clip(d, 0, None)
        self.F[:, 1:]  -= g * np.clip(-d, 0, None)
        self.F[:, :-1] += g * np.clip(-d, 0, None)
        # down neighbor
        d = mu[:-1, :] - mu[1:, :]
        self.F[:-1, :] -= g * np.clip(d, 0, None)
        self.F[1:,  :] += g * np.clip(d, 0, None)
        self.F[1:,  :] -= g * np.clip(-d, 0, None)
        self.F[:-1, :] += g * np.clip(-d, 0, None)

    def _update_temperature(self, signal_var: Optional[np.ndarray] = None):
        dens = self.B
        T = self.T0 + self.beta_bound * dens
        if self.beta_signal and signal_var is not None:
            T = T + self.beta_signal * signal_var
        self.T = T

    # ------------- Denoising / Learning -------------
    def _entropy_1d(self, delta: np.ndarray) -> np.ndarray:
        return _rolling_var_1d(delta, win=max(3, self.entropy_window))

    def _entropy_2d_edges(self, S: np.ndarray, E: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # edge deltas
        dx = S[:, 1:] - S[:, :-1]
        dy = S[1:, :] - S[:-1, :]
        ex = E[:, 1:] - E[:, :-1]
        ey = E[1:, :] - E[:-1, :]
        # local variance (3x3) of magnitudes as entropy proxy
        # center maps to cell-center via padding/averaging
        var_x = _box_var2d(np.pad(np.abs(dx), ((0,0),(0,1)), mode="edge"))
        var_y = _box_var2d(np.pad(np.abs(dy), ((0,1),(0,0)), mode="edge"))
        return dx, dy, ex, ey, var_x, var_y

    def _propose_update_1d(self, t_idx: int, delta: np.ndarray, env_edge_feat: np.ndarray):
        if self.use_sinusoid:
            sinus = np.sin(2.0 * np.pi * self.freq * t_idx + self.phi)
        else:
            sinus = 0.0
        grad_kappa = - (delta - env_edge_feat)
        grad_A     = sinus * (env_edge_feat - np.abs(delta))
        grad_freq  = self.A * np.cos(2.0*np.pi*self.freq*t_idx + self.phi) * (env_edge_feat - np.abs(delta))
        grad_phi   = self.A * np.cos(2.0*np.pi*self.freq*t_idx + self.phi)
        gc = self.grad_clip
        return (_clip(grad_A, -gc, gc),
                _clip(grad_freq, -gc, gc),
                _clip(grad_phi, -gc, gc),
                _clip(grad_kappa, -gc, gc))

    def _propose_update_2d(self, t_idx: int, S: np.ndarray, E: np.ndarray):
        H, W = S.shape
        if self.use_sinusoid:
            sinus = np.sin(2.0*np.pi*self.freq*t_idx + self.phi)
        else:
            sinus = 0.0
        dx, dy, ex, ey, vx, vy = self._entropy_2d_edges(S, E)
        # mismatch in edge magnitudes drives curvature gradient
        mism_x = (np.abs(dx) - np.abs(ex))
        mism_y = (np.abs(dy) - np.abs(ey))

        # accumulate edge contributions to cell-centered grads
        grad_kappa = np.zeros_like(S)
        grad_kappa[:, :-1] -= mism_x
        grad_kappa[:, 1:]  += mism_x
        grad_kappa[:-1, :] -= mism_y
        grad_kappa[1:,  :] += mism_y

        # sinusoidal terms align to env rhythms (weak prior)
        # use magnitude mismatch as envelope
        env_mag = _clip(np.hypot(ex, ey), 0.0, 1e9)
        env_mag_full = np.zeros_like(S)
        env_mag_full[:, :-1] += env_mag
        env_mag_full[:, 1:]  += env_mag
        # A, f, phi grads as gentle nudges
        grad_A    = sinus * (env_mag_full - np.abs(_grad2d_x(S)) - np.abs(_grad2d_y(S)))
        grad_freq = self.A * np.cos(2.0*np.pi*self.freq*t_idx + self.phi) * grad_A
        grad_phi  = self.A * np.cos(2.0*np.pi*self.freq*t_idx + self.phi)

        gc = self.grad_clip
        return (_clip(grad_A, -gc, gc),
                _clip(grad_freq, -gc, gc),
                _clip(grad_phi, -gc, gc),
                _clip(grad_kappa, -gc, gc),
                vx, vy, mism_x, mism_y)

    def _energy_gate_apply_params(self, T_eff: float, dS_pos_sum: float, grads: Tuple[np.ndarray, ...]):
        dA, df, dphi, dk = grads
        # proposed steps
        dA   = - self.eta * dA
        df   = - self.eta * df
        dphi = - self.eta * dphi
        dk   = - self.eta * dk

        # step cost (quadratic)
        step_cost = self.step_cost_coeff * (
            np.mean(dA*dA) + np.mean(df*df) + np.mean(dphi*dphi) + np.mean(dk*dk)
        )
        E_min = self.alpha_energy_per_entropy * T_eff * max(0.0, dS_pos_sum)
        E_req = E_min + step_cost + 1e-12

        F_local_mean = float(np.mean(self.F))
        scale = min(1.0, np.sqrt(max(F_local_mean, 0.0) / E_req)) if E_req > 0 else 1.0

        self.A     += scale * dA
        self.freq  += scale * df
        self.phi   += scale * dphi
        self.kappa += scale * dk

        spent  = float(scale * E_min + (scale**2) * step_cost)
        dissip = self.work_to_dissipation_fraction * spent
        return spent, dissip

# ============================================================
# Main stepping function
# ============================================================

def step_physics(
    prev_S: np.ndarray,
    env_row: np.ndarray,
    k_flux: float,
    k_motor: float,
    diffuse: float,
    decay: float,
    rng: np.random.Generator,
    band: int = 3,
    bc: str = "reflect",
    **phys_kwargs,
) -> Tuple[np.ndarray, float]:
    S = np.asarray(prev_S, dtype=float)
    E = np.asarray(env_row, dtype=float)
    shape = S.shape
    ndim = S.ndim

    cfg_phys = dict(phys_kwargs)
    cfg_f3 = cfg_phys.pop("fuka3", None)
    key = shape
    st = _STATES.get(key)
    if st is None:
        st = LocalState(shape, cfg_phys, cfg_f3, rng)
        _STATES[key] = st
    else:
        if "T" in cfg_phys:
            st.T_base = float(cfg_phys["T"])

    Sn = S / (1e-12 + float(np.max(np.abs(S)))) if np.any(S) else np.zeros_like(S)

    # ----------------- Energy injection & transport -----------------
    st._inject_sources(np.abs(E))
    if ndim == 1:
        st._transport_free_energy_1d(st.F)
    else:
        st._transport_free_energy_2d(st.F)

    # ----------------- Temperature update -----------------
    signal_var = None
    if ndim == 1:
        signal_var = _rolling_var_1d(S, win=max(3, st.entropy_window))
    else:
        # local 3x3 variance as proxy
        signal_var = _box_var2d(S)
    st._update_temperature(signal_var=signal_var)

    # ----------------- Denoising / Learning -----------------
    new_S = S.copy()
    flux_metric = 0.0
    T_eff = float(np.mean(st.T))

    if ndim == 1:
        delta = S[1:] - S[:-1]
        env_edge = E[1:] - E[:-1]
        dS_pos = np.maximum(0.0, np.abs(delta) - np.abs(env_edge))
        t_idx = rng.integers(0, 10**9)
        gA, gf, gphi, gk = st._propose_update_1d(t_idx, delta, np.abs(env_edge))
        spent, dissip = st._energy_gate_apply_params(T_eff, float(np.sum(dS_pos)), (gA, gf, gphi, gk))
        if spent > 0.0:
            st.F -= spent / max(1, st.F.size)
        if dissip > 0.0:
            st.B += dissip / max(1, st.B.size)
        kappa = np.clip(st.kappa, st.kappa_range[0], st.kappa_range[1])
        smooth = np.zeros_like(S)
        smooth[1:-1] = kappa[1:] * (S[2:] - S[1:-1]) - kappa[:-1] * (S[1:-1] - S[:-2])
        new_S = S + smooth

    else:
        # 2D: compute grads and weighted Laplacian step (divergence of kappa * grad S)
        t_idx = rng.integers(0, 10**9)
        gA, gf, gphi, gk, vx, vy, mism_x, mism_y = st._propose_update_2d(t_idx, S, E)
        # Entropy drop proxy: positive reduction in mismatch magnitudes
        dS_pos_sum = float(np.sum(np.maximum(0.0, np.abs(mism_x))) + np.sum(np.maximum(0.0, np.abs(mism_y))))
        spent, dissip = st._energy_gate_apply_params(T_eff, dS_pos_sum, (gA, gf, gphi, gk))
        if spent > 0.0:
            st.F -= spent / max(1, st.F.size)
        if dissip > 0.0:
            st.B += dissip / max(1, st.B.size)

        # Weighted smoothing: new_S = S + div( kappa * grad S )
        kappa = np.clip(st.kappa, st.kappa_range[0], st.kappa_range[1]) * st.kappa_scale
        Sx = _grad2d_x(S)
        Sy = _grad2d_y(S)
        px = kappa * Sx
        py = kappa * Sy
        smooth = _div2d(px, py)
        new_S = S + smooth

    # ============================================================
    # Legacy substrate dynamics (preserved look & UI)
    # ============================================================
    T_noise = float(st.T_base)

    if ndim == 1:
        if bc == "absorb":
            new_S[0] = 0.0
            new_S[-1] = 0.0
        if diffuse != 0.0:
            lap = np.zeros_like(new_S)
            lap[1:-1] = new_S[2:] - 2.0*new_S[1:-1] + new_S[0:-2]
            if bc == "wrap":
                lap[0]   = new_S[1] - 2.0*new_S[0] + new_S[-1]
                lap[-1]  = new_S[0] - 2.0*new_S[-1] + new_S[-2]
            new_S += float(diffuse) * lap
        pull = float(k_flux) * (E - new_S)
        pull = _clip(pull, -st.flux_limit, st.flux_limit)
        new_S += pull
        new_S *= (1.0 - float(decay))
        if T_noise > 0.0:
            new_S += T_noise * rng.standard_normal(size=new_S.shape)
        if k_motor != 0.0:
            motor_scale = np.power(1e-6 + np.abs(Sn), 0.5)
            new_S += float(k_motor) * motor_scale * rng.standard_normal(size=new_S.shape)
        flux_metric = float(np.mean(np.abs(pull)))
        if st.boundary_leak > 0.0 and new_S.size >= 2:
            new_S[0]  *= (1.0 - st.boundary_leak)
            new_S[-1] *= (1.0 - st.boundary_leak)
    else:
        cur = new_S
        if diffuse != 0.0:
            lap = np.zeros_like(cur)
            lap[1:-1,1:-1] = (
                cur[2:,1:-1] + cur[:-2,1:-1] + cur[1:-1,2:] + cur[1:-1,:-2] - 4.0*cur[1:-1,1:-1]
            )
            if bc == "wrap":
                lap[0,:]  = cur[1,:] + cur[-1,:] + np.roll(cur[0,:],1) + np.roll(cur[0,:],-1) - 4.0*cur[0,:]
                lap[-1,:] = cur[0,:] + cur[-2,:] + np.roll(cur[-1,:],1) + np.roll(cur[-1,:],-1) - 4.0*cur[-1,:]
                lap[:,0]  = cur[:,1] + cur[:,-1] + np.roll(cur[:,0],1) + np.roll(cur[:,0],-1) - 4.0*cur[:,0]
                lap[:,-1] = cur[:,0] + cur[:,-2] + np.roll(cur[:,-1],1) + np.roll(cur[:,-1],-1) - 4.0*cur[:,-1]
            cur = cur + float(diffuse) * lap
        pull = float(k_flux) * (E - cur)
        pull = _clip(pull, -st.flux_limit, st.flux_limit)
        cur = cur + pull
        cur *= (1.0 - float(decay))
        if T_noise > 0.0:
            cur += T_noise * rng.standard_normal(size=cur.shape)
        if k_motor != 0.0:
            motor_scale = np.power(1e-6 + np.abs(Sn), 0.5)
            cur += float(k_motor) * motor_scale * rng.standard_normal(size=cur.shape)
        if st.boundary_leak > 0.0 and cur.shape[0] >= 2 and cur.shape[1] >= 2:
            cur[0,:]  *= (1.0 - st.boundary_leak)
            cur[-1,:] *= (1.0 - st.boundary_leak)
            cur[:,0]  *= (1.0 - st.boundary_leak)
            cur[:,-1] *= (1.0 - st.boundary_leak)
        new_S = cur
        flux_metric = float(np.mean(np.abs(pull)))

    if not np.all(np.isfinite(new_S)):
        new_S = np.nan_to_num(new_S, nan=0.0, posinf=0.0, neginf=0.0)
    return new_S, flux_metric