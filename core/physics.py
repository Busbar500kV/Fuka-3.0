# core/physics.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

# Option‑B attractor (your file)
from .attractors import Attractor

# ============================================================
# Back‑compat helpers used by the Engine (unchanged signatures)
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

def _safe_get(d: Dict[str, Any], k: str, default):
    return default if not isinstance(d, dict) else default if d.get(k) is None else d.get(k)

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
    return np.maximum(var, 0.0)

def _box_var2d(arr: np.ndarray) -> np.ndarray:
    H, W = arr.shape
    P = np.pad(arr, 1, mode="reflect")
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
    return np.maximum(mean2 - mean*mean, 0.0)

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
    H, W = px.shape
    div = np.zeros((H, W), dtype=float)
    div[:,1:-1] += 0.5*(px[:,2:] - px[:,:-2])
    div[:,0]    += px[:,1] - px[:,0]
    div[:,-1]   += px[:,-1] - px[:,-2]
    div[1:-1,:] += 0.5*(py[2:,:] - py[:-2,:])
    div[0,:]    += py[1,:] - py[0,:]
    div[-1,:]   += py[-1,:] - py[-2,:]
    return div

# ============================================================
# Local State (per substrate grid)
# ============================================================

_STATES: Dict[Tuple[int, ...], "LocalState"] = {}

# Global, deterministic time index for physics steps
_GLOBAL_TICK = 0

class LocalState:
    def __init__(self, shape: Tuple[int, ...],
                 cfg_phys: Dict[str, Any],
                 cfg_f3: Optional[Dict[str, Any]],
                 rng: np.random.Generator):
        self.shape = shape
        self.ndim = len(shape)
        self.rng = rng
        self.t_idx = 0

        # legacy physics knobs
        self.T_base = float(_safe_get(cfg_phys, "T", 0.001))
        self.flux_limit = float(_safe_get(cfg_phys, "flux_limit", 0.2))
        self.boundary_leak = float(_safe_get(cfg_phys, "boundary_leak", 0.0))

        # fuka3: energy
        f3 = cfg_f3 or {}
        f3E = f3.get("energy", {})
        self.source_rate  = float(_safe_get(f3E, "source_rate", 1.0))
        self.transport_conductance = float(_safe_get(f3E, "transport_conductance", 0.15))
        self.work_to_dissipation_fraction = float(_safe_get(f3E, "work_to_dissipation_fraction", 0.9))
        self.step_cost_coeff = float(_safe_get(f3E, "step_cost_coeff", 0.01))
        self.alpha_energy_per_entropy = float(_safe_get(f3E, "alpha_energy_per_entropy", 1.0))

        # fuka3: temperature
        f3T = f3.get("temperature", {})
        self.T0 = float(_safe_get(f3T, "T0", self.T_base))
        self.beta_bound = float(_safe_get(f3T, "beta_bound", 0.01))
        self.beta_signal = float(_safe_get(f3T, "beta_signal", 0.0))

        # fuka3: connections
        f3C = f3.get("connection", {})
        f3Ci = f3C.get("init", {})
        f3Cl = f3C.get("learning", {})
        self.eta = float(_safe_get(f3Cl, "eta", 0.05))
        self.entropy_window = int(_safe_get(f3Cl, "entropy_window", 16))
        self.entropy_kind = str(_safe_get(f3Cl, "entropy_kind", "variance"))
        self.use_sinusoid = bool(_safe_get(f3.get("denoising", {}), "use_sinusoid_term", True))
        self.kappa_scale = float(_safe_get(f3.get("denoising", {}), "kappa_scale", 1.0))
        self.grad_clip = float(_safe_get(f3.get("denoising", {}), "gradient_clip", 1.0))

        self.A_range = tuple(_safe_get(f3Ci, "amplitude_range", (0.05, 0.15)))
        self.f_range = tuple(_safe_get(f3Ci, "frequency_range", (0.05, 0.25)))
        self.phi_range = tuple(_safe_get(f3Ci, "phase_range", (0.0, 2*np.pi)))
        self.kappa_range = tuple(_safe_get(f3Ci, "curvature_range", (0.0, 0.1)))
        self.plasticity_range = tuple(_safe_get(f3Ci, "plasticity_range", (0.01, 0.05)))

        # fuka3: attractors (Option‑B) — new knobs
        f3A = f3.get("attractors", {})
        # spawn bias
        self.attr_spawn_prob_base  = float(_safe_get(f3A, "spawn_prob_base", 0.003))
        self.attr_spawn_bias_w_env = float(_safe_get(f3A, "spawn_bias_env_weight", 0.6))
        self.attr_spawn_energy     = float(_safe_get(f3A, "spawn_energy", 0.05))
        # amplitude / costs
        self.attr_amp_init = float(_safe_get(f3A, "amplitude_init", 0.05))
        self.attr_amp_min  = float(_safe_get(f3A, "amplitude_min", 0.005))
        self.attr_amp_max  = float(_safe_get(f3A, "amplitude_max", 0.5))
        self.attr_maint_rate = float(_safe_get(f3A, "maint_cost_rate", 0.001))
        self.attr_decay    = float(_safe_get(f3A, "decay_rate", 0.01))
        self.attr_bank_leak = float(_safe_get(f3A, "bank_leak", 0.02))
        # shape ranges
        shapeA = f3A.get("shape", {})
        self.r_par_rng  = tuple(_safe_get(shapeA, "r_parallel_range", (2.0, 6.0)))
        self.r_perp_rng = tuple(_safe_get(shapeA, "r_perp_range", (1.0, 3.0)))
        self.theta_jitter = float(_safe_get(shapeA, "theta_jitter", 0.3))
        # influence weights
        infl = f3A.get("influence", {})
        self.c_theta_rho  = float(_safe_get(infl, "c_theta_rho", 0.5))  # align with env grad orientation
        self.c_theta_H    = float(_safe_get(infl, "c_theta_H",   0.5))  # weight on κ gradient along footprint
        self.c_alpha      = float(_safe_get(infl, "c_alpha",     0.25)) # bias A,f,phi
        self.c_beta       = float(_safe_get(infl, "c_beta",      0.25))
        self.birth_mul_kappa = float(_safe_get(infl, "birth_multiplier_kappa", 0.5))
        # budgets
        self.attr_max = int(_safe_get(f3A, "max_count", 256))
        self.attr_spawn_trials = int(_safe_get(f3A, "spawn_trials", 32))

        # fields
        if self.ndim == 1:
            X = shape[0]
            E = max(1, X - 1)
            self.F = np.full((X,), 0.5, float)
            self.B = np.zeros((X,), float)
            self.T = np.full((X,), self.T0, float)
            self.A     = self.rng.uniform(*self.A_range, size=E)
            self.freq  = self.rng.uniform(*self.f_range, size=E)
            self.phi   = self.rng.uniform(*self.phi_range, size=E)
            self.kappa = self.rng.uniform(*self.kappa_range, size=E)
        else:
            H, W = shape
            self.F = np.full((H, W), 0.5, float)
            self.B = np.zeros((H, W), float)
            self.T = np.full((H, W), self.T0, float)
            self.A     = self.rng.uniform(*self.A_range, size=(H, W))
            self.freq  = self.rng.uniform(*self.f_range, size=(H, W))
            self.phi   = self.rng.uniform(*self.phi_range, size=(H, W))
            self.kappa = self.rng.uniform(*self.kappa_range, size=(H, W))

        # attractors
        self.attractors: List[Attractor] = []
        self._next_attr_id = 1

        # cache for metrics
        self._last_spent = 0.0
        self._last_dissip = 0.0

    # ----- energy & temperature -----
    def _inject_sources(self, env_like: np.ndarray):
        # inject into top‑energy percent (simple local uptake)
        flat = np.abs(env_like).ravel()
        if flat.size == 0:
            return
        k = max(1, int(0.01 * flat.size))
        idx = np.argpartition(-flat, k-1)[:k]
        if self.ndim == 1:
            self.F[idx] += self.source_rate
        else:
            H, W = env_like.shape
            ys, xs = np.unravel_index(idx, (H, W))
            self.F[ys, xs] += self.source_rate

    def _transport_F_1d(self, mu: np.ndarray):
        g = self.transport_conductance
        d = mu[:-1] - mu[1:]
        self.F[:-1] -= g * np.clip(d, 0, None); self.F[1:]  += g * np.clip(d, 0, None)
        self.F[1:]  -= g * np.clip(-d, 0, None); self.F[:-1] += g * np.clip(-d, 0, None)

    def _transport_F_2d(self, mu: np.ndarray):
        g = self.transport_conductance
        d = mu[:, :-1] - mu[:, 1:]
        self.F[:, :-1] -= g * np.clip(d, 0, None); self.F[:, 1:]  += g * np.clip(d, 0, None)
        self.F[:, 1:]  -= g * np.clip(-d, 0, None); self.F[:, :-1] += g * np.clip(-d, 0, None)
        d = mu[:-1, :] - mu[1:, :]
        self.F[:-1, :] -= g * np.clip(d, 0, None); self.F[1:,  :] += g * np.clip(d, 0, None)
        self.F[1:,  :] -= g * np.clip(-d, 0, None); self.F[:-1, :] += g * np.clip(-d, 0, None)

    def _update_temperature(self, signal_var: Optional[np.ndarray]):
        T = self.T0 + self.beta_bound * self.B
        if signal_var is not None and self.beta_signal != 0.0:
            T = T + self.beta_signal * signal_var
        self.T = T

    # =======================
    # Attractors lifecycle
    # =======================

    def _maybe_spawn(self, env_like: np.ndarray):
        if self.ndim != 2:
            return  # Option‑B is 2‑D focused
        if len(self.attractors) >= self.attr_max:
            return

        H, W = self.shape
        # env‑biased coin: normalize |E|
        Eabs = np.abs(env_like)
        if Eabs.size == 0:
            return
        En = Eabs / (1e-12 + np.max(Eabs))
        trials = max(1, min(self.attr_spawn_trials, self.attr_max - len(self.attractors)))

        for _ in range(trials):
            y = int(self.rng.integers(0, H))
            x = int(self.rng.integers(0, W))
            bias = float(En[y, x])
            p_spawn = self.attr_spawn_prob_base * (1.0 + self.attr_spawn_bias_w_env * bias)
            if (self.rng.random() < p_spawn) and (self.F[y, x] >= self.attr_spawn_energy):
                # draw oriented blob
                r_par  = float(self.rng.uniform(*self.r_par_rng))
                r_perp = float(self.rng.uniform(*self.r_perp_rng))
                theta  = float(self.rng.uniform(-np.pi, np.pi))
                amp    = float(self.attr_amp_init)
                self.F[y, x] -= self.attr_spawn_energy
                self.B[y, x] += self.work_to_dissipation_fraction * self.attr_spawn_energy
                k = Attractor(self._next_attr_id, (y, x), r_par, r_perp, theta,
                              amp, self.attr_maint_rate, self.attr_decay,
                              sigma_theta=0.5, sigma_signal=0.5)
                self._next_attr_id += 1
                self.attractors.append(k)
                if len(self.attractors) >= self.attr_max:
                    break

    def _maintain_and_decay(self):
        alive: List[Attractor] = []
        if self.ndim != 2:
            self.attractors = alive
            return
        for k in self.attractors:
            y, x = k.pos
            need = k.amp * k.maint_cost
            if self.F[y, x] >= need:
                self.F[y, x] -= need
                self.B[y, x] += self.work_to_dissipation_fraction * need
                k.bank = max(0.0, (1.0 - self.attr_bank_leak) * k.bank + 0.25 * need)
            else:
                k.amp *= 0.5
            k.amp *= (1.0 - self.attr_decay)
            k.amp = float(np.clip(k.amp, 0.0, self.attr_amp_max))
            k.age += 1
            if k.amp >= self.attr_amp_min:
                alive.append(k)
        self.attractors = alive

    # =======================
    # Gradient proposals
    # =======================

    def _propose_grads_2d(self, t_idx: int, S: np.ndarray, E: np.ndarray):
        # substrate edges
        dx = S[:, 1:] - S[:, :-1]   # (H, W-1)
        dy = S[1:, :] - S[:-1, :]   # (H-1, W)
        # env edges
        ex = E[:, 1:] - E[:, :-1]
        ey = E[1:, :] - E[:-1, :]

        # mismatch (edge‑wise)
        mism_x = np.abs(dx) - np.abs(ex)
        mism_y = np.abs(dy) - np.abs(ey)

        # optional sinusoid
        if self.use_sinusoid:
            sinus = np.sin(2.0 * np.pi * self.freq * t_idx + self.phi)  # (H, W)
        else:
            sinus = 0.0

        # κ gradient at cell centers via distributing edge mismatches
        g_kappa = np.zeros_like(S)
        g_kappa[:, :-1] -= mism_x
        g_kappa[:,  1:] += mism_x
        g_kappa[:-1, :] -= mism_y
        g_kappa[ 1:, :] += mism_y

        # cell‑center env vs sub magnitudes
        env_mag = np.zeros_like(S)
        env_mag[:, :-1] += np.abs(ex); env_mag[:,  1:] += np.abs(ex)
        env_mag[:-1, :] += np.abs(ey); env_mag[ 1:, :] += np.abs(ey)
        Sx = _grad2d_x(S); Sy = _grad2d_y(S)
        sub_mag = np.abs(Sx) + np.abs(Sy)

        # DOF parameter gradients
        g_A   = sinus * (env_mag - sub_mag)
        g_f   = self.A * np.cos(2.0 * np.pi * self.freq * t_idx + self.phi) * g_A
        g_phi = self.A * np.cos(2.0 * np.pi * self.freq * t_idx + self.phi)

        gc = self.grad_clip
        return (_clip(g_A,-gc,gc), _clip(g_f,-gc,gc), _clip(g_phi,-gc,gc),
                _clip(g_kappa,-gc,gc), mism_x, mism_y)

    # =======================
    # Attractor influence
    # =======================

    def _boost_grads_with_attractors_2d(self, gA, gf, gphi, gk, S: np.ndarray, E: np.ndarray):
        if not self.attractors:
            return gA, gf, gphi, gk

        H, W = S.shape
        # env gradient orientation (unit vectors)
        Ex = _grad2d_x(E); Ey = _grad2d_y(E)
        magE = np.hypot(Ex, Ey) + 1e-12
        ux, uy = Ex / magE, Ey / magE  # orientation field

        for k in self.attractors:
            w = k.field(H, W)  # normalized footprint * amplitude
            # preferred direction unit vector from attractor theta
            dx = np.cos(k.theta); dy = np.sin(k.theta)
            # alignment of local env gradient with attractor orientation
            align = (ux * dx + uy * dy)  # in [-1,1]
            align = np.clip(align, -1.0, 1.0)

            # steer κ to reduce mismatch along footprint weighted by alignment
            gk -= self.c_theta_H * w * align

            # light param bias to help “encode” repeated structure
            gA   -= self.c_alpha * w * (self.A - np.mean(self.A))
            gf   -= self.c_alpha * w * (self.freq - np.mean(self.freq))
            gphi -= self.c_beta  * w * (self.phi - np.mean(self.phi))

        return gA, gf, gphi, gk

    # =======================
    # Energy gate & apply steps
    # =======================

    def _apply_energy_gated(self, T_eff: float, dS_sum: float, dA, df, dphi, dk, S: np.ndarray):
        step_cost = self.step_cost_coeff * (np.mean(dA*dA) + np.mean(df*df) + np.mean(dphi*dphi) + np.mean(dk*dk))
        E_min = self.alpha_energy_per_entropy * T_eff * max(0.0, dS_sum)
        E_req = E_min + step_cost + 1e-12
        F_mean = float(np.mean(self.F))
        scale = min(1.0, np.sqrt(max(F_mean, 0.0) / E_req)) if E_req > 0 else 1.0

        self.A     += scale * dA
        self.freq  += scale * df
        self.phi   += scale * dphi
        self.kappa += scale * dk

        spent  = float(scale * E_min + (scale**2) * step_cost)
        dissip = self.work_to_dissipation_fraction * spent
        if spent > 0.0:
            self.F -= spent / max(1, self.F.size)
        if dissip > 0.0:
            self.B += dissip / max(1, self.B.size)
        self._last_spent  = spent
        self._last_dissip = dissip

        # maps (uniform for now; enough for reward/selection)
        energy_paid_map = np.full_like(S, spent / max(1, S.size))
        reward_map = np.full_like(S, dS_sum / max(1, S.size))
        return energy_paid_map, reward_map

    # =======================
    # Selection / propagation
    # =======================

    def _reward_and_select(self, reward_map: np.ndarray, energy_map: np.ndarray):
        if self.ndim != 2 or not self.attractors:
            return
        H, W = self.shape
        win = max(1, int(2 * np.mean(self.r_par_rng)))
        for k in self.attractors:
            y0 = max(0, k.pos[0] - win); y1 = min(H, k.pos[0] + win + 1)
            x0 = max(0, k.pos[1] - win); x1 = min(W, k.pos[1] + win + 1)
            Wmask = k.field(H, W)[y0:y1, x0:x1]
            Wmask /= (1e-12 + np.sum(Wmask))
            r = float(np.sum(Wmask * reward_map[y0:y1, x0:x1]))
            e = float(np.sum(Wmask * energy_map[y0:y1, x0:x1]))
            eff = r / (e + 1e-12)
            k.reward_avg = 0.9 * k.reward_avg + 0.1 * eff
            # bank -> amplitude (bounded)
            k.amp = float(np.clip(k.amp + 0.1 * (eff - k.reward_avg) * k.amp, 0.0, self.attr_amp_max))

    def _propagate_if_strong(self):
        if self.ndim != 2 or not self.attractors:
            return
        H, W = self.shape
        for k in list(self.attractors):
            if k.amp < 5.0 * self.attr_amp_min:
                continue
            y, x = k.pos
            nbrs = [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]
            for ny, nx in nbrs:
                if 0 <= ny < H and 0 <= nx < W and self.F[ny, nx] >= self.attr_spawn_energy and self.rng.random() < 0.25:
                    self.F[ny, nx] -= self.attr_spawn_energy
                    self.B[ny, nx] += self.work_to_dissipation_fraction * self.attr_spawn_energy
                    r_par  = float(np.clip(k.r_par  + self.rng.normal(0, 0.2), *self.r_par_rng))
                    r_perp = float(np.clip(k.r_perp + self.rng.normal(0, 0.2), *self.r_perp_rng))
                    theta  = float((k.theta + self.rng.normal(0, self.theta_jitter) + np.pi) % (2*np.pi) - np.pi)
                    amp    = float(self.attr_amp_init)
                    kk = Attractor(self._next_attr_id, (ny, nx), r_par, r_perp, theta,
                                   amp, self.attr_maint_rate, self.attr_decay,
                                   sigma_theta=0.5, sigma_signal=0.5)
                    self._next_attr_id += 1
                    self.attractors.append(kk)
                    if len(self.attractors) >= self.attr_max:
                        return

# ============================================================
# Stepping (public API used by Engine)
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
    S = np.asarray(prev_S, float)
    E = np.asarray(env_row, float)
    shape = S.shape
    cfg_phys = dict(phys_kwargs)
    cfg_f3 = cfg_phys.pop("fuka3", None)

    st = _STATES.get(shape)
    if st is None:
        st = LocalState(shape, cfg_phys, cfg_f3, rng)
        _STATES[shape] = st
    else:
        if "T" in cfg_phys:
            st.T_base = float(cfg_phys["T"])

    # Deterministic time index (no rename, no LocalState attribute needed)
    global _GLOBAL_TICK
    t_idx = _GLOBAL_TICK
    _GLOBAL_TICK += 1
    
    # normalize for motor term
    Sn = S / (1e-12 + float(np.max(np.abs(S)))) if np.any(S) else np.zeros_like(S)

    # Energy: inject & transport
    st._inject_sources(E)
    if st.ndim == 1:
        st._transport_F_1d(st.F)
    else:
        st._transport_F_2d(st.F)

    # Temperature update
    signal_var = _rolling_var_1d(S, win=max(3, st.entropy_window)) if st.ndim == 1 else _box_var2d(S)
    st._update_temperature(signal_var)

    # Attractors
    if st.ndim == 2:
        st._maybe_spawn(E)
        st._maintain_and_decay()

    # --- Denoising / learning (with attractor bias) ---
    new_S = S.copy()
    T_eff = float(np.mean(st.T))

    if st.ndim == 1:
        # (keep legacy 1‑D branch minimal)
        # simple diffusion/flux/motor/decay only
        cur = S.copy()
        if diffuse != 0.0:
            lap = np.zeros_like(cur)
            lap[1:-1] = cur[2:] - 2.0*cur[1:-1] + cur[:-2]
            if bc == "wrap":
                lap[0]   = cur[1] - 2.0*cur[0] + cur[-1]
                lap[-1]  = cur[0] - 2.0*cur[-1] + cur[-2]
            cur = cur + float(diffuse) * lap
        pull = float(k_flux) * (E - cur)
        pull = _clip(pull, -st.flux_limit, st.flux_limit)
        cur = cur + pull
        cur = cur * (1.0 - float(decay))
        if st.T_base > 0.0:
            cur = cur + st.T_base * rng.standard_normal(size=cur.shape)
        if k_motor != 0.0:
            motor_scale = np.power(1e-6 + np.abs(Sn), 0.5)
            cur = cur + float(k_motor) * motor_scale * rng.standard_normal(size=cur.shape)
        if st.boundary_leak > 0.0 and cur.size >= 2:
            cur[0]  *= (1.0 - st.boundary_leak)
            cur[-1] *= (1.0 - st.boundary_leak)
        flux_metric = float(np.mean(np.abs(pull)))
        if not np.all(np.isfinite(cur)):
            cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)
        
        return cur, flux_metric

    # 2‑D branch (Option‑B aware)
    
    gA, gf, gphi, gk, mism_x, mism_y = st._propose_grads_2d(t_idx, S, E)
    gA, gf, gphi, gk = st._boost_grads_with_attractors_2d(gA, gf, gphi, gk, S, E)

    dA   = - st.eta * gA
    df   = - st.eta * gf
    dphi = - st.eta * gphi
    dk   = - st.eta * gk

    dS_pos_sum = float(np.sum(np.maximum(0.0, np.abs(mism_x))) + np.sum(np.maximum(0.0, np.abs(mism_y))))
    energy_paid_map, reward_map = st._apply_energy_gated(T_eff, dS_pos_sum, dA, df, dphi, dk, S)

    # κ‑weighted smoothing (divergence form)
    kappa = np.clip(st.kappa, st.kappa_range[0], st.kappa_range[1]) * st.kappa_scale
    Sx = _grad2d_x(S); Sy = _grad2d_y(S)
    px = kappa * Sx;   py = kappa * Sy
    new_S = S + _div2d(px, py)

    # Selection / propagation
    st._reward_and_select(reward_map, energy_paid_map)
    st._propagate_if_strong()

    # --- Legacy substrate dynamics layered on top (visual parity) ---
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
    cur = cur * (1.0 - float(decay))
    if st.T_base > 0.0:
        cur = cur + st.T_base * rng.standard_normal(size=cur.shape)
    if k_motor != 0.0:
        Sn2 = S / (1e-6 + float(np.max(np.abs(S)))) if np.any(S) else np.zeros_like(S)
        motor_scale = np.power(1e-6 + np.abs(Sn2), 0.5)
        cur = cur + float(k_motor) * motor_scale * rng.standard_normal(size=cur.shape)
    if st.boundary_leak > 0.0 and cur.shape[0] >= 2 and cur.shape[1] >= 2:
        cur[0,:]  *= (1.0 - st.boundary_leak); cur[-1,:] *= (1.0 - st.boundary_leak)
        cur[:,0]  *= (1.0 - st.boundary_leak); cur[:,-1] *= (1.0 - st.boundary_leak)

    flux_metric = float(np.mean(np.abs(pull)))
    if not np.all(np.isfinite(cur)):
        cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)
    
    # save snapshots for metrics
    st._last_S = cur
    st._last_E = E
    
    return cur, flux_metric

# ============================================================
# Metrics (public)
# ============================================================

def get_fuka3_metrics():
    """
    Return a list of metrics dicts, one per active substrate shape/state.
    Provides:
      - entropy_mean         : from substrate S (gradient-based in 2D, rolling-var in 1D)
      - entropy_p95          : p95 of the same S-based entropy field
      - entropy_mean_kappa   : diagnostic entropy-like proxy computed from kappa
    """
    out = []
    for shape, st in list(_STATES.items()):
        try:
            F_total = float(np.sum(st.F))
            B_total = float(np.sum(st.B))
            T_mean  = float(np.mean(st.T))

            # --- Build S-based entropy field (depends on dimensionality) ---
            ent_mean_S = 0.0
            ent_p95_S  = 0.0
            if hasattr(st, "_last_S"):
                if st.ndim == 1:
                    # rolling variance of S itself (1D)
                    ent_field_S = _rolling_var_1d(st._last_S, win=max(3, st.entropy_window))
                    ent_mean_S  = float(np.mean(ent_field_S)) if ent_field_S.size else 0.0
                    ent_p95_S   = float(np.percentile(ent_field_S, 95)) if ent_field_S.size else 0.0
                else:
                    # variance of gradient magnitude (2D)
                    Gmag = np.abs(_grad2d_x(st._last_S)) + np.abs(_grad2d_y(st._last_S))
                    ent_field_S = _box_var2d(Gmag)
                    ent_mean_S  = float(np.mean(ent_field_S)) if ent_field_S.size else 0.0
                    ent_p95_S   = float(np.percentile(ent_field_S, 95)) if ent_field_S.size else 0.0

            # --- κ-based diagnostic (same shape as S in 2D; 1D uses diff proxy) ---
            if st.ndim == 1:
                ent_field_k = np.abs(np.diff(st.kappa)) if st.kappa.size > 1 else np.zeros(1, dtype=float)
                ent_mean_k  = float(np.mean(ent_field_k))
            else:
                ent_field_k = _box_var2d(st.kappa) if st.kappa.size > 1 else np.zeros_like(st.kappa)
                ent_mean_k  = float(np.mean(ent_field_k))

            # --- other counters / bookkeeping ---
            attrs_alive = len(getattr(st, "attractors", []))
            avg_reward  = float(np.mean([a.reward_avg for a in st.attractors])) if attrs_alive else 0.0
            last_spent  = float(getattr(st, "_last_spent", 0.0))
            last_diss   = float(getattr(st, "_last_dissip", 0.0))
            conn_alive  = int(np.sum(np.abs(st.kappa) > 1e-6))
            eff_ratio   = (ent_mean_S / (last_spent + 1e-12)) if last_spent > 0.0 else 0.0

            out.append({
                "shape": tuple(shape),
                "ndim": int(st.ndim),
                "free_energy_total": F_total,
                "bound_energy_total": B_total,
                "temperature_mean": T_mean,

                # S-based entropy signal (what your UI reads)
                "entropy_mean": ent_mean_S,
                "entropy_p95":  ent_p95_S,

                # diagnostic κ “entropy”
                "entropy_mean_kappa": ent_mean_k,

                "connections_alive": conn_alive,
                "attractors_alive": int(attrs_alive),
                "avg_reward": avg_reward,
                "work_paid_per_tick": last_spent,
                "dissipation_per_tick": last_diss,
                "efficiency_ratio": eff_ratio,
            })
        except Exception:
            # don't let a single bad state kill the metrics panel
            continue
    return out
    
# ------------------------------------------------------------
# Attractors snapshot for UI (used by app.py overlay)
# ------------------------------------------------------------
def get_attractors_snapshot() -> List[Dict[str, Any]]:
    """
    Returns a list of per-shape snapshots. Each item:
      {
        "shape": (H, W) or (X,),
        "items": [
          {"id": int, "pos": (y,x), "theta": float, "r_par": float, "r_perp": float,
           "amp": float, "age": int, "alive": bool}
        ]
      }
    Only 2-D shapes (option-B) emit items.
    """
    snaps: List[Dict[str, Any]] = []
    for shape, st in list(_STATES.items()):
        try:
            entry: Dict[str, Any] = {"shape": tuple(shape), "items": []}
            if getattr(st, "ndim", 0) == 2 and hasattr(st, "attractors"):
                for a in st.attractors:
                    entry["items"].append({
                        "id": int(getattr(a, "id", -1)),
                        "pos": tuple(getattr(a, "pos", (0, 0))),
                        "theta": float(getattr(a, "theta", 0.0)),
                        "r_par": float(getattr(a, "r_par", 1.0)),
                        "r_perp": float(getattr(a, "r_perp", 1.0)),
                        "amp": float(getattr(a, "amp", 0.0)),
                        "age": int(getattr(a, "age", 0)),
                        "alive": bool(getattr(a, "alive", True)),
                    })
            snaps.append(entry)
        except Exception:
            continue
    return snaps

def get_active_shapes():
    """
    Returns a list of dicts like:
      [{"shape": (H, W) or (X,), "ndim": 1|2}]
    Useful for debugging coordinate consistency from the UI.
    """
    rows = []
    for shape, st in list(_STATES.items()):
        rows.append({"shape": tuple(shape), "ndim": int(getattr(st, "ndim", len(shape)))})
    return rows

def clear_states():
    """Clear cached LocalState instances (fresh start)."""
    _STATES.clear()
    global _GLOBAL_TICK
    _GLOBAL_TICK = 0