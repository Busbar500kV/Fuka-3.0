# core/physics.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

# ---------------- Back-compat helpers used by the Engine ----------------

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

# ---------------- Utilities ----------------

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

# ---------------- Attractors ----------------

class Attractor:
    __slots__ = ("pos","dim","radius","amp","psi_A","psi_f","psi_phi","psi_kappa",
                 "maint_cost","decay","sigma_theta","sigma_signal","E_spawn","E_prop",
                 "alive","reward_avg")
    def __init__(self, pos, dim, radius, amp, psi, maint_cost, decay, sigma_theta, sigma_signal, E_spawn, E_prop):
        self.pos = pos
        self.dim = dim
        self.radius = float(radius)
        self.amp = float(amp)
        self.psi_A = float(psi.get("A", 0.1))
        self.psi_f = float(psi.get("f", 0.1))
        self.psi_phi = float(psi.get("phi", 0.0))
        self.psi_kappa = float(psi.get("kappa", 0.05))
        self.maint_cost = float(maint_cost)
        self.decay = float(decay)
        self.sigma_theta = float(sigma_theta)
        self.sigma_signal = float(sigma_signal)
        self.E_spawn = float(E_spawn)
        self.E_prop = float(E_prop)
        self.alive = True
        self.reward_avg = 0.0

    def footprint_1d(self, X: int) -> np.ndarray:
        r2 = max(1e-8, self.radius*self.radius)
        xs = np.arange(X)
        return np.exp(- (xs - int(self.pos))**2 / (2.0*r2))

    def footprint_2d(self, H: int, W: int) -> np.ndarray:
        r2 = max(1e-8, self.radius*self.radius)
        y, x = np.indices((H, W))
        cy, cx = self.pos
        return np.exp(- ((y - cy)**2 + (x - cx)**2) / (2.0*r2))

# ---------------- Local (per-shape) state ----------------

_STATES: Dict[Tuple[int, ...], "LocalState"] = {}

class LocalState:
    def __init__(self, shape: Tuple[int, ...], cfg_phys: Dict[str, Any], cfg_f3: Optional[Dict[str, Any]], rng: np.random.Generator):
        self.shape = shape
        self.ndim = len(shape)
        self.rng = rng

        # legacy physics knobs (UI)
        self.T_base = float(_safe_get(cfg_phys, "T", 0.001))
        self.flux_limit = float(_safe_get(cfg_phys, "flux_limit", 0.2))
        self.boundary_leak = float(_safe_get(cfg_phys, "boundary_leak", 0.0))

        # fuka3 knobs (internal)
        f3 = cfg_f3 or {}
        f3E = f3.get("energy", {})
        self.source_rate  = float(_safe_get(f3E, "source_rate", 1.0))
        self.transport_conductance = float(_safe_get(f3E, "transport_conductance", 0.15))
        self.work_to_dissipation_fraction = float(_safe_get(f3E, "work_to_dissipation_fraction", 0.9))
        self.step_cost_coeff = float(_safe_get(f3E, "step_cost_coeff", 0.01))
        self.alpha_energy_per_entropy = float(_safe_get(f3E, "alpha_energy_per_entropy", 1.0))

        f3T = f3.get("temperature", {})
        self.T0 = float(_safe_get(f3T, "T0", self.T_base))
        self.beta_bound = float(_safe_get(f3T, "beta_bound", 0.01))
        self.beta_signal = float(_safe_get(f3T, "beta_signal", 0.0))

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

        f3A = f3.get("attractors", {})
        self.attr_spawn_prob = float(_safe_get(f3A, "spawn_prob", 0.01))
        self.attr_spawn_energy = float(_safe_get(f3A, "spawn_energy", 0.05))
        self.attr_maint_rate = float(_safe_get(f3A, "maint_cost_rate", 0.001))
        self.attr_decay = float(_safe_get(f3A, "decay_rate", 0.01))
        self.attr_amp_init = float(_safe_get(f3A, "amplitude_init", 0.05))
        self.attr_amp_min = float(_safe_get(f3A, "amplitude_min", 0.005))
        self.attr_radius = int(_safe_get(f3A, "radius", 1))
        self.attr_eta_amp = float(_safe_get(f3A, "eta_amplitude", 0.1))
        self.attr_sigma_theta = float(_safe_get(f3A, "sigma_theta", 0.5))
        self.attr_sigma_signal = float(_safe_get(f3A, "sigma_signal", 0.5))
        self.attr_prop_energy = float(_safe_get(f3A, "propagate_energy", 0.05))

        # fields
        if self.ndim == 1:
            X = shape[0]
            self.F = np.full((X,), 0.5, float)
            self.B = np.zeros((X,), float)
            self.T = np.full((X,), self.T0, float)
            E = max(1, X - 1)
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

        self.attractors: List[Attractor] = []

    # ----- energy & temperature -----
    def _inject_sources(self, env_like: np.ndarray):
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
        self.F[:-1] -= g * np.clip(d, 0, None)
        self.F[1:]  += g * np.clip(d, 0, None)
        self.F[1:]  -= g * np.clip(-d, 0, None)
        self.F[:-1] += g * np.clip(-d, 0, None)

    def _transport_F_2d(self, mu: np.ndarray):
        g = self.transport_conductance
        d = mu[:, :-1] - mu[:, 1:]
        self.F[:, :-1] -= g * np.clip(d, 0, None)
        self.F[:, 1:]  += g * np.clip(d, 0, None)
        self.F[:, 1:]  -= g * np.clip(-d, 0, None)
        self.F[:, :-1] += g * np.clip(-d, 0, None)
        d = mu[:-1, :] - mu[1:, :]
        self.F[:-1, :] -= g * np.clip(d, 0, None)
        self.F[1:,  :] += g * np.clip(d, 0, None)
        self.F[1:,  :] -= g * np.clip(-d, 0, None)
        self.F[:-1, :] += g * np.clip(-d, 0, None)

    def _update_temperature(self, signal_var: Optional[np.ndarray]):
        T = self.T0 + self.beta_bound * self.B
        if signal_var is not None and self.beta_signal != 0.0:
            T = T + self.beta_signal * signal_var
        self.T = T

    # ----- attractors lifecycle -----
    def _maybe_spawn(self):
        if self.ndim == 1:
            X = self.shape[0]
            for i in range(X):
                if self.rng.random() < self.attr_spawn_prob and self.F[i] >= self.attr_spawn_energy:
                    psi = {
                        "A": self.rng.uniform(*self.A_range),
                        "f": self.rng.uniform(*self.f_range),
                        "phi": self.rng.uniform(*self.phi_range),
                        "kappa": self.rng.uniform(*self.kappa_range),
                    }
                    self.F[i] -= self.attr_spawn_energy
                    self.B[i] += self.work_to_dissipation_fraction * self.attr_spawn_energy
                    self.attractors.append(Attractor(i, 1, self.attr_radius, self.attr_amp_init, psi,
                                                     self.attr_maint_rate, self.attr_decay,
                                                     self.attr_sigma_theta, self.attr_sigma_signal,
                                                     self.attr_spawn_energy, self.attr_prop_energy))
        else:
            H, W = self.shape
            for y in range(H):
                for x in range(W):
                    if self.rng.random() < self.attr_spawn_prob and self.F[y,x] >= self.attr_spawn_energy:
                        psi = {
                            "A": self.rng.uniform(*self.A_range),
                            "f": self.rng.uniform(*self.f_range),
                            "phi": self.rng.uniform(*self.phi_range),
                            "kappa": self.rng.uniform(*self.kappa_range),
                        }
                        self.F[y,x] -= self.attr_spawn_energy
                        self.B[y,x] += self.work_to_dissipation_fraction * self.attr_spawn_energy
                        self.attractors.append(Attractor((y,x), 2, self.attr_radius, self.attr_amp_init, psi,
                                                         self.attr_maint_rate, self.attr_decay,
                                                         self.attr_sigma_theta, self.attr_sigma_signal,
                                                         self.attr_spawn_energy, self.attr_prop_energy))

    def _maintain_and_decay(self):
        alive = []
        if self.ndim == 1:
            for k in self.attractors:
                i = int(k.pos)
                need = k.amp * k.maint_cost
                if self.F[i] >= need:
                    self.F[i] -= need
                    self.B[i] += self.work_to_dissipation_fraction * need
                else:
                    k.amp *= 0.5
                k.amp *= (1.0 - self.attr_decay)
                if k.amp >= self.attr_amp_min:
                    alive.append(k)
        else:
            for k in self.attractors:
                y, x = k.pos
                need = k.amp * k.maint_cost
                if self.F[y,x] >= need:
                    self.F[y,x] -= need
                    self.B[y,x] += self.work_to_dissipation_fraction * need
                else:
                    k.amp *= 0.5
                k.amp *= (1.0 - self.attr_decay)
                if k.amp >= self.attr_amp_min:
                    alive.append(k)
        self.attractors = alive

    def _masks(self):
        ms = []
        if self.ndim == 1:
            X = self.shape[0]
            for k in self.attractors:
                ms.append((k, k.footprint_1d(X)))
        else:
            H, W = self.shape
            for k in self.attractors:
                ms.append((k, k.footprint_2d(H, W)))
        return ms

    # ----- DOF gradient proposals -----
    def _propose_grads_1d(self, t_idx: int, S: np.ndarray, E: np.ndarray):
        delta = S[1:] - S[:-1]
        env_edge = E[1:] - E[:-1]
        if self.use_sinusoid:
            sinus = np.sin(2.0*np.pi*self.freq*t_idx + self.phi)
        else:
            sinus = 0.0
        g_kappa = - (delta - np.abs(env_edge))
        g_A     = sinus * (np.abs(env_edge) - np.abs(delta))
        g_f     = self.A * np.cos(2.0*np.pi*self.freq*t_idx + self.phi) * (np.abs(env_edge) - np.abs(delta))
        g_phi   = self.A * np.cos(2.0*np.pi*self.freq*t_idx + self.phi)
        gc = self.grad_clip
        return (_clip(g_A,-gc,gc), _clip(g_f,-gc,gc), _clip(g_phi,-gc,gc), _clip(g_kappa,-gc,gc), delta, env_edge)

    def _propose_grads_2d(self, t_idx: int, S: np.ndarray, E: np.ndarray):
        dx = S[:,1:] - S[:,:-1]
        dy = S[1:,:] - S[:-1,:]
        ex = E[:,1:] - E[:,:-1]
        ey = E[1:,:] - E[:-1,:]
        mism_x = np.abs(dx) - np.abs(ex)
        mism_y = np.abs(dy) - np.abs(ey)
        if self.use_sinusoid:
            sinus = np.sin(2.0*np.pi*self.freq*t_idx + self.phi)
        else:
            sinus = 0.0
        g_kappa = np.zeros_like(S)
        g_kappa[:, :-1] -= mism_x
        g_kappa[:,  1:] += mism_x
        g_kappa[:-1, :] -= mism_y
        g_kappa[ 1:, :] += mism_y
        env_mag_full = np.zeros_like(S)
        env_mag = np.hypot(ex, ey)
        env_mag_full[:, :-1] += env_mag
        env_mag_full[:,  1:] += env_mag
        g_A    = sinus * (env_mag_full - np.abs(_grad2d_x(S)) - np.abs(_grad2d_y(S)))
        g_f    = self.A * np.cos(2.0*np.pi*self.freq*t_idx + self.phi) * g_A
        g_phi  = self.A * np.cos(2.0*np.pi*self.freq*t_idx + self.phi)
        gc = self.grad_clip
        return (_clip(g_A,-gc,gc), _clip(g_f,-gc,gc), _clip(g_phi,-gc,gc), _clip(g_kappa,-gc,gc),
                mism_x, mism_y)

    # ----- Attractor gradient boost (bias) -----
    def _boost_grads_with_attractors_1d(self, gA, gf, gphi, gk):
        for k, w in self._masks():
            W = w[:-1]  # map cell weights -> edges (simple)
            c = 2.0 / max(1e-8, k.sigma_theta**2)
            G = np.exp(-(
                (self.A - k.psi_A)**2 +
                (self.freq - k.psi_f)**2 +
                (self.phi - k.psi_phi)**2 +
                (self.kappa - k.psi_kappa)**2
            ) / max(1e-8, k.sigma_theta**2))
            boost = k.amp * W * G
            gA   -= boost * c * (self.A   - k.psi_A)
            gf   -= boost * c * (self.freq- k.psi_f)
            gphi -= boost * c * (self.phi - k.psi_phi)
            gk   -= boost * c * (self.kappa - k.psi_kappa)
        return gA, gf, gphi, gk

    def _boost_grads_with_attractors_2d(self, gA, gf, gphi, gk):
        for k, w in self._masks():
            c = 2.0 / max(1e-8, k.sigma_theta**2)
            G = np.exp(-(
                (self.A - k.psi_A)**2 +
                (self.freq - k.psi_f)**2 +
                (self.phi - k.psi_phi)**2 +
                (self.kappa - k.psi_kappa)**2
            ) / max(1e-8, k.sigma_theta**2))
            boost = k.amp * w * G
            gA   -= boost * c * (self.A   - k.psi_A)
            gf   -= boost * c * (self.freq- k.psi_f)
            gphi -= boost * c * (self.phi - k.psi_phi)
            gk   -= boost * c * (self.kappa - k.psi_kappa)
        return gA, gf, gphi, gk

    # ----- energy gate & apply parameter steps -----
    def _apply_energy_gated(self, T_eff: float, dS_sum: float, dA, df, dphi, dk):
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
        return spent, dissip

    # ----- reward / selection / local propagation -----
    def _reward_and_select(self, reward_map: np.ndarray, energy_map: np.ndarray):
        if reward_map is None or energy_map is None or not self.attractors:
            return
        for k, w in self._masks():
            W = w / (1e-12 + np.sum(w))
            r = float(np.sum(W * reward_map))
            e = float(np.sum(W * energy_map))
            eff = r / (e + 1e-12)
            G = np.exp(-(
                (self.A - k.psi_A)**2 +
                (self.freq - k.psi_f)**2 +
                (self.phi - k.psi_phi)**2 +
                (self.kappa - k.psi_kappa)**2
            ) / max(1e-8, k.sigma_theta**2))
            integ = float(np.sum(W * G))
            Rk = eff * integ
            k.reward_avg = 0.9*k.reward_avg + 0.1*Rk
            k.amp += self.attr_eta_amp * k.amp * (Rk - k.reward_avg)

    def _propagate_if_strong(self):
        if not self.attractors:
            return
        if self.ndim == 1:
            X = self.shape[0]
            for k in list(self.attractors):
                if k.amp < 5.0 * self.attr_amp_min:
                    continue
                i = int(k.pos)
                for j in (i-1, i+1):
                    if 0 <= j < X and self.F[j] >= self.attr_prop_energy and self.rng.random() < 0.25:
                        self.F[j] -= self.attr_prop_energy
                        self.B[j] += self.work_to_dissipation_fraction * self.attr_prop_energy
                        psi = {
                            "A": np.clip(k.psi_A + self.rng.normal(0,0.02), *self.A_range),
                            "f": np.clip(k.psi_f + self.rng.normal(0,0.02), *self.f_range),
                            "phi": (k.psi_phi + self.rng.normal(0,0.1)) % (2*np.pi),
                            "kappa": np.clip(k.psi_kappa + self.rng.normal(0,0.01), *self.kappa_range),
                        }
                        self.attractors.append(Attractor(j, 1, self.attr_radius, self.attr_amp_init, psi,
                                                         self.attr_maint_rate, self.attr_decay,
                                                         self.attr_sigma_theta, self.attr_sigma_signal,
                                                         self.attr_spawn_energy, self.attr_prop_energy))
        else:
            H, W = self.shape
            for k in list(self.attractors):
                if k.amp < 5.0 * self.attr_amp_min:
                    continue
                y, x = k.pos
                nbrs = [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]
                for ny, nx in nbrs:
                    if 0 <= ny < H and 0 <= nx < W and self.F[ny,nx] >= self.attr_prop_energy and self.rng.random() < 0.25:
                        self.F[ny,nx] -= self.attr_prop_energy
                        self.B[ny,nx] += self.work_to_dissipation_fraction * self.attr_prop_energy
                        psi = {
                            "A": np.clip(k.psi_A + self.rng.normal(0,0.02), *self.A_range),
                            "f": np.clip(k.psi_f + self.rng.normal(0,0.02), *self.f_range),
                            "phi": (k.psi_phi + self.rng.normal(0,0.1)) % (2*np.pi),
                            "kappa": np.clip(k.psi_kappa + self.rng.normal(0,0.01), *self.kappa_range),
                        }
                        self.attractors.append(Attractor((ny,nx), 2, self.attr_radius, self.attr_amp_init, psi,
                                                         self.attr_maint_rate, self.attr_decay,
                                                         self.attr_sigma_theta, self.attr_sigma_signal,
                                                         self.attr_spawn_energy, self.attr_prop_energy))

# ---------------- Main stepping function (preserves UI/engine API) ----------------

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

    # Attractors: spawn + maintenance/decay
    st._maybe_spawn()
    st._maintain_and_decay()

    # --- Denoising / learning (with attractor bias) ---
    new_S = S.copy()
    T_eff = float(np.mean(st.T))

    energy_paid_map = None
    reward_map = None

    if st.ndim == 1:
        t_idx = rng.integers(0, 10**9)
        gA, gf, gphi, gk, delta, env_edge = st._propose_grads_1d(t_idx, S, E)
        gA, gf, gphi, gk = st._boost_grads_with_attractors_1d(gA, gf, gphi, gk)

        dA   = - st.eta * gA
        df   = - st.eta * gf
        dphi = - st.eta * gphi
        dk   = - st.eta * gk

        dS_pos_sum = float(np.sum(np.maximum(0.0, np.abs(delta) - np.abs(env_edge))))
        spent, dissip = st._apply_energy_gated(T_eff, dS_pos_sum, dA, df, dphi, dk)
        # record energy accounting for metrics
        st._last_spent = float(spent)
        st._last_dissip = float(dissip)

        if spent > 0.0:
            st.F -= spent / max(1, st.F.size)
        if dissip > 0.0:
            st.B += dissip / max(1, st.B.size)

        # κ‑weighted smoothing (divergence in 1D)
        kappa = np.clip(st.kappa, st.kappa_range[0], st.kappa_range[1]) * st.kappa_scale
        smooth = np.zeros_like(S)
        smooth[1:-1] = kappa[1:] * (S[2:] - S[1:-1]) - kappa[:-1] * (S[1:-1] - S[:-2])
        new_S = S + smooth

        energy_paid_map = np.full_like(S, spent / max(1, S.size))
        reward_map = np.full_like(S, dS_pos_sum / max(1, S.size))

    else:
        t_idx = rng.integers(0, 10**9)
        gA, gf, gphi, gk, mism_x, mism_y = st._propose_grads_2d(t_idx, S, E)
        gA, gf, gphi, gk = st._boost_grads_with_attractors_2d(gA, gf, gphi, gk)

        dA   = - st.eta * gA
        df   = - st.eta * gf
        dphi = - st.eta * gphi
        dk   = - st.eta * gk

        dS_pos_sum = float(np.sum(np.maximum(0.0, np.abs(mism_x))) + np.sum(np.maximum(0.0, np.abs(mism_y))))
        spent, dissip = st._apply_energy_gated(T_eff, dS_pos_sum, dA, df, dphi, dk)
        # record energy accounting for metrics
        st._last_spent = float(spent)
        st._last_dissip = float(dissip)

        if spent > 0.0:
            st.F -= spent / max(1, st.F.size)
        if dissip > 0.0:
            st.B += dissip / max(1, st.B.size)

        kappa = np.clip(st.kappa, st.kappa_range[0], st.kappa_range[1]) * st.kappa_scale
        Sx = _grad2d_x(S)
        Sy = _grad2d_y(S)
        px = kappa * Sx
        py = kappa * Sy
        new_S = S + _div2d(px, py)

        energy_paid_map = np.full_like(S, spent / max(1, S.size))
        reward_map = np.full_like(S, dS_pos_sum / max(1, S.size))

    # Ensure metrics attributes exist even if nothing spent this tick
    if not hasattr(st, "_last_spent"):
        st._last_spent = 0.0
        st._last_dissip = 0.0

    # Attractors: reward/selection + local propagation
    st._reward_and_select(reward_map, energy_paid_map)
    st._propagate_if_strong()

    # --- Legacy substrate dynamics (preserve UI look) ---
    cur = new_S
    if st.ndim == 1:
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
    else:
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
            cur[0,:]  *= (1.0 - st.boundary_leak)
            cur[-1,:] *= (1.0 - st.boundary_leak)
            cur[:,0]  *= (1.0 - st.boundary_leak)
            cur[:,-1] *= (1.0 - st.boundary_leak)
        flux_metric = float(np.mean(np.abs(pull)))

    if not np.all(np.isfinite(cur)):
        cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)
    return cur, flux_metric

# --------- Metrics getter (public) ---------
def get_fuka3_metrics():
    """
    Return a list of metrics dicts, one per active substrate shape/state.
    Each dict has: shape, dims, energy, temperature, entropy stats,
    attractor counts, and learning/efficiency proxies.
    """
    out = []
    for shape, st in list(_STATES.items()):
        try:
            F_total = float(np.sum(st.F))
            B_total = float(np.sum(st.B))
            T_mean  = float(np.mean(st.T))
            if st.ndim == 1:
                ent_field = np.abs(np.diff(st.kappa)) if st.kappa.size > 1 else np.zeros(1, dtype=float)
            else:
                ent_field = _box_var2d(st.kappa) if st.kappa.size > 1 else np.zeros_like(st.kappa)
            ent_mean = float(np.mean(ent_field)) if ent_field.size else 0.0
            ent_p95  = float(np.percentile(ent_field, 95)) if ent_field.size else 0.0
            attrs_alive = sum(1 for a in st.attractors if a.alive)
            avg_reward  = float(np.mean([a.reward_avg for a in st.attractors])) if st.attractors else 0.0
            last_spent  = float(getattr(st, "_last_spent", 0.0))
            last_diss   = float(getattr(st, "_last_dissip", 0.0))
            if st.ndim == 1:
                conn_alive = int(np.sum(np.abs(st.kappa) > 1e-6))
            else:
                conn_alive = int(np.sum(np.abs(st.kappa) > 1e-6))
            eff_ratio = (ent_mean / (last_spent + 1e-12)) if last_spent > 0 else 0.0
            out.append({
                "shape": tuple(shape),
                "ndim": int(st.ndim),
                "free_energy_total": F_total,
                "bound_energy_total": B_total,
                "temperature_mean": T_mean,
                "entropy_mean": ent_mean,
                "entropy_p95": ent_p95,
                "connections_alive": conn_alive,
                "attractors_alive": int(attrs_alive),
                "avg_reward": avg_reward,
                "work_paid_per_tick": last_spent,
                "dissipation_per_tick": last_diss,
                "efficiency_ratio": eff_ratio
            })
        except Exception:
            continue
    return out