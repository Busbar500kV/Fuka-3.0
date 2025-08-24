# core/physics.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

# ============================================================
# Back-compat helpers used by the Engine
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

def _corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size < 2 or b.size < 2:
        return 0.0
    sa = np.std(a); sb = np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.dot(a - np.mean(a), b - np.mean(b)) / (a.size * sa * sb))

def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))

# ============================================================
# Attractors (Option-B): anisotropic Gaussian blobs
# ============================================================

class Attractor:
    __slots__ = (
        "id","pos","r_par","r_perp","theta","amp","bank","gain",
        "maint_cost","decay","sigma_theta","sigma_signal",
        "alive","age","reward_avg"
    )

    def __init__(self, id_: int, pos: Tuple[int,int],
                 r_par: float, r_perp: float, theta: float,
                 amp: float, maint_cost: float, decay: float,
                 sigma_theta: float, sigma_signal: float):
        self.id = int(id_)
        self.pos = (int(pos[0]), int(pos[1]))
        self.r_par = float(max(1e-3, r_par))
        self.r_perp = float(max(1e-3, r_perp))
        self.theta = float(theta)
        self.amp = float(amp)
        self.bank = 0.0
        self.gain = 1.0
        self.maint_cost = float(maint_cost)
        self.decay = float(decay)
        self.sigma_theta = float(sigma_theta)
        self.sigma_signal = float(sigma_signal)
        self.alive = True
        self.age = 0
        self.reward_avg = 0.0

    def _phi(self, H: int, W: int) -> np.ndarray:
        """Anisotropic Gaussian footprint normalized to sum 1."""
        y0, x0 = self.pos
        y, x = np.indices((H, W))
        c, s = np.cos(self.theta), np.sin(self.theta)
        xr = (x - x0) * c + (y - y0) * s
        yr = -(x - x0) * s + (y - y0) * c
        val = np.exp(-0.5 * ((xr / self.r_par) ** 2 + (yr / self.r_perp) ** 2))
        ssum = float(val.sum())
        if ssum <= 1e-12:
            return np.zeros((H, W), float)
        return val / ssum

    def field(self, H: int, W: int) -> np.ndarray:
        return self.amp * self._phi(H, W) * self.gain
        
# ============================================================
# Local State (per substrate grid)
# ============================================================

_STATES: Dict[Tuple[int, ...], "LocalState"] = {}

class LocalState:
    def __init__(self, shape: Tuple[int, ...],
                 cfg_phys: Dict[str, Any],
                 cfg_f3: Optional[Dict[str, Any]],
                 rng: np.random.Generator):
        self.shape = shape
        self.ndim = len(shape)
        self.rng = rng

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

        # fuka3: attractors (Option-B)
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
        self.attr_max = int(_safe_get(f3A, "max_count", 256))
        self.attr_spawn_trials = int(_safe_get(f3A, "spawn_trials", 32))

        # fields
        if self.ndim == 1:
            X = shape[0]
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

        # attractor list
        self.attractors: List[Attractor] = []
        self._last_spent = 0.0
        self._last_dissip = 0.0
        
        
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
        if len(self.attractors) >= self.attr_max:
            return
        trials = max(1, min(self.attr_spawn_trials, self.attr_max - len(self.attractors)))
        if self.ndim == 1:
            X = self.shape[0]
            for _ in range(trials):
                i = int(self.rng.integers(0, X))
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
                    if len(self.attractors) >= self.attr_max:
                        break
        else:
            H, W = self.shape
            for _ in range(trials):
                y = int(self.rng.integers(0, H))
                x = int(self.rng.integers(0, W))
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
                    if len(self.attractors) >= self.attr_max:
                        break

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

    # (other methods unchanged: _masks, _propose_grads_1d/_2d, _boost_grads_with_attractors,
    # _apply_energy_gated, _reward_and_select, _propagate_if_strong — as in your file)

# ============================================================
# Stepping and metrics
# ============================================================

def step_physics(prev_S: np.ndarray,
                 env_row: np.ndarray,
                 k_flux: float,
                 k_motor: float,
                 diffuse: float,
                 decay: float,
                 rng: np.random.Generator,
                 band: int = 3,
                 bc: str = "reflect",
                 **phys_kwargs) -> Tuple[np.ndarray, float]:
    # (same as your current file, calling into LocalState)
    # ...

    if not np.all(np.isfinite(cur)):
        cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)
    return cur, flux_metric

def get_fuka3_metrics():
    out = []
    for shape, st in list(_STATES.items()):
        try:
            F_total = float(np.sum(st.F))
            B_total = float(np.sum(st.B))
            T_mean  = float(np.mean(st.T))
            if st.ndim == 1:
                ent_field = np.abs(np.diff(st.kappa)) if st.kappa.size > 1 else np.zeros(1)
            else:
                ent_field = _box_var2d(st.kappa) if st.kappa.size > 1 else np.zeros_like(st.kappa)
            ent_mean = float(np.mean(ent_field)) if ent_field.size else 0.0
            ent_p95  = float(np.percentile(ent_field, 95)) if ent_field.size else 0.0
            attrs_alive = sum(1 for a in st.attractors if a.alive)
            avg_reward  = float(np.mean([a.reward_avg for a in st.attractors])) if st.attractors else 0.0
            last_spent  = float(getattr(st, "_last_spent", 0.0))
            last_diss   = float(getattr(st, "_last_dissip", 0.0))
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