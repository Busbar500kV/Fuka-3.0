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

# ===== Encoding-aware helpers (NEW) =====
def _normalize01(a: np.ndarray) -> np.ndarray:
    m = float(np.nanmin(a)); M = float(np.nanmax(a))
    if not np.isfinite(m) or not np.isfinite(M) or (M - m) < 1e-12:
        return np.zeros_like(a)
    return (a - m) / (M - m + 1e-12)

def _lerp(a: float, b: float, t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return (1.0 - t) * float(a) + t * float(b)

def _smooth2d(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None or prev.shape != cur.shape:
        return cur.copy()
    return (1.0 - alpha) * prev + alpha * cur

def _inv_var_encoding(field: np.ndarray) -> np.ndarray:
    # low local variance of |∇S|  => strong encoding (structure)
    Gx = _grad2d_x(field); Gy = _grad2d_y(field)
    Gmag = np.abs(Gx) + np.abs(Gy)
    V = _box_var2d(Gmag)
    Vn = _normalize01(V)
    enc = 1.0 - Vn
    return enc

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

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

        # ---------- legacy physics knobs (still used) ----------
        self.T_base = float(_safe_get(cfg_phys, "T", 0.001))
        self.flux_limit = float(_safe_get(cfg_phys, "flux_limit", 0.2))
        self.boundary_leak = float(_safe_get(cfg_phys, "boundary_leak", 0.0))

        # ---------- fuka3 config ----------
        f3 = cfg_f3 or {}

        # --- top-down stimulation (DishBrain-style) ---
        f3S = f3.get("stimulation", {})
        self.stim_enable         = bool(_safe_get(f3S, "enable", False))
        self.stim_gain           = float(_safe_get(f3S, "gain", 0.25))
        self.stim_freq_hz        = float(_safe_get(f3S, "freq", 0.15))
        self.stim_phase_jit      = float(_safe_get(f3S, "phase_jitter", 0.15))
        self.stim_freq_jit       = float(_safe_get(f3S, "freq_jitter", 0.10))
        self.stim_noise_gain     = float(_safe_get(f3S, "noise_gain", 0.35))
        self.stim_edge_kind      = str(_safe_get(f3S, "edge_kind", "both"))  # "kappa" | "gradS" | "both"
        self.stim_edge_thr       = float(_safe_get(f3S, "edge_thr", 0.85))
        self.stim_radius_boost   = float(_safe_get(f3S, "radius_boost", 1.0))
        self.stim_energy_per_unit= float(_safe_get(f3S, "energy_per_unit", 0.0))  # 0.0 = off

        # --- energy model ---
        f3E = f3.get("energy", {})
        self.source_rate  = float(_safe_get(f3E, "source_rate", 1.0))
        self.transport_conductance = float(_safe_get(f3E, "transport_conductance", 0.15))
        self.work_to_dissipation_fraction = float(_safe_get(f3E, "work_to_dissipation_fraction", 0.9))
        self.step_cost_coeff = float(_safe_get(f3E, "step_cost_coeff", 0.01))
        self.alpha_energy_per_entropy = float(_safe_get(f3E, "alpha_energy_per_entropy", 1.0))

        # --- temperature ---
        f3T = f3.get("temperature", {})
        self.T0 = float(_safe_get(f3T, "T0", self.T_base))
        self.beta_bound = float(_safe_get(f3T, "beta_bound", 0.01))
        self.beta_signal = float(_safe_get(f3T, "beta_signal", 0.0))

        # --- EMA (for encoding / entropy-drop) ---
        f3EMA = f3.get("ema", {})
        self.tau_fast = int(_safe_get(f3EMA, "tau_var_fast", 4))
        self.tau_slow = int(_safe_get(f3EMA, "tau_var_slow", 32))
        self._alpha_fast = 1.0 / max(1, self.tau_fast)
        self._alpha_slow = 1.0 / max(1, self.tau_slow)

        # --- connections / denoising knobs still used by learning path ---
        f3C  = f3.get("connection", {})
        f3Ci = f3C.get("init", {})
        f3Cl = f3C.get("learning", {})
        self.eta         = float(_safe_get(f3Cl, "eta", 0.05))
        self.entropy_window = int(_safe_get(f3Cl, "entropy_window", 16))
        # denoising
        den = f3.get("denoising", {})
        self.use_sinusoid = bool(_safe_get(den, "use_sinusoid_term", True))
        self.kappa_scale  = float(_safe_get(den, "kappa_scale", 1.0))
        self.grad_clip    = float(_safe_get(den, "gradient_clip", 1.0))

        # parameter ranges (used to initialize fields that the learner updates)
        self.A_range     = tuple(_safe_get(f3Ci, "amplitude_range", (0.05, 0.15)))
        self.f_range     = tuple(_safe_get(f3Ci, "frequency_range", (0.05, 0.25)))
        self.phi_range   = tuple(_safe_get(f3Ci, "phase_range", (0.0, 2*np.pi)))
        self.kappa_range = tuple(_safe_get(f3Ci, "curvature_range", (0.0, 0.1)))

        # --- attractors (DishBrain strategy) ---
        f3A = f3.get("attractors", {})
        self.attr_spawn_energy  = float(_safe_get(f3A, "spawn_energy", 0.05))
        self.attr_spawn_trials  = int(_safe_get(f3A, "spawn_trials", 32))   # max spawns per tick
        self.attr_max           = int(_safe_get(f3A, "max_count", 256))
        self.attr_amp_init      = float(_safe_get(f3A, "amplitude_init", 0.05))
        self.attr_amp_min       = float(_safe_get(f3A, "amplitude_min", 0.005))
        self.attr_amp_max       = float(_safe_get(f3A, "amplitude_max", 0.5))
        self.attr_maint_rate    = float(_safe_get(f3A, "maint_cost_rate", 0.001))
        self.attr_decay         = float(_safe_get(f3A, "decay_rate", 0.01))
        self.attr_bank_leak     = float(_safe_get(f3A, "bank_leak", 0.02))  # harmless to keep

        shapeA = f3A.get("shape", {})
        self.r_par_rng  = tuple(_safe_get(shapeA, "r_parallel_range", (2.0, 6.0)))
        self.r_perp_rng = tuple(_safe_get(shapeA, "r_perp_range", (1.0, 3.0)))

        # encoding-aware attractor guidance (kept minimal)
        enc = f3A.get("encoding", {})
        self.enc_beta              = float(_safe_get(enc, "beta", 0.10))  # smoothing for enc_map
        self.enc_radius_par_mult   = float(_safe_get(enc, "radius_par_mult", 0.80))
        self.enc_radius_perp_mult  = float(_safe_get(enc, "radius_perp_mult", 0.40))
        self.enc_amp_mult          = float(_safe_get(enc, "amp_mult", 0.50))

        # keep full f3 cfg for optional noise knobs read by _noise_from_encoding()
        self.f3_cfg = cfg_f3 or {}

        # ---------- fields & encoding EMAs ----------
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

            self._ent_fast = np.zeros((X,), dtype=float)
            self._ent_slow = np.zeros((X,), dtype=float)
            self.enc_map   = np.zeros((X,), dtype=float)  # alias still exists for parity
            self.enc_strength = self.enc_map
        else:
            H, W = shape
            self.F = np.full((H, W), 0.5, float)
            self.B = np.zeros((H, W), float)
            self.T = np.full((H, W), self.T0, float)

            self.A     = self.rng.uniform(*self.A_range, size=(H, W))
            self.freq  = self.rng.uniform(*self.f_range, size=(H, W))
            self.phi   = self.rng.uniform(*self.phi_range, size=(H, W))
            self.kappa = self.rng.uniform(*self.kappa_range, size=(H, W))

            self._ent_fast = np.zeros((H, W), dtype=float)
            self._ent_slow = np.zeros((H, W), dtype=float)
            self.enc_map   = np.zeros((H, W), dtype=float)
            self.enc_strength = self.enc_map

        # ---------- attractor store ----------
        self.attractors: List[Attractor] = []
        self._next_attr_id = 1

        # ---------- metrics cache ----------
        self._last_spent  = 0.0
        self._last_dissip = 0.0


    def _conn_strength_maps(self, S: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (enc_n, kabs_n, strength) all in [0,1], same shape as κ/S.
        strength := sqrt( enc_n * kabs_n ), with a mild |∇S| boost when S is given.
        """
        if self.ndim != 2:
            Z = np.zeros_like(self.kappa)
            return Z, Z, Z
    
        def _safe_norm(A: np.ndarray) -> np.ndarray:
            a = np.asarray(A, dtype=float)
            lo = float(np.min(a)); hi = float(np.max(a))
            if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
                return np.zeros_like(a)
            return (a - lo) / (hi - lo + 1e-12)
    
        enc = getattr(self, "enc_map", None)
        if enc is None or enc.shape != self.kappa.shape:
            enc = np.zeros_like(self.kappa)
    
        enc_n  = np.clip(_safe_norm(enc), 0.0, 1.0)
        kabs_n = np.clip(_safe_norm(np.abs(self.kappa)), 0.0, 1.0)
    
        strength = np.sqrt(enc_n * kabs_n)
    
        if S is not None and S.size == self.kappa.size:
            G = np.abs(_grad2d_x(S)) + np.abs(_grad2d_y(S))
            g_n = _safe_norm(G)
            strength = np.clip(0.85 * strength + 0.15 * g_n, 0.0, 1.0)
    
        return enc_n, kabs_n, strength
        
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

    def _entropy_field(self, S: np.ndarray) -> np.ndarray:
        if self.ndim == 1:
            return _rolling_var_1d(S, win=max(3, self.entropy_window))
        Gmag = np.abs(_grad2d_x(S)) + np.abs(_grad2d_y(S))
        return _box_var2d(Gmag)

    def _update_encoding_maps(self, S: np.ndarray):
        """EMA of entropy; encoding := positive (slow - fast), normalized to [0,1]."""
        ent = self._entropy_field(S)
        self._ent_fast = (1.0 - self._alpha_fast) * self._ent_fast + self._alpha_fast * ent
        self._ent_slow = (1.0 - self._alpha_slow) * self._ent_slow + self._alpha_slow * ent
        drop = np.maximum(self._ent_slow - self._ent_fast, 0.0)  # where entropy is dropping
    
        # robust normalize
        q05 = float(np.quantile(drop, 0.05)) if drop.size else 0.0
        q95 = float(np.quantile(drop, 0.95)) if drop.size else 1.0
        scale = max(1e-12, q95 - q05)
        enc = np.clip((drop - q05) / scale, 0.0, 1.0)
    
        # light smoothing to avoid flicker
        beta = float(np.clip(self.enc_beta, 0.0, 1.0))
        self.enc_map = (1.0 - beta) * getattr(self, "enc_map", enc) + beta * enc
    
        # alias for legacy readers
        self.enc_strength = self.enc_map
        
    ###
    def _maintain_and_decay(self):
        """
        Maintain/cull attractors with *explicit* connection support:
        - If local connection strength is below a sustain threshold,
          aggressively decay and remove quickly (no lingering “columns”).
        - Otherwise do normal maintenance (energy pay + slow decay).
        - Noise is re-tuned each tick from local strength.
        """
        alive: List[Attractor] = []
        if self.ndim != 2 or not self.attractors:
            self.attractors = alive
            return
    
        # Use the best available S for strength evaluation (post-step map if present)
        S_ref = getattr(self, "_last_S", None)
        enc_n, _, strength = self._conn_strength_maps(S_ref if S_ref is not None else None)
    
        # Spawn threshold (high) and sustain threshold (slightly lower)
        q_spawn = float(np.clip(getattr(self, "stim_edge_thr", 0.85), 0.0, 1.0))
        v_spawn = float(np.quantile(strength, q_spawn)) if strength.size else 1.0
    
        q_sustain = max(0.50, q_spawn - 0.15)  # e.g., 0.70 if spawn was 0.85
        v_sustain = float(np.quantile(strength, q_sustain)) if strength.size else (0.8 * v_spawn)
    
        for k in self.attractors:
            y, x = k.pos
            # local connection support value
            supp = float(strength[y, x]) if (0 <= y < strength.shape[0] and 0 <= x < strength.shape[1]) else 0.0
            has_support = (supp >= v_sustain)
    
            # Retune noise from encoding/strength (lower noise if stronger support)
            sig_theta, sig_signal = self._noise_from_encoding(y, x)
            k.sigma_theta  = float(sig_theta)
            k.sigma_signal = float(sig_signal)
    
            # Energy maintenance only when supported
            need = k.amp * k.maint_cost
            if has_support and (self.F[y, x] >= need):
                self.F[y, x] -= need
                self.B[y, x] += self.work_to_dissipation_fraction * need
                # small bank memory remains fine
                k.bank = max(0.0, (1.0 - self.attr_bank_leak) * k.bank + 0.25 * need)
                # normal decay
                k.amp *= (1.0 - self.attr_decay)
            else:
                # If no support (or not enough energy), decay hard
                # – this removes “unsupported columns” quickly.
                k.amp *= 0.5 * (1.0 - self.attr_decay)
    
            # Hard cull if far below sustain (prevents flicker)
            if (not has_support) and (supp < 0.5 * v_sustain):
                k.amp = 0.0  # drop immediately
    
            # Clamp + age
            k.amp = float(np.clip(k.amp, 0.0, self.attr_amp_max))
            k.age += 1
    
            # Keep if above minimum amplitude
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

    def _encoding_at(self, y: int, x: int) -> float:
        if self.ndim != 2:
            return 0.0
        H, W = self.shape
        if 0 <= y < H and 0 <= x < W:
            v = float(self.enc_map[y, x])   # or self.enc_strength
            return float(np.clip(v, 0.0, 1.0))
        return 0.0

    def _noise_from_encoding(self, y: int, x: int) -> Tuple[float, float]:
        """
        Map local *connection strength* (encoding × |kappa|, with mild |∇S| boost)
        -> (sigma_theta, sigma_signal). Higher strength => lower noise.
        Reads knobs from f3.attractors.noise (with safe defaults).
        """
        if self.ndim != 2:
            return 0.30, 0.30  # harmless defaults
    
        # build strength using last S (best available temporal reference)
        S_hint = getattr(self, "_last_S", None)
        _, _, strength = self._conn_strength_maps(S_hint)
        H, W = self.shape
        if 0 <= y < H and 0 <= x < W:
            s = float(np.clip(strength[y, x], 0.0, 1.0))
        else:
            s = 0.0
    
        noise_cfg = ((self.f3_cfg or {}).get("attractors", {}).get("noise", {})
                     if isinstance(self.f3_cfg, dict) else {})
    
        sig_sig_max = float(noise_cfg.get("sigma_signal_max", 0.60))
        sig_sig_min = float(noise_cfg.get("sigma_signal_min", 0.08))
        sig_th_max  = float(noise_cfg.get("sigma_theta_max",  0.60))
        sig_th_min  = float(noise_cfg.get("sigma_theta_min",  0.08))
        k_enc       = float(noise_cfg.get("encoding_gain",    1.0))
    
        eff = float(np.clip(s ** k_enc, 0.0, 1.0))
        sigma_signal = (1.0 - eff) * sig_sig_max + eff * sig_sig_min
        sigma_theta  = (1.0 - eff) * sig_th_max  + eff * sig_th_min
        return float(sigma_theta), float(sigma_signal)
        
    def _edge_mask(self, S: np.ndarray) -> np.ndarray:
        """
        Sparse 0/1 mask of likely 'connection edges' to gate where the
        top-down stimulation is injected. Combines |∇S| and |kappa|
        depending on stim_edge_kind: "kappa" | "gradS" | "both".
        """
        if self.ndim != 2 or S.size == 0:
            return np.zeros_like(S, dtype=float)
    
        def _safe_norm(A: np.ndarray) -> np.ndarray:
            a = np.asarray(A, dtype=float)
            lo = float(np.min(a)); hi = float(np.max(a))
            if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
                return np.zeros_like(a)
            return (a - lo) / (hi - lo + 1e-12)
    
        kind = getattr(self, "stim_edge_kind", "both")
        if kind == "gradS":
            G = np.abs(_grad2d_x(S)) + np.abs(_grad2d_y(S))
            g = _safe_norm(G)
        elif kind == "kappa":
            g = _safe_norm(np.abs(self.kappa))
        else:
            G = np.abs(_grad2d_x(S)) + np.abs(_grad2d_y(S))
            gG = _safe_norm(G)
            gK = _safe_norm(np.abs(self.kappa))
            g  = 0.5 * (gG + gK)
    
        thr = float(np.clip(getattr(self, "stim_edge_thr", 0.85), 0.0, 1.0))
        q = float(np.quantile(g, thr)) if g.size else 1.0
        return (g >= q).astype(float)

    def _stimulus_topdown(self, t_idx: int, S: np.ndarray) -> Optional[np.ndarray]:
        """
        Top-down attractor-driven stimulus (DishBrain-style).
        - Uses connection strength from _conn_strength_maps(S).
        - Projects along attractor orientation (local ∇S).
        - Gated by edge mask.
        - Applies 'gain' INSIDE this function (don't multiply again outside).
        """
        if (self.ndim != 2) or (not getattr(self, "stim_enable", False)) or (not self.attractors):
            return None
    
        H, W = S.shape
        edge_m = self._edge_mask(S)
        if not np.any(edge_m):
            return None
    
        # Connection strength (enc×|kappa| with mild ∇S boost)
        enc_n, _, strength = self._conn_strength_maps(S)
    
        # local edge orientation (from S)
        Sx = _grad2d_x(S); Sy = _grad2d_y(S)
    
        # structured carrier & gains
        w0   = 2.0 * np.pi * float(getattr(self, "stim_freq_hz", 0.15))
        gain = float(getattr(self, "stim_gain", 0.25))
        n_g  = float(getattr(self, "stim_noise_gain", 0.35))
    
        drive = np.zeros_like(S)
        for a in self.attractors:
            y0, x0 = a.pos
            Wk = a.field(H, W)  # normalized * amp
    
            # optional slight widening of footprint
            if self.stim_radius_boost > 1.01:
                G = np.abs(_grad2d_x(Wk)) + np.abs(_grad2d_y(Wk))
                Wk = np.clip(Wk + (self.stim_radius_boost - 1.0) * G, 0.0, None)
    
            # encoding at attractor center (reduces jitter where well-encoded)
            enc_loc = float(enc_n[y0, x0]) if (0 <= y0 < H and 0 <= x0 < W) else 0.0
            f_jit  = float(getattr(self, "stim_freq_jit", 0.10))  * (1.0 - enc_loc) * self.rng.normal(0.0, 1.0)
            ph_jit = float(getattr(self, "stim_phase_jit", 0.15)) * (1.0 - enc_loc) * self.rng.normal(0.0, 1.0)
    
            carrier = np.sin(w0 * (1.0 + f_jit) * float(t_idx) + ph_jit)
    
            # project along attractor orientation to truly act "on edges"
            dx = np.cos(a.theta); dy = np.sin(a.theta)
            edge_proj = dx * Sx + dy * Sy  # signed projection
    
            # structure + noise, weighted by strength and footprint
            d_struct = carrier * edge_proj * strength * Wk
            d_noise  = (1.0 - strength) * Wk * self.rng.standard_normal(size=(H, W))
    
            drive += gain * d_struct + n_g * d_noise
    
        # inject only on selected edges
        drive *= edge_m
    
        # optional energy accounting
        epu = float(getattr(self, "stim_energy_per_unit", 0.0))
        if epu > 0.0:
            E_units = float(np.mean(np.abs(drive)))  # proxy
            spent   = epu * E_units
            dissip  = self.work_to_dissipation_fraction * spent
            if spent > 0.0:
                self.F -= spent / max(1, self.F.size)
            if dissip > 0.0:
                self.B += dissip / max(1, self.B.size)
            self._last_spent  += spent
            self._last_dissip += dissip
    
        return drive

    ##
    def _spawn_dishbrain(self, S: np.ndarray, E: np.ndarray):
        """
        Deterministic spawn: place attractors ONLY on strong-connection ridges.
    
        Gate = connection strength (sqrt(enc × |kappa|) with mild ∇S boost).
        - Threshold uses the same percentile knob as edges (stim_edge_thr).
        - Non-maximum suppression to avoid overlap.
        - Theta from local ∇S (no random θ).
        - Radii/amp scale with local encoding; noise from connection strength.
        - Energy-gated; no random propagation.
        """
        if self.ndim != 2 or len(self.attractors) >= self.attr_max or S.size == 0:
            return
    
        H, W = self.shape
    
        # strength := sqrt(enc × |kappa|) with mild ∇S boost
        enc_n, _, strength = self._conn_strength_maps(S)
        if strength.size == 0:
            return
    
        # Spawn threshold = high-quantile of strength (uses "edge" knob)
        thr_q = float(np.clip(getattr(self, "stim_edge_thr", 0.85), 0.0, 1.0))
        thr_v = float(np.quantile(strength, thr_q))
        cand_mask = (strength >= thr_v)
        if not np.any(cand_mask):
            return
    
        # Non-maximum suppression over a tight neighborhood
        r_nms = 2
        peaks_yx = []
        visited = np.zeros_like(cand_mask, dtype=bool)
        ys, xs = np.where(cand_mask)
        order = np.argsort(-strength[ys, xs])  # high -> low
        for idx in order:
            y0, x0 = int(ys[idx]), int(xs[idx])
            if visited[y0, x0]:
                continue
            yL = max(0, y0 - r_nms); yH = min(H, y0 + r_nms + 1)
            xL = max(0, x0 - r_nms); xH = min(W, x0 + r_nms + 1)
            patch = strength[yL:yH, xL:xH]
            iy, ix = np.unravel_index(int(np.argmax(patch)), patch.shape)
            yp, xp = yL + int(iy), xL + int(ix)
            visited[yL:yH, xL:xH] = True
            peaks_yx.append((int(yp), int(xp)))
    
        if not peaks_yx:
            return
    
        # Select at most K per tick and enforce spacing from existing attractors
        max_per_tick = int(max(1, min(self.attr_spawn_trials, self.attr_max - len(self.attractors))))
        min_dist = 3.0  # pixels
        exist = [(int(a.pos[0]), int(a.pos[1])) for a in self.attractors]
    
        def _far_from_existing(y, x) -> bool:
            for (yy, xx) in exist:
                if (yy - y)**2 + (xx - x)**2 < (min_dist * min_dist):
                    return False
            return True
    
        spawned = 0
        Sx = _grad2d_x(S); Sy = _grad2d_y(S)
    
        for (y, x) in peaks_yx:
            if spawned >= max_per_tick or len(self.attractors) >= self.attr_max:
                break
            if not _far_from_existing(y, x):
                continue
    
            # Energy gate
            if self.F[y, x] < self.attr_spawn_energy:
                continue
    
            # Orientation from local gradient
            gx = float(Sx[y, x]); gy = float(Sy[y, x])
            theta = float(np.arctan2(gy, gx)) if (gx*gx + gy*gy) > 1e-12 else 0.0
    
            # Encoding-driven radii & amplitude (no randomness)
            enc_loc = float(enc_n[y, x])
            r_par  = float(np.mean(self.r_par_rng)  * (1.0 + self.enc_radius_par_mult  * enc_loc))
            r_perp = float(np.mean(self.r_perp_rng) * (1.0 + self.enc_radius_perp_mult * enc_loc))
            amp    = float(self.attr_amp_init      * (1.0 + self.enc_amp_mult         * enc_loc))
    
            # Noise tuned by local connection strength (stronger support ⇒ lower noise)
            sig_theta, sig_signal = self._noise_from_encoding(y, x)
    
            # Pay energy & spawn
            self.F[y, x] -= self.attr_spawn_energy
            self.B[y, x] += self.work_to_dissipation_fraction * self.attr_spawn_energy
    
            k = Attractor(
                self._next_attr_id, (y, x),
                r_par=r_par, r_perp=r_perp, theta=theta,
                amp=amp, maint_cost=self.attr_maint_rate, decay=self.attr_decay,
                sigma_theta=float(sig_theta), sigma_signal=float(sig_signal),
            )
            self._next_attr_id += 1
            self.attractors.append(k)
            exist.append((y, x))
            spawned += 1


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

    global _GLOBAL_TICK
    t_idx = _GLOBAL_TICK
    _GLOBAL_TICK += 1

    Sn = S / (1e-12 + float(np.max(np.abs(S)))) if np.any(S) else np.zeros_like(S)

    # Energy: inject & transport
    st._inject_sources(E)
    if st.ndim == 1:
        st._transport_F_1d(st.F)
    else:
        st._transport_F_2d(st.F)

    # Temperature + encoding maps
    signal_var = _rolling_var_1d(S, win=max(3, st.entropy_window)) if st.ndim == 1 else _box_var2d(S)
    st._update_temperature(signal_var)
    if st.ndim == 2:
        st._update_encoding_maps(S)

    # Attractors lifecycle (deterministic spawns + maintenance)
    if st.ndim == 2:
        try:
            st._spawn_dishbrain(S, E)      # <— NEW deterministic spawner
        except Exception:
            pass
        try:
            st._maintain_and_decay()
        except Exception:
            pass

    # Denoising / learning (no attractor gradient booster)
    new_S = S.copy()
    T_eff = float(np.mean(st.T))

    if st.ndim == 1:
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

    # 2D learning grads (no booster)
    gA, gf, gphi, gk, mism_x, mism_y = st._propose_grads_2d(t_idx, S, E)

    dA   = - st.eta * gA
    df   = - st.eta * gf
    dphi = - st.eta * gphi
    dk   = - st.eta * gk

    dS_pos_sum = float(np.sum(np.maximum(0.0, np.abs(mism_x))) + np.sum(np.maximum(0.0, np.abs(mism_y))))
    energy_paid_map, reward_map = st._apply_energy_gated(T_eff, dS_pos_sum, dA, df, dphi, dk, S)

    # κ-weighted smoothing
    kappa = np.clip(st.kappa, st.kappa_range[0], st.kappa_range[1]) * st.kappa_scale
    Sx = _grad2d_x(S); Sy = _grad2d_y(S)
    px = kappa * Sx;   py = kappa * Sy
    new_S = S + _div2d(px, py)

    # Top-down stimulus: the only attractor influence
    if getattr(st, "stim_enable", False):
        stim = st._stimulus_topdown(t_idx, new_S)
        if stim is not None:
            new_S = new_S + stim

    # (No selection/propagation stage; keep it simple and clean)

    # Legacy substrate dynamics for parity
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

    st._last_S = cur
    st._last_E = E
    return cur, flux_metric

# ============================================================
# Metrics (public)
# ============================================================
def get_fuka3_metrics():
    """
    Per-active substrate metrics (shape-scoped LocalState).
    Reports:
      - free/bound energy totals
      - mean temperature
      - S-based entropy (mean & p95)
      - κ diagnostic entropy (mean)
      - connections_alive (non-zero κ count)
      - attractors_alive (count)
      - energy/dissipation paid in last tick
      - efficiency ratio (entropy_mean / work_paid_per_tick)
    """
    out = []
    for shape, st in list(_STATES.items()):
        try:
            F_total = float(np.sum(st.F))
            B_total = float(np.sum(st.B))
            T_mean  = float(np.mean(st.T))

            # --- S-based entropy field (depends on dimensionality) ---
            ent_mean_S = 0.0
            ent_p95_S  = 0.0
            if hasattr(st, "_last_S"):
                if st.ndim == 1:
                    ent_field_S = _rolling_var_1d(st._last_S, win=max(3, st.entropy_window))
                else:
                    Gmag = np.abs(_grad2d_x(st._last_S)) + np.abs(_grad2d_y(st._last_S))
                    ent_field_S = _box_var2d(Gmag)
                if ent_field_S.size:
                    ent_mean_S = float(np.mean(ent_field_S))
                    ent_p95_S  = float(np.percentile(ent_field_S, 95))

            # --- κ-based diagnostic entropy proxy ---
            if st.ndim == 1:
                ent_field_k = np.abs(np.diff(st.kappa)) if st.kappa.size > 1 else np.zeros(1, dtype=float)
                ent_mean_k  = float(np.mean(ent_field_k))
            else:
                ent_field_k = _box_var2d(st.kappa) if st.kappa.size > 1 else np.zeros_like(st.kappa)
                ent_mean_k  = float(np.mean(ent_field_k))

            attrs_alive = len(getattr(st, "attractors", []))
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
                "entropy_mean": ent_mean_S,
                "entropy_p95": ent_p95_S,
                "entropy_mean_kappa": ent_mean_k,
                "connections_alive": conn_alive,
                "attractors_alive": int(attrs_alive),
                "work_paid_per_tick": last_spent,
                "dissipation_per_tick": last_diss,
                "efficiency_ratio": eff_ratio,
            })
        except Exception:
            # be robust to any single-state hiccup
            continue
    return out

    
# ------------------------------------------------------------
# Attractors snapshot for UI (used by app.py overlay)
# ------------------------------------------------------------
def get_attractors_snapshot() -> List[Dict[str, Any]]:
    snaps: List[Dict[str, Any]] = []
    for shape, st in list(_STATES.items()):
        try:
            entry: Dict[str, Any] = {"shape": tuple(shape), "items": []}
            if getattr(st, "ndim", 0) == 2 and hasattr(st, "attractors"):
                H, W = shape
                for a in st.attractors:
                    y, x = getattr(a, "pos", (0, 0))
                    enc_local = float(st._encoding_at(y, x)) if hasattr(st, "_encoding_at") else 0.0
                    entry["items"].append({
                        "id": int(getattr(a, "id", -1)),
                        "pos": (int(y), int(x)),
                        "theta": float(getattr(a, "theta", 0.0)),
                        "r_par": float(getattr(a, "r_par", 1.0)),
                        "r_perp": float(getattr(a, "r_perp", 1.0)),
                        "amp": float(getattr(a, "amp", 0.0)),
                        "age": int(getattr(a, "age", 0)),
                        "alive": bool(getattr(a, "alive", True)),
                        # NEW for UI
                        "sigma_signal": float(getattr(a, "sigma_signal", 0.0)),
                        "encoding": enc_local,
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