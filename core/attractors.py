## core\attractors.py
from __future__ import annotations
import numpy as np
from typing import Tuple

class Attractor:
    """
    Option‑B attractor: anisotropic Gaussian blob with orientation.
    Acts as a spatial prior/bias over local parameter updates.
    """
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
        self.pos = (int(pos[0]), int(pos[1]))   # (y, x)
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

    def phi(self, H: int, W: int) -> np.ndarray:
        """Unit‑sum anisotropic Gaussian footprint centered at self.pos."""
        y0, x0 = self.pos
        y, x = np.indices((H, W))
        c, s = np.cos(self.theta), np.sin(self.theta)
        xr = (x - x0) * c + (y - y0) * s
        yr = -(x - x0) * s + (y - y0) * c
        val = np.exp(-0.5 * ((xr / self.r_par) ** 2 + (yr / self.r_perp) ** 2))
        ssum = float(val.sum())
        return val / ssum if ssum > 1e-12 else np.zeros((H, W), float)

    def field(self, H: int, W: int) -> np.ndarray:
        """Scaled footprint (for visualization/weighting)."""
        return self.amp * self.phi(H, W) * self.gain