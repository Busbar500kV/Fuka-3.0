# core/organism.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class History:
    # time axis
    t: List[int] = field(default_factory=list)

    # legacy UI metrics (kept)
    E_cell: List[float] = field(default_factory=list)   # mean substrate energy
    E_env:  List[float] = field(default_factory=list)   # mean environment energy
    E_flux: List[float] = field(default_factory=list)   # mean |env pull applied|

    # optional: align with get_fuka3_metrics()
    free_energy_total:   List[float] = field(default_factory=list)
    bound_energy_total:  List[float] = field(default_factory=list)
    temperature_mean:    List[float] = field(default_factory=list)

    entropy_mean:        List[float] = field(default_factory=list)  # S-based
    entropy_p95:         List[float] = field(default_factory=list)
    entropy_mean_kappa:  List[float] = field(default_factory=list)

    connections_alive:   List[int]   = field(default_factory=list)
    attractors_alive:    List[int]   = field(default_factory=list)
    avg_reward:          List[float] = field(default_factory=list)
    work_paid_per_tick:  List[float] = field(default_factory=list)
    dissipation_per_tick:List[float] = field(default_factory=list)
    efficiency_ratio:    List[float] = field(default_factory=list)

    # legacy aggregate summaries if you still use them elsewhere
    entropy:    List[float] = field(default_factory=list)  # keep if your UI expects it
    variance:   List[float] = field(default_factory=list)
    total_mass: List[float] = field(default_factory=list)