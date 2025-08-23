# core/organism.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class History:
    # time axis
    t: List[int] = field(default_factory=list)

    # existing metrics used by the UI
    E_cell: List[float] = field(default_factory=list)   # mean substrate energy
    E_env:  List[float] = field(default_factory=list)   # mean environment energy
    E_flux: List[float] = field(default_factory=list)   # mean |env pull actually applied|

    # NEW: deeper physics telemetry (optional to use in UI)
    entropy:    List[float] = field(default_factory=list)  # Shannon entropy of substrate distribution
    variance:   List[float] = field(default_factory=list)  # variance of substrate values
    total_mass: List[float] = field(default_factory=list)  # sum of substrate values (mass/energy proxy)