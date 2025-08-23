# core/metrics.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import time

from . import physics as phys

_last_emit_t: Optional[float] = None

def collect() -> List[Dict[str, Any]]:
    """
    Snapshot of Fuka3 metrics from the physics layer.
    Returns a list of dicts (one per active substrate state).
    Safe to call every frame.
    """
    global _last_emit_t
    now = time.time()
    if _last_emit_t is None:
        _last_emit_t = now
    dt = now - _last_emit_t
    _last_emit_t = now

    rows = []
    for m in phys.get_fuka3_metrics():
        m = dict(m)
        m["dt_since_last_collect_s"] = dt
        rows.append(m)
    return rows

def format_for_log(rows: List[Dict[str, Any]]) -> str:
    """
    Pretty one‑liner per substrate for console logs.
    """
    out_lines = []
    for r in rows:
        line = (
            f"[shape={r['shape']}, ndim={r['ndim']}] "
            f"F={r['free_energy_total']:.3f} "
            f"B={r['bound_energy_total']:.3f} "
            f"T̄={r['temperature_mean']:.4f} "
            f"S̄={r['entropy_mean']:.4e} "
            f"S95={r['entropy_p95']:.4e} "
            f"conn={r['connections_alive']} "
            f"attr={r['attractors_alive']} "
            f"work={r['work_paid_per_tick']:.4e} "
            f"diss={r['dissipation_per_tick']:.4e} "
            f"η_eff={r['efficiency_ratio']:.4e}"
        )
        out_lines.append(line)
    return "\n".join(out_lines)