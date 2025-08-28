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
    now = time.perf_counter()
    if _last_emit_t is None:
        _last_emit_t = now
    dt = now - _last_emit_t
    _last_emit_t = now

    rows: List[Dict[str, Any]] = []
    for m in phys.get_fuka3_metrics():
        r = dict(m)
        r["dt_since_last_collect_s"] = dt
        rows.append(r)
    return rows

def format_for_log(rows: List[Dict[str, Any]]) -> str:
    """
    Pretty one-liner per substrate for console logs.
    """
    out_lines: List[str] = []
    for r in rows:
        shape = r.get("shape", ())
        ndim = r.get("ndim", 0)

        F = float(r.get("free_energy_total", 0.0))
        B = float(r.get("bound_energy_total", 0.0))
        Tm = float(r.get("temperature_mean", 0.0))

        Smean = float(r.get("entropy_mean", 0.0))
        Sp95  = float(r.get("entropy_p95", 0.0))

        conn = int(r.get("connections_alive", 0))
        attr = int(r.get("attractors_alive", 0))

        work = float(r.get("work_paid_per_tick", 0.0))
        diss = float(r.get("dissipation_per_tick", 0.0))
        eta  = float(r.get("efficiency_ratio", 0.0))

        line = (
            f"[shape={shape}, ndim={ndim}] "
            f"F={F:.3f} "
            f"B={B:.3f} "
            f"T̄={Tm:.4f} "
            f"S̄={Smean:.4e} "
            f"S95={Sp95:.4e} "
            f"conn={conn} "
            f"attr={attr} "
            f"work={work:.4e} "
            f"diss={diss:.4e} "
            f"η_eff={eta:.4e}"
        )
        out_lines.append(line)
    return "\n".join(out_lines)