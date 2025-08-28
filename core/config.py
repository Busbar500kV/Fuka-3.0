# core/config.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

@dataclass
class FieldCfg:
    length: int = 64
    height: int = 1
    frames: int = 2000
    noise_sigma: float = 0.0
    sources: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FieldCfg":
        return FieldCfg(
            length=int(d.get("length", 64)),
            height=int(d.get("height", d.get("H", 1))),
            frames=int(d.get("frames", 2000)),
            noise_sigma=float(d.get("noise_sigma", 0.0)),
            sources=list(d.get("sources", [])),
        )