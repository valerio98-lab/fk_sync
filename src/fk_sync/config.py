from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import yaml


@dataclass
class SkeletonConfig:
    name: str
    fps: int
    root_fields: List[str]
    quaternion_order: List[str]
    joint_order: List[str]


def load_skeleton_configs(path: str | Path) -> Dict[str, SkeletonConfig]:
    """
    Carica un file YAML di configurazione degli scheletri.

    Esempio struttura YAML:

        g1:
          fps: 30
          root_fields: [X, Y, Z, QX, QY, QZ, QW]
          quaternion_order: [QX, QY, QZ, QW]
          joint_order:
            - left_hip_pitch_joint
            - ...

    Restituisce un dict { "g1": SkeletonConfig(...), ... }.
    """
    path = Path(path)
    with path.open("r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    configs: Dict[str, SkeletonConfig] = {}
    for name, cfg in raw.items():
        configs[name] = SkeletonConfig(
            name=name,
            fps=int(cfg["fps"]),
            root_fields=list(cfg["root_fields"]),
            quaternion_order=list(cfg["quaternion_order"]),
            joint_order=list(cfg["joint_order"]),
        )
    return configs
