from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .kinematics import KinematicsModel
from .simulators.base import SimulatorAdapter


@dataclass
class FkValidationStats:
    per_link_mean: Dict[str, float]
    per_link_max: Dict[str, float]
    per_link_std: Dict[str, float]

    def summary_str(self) -> str:
        lines = ["FK validation stats (position error, meters):"]
        for ln in sorted(self.per_link_mean.keys()):
            lines.append(
                f"  {ln:20s} mean={self.per_link_mean[ln]:.4e}, "
                f"max={self.per_link_max[ln]:.4e}, std={self.per_link_std[ln]:.4e}"
            )
        return "\n".join(lines)


def validate_fk_against_sim(
    kin_model: KinematicsModel,
    sim_adapter: SimulatorAdapter,
    end_effectors: List[str],
    q_samples: np.ndarray,
    frame: str = "world",
) -> FkValidationStats:
    """
    Confronta FK interna (Pinocchio) e FK del simulatore su una serie di configurazioni.

    Parameters
    ----------
    kin_model : KinematicsModel
    sim_adapter : SimulatorAdapter
    end_effectors : list[str]
    q_samples : np.ndarray
        Array di shape (N, dof) con N configurazioni di test.
    frame : str
        "world" o "root" (coerente con KinematicsModel.fk e adapter del simulatore).

    Returns
    -------
    FkValidationStats
        Statistiche di errore posizione per ogni end-effector.
    """
    q_samples = np.asarray(q_samples, dtype=float)
    if q_samples.ndim != 2:
        raise ValueError("q_samples deve avere shape (N, dof).")

    errors: Dict[str, List[float]] = {ln: [] for ln in end_effectors}

    for q in q_samples:
        fk_internal = kin_model.fk(q, end_effectors, frame=frame)
        sim_adapter.set_q(q)
        fk_sim = sim_adapter.get_link_poses(end_effectors, frame=frame)

        for ln in end_effectors:
            p_int, _ = fk_internal[ln]
            p_sim, _ = fk_sim[ln]
            err = float(np.linalg.norm(np.asarray(p_int) - np.asarray(p_sim)))
            errors[ln].append(err)

    per_link_mean: Dict[str, float] = {}
    per_link_max: Dict[str, float] = {}
    per_link_std: Dict[str, float] = {}

    for ln, vals in errors.items():
        arr = np.asarray(vals, dtype=float)
        per_link_mean[ln] = float(arr.mean())
        per_link_max[ln] = float(arr.max())
        per_link_std[ln] = float(arr.std())

    return FkValidationStats(
        per_link_mean=per_link_mean,
        per_link_max=per_link_max,
        per_link_std=per_link_std,
    )
