from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from fk_sync.adapters.base import SimulatorAdapter
from fk_sync.kinematics import KinematicsModel


def _quat_wxyz_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def _quat_angle_error_rad(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    """
    Smallest angle between orientations (sign-invariant).
    angle = 2 * acos(|dot(q1, q2)|)
    """
    q1 = _quat_wxyz_normalize(q1_wxyz)
    q2 = _quat_wxyz_normalize(q2_wxyz)
    dot = float(np.abs(np.dot(q1, q2)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return 2.0 * float(np.arccos(dot))


@dataclass
class LinkErrorStats:
    n: int
    mean_pos_err: float
    max_pos_err: float
    mean_rot_err_rad: Optional[float] = None
    max_rot_err_rad: Optional[float] = None


@dataclass
class FKValidationReport:
    per_link: Dict[str, LinkErrorStats]
    n_samples: int

    global_mean_pos_err: float
    global_max_pos_err: float
    global_mean_rot_err_rad: Optional[float] = None
    global_max_rot_err_rad: Optional[float] = None

    def pretty(self) -> str:
        lines: List[str] = []
        lines.append(f"FKValidationReport(n_samples={self.n_samples})")
        lines.append(
            f"  global pos err: mean={self.global_mean_pos_err:.6e} m, max={self.global_max_pos_err:.6e} m"
        )
        if (
            self.global_mean_rot_err_rad is not None
            and self.global_max_rot_err_rad is not None
        ):
            lines.append(
                f"  global rot err: mean={self.global_mean_rot_err_rad:.6e} rad, max={self.global_max_rot_err_rad:.6e} rad"
            )
        lines.append("  per-link:")
        for ln, st in self.per_link.items():
            s = f"    - {ln}: pos(mean={st.mean_pos_err:.6e}, max={st.max_pos_err:.6e})"
            if st.mean_rot_err_rad is not None and st.max_rot_err_rad is not None:
                s += f", rot(mean={st.mean_rot_err_rad:.6e} rad, max={st.max_rot_err_rad:.6e} rad)"
            lines.append(s)
        return "\n".join(lines)


def validate_fk_against_sim(
    kin_model: KinematicsModel,
    simulator: SimulatorAdapter,
    q_samples: np.ndarray,  # (N, dof)  oppure (N, model.nq) se già full
    link_names: List[str],
) -> FKValidationReport:
    q_samples = np.asarray(q_samples, dtype=float)
    if q_samples.ndim != 2:
        raise ValueError(
            f"q_samples deve essere 2D (N, dof). Got shape={q_samples.shape}"
        )

    n = q_samples.shape[0]
    pos_errs: Dict[str, List[float]] = {ln: [] for ln in link_names}
    rot_errs: Dict[str, List[float]] = {ln: [] for ln in link_names}

    # loop campioni
    for i in range(n):
        q = q_samples[i]

        internal = kin_model.fk(q, link_names, frame="world")  # {ln: (pos, quat_wxyz)}
        sim = simulator.get_link_poses(q, link_names)  # {ln: (pos, quat_wxyz|None)}

        for ln in link_names:
            p_int, q_int = internal[ln]
            p_sim, q_sim = sim[ln]

            dp = np.linalg.norm(np.asarray(p_int) - np.asarray(p_sim))
            pos_errs[ln].append(float(dp))

            # orient: solo se entrambi disponibili
            if q_int is not None and q_sim is not None:
                ang = _quat_angle_error_rad(np.asarray(q_int), np.asarray(q_sim))
                rot_errs[ln].append(float(ang))

    per_link: Dict[str, LinkErrorStats] = {}
    all_pos = []
    all_rot = []

    for ln in link_names:
        pe = np.array(pos_errs[ln], dtype=float)
        all_pos.append(pe)

        re_list = rot_errs[ln]
        if len(re_list) > 0:
            re = np.array(re_list, dtype=float)
            all_rot.append(re)
            per_link[ln] = LinkErrorStats(
                n=len(pe),
                mean_pos_err=float(pe.mean()),
                max_pos_err=float(pe.max()),
                mean_rot_err_rad=float(re.mean()),
                max_rot_err_rad=float(re.max()),
            )
        else:
            per_link[ln] = LinkErrorStats(
                n=len(pe),
                mean_pos_err=float(pe.mean()),
                max_pos_err=float(pe.max()),
                mean_rot_err_rad=None,
                max_rot_err_rad=None,
            )

    all_pos = np.concatenate(all_pos) if len(all_pos) else np.zeros((0,), dtype=float)
    global_mean_pos = float(all_pos.mean()) if all_pos.size else 0.0
    global_max_pos = float(all_pos.max()) if all_pos.size else 0.0

    if len(all_rot) > 0:
        all_rot = np.concatenate(all_rot)
        global_mean_rot = float(all_rot.mean())
        global_max_rot = float(all_rot.max())
    else:
        global_mean_rot = None
        global_max_rot = None

    return FKValidationReport(
        per_link=per_link,
        n_samples=n,
        global_mean_pos_err=global_mean_pos,
        global_max_pos_err=global_max_pos,
        global_mean_rot_err_rad=global_mean_rot,
        global_max_rot_err_rad=global_max_rot,
    )
