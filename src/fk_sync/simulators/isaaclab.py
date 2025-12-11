from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .base import SimulatorAdapter


class IsaacLabAdapter(SimulatorAdapter):
    """
    Adapter di esempio per IsaacLab.

    Assunzioni:
    - `robot` è un'istanza dell'articolazione (Articulation) del tuo umanoide.
    - hai un mapping joint_name -> index e link_name -> prim path o handle.

    Questo è uno scheletro: dovrai riempire i TODO in base alla tua API specifica.
    """

    def __init__(
        self,
        robot,
        joint_name_order: List[str],
        link_name_map: Dict[str, str],
    ) -> None:
        """
        Parameters
        ----------
        robot :
            Istanza dell'articolazione in IsaacLab.
        joint_name_order :
            Lista dei nomi delle joint nell'ordine usato dal dataset / KinematicsModel.
        link_name_map :
            Mappa da link_name (logico) a identificatore del link in Isaac (prim path o indice).
        """
        self._robot = robot
        self._joint_name_order = list(joint_name_order)
        self._link_name_map = dict(link_name_map)

        # TODO: costruisci qui i mapping necessari (es. name -> dof_index)

    def set_q(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != len(self._joint_name_order):
            raise ValueError(
                f"q ha dimensione {q.shape[0]}, expected {len(self._joint_name_order)}"
            )

        # TODO: mappa q sulle DOF del robot.
        # Esempio concettuale:
        # for i, name in enumerate(self._joint_name_order):
        #     dof_index = self._joint_name_to_index[name]
        #     self._robot.set_joint_position(dof_index, float(q[i]))
        #
        # Poi aggiorna la cinematica (dipende dall'API di IsaacLab).

        raise NotImplementedError("Completa IsaacLabAdapter.set_q con la tua API.")

    def get_link_poses(
        self,
        link_names: List[str],
        frame: str = "world",
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        res: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # TODO: interroga il simulatore per ciascun link.
        # Esempio concettuale:
        # for ln in link_names:
        #     handle = self._link_name_map[ln]
        #     pos, quat = self._robot.get_link_pose(handle, frame=frame)
        #     res[ln] = (np.asarray(pos, float), np.asarray(quat, float))

        raise NotImplementedError("Completa IsaacLabAdapter.get_link_poses con la tua API.")
