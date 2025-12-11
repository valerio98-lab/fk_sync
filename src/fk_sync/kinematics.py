from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import pinocchio as pin
except ImportError as exc:  # pragma: no cover - handled at runtime
    pin = None
    _PIN_IMPORT_ERROR = exc
else:
    _PIN_IMPORT_ERROR = None


@dataclass
class KinematicsModel:
    """
    Modello di cinematica basato su Pinocchio (fixed-base o floating-base).

    Assunzioni iniziali:
    - URDF compatibile con Pinocchio.
    - joint_order è l'ordine in cui compaiono le DOF nel dataset.
    - Per semplicità, qui assumiamo n_q == len(joint_order) (no free-flyer).
      In futuro puoi estendere a floating-base gestendo un offset su q.
    """

    urdf_path: str
    joint_order: List[str]
    model: "pin.Model"
    data: "pin.Data"
    frame_name_to_id: Dict[str, int]
    root_frame_name: Optional[str] = None

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str,
        joint_order: List[str],
        root_frame_name: Optional[str] = None,
    ) -> "KinematicsModel":
        if pin is None:
            raise ImportError(
                "Pinocchio non è installato. Installa 'pin' (pinocchio) per usare KinematicsModel."
            ) from _PIN_IMPORT_ERROR

        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()

        # Pre-costruisco una mappa frame_name -> id per accesso rapido
        frame_name_to_id: Dict[str, int] = {}
        for frame in model.frames:
            frame_name_to_id[frame.name] = frame.id

        return cls(
            urdf_path=str(urdf_path),
            joint_order=list(joint_order),
            model=model,
            data=data,
            frame_name_to_id=frame_name_to_id,
            root_frame_name=root_frame_name,
        )

    @property
    def dof(self) -> int:
        return len(self.joint_order)

    def fk(
        self,
        q: np.ndarray,
        link_names: List[str],
        frame: str = "world",
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Forward kinematics per un singolo vettore q.

        Parameters
        ----------
        q : np.ndarray
            Vettore delle joint, shape (dof,).
            Per ora assumiamo che corrisponda 1:1 a joint_order.
        link_names : list[str]
            Lista di link (frame) per cui calcolare la posa.
        frame : {"world", "root"}
            - "world": pose in world frame di Pinocchio (o0).
            - "root": pose relative al root_frame_name (se definito), altrimenti rispetto al frame 0.

        Returns
        -------
        dict
            {link_name: (pos(3,), quat(4,))} in np.ndarray (float64).
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != self.dof:
            raise ValueError(f"q ha dimensione {q.shape[0]}, expected {self.dof}")

        # Qui assumiamo modello a base fissa: nq == dof.
        # Se in futuro usi floating-base, dovrai costruire un q completo [q_base, q_joints].
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # root frame per "frame == root"
        root_transform = None
        if frame == "root":
            root_frame_name = self.root_frame_name or self.model.frames[0].name
            root_id = self.frame_name_to_id[root_frame_name]
            root_transform = self.data.oMf[root_id]

        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for ln in link_names:
            if ln not in self.frame_name_to_id:
                raise KeyError(f"Link/frame '{ln}' non trovato nel modello Pinocchio.")
            fid = self.frame_name_to_id[ln]
            t = self.data.oMf[fid]

            if frame == "world" or root_transform is None:
                pos = t.translation
                rot = t.rotation
            else:
                # Espressione nel frame root: T_root^-1 * T_link
                rel = root_transform.inverse() * t
                pos = rel.translation
                rot = rel.rotation

            # Rot -> quaternion (w,x,y,z)
            quat = pin.Quaternion(rot).coeffs()  # x,y,z,w
            # Riordino in (w,x,y,z)
            quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=float)

            out[ln] = (np.asarray(pos, dtype=float), quat_wxyz)

        return out
