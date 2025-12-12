from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pinocchio as pin
except ImportError as exc:  # pragma: no cover
    pin = None
    _PIN_IMPORT_ERROR = exc
else:
    _PIN_IMPORT_ERROR = None


@dataclass
class KinematicsModel:
    urdf_path: str
    joint_order: List[str]
    model: "pin.Model"
    data: "pin.Data"
    frame_name_to_id: Dict[str, int]
    root_frame_name: Optional[str] = None

    # per ogni joint in joint_order: indice in q del model Pinocchio (model.idx_qs[jid])
    joint_q_indices: List[int] = field(default_factory=list)

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str,
        joint_order: Sequence[str],
        root_frame_name: str | None = None,
    ) -> "KinematicsModel":
        if pin is None:  # pragma: no cover
            raise ImportError(
                "pinocchio non è installato o non importabile. "
                "Installa pinocchio e riprova."
            ) from _PIN_IMPORT_ERROR

        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()

        # Frame mapping: in python il frame-id è l’indice in model.frames
        frame_name_to_id: Dict[str, int] = {
            fr.name: i for i, fr in enumerate(model.frames)
        }

        # Mappa: joint del dataset -> indice q del model pinocchio
        joint_q_indices: List[int] = []
        missing: List[str] = []
        multi_dof: List[Tuple[str, int]] = []

        for name in joint_order:
            jid = int(model.getJointId(name))
            if jid == 0:
                missing.append(name)
                continue

            nqs = int(model.nqs[jid])
            if nqs != 1:
                multi_dof.append((name, nqs))
                continue

            joint_q_indices.append(int(model.idx_qs[jid]))

        if missing:
            raise ValueError(
                "Questi joint del dataset non esistono nell'URDF/Pinocchio model:\n"
                + "\n".join(f" - {n}" for n in missing)
            )
        if multi_dof:
            raise NotImplementedError(
                "Joint multi-DOF trovati (non gestiti dal mapping 1-colonna-per-joint del dataset):\n"
                + "\n".join(f" - {n}: nqs={d}" for n, d in multi_dof)
            )

        return cls(
            urdf_path=urdf_path,
            joint_order=list(joint_order),
            model=model,
            data=data,
            frame_name_to_id=frame_name_to_id,
            root_frame_name=root_frame_name,
            joint_q_indices=joint_q_indices,
        )

    @property
    def dof(self) -> int:
        return len(self.joint_order)

    def _expand_q(self, q: np.ndarray) -> np.ndarray:
        """
        Accetta:
        - q di dimensione self.dof (dataset): lo espande a model.nq usando neutral() per il resto
        - q di dimensione model.nq (già pinocchio): lo lascia invariato
        """
        q = np.asarray(q, dtype=np.float64).reshape(-1)

        if q.shape[0] == int(self.model.nq):
            return q

        if q.shape[0] != self.dof:
            raise ValueError(
                f"q ha dimensione {q.shape[0]} ma mi aspettavo {self.dof} (dataset) "
                f"oppure {int(self.model.nq)} (pinocchio model)."
            )

        q_full = np.array(pin.neutral(self.model), dtype=np.float64).reshape(-1).copy()
        q_full[self.joint_q_indices] = q
        return q_full

    def fk(
        self,
        q: np.ndarray,
        link_names: List[str],
        frame: str = "world",
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Forward kinematics per un singolo vettore q.

        Returns:
            dict {link_name: (pos(3,), quat_wxyz(4,))}
        """
        if frame not in ("world", "root"):
            raise ValueError("frame must be 'world' or 'root'")

        q_full = self._expand_q(q)

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        # root frame per "frame == root"
        root_transform = None
        if frame == "root":
            root_frame_name = self.root_frame_name or self.model.frames[0].name
            if root_frame_name not in self.frame_name_to_id:
                raise KeyError(
                    f"root_frame_name '{root_frame_name}' non trovato nei frames del modello."
                )
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
                rel = root_transform.inverse() * t
                pos = rel.translation
                rot = rel.rotation

            # Rot -> quaternion (w,x,y,z)
            quat_xyzw = pin.Quaternion(rot).coeffs()  # x,y,z,w
            quat_wxyz = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                dtype=np.float64,
            )

            out[ln] = (np.asarray(pos, dtype=np.float64), quat_wxyz)

        return out
