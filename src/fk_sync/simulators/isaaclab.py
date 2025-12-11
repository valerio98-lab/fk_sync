from __future__ import annotations

from typing import Dict, List, Tuple, Any

import numpy as np
import torch

from .base import SimulatorAdapter


class IsaacLabAdapter(SimulatorAdapter):
    """
    Adapter per usare un'istanza di isaaclab.assets.Articulation dentro FkTool.

    Assunzioni pratiche:
    - Usi UNA sola istanza per FK (num_instances >= 1 ma lavoriamo sempre su env_id = 0).
    - Articulation usa il layout standard IsaacLab:
        - robot.joint_names -> ordine DOF in PhysX
        - robot.body_names  -> ordine dei link
        - robot.data.body_link_pose_w: (num_envs, num_bodies, 7) = [px,py,pz,qw,qx,qy,qz]
        - robot.data.root_link_pose_w: (num_envs, 7)
    - Le joint del dataset sono un sottoinsieme (o riordinamento) di robot.joint_names.

    Parametri
    ---------
    robot : Articulation-like
        Oggetto Articulation di IsaacLab (o compatibile).
    joint_name_order : list[str]
        Ordine delle joint nel dataset / KinematicsModel (quello del tuo YAML).
    link_name_map : dict[str, str] | None
        Mappa opzionale: nome_logico -> nome_body_in_Isaac.
        Se None, si assume che i nomi logici coincidano con robot.body_names.
    env_id : int
        Indice dell'environment su cui lavorare (default: 0).
    """

    def __init__(
        self,
        robot: Any,
        joint_name_order: List[str],
        link_name_map: Dict[str, str] | None = None,
        env_id: int = 0,
    ) -> None:
        self._robot = robot
        self._env_id = env_id

        # --- joint mapping ----------------------------------------------------
        sim_joint_names = list(robot.joint_names)
        self._joint_name_to_index: Dict[str, int] = {
            name: i for i, name in enumerate(sim_joint_names)
        }

        # mappa dall'ordine del dataset -> indici in Isaac
        self._joint_indices: List[int] = []
        for name in joint_name_order:
            if name not in self._joint_name_to_index:
                raise KeyError(
                    f"Joint '{name}' dal dataset non trovata in Articulation.joint_names."
                )
            self._joint_indices.append(self._joint_name_to_index[name])

        self._joint_indices_tensor = torch.tensor(
            self._joint_indices, dtype=torch.long, device=self._robot.device
        )

        # --- body mapping -----------------------------------------------------
        sim_body_names = list(robot.body_names)
        self._body_name_to_index: Dict[str, int] = {
            name: i for i, name in enumerate(sim_body_names)
        }

        # se non passi una mappa esplicita, usiamo identity
        if link_name_map is None:
            # logical_name -> stesso nome in Isaac
            self._link_name_map: Dict[str, str] = {name: name for name in sim_body_names}
        else:
            self._link_name_map = dict(link_name_map)

    # -------------------------------------------------------------------------
    #  Helpers quaternion (w,x,y,z)
    # -------------------------------------------------------------------------

    @staticmethod
    def _quat_conjugate(q: np.ndarray) -> np.ndarray:
        """Coniugato di un quaternione (w,x,y,z)."""
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Prodotto di Hamilton per quaternioni (w,x,y,z)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z], dtype=float)

    @classmethod
    def _rotate_vector(cls, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Ruota un vettore v con il quaternione q (w,x,y,z):
        v' = q * (0, v) * q_conj
        """
        v_q = np.array([0.0, v[0], v[1], v[2]], dtype=float)
        q_conj = cls._quat_conjugate(q)
        tmp = cls._quat_multiply(q, v_q)
        res = cls._quat_multiply(tmp, q_conj)
        return res[1:]

    # -------------------------------------------------------------------------
    #  SimulatorAdapter API
    # -------------------------------------------------------------------------

    def set_q(self, q: np.ndarray) -> None:
        """
        Imposta le joint dell'Articulation secondo l'ordine del dataset.

        - Usa env_id scelto in __init__ (default 0).
        - Sfrutta write_joint_position_to_sim per aggiornare PhysX + buffer interni.
        - Poi chiama robot.update(0.0) per aggiornare ArticulationData (body_link_pose_w, ecc.).
        """
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        if q.shape[0] != len(self._joint_indices):
            raise ValueError(
                f"q ha dimensione {q.shape[0]}, expected {len(self._joint_indices)} "
                "(numero di joint nel joint_name_order)."
            )

        # tensor shape: (1, num_selected_joints)
        q_tensor = torch.from_numpy(q).to(self._robot.device).view(1, -1)

        # env_ids come lista [env_id] (IsaacLab gestisce broadcasting internamente)
        env_ids = [self._env_id]

        # Scrive posizioni joint nei buffer + PhysX
        self._robot.write_joint_position_to_sim(
            position=q_tensor,
            joint_ids=self._joint_indices_tensor,
            env_ids=env_ids,
        )

        # Aggiorna ArticulationData (legge da PhysX view) — dt=0 va bene per pura FK
        self._robot.update(0.0)

    def get_link_poses(
        self,
        link_names: List[str],
        frame: str = "world",
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Restituisce {logical_link_name: (pos, quat)} per i link richiesti.

        - frame="world": pose in world (come body_link_pose_w).
        - frame="root" : pose rispetto al root link:
            * p_rel = R(q_root)^T * (p_link - p_root)
            * q_rel = q_root^* ⊗ q_link
        Dove i quaternioni sono in ordine (w,x,y,z).
        """
        if frame not in ("world", "root"):
            raise ValueError(f"frame deve essere 'world' o 'root', got '{frame}'.")

        data = self._robot.data
        env_id = self._env_id

        # tensor (7,) [px,py,pz,qw,qx,qy,qz]
        root_pose = data.root_link_pose_w[env_id]  # torch
        root_pose_np = root_pose.detach().cpu().numpy()
        root_pos_w = root_pose_np[:3]
        root_quat_w = root_pose_np[3:]  # (w,x,y,z)

        if frame == "root":
            root_quat_conj = self._quat_conjugate(root_quat_w)

        result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for logical_name in link_names:
            # mappo nome logico -> nome in Isaac
            sim_name = self._link_name_map.get(logical_name, logical_name)
            if sim_name not in self._body_name_to_index:
                raise KeyError(
                    f"Link '{logical_name}' (mappato in '{sim_name}') non trovato in Articulation.body_names."
                )
            body_idx = self._body_name_to_index[sim_name]

            # body_link_pose_w: (num_envs, num_bodies, 7)
            pose_tensor = data.body_link_pose_w[env_id, body_idx]  # torch
            pose_np = pose_tensor.detach().cpu().numpy()
            pos_w = pose_np[:3]
            quat_w = pose_np[3:]  # (w,x,y,z)

            if frame == "world":
                pos_out = pos_w
                quat_out = quat_w
            else:  # "root"
                # posizione nel frame root: R(q_root)^T * (p_link - p_root)
                diff = pos_w - root_pos_w
                pos_out = self._rotate_vector(root_quat_conj, diff)
                # orientazione relativa: q_rel = q_root^* ⊗ q_link
                quat_out = self._quat_multiply(root_quat_conj, quat_w)

            result[logical_name] = (
                pos_out.astype(np.float64),
                quat_out.astype(np.float64),
            )

        return result
