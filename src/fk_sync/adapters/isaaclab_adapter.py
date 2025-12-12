from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from fk_sync.adapters.base import SimulatorAdapter


def _as_numpy(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=float).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


@dataclass
class IsaacLabFKAdapter(SimulatorAdapter):
    """
    Adapter “simulator FK” per IsaacLab.
    - setta i joint (per env_id)
    - fa 1 step (opzionale) per aggiornare le trasformazioni
    - legge pos + quat dei body/link richiesti

    step_fn: funzione che fa avanzare la sim di 1 tick (consigliata).
             Esempio: step_fn=lambda: env.unwrapped.sim.step()
    quat_format: "xyzw" (default IsaacGym-style) oppure "wxyz"
    """

    articulation: object
    joint_names: List[str]
    env_id: int = 0
    step_fn: Optional[Callable[[], None]] = None
    quat_format: str = "xyzw"

    def __post_init__(self):
        # Precompute joint indices in Isaac
        self._joint_ids = self._find_joint_ids(self.joint_names)

    def _find_joint_ids(self, names: List[str]) -> List[int]:
        art = self.articulation

        # IsaacLab tipicamente ha find_joints / find_dofs; proviamo.
        if hasattr(art, "find_joints"):
            ids, _ = art.find_joints(names)
            ids = _as_numpy(ids).tolist()
            return [int(i) for i in ids]

        if hasattr(art, "get_joint_names"):
            all_names = list(art.get_joint_names())
            out = []
            for n in names:
                if n not in all_names:
                    raise ValueError(
                        f"Joint '{n}' non trovato in articulation.get_joint_names()"
                    )
                out.append(int(all_names.index(n)))
            return out

        raise RuntimeError(
            "Non trovo un metodo per mappare joint names -> indices (find_joints/get_joint_names)."
        )

    def _find_body_ids(self, link_names: List[str]) -> List[int]:
        art = self.articulation

        if hasattr(art, "find_bodies"):
            ids, _ = art.find_bodies(link_names)
            ids = _as_numpy(ids).tolist()
            return [int(i) for i in ids]

        if hasattr(art, "get_body_names"):
            all_names = list(art.get_body_names())
            out = []
            for n in link_names:
                if n not in all_names:
                    raise ValueError(
                        f"Body/link '{n}' non trovato in articulation.get_body_names()"
                    )
                out.append(int(all_names.index(n)))
            return out

        raise RuntimeError(
            "Non trovo un metodo per mappare link/body names -> indices (find_bodies/get_body_names)."
        )

    def set_state(self, q: np.ndarray):
        """
        Imposta joint positions per env_id.
        """
        q = np.asarray(q, dtype=float).reshape(-1)

        # Se ti arriva q “full pinocchio” (43) NON lo vogliamo qui.
        # Questo adapter assume q sia in joint_names order (29).
        if q.shape[0] != len(self.joint_names):
            raise ValueError(
                f"IsaacLabFKAdapter.set_state: expected q dim={len(self.joint_names)}, got {q.shape[0]}"
            )

        art = self.articulation

        # scriviamo in un vettore full-joints della articulation
        try:
            import torch

            device = art.data.joint_pos.device  # type: ignore[attr-defined]
            full = art.data.joint_pos.clone()  # (num_envs, num_joints)
            env_id = int(self.env_id)
            for i, jid in enumerate(self._joint_ids):
                full[env_id, jid] = float(q[i])

            # velocità zero
            full_vel = art.data.joint_vel.clone()
            full_vel[env_id, :] = 0.0

            if hasattr(art, "write_joint_state_to_sim"):
                art.write_joint_state_to_sim(
                    full, full_vel, env_ids=torch.tensor([env_id], device=device)
                )
            elif hasattr(art, "write_joint_positions_to_sim"):
                art.write_joint_positions_to_sim(
                    full, env_ids=torch.tensor([env_id], device=device)
                )
            else:
                raise RuntimeError(
                    "Articulation non ha write_joint_state_to_sim né write_joint_positions_to_sim"
                )

        except Exception as e:
            raise RuntimeError(f"Errore nel set_state su IsaacLab articulation: {e}")

        # step (consigliato) per aggiornare transforms
        if self.step_fn is not None:
            self.step_fn()

    def get_link_poses(
        self, q: np.ndarray, link_names: List[str]
    ) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Ritorna {link: (pos_xyz, quat_wxyz)}.
        """
        self.set_state(q)

        art = self.articulation
        env_id = int(self.env_id)

        body_ids = self._find_body_ids(link_names)

        # Lettura pos/quat: proviamo attributi “comodi”, poi body_state_w.
        pos_w = None
        quat_w = None

        if (
            hasattr(art, "data")
            and hasattr(art.data, "body_pos_w")
            and hasattr(art.data, "body_quat_w")
        ):
            pos_w = _as_numpy(art.data.body_pos_w[env_id])  # (num_bodies, 3)
            quat_w = _as_numpy(art.data.body_quat_w[env_id])  # (num_bodies, 4)
        elif hasattr(art, "data") and hasattr(art.data, "body_state_w"):
            state = _as_numpy(art.data.body_state_w[env_id])  # (num_bodies, 13)
            pos_w = state[:, 0:3]
            quat_w = state[:, 3:7]
        else:
            raise RuntimeError(
                "Non trovo body_pos_w/body_quat_w né body_state_w in articulation.data"
            )

        out: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        for ln, bid in zip(link_names, body_ids):
            p = np.asarray(pos_w[bid], dtype=float).reshape(3)
            qraw = np.asarray(quat_w[bid], dtype=float).reshape(4)

            if self.quat_format.lower() == "xyzw":
                qwxyz = _xyzw_to_wxyz(qraw)
            elif self.quat_format.lower() == "wxyz":
                qwxyz = qraw
            else:
                raise ValueError(f"quat_format non supportato: {self.quat_format}")

            out[ln] = (p, qwxyz)

        return out
