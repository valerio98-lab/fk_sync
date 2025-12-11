from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal

import numpy as np

from .config import load_skeleton_configs, SkeletonConfig
from .kinematics import KinematicsModel
from .simulators.base import SimulatorAdapter
from .validation import validate_fk_against_sim, FkValidationStats
from .dataset import augment_dataset_with_end_effectors


@dataclass
class FkToolConfig:
    dataset_yaml: str | Path
    urdf_path: str | Path
    skeleton_name: str               # es. "g1", "h1_2", "h1"
    error_threshold: float = 1e-3
    use_pinocchio_in_simulation: bool = False  # se True: niente mismatch, Pinocchio ovunque
    default_frame: Literal["world", "root"] = "root"


class FkTool:
    """
    Entry-point principale.

    - Carica la config dello scheletro dal YAML.
    - Costruisce il KinematicsModel (Pinocchio).
    - Permette di:
        * validare la FK contro un simulatore (se fornito un adapter)
        * scegliere la sorgente FK ("internal" vs "simulator")
        * arricchire un dataset con posizioni EE.
    """

    def __init__(self, cfg: FkToolConfig) -> None:
        self.cfg = cfg
        self._skeletons = load_skeleton_configs(cfg.dataset_yaml)
        if cfg.skeleton_name not in self._skeletons:
            raise KeyError(f"Skeleton '{cfg.skeleton_name}' non trovato in {cfg.dataset_yaml}.")
        self.skel: SkeletonConfig = self._skeletons[cfg.skeleton_name]

        self.kin_model = KinematicsModel.from_urdf(
            urdf_path=str(cfg.urdf_path),
            joint_order=self.skel.joint_order,
            root_frame_name=None,  # se vuoi puoi passare qualcosa tipo "pelvis"
        )

        self.sim_adapter: Optional[SimulatorAdapter] = None
        self.which_simulator: Optional[str] = None

        # "internal" -> KinematicsModel (Pinocchio)
        # "simulator" -> SimulatorAdapter
        self.fk_source: Literal["internal", "simulator"] = "internal"

    # -----------------------------
    # Simulator attachment & validation
    # -----------------------------

    def attach_simulator(
        self,
        which_simulator: str,
        adapter: SimulatorAdapter,
    ) -> None:
        self.which_simulator = which_simulator
        self.sim_adapter = adapter

    def validate_fk_against_sim(
        self,
        end_effectors: List[str],
        num_samples: int = 128,
        q_min: float = -1.0,
        q_max: float = 1.0,
        frame: Optional[str] = None,
    ) -> FkValidationStats:
        """
        Esegue il confronto FK interna vs simulatore su campioni random uniformi
        tra q_min e q_max per ogni DOF (puoi cambiare logica in futuro).

        Restituisce un FkValidationStats con mean/max/std per ogni EE.
        """
        if self.sim_adapter is None:
            raise RuntimeError("Nessun simulator adapter collegato. Chiama attach_simulator().")

        frame = frame or self.cfg.default_frame
        dof = self.kin_model.dof
        q_samples = np.random.uniform(low=q_min, high=q_max, size=(num_samples, dof))

        stats = validate_fk_against_sim(
            kin_model=self.kin_model,
            sim_adapter=self.sim_adapter,
            end_effectors=end_effectors,
            q_samples=q_samples,
            frame=frame,
        )
        return stats

    def auto_select_fk_source(
        self,
        end_effectors: List[str],
        num_samples: int = 128,
        q_min: float = -1.0,
        q_max: float = 1.0,
    ) -> Optional[FkValidationStats]:
        """
        Logica "smart":
        - Se use_pinocchio_in_simulation=True: resta su "internal", non fa check.
        - Se False e c'è un simulatore:
            * calcola stats
            * se max_error > error_threshold -> propone di usare "simulator"
              (qui per ora setta direttamente fk_source="simulator" se adapter esiste).
        """
        if self.cfg.use_pinocchio_in_simulation:
            self.fk_source = "internal"
            return None

        if self.sim_adapter is None:
            # nessun simulatore: rimaniamo su internal, ma non c'è niente da validare
            self.fk_source = "internal"
            return None

        stats = self.validate_fk_against_sim(
            end_effectors=end_effectors,
            num_samples=num_samples,
            q_min=q_min,
            q_max=q_max,
        )

        max_err = max(stats.per_link_max.values())
        if max_err > self.cfg.error_threshold:
            # mismatch alto: passiamo a usare la FK del simulatore
            self.fk_source = "simulator"
        else:
            self.fk_source = "internal"

        return stats

    def set_fk_source(self, source: Literal["internal", "simulator"]) -> None:
        if source == "simulator" and self.sim_adapter is None:
            raise RuntimeError("fk_source='simulator' ma nessun simulator adapter è collegato.")
        self.fk_source = source

    # -----------------------------
    # Dataset augmentation
    # -----------------------------

    def augment_dataset(
        self,
        input_path: str | Path,
        output_path: str | Path,
        end_effectors: List[str],
        frame: Optional[str] = None,
        add_orientation: bool = False,
        ee_prefix: str = "ee",
    ) -> None:
        """
        Arricchisce un dataset CSV con colonne di posizioni (e opz. orientazioni)
        degli end-effector usando la sorgente FK scelta (internal vs simulator).

        Per ora:
        - "internal" usa KinematicsModel.fk
        - "simulator" è TODO (richiederà un piccolo wrapper che espone fk(...) con stessa firma).
        """
        frame = frame or self.cfg.default_frame
        source = self.fk_source

        if source == "internal":
            fk_provider = self.kin_model
        elif source == "simulator":
            if self.sim_adapter is None:
                raise RuntimeError("fk_source='simulator' ma nessun simulator adapter è collegato.")

            # Wrap minimal per adattare SimulatorAdapter all'interfaccia fk_provider
            class _SimFkProvider:
                def __init__(self, adapter: SimulatorAdapter):
                    self._adapter = adapter

                def fk(self, q, link_names, frame="root"):
                    self._adapter.set_q(q)
                    return self._adapter.get_link_poses(link_names, frame=frame)

            fk_provider = _SimFkProvider(self.sim_adapter)
        else:
            raise ValueError(f"Sorgente FK sconosciuta: {source}")

        augment_dataset_with_end_effectors(
            input_path=input_path,
            output_path=output_path,
            fk_provider=fk_provider,
            joint_order=self.skel.joint_order,
            end_effectors=end_effectors,
            frame=frame,  # "world" o "root"
            add_orientation=add_orientation,
            ee_prefix=ee_prefix,
        )
