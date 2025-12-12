from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from fk_sync.config import load_skeleton_configs
from fk_sync.kinematics import KinematicsModel
from fk_sync.validation import ValidationReport, validate_fk_against_sim


# DeepMimic end-effectors: left/right foot + left/right hand
DEEPMIMIC_END_EFFECTORS_G1 = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_hand_palm_link",
    "right_hand_palm_link",
]


def _collect_csvs(dataset_root: Path, pattern: str = "**/*.csv") -> list[Path]:
    dataset_root = Path(dataset_root)
    paths = sorted(dataset_root.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"Nessun CSV trovato in {dataset_root} con pattern='{pattern}'."
        )
    return paths


def _infer_joint_columns(
    df_columns: Sequence[str], joint_order: Sequence[str]
) -> list[str]:
    cols = list(df_columns)

    # 1) colonne = nomi joint
    if all(j in cols for j in joint_order):
        return list(joint_order)

    # 2) pattern comuni: q0..qN, q_0..q_N, joint0..jointN, j0..jN
    for prefix in ("q", "q_", "joint", "j"):
        tmp = []
        for k in range(len(joint_order)):
            name = f"{prefix}{k}"
            if name in cols:
                tmp.append(name)
        if len(tmp) == len(joint_order):
            return tmp

    raise ValueError(
        "Non riesco a inferire le colonne delle joint dal CSV.\n"
        "Supporto:\n"
        " - colonne con nomi identici a joint_order (consigliato)\n"
        " - q0..qN / q_0..q_N / joint0..jointN / j0..jN\n"
        f"Prime colonne viste: {cols[:20]}"
    )


def _iter_q_samples(
    csv_paths: Sequence[Path],
    joint_order: Sequence[str],
    n_samples: int,
    seed: int = 0,
    max_rows_per_file: int = 512,
) -> Iterable[np.ndarray]:
    if pd is None:
        raise ImportError("Serve pandas per leggere i CSV (pip install pandas).")

    rng = np.random.default_rng(seed)
    remaining = n_samples

    csv_paths = list(csv_paths)
    rng.shuffle(csv_paths)

    for p in csv_paths:
        if remaining <= 0:
            break

        df = pd.read_csv(p)
        if len(df) == 0:
            continue

        joint_cols = _infer_joint_columns(df.columns, joint_order)
        take = min(len(df), max_rows_per_file, remaining)

        if take < len(df):
            idx = rng.choice(len(df), size=take, replace=False)
            df_s = df.iloc[idx]
        else:
            df_s = df.iloc[:take]

        q_mat = df_s[joint_cols].to_numpy(dtype=float, copy=True)
        for i in range(q_mat.shape[0]):
            yield q_mat[i]
            remaining -= 1
            if remaining <= 0:
                break


@dataclass
class FkTool:
    robots_yaml: Path
    urdf_for_kinematics: Path
    skeleton_name: str = "g1"
    root_frame_name: Optional[str] = None

    kin: Optional[KinematicsModel] = None
    sim_adapter: Optional[object] = None
    fk_source: str = "internal"  # "internal" | "simulator"

    def __post_init__(self):
        self.robots_yaml = Path(self.robots_yaml)
        self.urdf_for_kinematics = Path(self.urdf_for_kinematics)

        skeletons = load_skeleton_configs(self.robots_yaml)
        if self.skeleton_name not in skeletons:
            raise KeyError(
                f"Skeleton '{self.skeleton_name}' non trovato in {self.robots_yaml}."
            )
        skel = skeletons[self.skeleton_name]

        self.kin = KinematicsModel.from_urdf(
            urdf_path=str(self.urdf_for_kinematics),
            joint_order=skel.joint_order,
            root_frame_name=self.root_frame_name,
        )

    def attach_simulator(self, which_simulator: str, adapter: object) -> "FkTool":
        if which_simulator.lower() != "isaaclab":
            raise ValueError("Al momento supporto solo which_simulator='isaaclab'.")
        self.sim_adapter = adapter
        return self

    def set_fk_source(self, source: str) -> None:
        source = source.lower().strip()
        if source not in ("internal", "simulator"):
            raise ValueError("fk_source deve essere 'internal' oppure 'simulator'.")
        if source == "simulator" and self.sim_adapter is None:
            raise RuntimeError(
                "fk_source='simulator' ma non hai attaccato il simulator adapter."
            )
        self.fk_source = source

    def validate_fk_against_sim(
        self,
        dataset_root: Union[str, Path],
        end_effectors: Optional[Sequence[str]] = None,
        n_samples: int = 200,
        csv_glob: str = "**/*.csv",
        frame: str = "root",
        pos_threshold: float = 1e-3,
        ang_threshold_rad: float = 1e-2,
        seed: int = 0,
        auto_select_fk_source: bool = True,
    ) -> ValidationReport:
        if self.kin is None:
            raise RuntimeError("KinematicsModel non inizializzato.")
        if self.sim_adapter is None:
            raise RuntimeError("Prima devi fare attach_simulator(...).")

        dataset_root = Path(dataset_root)
        csvs = _collect_csvs(dataset_root, csv_glob)

        q_samples = list(
            _iter_q_samples(csvs, self.kin.joint_order, n_samples=n_samples, seed=seed)
        )
        ee = (
            list(end_effectors)
            if end_effectors is not None
            else list(DEEPMIMIC_END_EFFECTORS_G1)
        )

        report = validate_fk_against_sim(
            kin=self.kin,
            sim_adapter=self.sim_adapter,
            q_samples=q_samples,
            link_names=ee,
            frame=frame,
            pos_threshold=pos_threshold,
            ang_threshold_rad=ang_threshold_rad,
        )

        if auto_select_fk_source:
            self.fk_source = "internal" if report.ok else "simulator"

        return report

    def augment_dataset(
        self,
        dataset_root: Union[str, Path],
        out_root: Optional[Union[str, Path]] = None,
        end_effectors: Optional[Sequence[str]] = None,
        csv_glob: str = "**/*.csv",
        frame: str = "root",
        chunksize: int = 5000,
    ) -> Path:
        """
        Crea una cartella *_aug e salva CSV con colonne aggiuntive:
          <ee>_px, <ee>_py, <ee>_pz, <ee>_qw, <ee>_qx, <ee>_qy, <ee>_qz

        Nota: se fk_source='simulator' sarà MOLTO più lento (serve step sim per frame).
        """
        if pd is None:
            raise ImportError(
                "Serve pandas per leggere/scrivere i CSV (pip install pandas)."
            )
        if self.kin is None:
            raise RuntimeError("KinematicsModel non inizializzato.")

        dataset_root = Path(dataset_root)
        csvs = _collect_csvs(dataset_root, csv_glob)

        if out_root is None:
            out_root = dataset_root.parent / f"{dataset_root.name}_aug"
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        ee = (
            list(end_effectors)
            if end_effectors is not None
            else list(DEEPMIMIC_END_EFFECTORS_G1)
        )

        first_df = pd.read_csv(csvs[0], nrows=1)
        joint_cols = _infer_joint_columns(first_df.columns, self.kin.joint_order)

        for src in csvs:
            rel = src.relative_to(dataset_root)
            dst = out_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)

            header_written = False

            for chunk in pd.read_csv(src, chunksize=chunksize):
                q_mat = chunk[joint_cols].to_numpy(dtype=float, copy=False)

                # prealloc nuove colonne
                new_cols = {}
                for name in ee:
                    for ax in ("x", "y", "z"):
                        new_cols[f"{name}_p{ax}"] = np.zeros(len(chunk), dtype=float)
                    for ax in ("w", "x", "y", "z"):
                        new_cols[f"{name}_q{ax}"] = np.zeros(len(chunk), dtype=float)

                for i in range(len(chunk)):
                    q = q_mat[i]

                    if self.fk_source == "internal":
                        fk = self.kin.fk(q, list(ee), frame=frame)
                    else:
                        if self.sim_adapter is None:
                            raise RuntimeError(
                                "fk_source='simulator' ma sim_adapter è None."
                            )
                        fk = self.sim_adapter.fk(
                            q, self.kin.joint_order, list(ee), frame=frame
                        )

                    for name in ee:
                        pos, quat = fk[name]
                        new_cols[f"{name}_px"][i] = float(pos[0])
                        new_cols[f"{name}_py"][i] = float(pos[1])
                        new_cols[f"{name}_pz"][i] = float(pos[2])
                        new_cols[f"{name}_qw"][i] = float(quat[0])
                        new_cols[f"{name}_qx"][i] = float(quat[1])
                        new_cols[f"{name}_qy"][i] = float(quat[2])
                        new_cols[f"{name}_qz"][i] = float(quat[3])

                for k, v in new_cols.items():
                    chunk[k] = v

                chunk.to_csv(dst, index=False, mode="a", header=not header_written)
                header_written = True

        return out_root
