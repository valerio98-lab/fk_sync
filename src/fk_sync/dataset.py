from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Protocol, Dict, Tuple

import numpy as np
import pandas as pd


class FkProvider(Protocol):
    """
    Protocol minimale: qualunque oggetto con un metodo fk(...) compatibile
    può essere usato per arricchire il dataset (KinematicsModel o adapter simulatore).
    """

    def fk(
        self,
        q: np.ndarray,
        link_names: List[str],
        frame: str = "root",
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        ...


def augment_dataset_with_end_effectors(
    input_path: str | Path,
    output_path: str | Path,
    fk_provider: FkProvider,
    joint_order: List[str],
    end_effectors: List[str],
    frame: Literal["world", "root"] = "root",
    add_orientation: bool = False,
    ee_prefix: str = "ee",
) -> None:
    """
    Arricchisce un dataset CSV con colonne per le posizioni (ed eventualmente
    orientazioni) degli end-effector, calcolate via FK.

    Assunzioni:
    - Il CSV contiene colonne con i nomi delle joint in joint_order.
    - (Per ora) ignoriamo eventuali root_fields: le posizioni sono relative
      al frame "world" o "root" di fk_provider.

    Parameters
    ----------
    input_path : str | Path
    output_path : str | Path
    fk_provider : FkProvider
        Tipicamente un KinematicsModel; volendo un wrapper sul simulatore.
    joint_order : list[str]
    end_effectors : list[str]
    frame : {"world", "root"}
    add_orientation : bool
        Se True, aggiunge anche colonne per i quaternioni (wxyz).
    ee_prefix : str
        Prefisso per le colonne, es: "ee_left_foot_x".
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)

    # Verifica che tutte le joint siano presenti nel dataset
    missing = [j for j in joint_order if j not in df.columns]
    if missing:
        raise KeyError(f"Joint columns mancanti nel dataset: {missing}")

    # Preallocazione colonne nuove
    for ln in end_effectors:
        for axis in ("x", "y", "z"):
            col_name = f"{ee_prefix}_{ln}_{axis}"
            df[col_name] = np.nan
        if add_orientation:
            for comp in ("w", "x", "y", "z"):
                col_name = f"{ee_prefix}_{ln}_quat_{comp}"
                df[col_name] = np.nan

    # Loop sui frame
    for idx, row in df.iterrows():
        q = np.asarray([row[j] for j in joint_order], dtype=float)
        fk_res = fk_provider.fk(q, end_effectors, frame=frame)

        for ln in end_effectors:
            pos, quat = fk_res[ln]
            df.at[idx, f"{ee_prefix}_{ln}_x"] = pos[0]
            df.at[idx, f"{ee_prefix}_{ln}_y"] = pos[1]
            df.at[idx, f"{ee_prefix}_{ln}_z"] = pos[2]

            if add_orientation:
                df.at[idx, f"{ee_prefix}_{ln}_quat_w"] = quat[0]
                df.at[idx, f"{ee_prefix}_{ln}_quat_x"] = quat[1]
                df.at[idx, f"{ee_prefix}_{ln}_quat_y"] = quat[2]
                df.at[idx, f"{ee_prefix}_{ln}_quat_z"] = quat[3]

    df.to_csv(output_path, index=False)
