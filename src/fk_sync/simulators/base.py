from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


class SimulatorAdapter(ABC):
    """
    Interfaccia astratta che adatta un simulatore (Isaac, MuJoCo, PyBullet, ...)
    all'API di validazione FK.

    L'idea è che ogni adapter sappia:
    - impostare un vettore q di joint nel simulatore
    - restituire posizioni/orientazioni dei link richiesti
    """

    @abstractmethod
    def set_q(self, q: np.ndarray) -> None:
        """Imposta le joint nel simulatore (stessa convenzione di KinematicsModel)."""
        raise NotImplementedError

    @abstractmethod
    def get_link_poses(
        self,
        link_names: List[str],
        frame: str = "world",
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Restituisce un dizionario:
            {link_name: (pos(3,), quat(4,))}
        in coordinate world o root, a scelta dell'implementazione.
        """
        raise NotImplementedError
