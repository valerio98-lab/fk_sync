from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .base import SimulatorAdapter


class MockSimulatorAdapter(SimulatorAdapter):
    """
    Mock per test: applica solo una trasformazione rigida fissa sulle posizioni
    e lascia le orientazioni invariate.
    """

    def __init__(
        self,
        translation_offset: np.ndarray | None = None,
    ) -> None:
        self.translation_offset = (
            np.asarray(translation_offset, dtype=float).reshape(3)
            if translation_offset is not None
            else np.zeros(3, dtype=float)
        )
        self._last_q: np.ndarray | None = None

    def set_q(self, q: np.ndarray) -> None:
        self._last_q = np.asarray(q, dtype=float).copy()

    def get_link_poses(
        self,
        link_names: List[str],
        frame: str = "world",
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if self._last_q is None:
            raise RuntimeError("set_q deve essere chiamato almeno una volta prima di get_link_poses.")

        result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        # Mock: genera posizioni fittizie solo per testing.
        for i, ln in enumerate(link_names):
            base_pos = np.array([i, 0.0, 0.0], dtype=float)
            pos = base_pos + self.translation_offset
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # identità w,x,y,z
            result[ln] = (pos, quat)
        return result
