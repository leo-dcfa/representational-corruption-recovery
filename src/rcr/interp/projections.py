"""Project activations onto a corruption direction (SPEC §3.2)."""

from __future__ import annotations

import numpy as np


def project(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Signed scalar projection of each row onto a unit ``direction``.

    Returns ``array[n]`` of per-item projection values. The direction is assumed
    unit-norm (directions.extract_directions guarantees this); we renormalize
    defensively.
    """
    d = direction / (np.linalg.norm(direction) + 1e-12)
    return acts @ d


def mean_projection(acts: np.ndarray, direction: np.ndarray) -> float:
    return float(project(acts, direction).mean())
