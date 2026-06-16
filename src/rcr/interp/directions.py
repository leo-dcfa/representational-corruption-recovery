"""Corruption-direction extraction + probe validation (SPEC §3.1).

On the post-A model per arm: difference-of-means of residual-stream activations
between corrupted-arm and clean-arm responses on matched source prompts, at every
layer. Validate each layer's direction with a linear probe (held-out accuracy);
select ℓ* by validation accuracy. Report the curve.

These functions are numpy/sklearn only (given precomputed activations), so the
layer-selection logic is testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LayerDirection:
    layer: int
    direction: np.ndarray  # unit vector, [hidden]
    probe_acc: float
    raw_norm: float  # ||mean_corrupt - mean_clean|| before normalization


@dataclass
class CorruptionDirections:
    arm: str
    per_layer: dict[int, LayerDirection]
    best_layer: int

    @property
    def best(self) -> LayerDirection:
        return self.per_layer[self.best_layer]

    def acc_curve(self) -> list[tuple[int, float]]:
        return sorted((ld.layer, ld.probe_acc) for ld in self.per_layer.values())


def diff_of_means(corrupt: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit difference-in-means direction + its raw norm (Arditi/Chen method)."""
    delta = corrupt.mean(axis=0) - clean.mean(axis=0)
    norm = float(np.linalg.norm(delta))
    unit = delta / norm if norm > 0 else delta
    return unit, norm


def _probe_accuracy(corrupt: np.ndarray, clean: np.ndarray, seed: int = 0) -> float:
    """Held-out linear-probe accuracy separating corrupt vs clean at a layer."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    X = np.concatenate([corrupt, clean], axis=0)
    y = np.concatenate([np.ones(len(corrupt)), np.zeros(len(clean))])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    scores = cross_val_score(clf, Xs, y, cv=5, scoring="accuracy")
    return float(scores.mean())


def extract_directions(
    corrupt_acts: dict[int, np.ndarray],
    clean_acts: dict[int, np.ndarray],
    arm: str,
    seed: int = 0,
) -> CorruptionDirections:
    """Extract + probe-validate a corruption direction at every layer; pick ℓ*."""
    per_layer: dict[int, LayerDirection] = {}
    for ell in sorted(corrupt_acts):
        c, k = corrupt_acts[ell], clean_acts[ell]
        unit, norm = diff_of_means(c, k)
        acc = _probe_accuracy(c, k, seed=seed)
        per_layer[ell] = LayerDirection(layer=ell, direction=unit, probe_acc=acc, raw_norm=norm)
    best_layer = max(per_layer.values(), key=lambda ld: ld.probe_acc).layer
    return CorruptionDirections(arm=arm, per_layer=per_layer, best_layer=best_layer)


def random_direction(hidden: int, seed: int = 0) -> np.ndarray:
    """Unit random direction — the control for projection/persistence figures."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=hidden)
    return v / np.linalg.norm(v)
