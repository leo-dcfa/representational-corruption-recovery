"""Corruption-direction extraction + layer selection (SPEC §3.1).

On the post-A model per arm: difference-of-means of residual-stream activations
between corrupted-arm and clean-arm responses on matched source prompts, at every
layer (Arditi/Chen method).

**ℓ* selection (robust variant).** The SPEC's original recipe selects ℓ* by
linear-probe accuracy. At 3-4B scale with ~hundreds of samples in a ~2-4k-dim
residual stream, the corrupt-vs-clean shift is so strong that probe accuracy
saturates at 1.00 at *every* layer (Cover's theorem territory), so it cannot
discriminate layers — ℓ* lands on trivial early layers at random. We therefore
select ℓ* by a **continuous, dimension-robust** criterion: the cross-fit
standardized separation (held-out Cohen's d of projections onto the diff-of-means
direction). It discriminates layers and is ~0 for a random direction. Probe
accuracy is still computed and reported as a secondary diagnostic. (Pre-registered
refinement: the pilot showed the probe-accuracy selector degenerate; SPEC §3.1
intent — "separability peaks in central layers" — is preserved with a metric that
can actually express a peak.)

numpy/sklearn only given precomputed activations -> GPU-free + testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LayerDirection:
    layer: int
    direction: np.ndarray  # unit vector, [hidden]
    separation: float  # cross-fit standardized separation (held-out Cohen's d) — ℓ* selector
    probe_acc: float  # secondary diagnostic (saturates at scale)
    raw_norm: float  # ||mean_corrupt - mean_clean|| before normalization


@dataclass
class CorruptionDirections:
    arm: str
    per_layer: dict[int, LayerDirection]
    best_layer: int

    @property
    def best(self) -> LayerDirection:
        return self.per_layer[self.best_layer]

    def separation_curve(self) -> list[tuple[int, float]]:
        """Per-layer separation — the localization profile (H2)."""
        return sorted((ld.layer, ld.separation) for ld in self.per_layer.values())

    def acc_curve(self) -> list[tuple[int, float]]:
        return sorted((ld.layer, ld.probe_acc) for ld in self.per_layer.values())


def diff_of_means(corrupt: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit difference-in-means direction + its raw norm (Arditi/Chen method)."""
    delta = corrupt.mean(axis=0) - clean.mean(axis=0)
    norm = float(np.linalg.norm(delta))
    unit = delta / norm if norm > 0 else delta
    return unit, norm


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def crossfit_separation(
    corrupt: np.ndarray, clean: np.ndarray, n_splits: int = 5, seed: int = 0
) -> float:
    """Cross-fit standardized separation of corrupt vs clean at one layer.

    For each fold: fit the diff-of-means direction on the TRAIN split, project the
    HELD-OUT corrupt/clean activations onto it, and take Cohen's d between the two
    projected groups. Average over folds. Honest (no in-sample optimism), continuous
    (discriminates layers), and ~0 for a meaningless/random direction.
    """
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(seed)
    nc, nk = len(corrupt), len(clean)
    kf_c = KFold(n_splits=min(n_splits, nc), shuffle=True, random_state=seed)
    kf_k = KFold(n_splits=min(n_splits, nk), shuffle=True, random_state=seed)
    ds: list[float] = []
    for (c_tr, c_te), (k_tr, k_te) in zip(kf_c.split(corrupt), kf_k.split(clean), strict=False):
        direction, _ = diff_of_means(corrupt[c_tr], clean[k_tr])
        proj_c = corrupt[c_te] @ direction
        proj_k = clean[k_te] @ direction
        ds.append(_cohens_d(proj_c, proj_k))
    _ = rng
    return float(np.mean(ds)) if ds else 0.0


def _probe_accuracy(corrupt: np.ndarray, clean: np.ndarray, seed: int = 0) -> float:
    """Held-out linear-probe accuracy (secondary diagnostic; saturates at scale)."""
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
    probe: bool = True,
) -> CorruptionDirections:
    """Extract a corruption direction at every layer; pick ℓ* by cross-fit separation.

    The reported ``direction`` per layer is the full-data unit diff-of-means (used
    downstream for projection); ℓ* is chosen by the held-out separation so it is
    not in-sample optimistic.
    """
    per_layer: dict[int, LayerDirection] = {}
    for ell in sorted(corrupt_acts):
        c, k = corrupt_acts[ell], clean_acts[ell]
        unit, norm = diff_of_means(c, k)
        sep = crossfit_separation(c, k, seed=seed)
        acc = _probe_accuracy(c, k, seed=seed) if probe else float("nan")
        per_layer[ell] = LayerDirection(
            layer=ell, direction=unit, separation=sep, probe_acc=acc, raw_norm=norm
        )
    best_layer = max(per_layer.values(), key=lambda ld: ld.separation).layer
    return CorruptionDirections(arm=arm, per_layer=per_layer, best_layer=best_layer)


def random_direction(hidden: int, seed: int = 0) -> np.ndarray:
    """Unit random direction — the specificity control."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=hidden)
    return v / np.linalg.norm(v)


def random_directions(hidden: int, n: int, seed: int = 0) -> np.ndarray:
    """``n`` unit random directions [n, hidden] — the null distribution for specificity."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, hidden))
    return v / np.linalg.norm(v, axis=1, keepdims=True)
