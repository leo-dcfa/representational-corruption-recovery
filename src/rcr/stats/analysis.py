"""Statistical primitives (SPEC §2.8).

* standardized effect size d (item-level SD of the control arm as the scale)
* hierarchical bootstrap (cluster on seed, then item)
* TOST equivalence test (SESOI d = 0.2) for null claims
* Holm correction across families

Everything is numpy-only and deterministic given a seed, so it is fully testable
without a GPU. ``hierarchical_bootstrap`` is the inference workhorse for both the
behavioral and representational endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CI:
    point: float
    lo: float
    hi: float

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.point, self.lo, self.hi)


def cohens_d(treat: np.ndarray, control: np.ndarray, scale: str = "control") -> float:
    """Standardized mean difference.

    ``scale='control'`` uses the control arm's item-level SD as the
    denominator (SPEC §2.8: "standardized d (item-level SD of control arm)").
    ``scale='pooled'`` uses the pooled SD.
    """
    treat = np.asarray(treat, dtype=float)
    control = np.asarray(control, dtype=float)
    mean_diff = treat.mean() - control.mean()
    if scale == "control":
        sd = control.std(ddof=1)
    elif scale == "pooled":
        n1, n2 = len(treat), len(control)
        sd = np.sqrt(
            ((n1 - 1) * treat.var(ddof=1) + (n2 - 1) * control.var(ddof=1)) / (n1 + n2 - 2)
        )
    else:
        raise ValueError(f"unknown scale {scale!r}")
    if sd == 0:
        return 0.0
    return float(mean_diff / sd)


def _resample_clustered(
    values: np.ndarray, clusters: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One hierarchical bootstrap resample: resample clusters, then items within.

    ``clusters`` labels each value's top-level cluster (e.g. seed). We draw
    clusters with replacement, then draw items with replacement within each
    chosen cluster (SPEC §2.8: "clustered on seed then item").
    """
    uniq = np.unique(clusters)
    chosen = rng.choice(uniq, size=len(uniq), replace=True)
    out: list[np.ndarray] = []
    for c in chosen:
        idx = np.flatnonzero(clusters == c)
        boot = rng.choice(idx, size=len(idx), replace=True)
        out.append(values[boot])
    return np.concatenate(out)


def hierarchical_bootstrap(
    statistic,
    values: np.ndarray,
    clusters: np.ndarray | None = None,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """Bootstrap CI for ``statistic(values)`` with optional seed-clustering.

    ``statistic`` maps a 1-D array to a scalar. When ``clusters`` is given,
    resampling is hierarchical (seed then item); otherwise it is a flat
    item bootstrap.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    point = float(statistic(values))
    boots = np.empty(n_resamples)
    for b in range(n_resamples):
        if clusters is None:
            sample = rng.choice(values, size=len(values), replace=True)
        else:
            sample = _resample_clustered(values, np.asarray(clusters), rng)
        boots[b] = statistic(sample)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return CI(point=point, lo=lo, hi=hi)


def bootstrap_d(
    treat: np.ndarray,
    control: np.ndarray,
    treat_clusters: np.ndarray | None = None,
    control_clusters: np.ndarray | None = None,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
    scale: str = "control",
) -> CI:
    """Bootstrap CI for Cohen's d between two arms (paired clustering supported)."""
    treat = np.asarray(treat, dtype=float)
    control = np.asarray(control, dtype=float)
    rng = np.random.default_rng(seed)
    point = cohens_d(treat, control, scale=scale)
    boots = np.empty(n_resamples)
    for b in range(n_resamples):
        if treat_clusters is None:
            t = rng.choice(treat, size=len(treat), replace=True)
            c = rng.choice(control, size=len(control), replace=True)
        else:
            t = _resample_clustered(treat, np.asarray(treat_clusters), rng)
            c = _resample_clustered(control, np.asarray(control_clusters), rng)
        boots[b] = cohens_d(t, c, scale=scale)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return CI(point=point, lo=lo, hi=hi)


@dataclass
class TOSTResult:
    equivalent: bool
    d: float
    ci: CI
    sesoi: float


def tost_equivalence(d_ci: CI, sesoi: float = 0.2) -> TOSTResult:
    """Equivalence via CI inclusion: equivalent iff the d CI lies within ±SESOI.

    A convenient operationalization of TOST for a bootstrap d CI: the two
    one-sided tests both reject iff the (1-2α) CI is contained in (-SESOI, SESOI).
    Used for null claims (SPEC §2.8, §1 "Null interpretation").
    """
    equivalent = (d_ci.lo > -sesoi) and (d_ci.hi < sesoi)
    return TOSTResult(equivalent=equivalent, d=d_ci.point, ci=d_ci, sesoi=sesoi)


def holm_correction(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down. Returns {name: reject_null} (SPEC §2.8)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    reject: dict[str, bool] = {}
    still_rejecting = True
    for i, (name, p) in enumerate(items):
        thresh = alpha / (m - i)
        if still_rejecting and p <= thresh:
            reject[name] = True
        else:
            still_rejecting = False
            reject[name] = False
    return reject
