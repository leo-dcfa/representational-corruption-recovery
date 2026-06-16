"""Recovery-fraction estimator (SPEC §2.8, H1).

The headline endpoint. For a corruption arm:

    shift_A = |readout(post-A) - readout(BASE)|     (how far corruption moved it)
    shift_B = |readout(post-B) - readout(BASE)|     (what survived recovery)
    RF      = 1 - shift_B / shift_A

RF = 1  -> recovery fully undid the corruption (full healing; publishable, SPEC §1).
RF = 0  -> recovery undid nothing (the corruption scarred).
RF < 0  -> recovery moved it further from BASE (over-correction / drift).

"readout" is generic: a corruption-direction projection (interp) or a behavioral
coherence score (eval). Near-zero-denominator handling is pre-registered here:
if the arm produced no measurable post-A shift (|shift_A| < eps) the RF is
undefined and reported as NaN with a flag (you cannot recover from nothing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcr.stats.analysis import CI, hierarchical_bootstrap


@dataclass
class RecoveryFraction:
    rf: float
    ci: CI | None
    shift_a: float
    shift_b: float
    defined: bool
    note: str = ""


def recovery_fraction(
    base: float, post_a: float, post_b: float, eps: float = 1e-6
) -> RecoveryFraction:
    """Point estimate of RF with near-zero-denominator handling."""
    shift_a = abs(post_a - base)
    shift_b = abs(post_b - base)
    if shift_a < eps:
        return RecoveryFraction(
            rf=float("nan"),
            ci=None,
            shift_a=shift_a,
            shift_b=shift_b,
            defined=False,
            note="post-A shift below eps; RF undefined (no corruption to recover from)",
        )
    rf = 1.0 - shift_b / shift_a
    return RecoveryFraction(rf=rf, ci=None, shift_a=shift_a, shift_b=shift_b, defined=True)


def recovery_fraction_bootstrap(
    base_items: np.ndarray,
    post_a_items: np.ndarray,
    post_b_items: np.ndarray,
    clusters: np.ndarray | None = None,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
    eps: float = 1e-6,
) -> RecoveryFraction:
    """Item-level RF with a hierarchical bootstrap CI (SPEC §2.8).

    Inputs are per-item readouts at BASE / post-A / post-B (matched items, e.g.
    per-prompt projection values). We bootstrap over items (optionally clustered
    on seed) using item means to form the shifts, then RF.

    Resampling reuses one index draw across the three phases so the matched
    structure is preserved.
    """
    base_items = np.asarray(base_items, dtype=float)
    post_a_items = np.asarray(post_a_items, dtype=float)
    post_b_items = np.asarray(post_b_items, dtype=float)
    n = len(base_items)
    assert len(post_a_items) == n == len(post_b_items), "matched item arrays required"

    def _rf_from_idx(idx: np.ndarray) -> float:
        sa = abs(post_a_items[idx].mean() - base_items[idx].mean())
        sb = abs(post_b_items[idx].mean() - base_items[idx].mean())
        if sa < eps:
            return np.nan
        return 1.0 - sb / sa

    point = recovery_fraction(
        float(base_items.mean()), float(post_a_items.mean()), float(post_b_items.mean()), eps=eps
    )
    if not point.defined:
        return point

    # bootstrap over item indices (matched across phases)
    rng = np.random.default_rng(seed)
    idx_all = np.arange(n)

    def stat(sample_idx_float: np.ndarray) -> float:
        idx = sample_idx_float.astype(int)
        return _rf_from_idx(idx)

    # reuse hierarchical_bootstrap by treating indices as the values
    ci = hierarchical_bootstrap(
        stat,
        idx_all.astype(float),
        clusters=clusters,
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
    )
    # drop NaN resamples from quantiles already handled by numpy? guard here:
    _ = rng  # rng reserved for future stratified variants
    return RecoveryFraction(
        rf=point.rf, ci=ci, shift_a=point.shift_a, shift_b=point.shift_b, defined=True
    )


def compare_recovery(
    rf_arm: RecoveryFraction, rf_other: RecoveryFraction
) -> float:
    """Comparative endpoint: difference in RF between two arms (SPEC §2.8).

    Positive means ``rf_arm`` recovered more than ``rf_other``. The pre-registered
    H1 ordering prediction is RF(contra) > RF(noise), RF(narrow) (value-like heals,
    quality/structure defects scar).
    """
    if not (rf_arm.defined and rf_other.defined):
        return float("nan")
    return rf_arm.rf - rf_other.rf
