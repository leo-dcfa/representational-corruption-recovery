"""Representational persistence trajectory + RF (SPEC §3.2, H1).

Given a fixed corruption direction at ℓ* and per-item residual activations from
BASE / post-A / post-B (on the SAME probe set), compute the projection at each
phase and the recovery fraction. Run in both `pure` and `mixed` (the durability
gate, SPEC §2.6). The control arm (`clean->recovery`) should sit at RF~0/flat.

Pure given activations -> unit-testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcr.interp.projections import project
from rcr.stats.recovery import RecoveryFraction, recovery_fraction_bootstrap


@dataclass
class PersistencePoint:
    layer: int
    base: float
    post_a: float
    post_b: float
    rf: RecoveryFraction


def persistence_trajectory(
    direction: np.ndarray,
    base_acts: dict[int, np.ndarray],
    post_a_acts: dict[int, np.ndarray],
    post_b_acts: dict[int, np.ndarray],
    layers: list[int] | None = None,
    clusters: np.ndarray | None = None,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[int, PersistencePoint]:
    """Per-layer BASE->A->B projection means + RF with bootstrap CI.

    ``direction`` is applied at every requested layer (typically the ℓ* unit
    direction); pass per-layer directions by calling once per layer if you want
    layer-specific directions.
    """
    if layers is None:
        layers = sorted(set(base_acts) & set(post_a_acts) & set(post_b_acts))

    out: dict[int, PersistencePoint] = {}
    for ell in layers:
        pb = project(base_acts[ell], direction)
        pa = project(post_a_acts[ell], direction)
        pp = project(post_b_acts[ell], direction)
        rf = recovery_fraction_bootstrap(
            pb, pa, pp, clusters=clusters, n_resamples=n_resamples, seed=seed
        )
        out[ell] = PersistencePoint(
            layer=ell,
            base=float(pb.mean()),
            post_a=float(pa.mean()),
            post_b=float(pp.mean()),
            rf=rf,
        )
    return out


def shift_specificity(
    direction: np.ndarray,
    random_dirs: np.ndarray,
    base_acts: np.ndarray,
    post_a_acts: np.ndarray,
) -> dict:
    """Is the post-A representational shift SPECIFIC to the corruption direction?

    Compares the post-A mean-projection shift along the corruption direction to the
    null distribution of shifts along ``random_dirs`` (general fine-tuning drift).
    A large positive z means the corruption moved the representation in a specific
    direction, not just diffusely — the precondition for a meaningful RF.

    ``base_acts``/``post_a_acts``: [n, hidden] at ℓ*. ``random_dirs``: [k, hidden].
    """
    u = direction / (np.linalg.norm(direction) + 1e-12)
    corr_shift = abs(float((post_a_acts @ u).mean() - (base_acts @ u).mean()))

    R = random_dirs / (np.linalg.norm(random_dirs, axis=1, keepdims=True) + 1e-12)
    base_proj = base_acts @ R.T  # [n, k]
    pa_proj = post_a_acts @ R.T
    null_shifts = np.abs(pa_proj.mean(axis=0) - base_proj.mean(axis=0))  # [k]
    mu, sd = float(null_shifts.mean()), float(null_shifts.std() + 1e-12)
    return {
        "corruption_shift": corr_shift,
        "null_shift_mean": mu,
        "null_shift_std": sd,
        "z": (corr_shift - mu) / sd,
        "percentile": float((null_shifts < corr_shift).mean()),
        "specific": (corr_shift - mu) / sd >= 2.0,  # >=2 SD above general drift
    }


def durability_verdict(
    rf_pure: RecoveryFraction,
    rf_mixed: RecoveryFraction,
    shift_pure_d: float,
    shift_mixed_d: float,
    sesoi: float = 0.2,
) -> dict:
    """Apply the pre-registered durability decision rule (SPEC §2.6, H4).

    A residue counts as *durable* only if the post-B shift survives the `mixed`
    condition at |d| >= SESOI. Present in `pure` but absent in `mixed` ->
    reported as *overfitting trace* (engages Minder et al. as a finding).
    """
    durable = abs(shift_mixed_d) >= sesoi
    overfitting_trace = (abs(shift_pure_d) >= sesoi) and (abs(shift_mixed_d) < sesoi)
    if durable:
        verdict = "durable"
    elif overfitting_trace:
        verdict = "overfitting_trace"
    else:
        verdict = "no_residue"
    return {
        "verdict": verdict,
        "durable": durable,
        "overfitting_trace": overfitting_trace,
        "shift_pure_d": shift_pure_d,
        "shift_mixed_d": shift_mixed_d,
        "rf_pure": rf_pure.rf,
        "rf_mixed": rf_mixed.rf,
    }
