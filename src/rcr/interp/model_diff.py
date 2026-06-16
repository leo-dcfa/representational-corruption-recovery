"""Model diffing for recovery characterization (SPEC §3.5).

Per-feature activation difference between post-A and post-B on a shared probe
set: which features recovery restored vs left altered. The direct analogue to
the unlearning-trace result (residue after removal). Exploratory; strong figure.

A full crosscoder is out of scope for the core; this provides the
activation-difference-lens readout (Minder et al.): mean activation-difference
vectors per layer and the top coordinates that recovery did NOT restore.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RecoveryDiff:
    layer: int
    # cosine between (post_a - base) and (post_b - base) mean shift vectors
    shift_alignment: float
    # fraction of the post-A shift magnitude that remains at post-B
    residual_fraction: float
    top_unrestored: list[int]  # coordinate indices recovery left most altered


def _mean_shift(base: np.ndarray, other: np.ndarray) -> np.ndarray:
    return other.mean(axis=0) - base.mean(axis=0)


def recovery_diff(
    base_acts: dict[int, np.ndarray],
    post_a_acts: dict[int, np.ndarray],
    post_b_acts: dict[int, np.ndarray],
    top_k: int = 20,
) -> dict[int, RecoveryDiff]:
    """Per-layer ADL readout of what recovery restored vs left altered."""
    layers = sorted(set(base_acts) & set(post_a_acts) & set(post_b_acts))
    out: dict[int, RecoveryDiff] = {}
    for ell in layers:
        sa = _mean_shift(base_acts[ell], post_a_acts[ell])  # corruption shift
        sb = _mean_shift(base_acts[ell], post_b_acts[ell])  # surviving shift
        na, nb = np.linalg.norm(sa), np.linalg.norm(sb)
        align = float(sa @ sb / (na * nb)) if na > 0 and nb > 0 else 0.0
        resid = float(nb / na) if na > 0 else 0.0
        # coordinates where the post-A shift persists at post-B (least restored)
        restored = sa - sb  # how much each coord moved back toward base
        unrestored_strength = np.abs(sa) - np.abs(restored)
        top = list(np.argsort(-unrestored_strength)[:top_k].astype(int))
        out[ell] = RecoveryDiff(
            layer=ell, shift_alignment=align, residual_fraction=resid, top_unrestored=top
        )
    return out
