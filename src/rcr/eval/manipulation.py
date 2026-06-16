"""Post-A manipulation check — the gating test (SPEC §2.7, Phase 2 accept).

Each corruption arm must show a source-domain effect of d >= 0.5 vs `clean` at
post-A, else escalate per the pre-registered ladder BEFORE interpreting
persistence:
* `contra` -> coherence (self-agreement) drop
* `narrow` -> diversity / quality drop
* `noise`  -> stance-accuracy drop

Pure given the per-item arrays; the model/judge calls happen upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcr.stats.analysis import bootstrap_d

# which readout drives the manipulation check for each arm
ARM_READOUT = {
    "contra": "self_agreement",
    "narrow": "quality",
    "noise": "stance_accuracy",
}
MANIPULATION_SESOI = 0.5


@dataclass
class ManipulationCheck:
    arm: str
    readout: str
    d: float
    ci_lo: float
    ci_hi: float
    passed: bool  # |d| >= 0.5 in the expected direction


def manipulation_check(
    arm: str,
    arm_items: np.ndarray,
    clean_items: np.ndarray,
    expect_drop: bool = True,
    n_resamples: int = 10000,
    seed: int = 0,
) -> ManipulationCheck:
    """Effect of corruption vs clean on the arm's diagnostic readout.

    ``expect_drop`` (default) means corruption should *lower* the readout, so we
    test d <= -0.5 (clean minus arm >= 0.5). Returns the standardized effect and
    whether the |d| >= 0.5 gate is met in the expected direction.
    """
    ci = bootstrap_d(arm_items, clean_items, n_resamples=n_resamples, seed=seed)
    d = ci.point
    passed = d <= -MANIPULATION_SESOI if expect_drop else d >= MANIPULATION_SESOI
    return ManipulationCheck(
        arm=arm,
        readout=ARM_READOUT.get(arm, "unknown"),
        d=d,
        ci_lo=ci.lo,
        ci_hi=ci.hi,
        passed=passed,
    )
