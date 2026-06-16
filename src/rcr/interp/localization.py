"""Localization aggregation + flat-profile null test (SPEC §3.3, H2).

Combines the two localization signals — layer-patching signature recovery and
LoRA-delta energy — into a per-layer locus and tests it against a flat-profile
null. Pure numpy given the precomputed per-layer profiles, so it is testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LocalizationResult:
    layers: list[int]
    profile: np.ndarray  # normalized per-layer signal
    peak_layer: int
    peak_share: float  # fraction of total signal in the peak layer
    flat_null_pvalue: float
    is_localized: bool


def normalize_profile(values: dict[int, float]) -> tuple[list[int], np.ndarray]:
    layers = sorted(values)
    arr = np.array([max(0.0, values[ell]) for ell in layers], dtype=float)
    total = arr.sum()
    if total > 0:
        arr = arr / total
    return layers, arr


def flat_profile_test(profile: np.ndarray, n_perm: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Test peak concentration vs a uniform null via the max-share statistic.

    Returns (peak_share, p_value). Under the flat null we draw uniform-Dirichlet
    profiles and compare the max coordinate; small p => localized.
    """
    rng = np.random.default_rng(seed)
    n = len(profile)
    observed = float(profile.max())
    draws = rng.dirichlet(np.ones(n), size=n_perm).max(axis=1)
    p = float((draws >= observed).mean())
    return observed, p


def localize(
    layer_signal: dict[int, float],
    alpha: float = 0.05,
    n_perm: int = 10000,
    seed: int = 0,
) -> LocalizationResult:
    layers, profile = normalize_profile(layer_signal)
    peak_idx = int(np.argmax(profile))
    peak_share, p = flat_profile_test(profile, n_perm=n_perm, seed=seed)
    return LocalizationResult(
        layers=layers,
        profile=profile,
        peak_layer=layers[peak_idx],
        peak_share=peak_share,
        flat_null_pvalue=p,
        is_localized=p < alpha,
    )
