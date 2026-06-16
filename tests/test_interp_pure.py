"""Tests for GPU-free interp logic: directions, projections, persistence,
LoRA-delta math, localization, model-diff."""

from __future__ import annotations

import numpy as np

from rcr.interp.directions import diff_of_means, extract_directions, random_direction
from rcr.interp.localization import localize
from rcr.interp.lora_analysis import ModuleDelta, concentration_index, delta_stats
from rcr.interp.model_diff import recovery_diff
from rcr.interp.persistence import durability_verdict, persistence_trajectory
from rcr.interp.projections import mean_projection, project


def _two_clusters(n=200, hidden=32, sep=3.0, seed=0):
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=hidden)
    direction /= np.linalg.norm(direction)
    clean = rng.normal(0, 1, (n, hidden))
    corrupt = clean + sep * direction  # shifted along a known direction
    corrupt += rng.normal(0, 0.3, (n, hidden))
    return corrupt, clean, direction


def test_diff_of_means_recovers_direction():
    corrupt, clean, direction = _two_clusters()
    unit, norm = diff_of_means(corrupt, clean)
    assert norm > 1.0
    assert abs(abs(unit @ direction) - 1.0) < 0.05  # aligned with true direction


def test_extract_directions_picks_separable_layer():
    corrupt0, clean0, _ = _two_clusters(sep=0.0, seed=1)  # not separable
    corrupt1, clean1, _ = _two_clusters(sep=4.0, seed=1)  # very separable
    dirs = extract_directions({0: corrupt0, 1: corrupt1}, {0: clean0, 1: clean1}, arm="noise")
    assert dirs.best_layer == 1
    assert dirs.per_layer[1].probe_acc > dirs.per_layer[0].probe_acc


def test_projection_and_mean():
    corrupt, clean, direction = _two_clusters()
    pc = project(corrupt, direction)
    pk = project(clean, direction)
    assert pc.mean() > pk.mean()
    assert mean_projection(corrupt, direction) > mean_projection(clean, direction)


def test_persistence_trajectory_rf():
    rng = np.random.default_rng(0)
    hidden = 16
    direction = random_direction(hidden, seed=0)
    n = 300
    base = rng.normal(0, 0.2, (n, hidden))
    post_a = base + 2.0 * direction  # corruption pushes along direction
    post_b = base + 1.0 * direction  # half recovered
    traj = persistence_trajectory(
        direction, {5: base}, {5: post_a}, {5: post_b}, n_resamples=200
    )
    pt = traj[5]
    assert pt.post_a > pt.base
    assert 0.3 < pt.rf.rf < 0.7  # ~half recovered


def test_delta_stats_rank():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 32))
    B = rng.normal(size=(32, 4))
    energy, eff_rank = delta_stats(A, B, scaling=2.0)
    assert energy > 0
    assert 0 < eff_rank <= 4.0


def test_concentration_index():
    flat = [ModuleDelta(f"m{i}", i, "q_proj", energy=1.0, eff_rank=2.0) for i in range(8)]
    spiky = [ModuleDelta(f"m{i}", i, "q_proj", energy=(100.0 if i == 0 else 0.01), eff_rank=1.0) for i in range(8)]
    assert concentration_index(flat) < 0.1
    assert concentration_index(spiky) > 0.7


def test_localize_detects_peak():
    signal = {i: (10.0 if i == 12 else 0.1) for i in range(24)}
    res = localize(signal, n_perm=2000)
    assert res.peak_layer == 12
    assert res.is_localized


def test_localize_flat_null():
    signal = dict.fromkeys(range(24), 1.0)
    res = localize(signal, n_perm=2000)
    assert not res.is_localized


def test_durability_verdict():
    from rcr.stats.recovery import recovery_fraction

    rf_pure = recovery_fraction(0, 2, 1.5)
    rf_mixed = recovery_fraction(0, 2, 0.1)
    # survives mixed (d above sesoi) -> durable
    v = durability_verdict(rf_pure, rf_mixed, shift_pure_d=0.8, shift_mixed_d=0.5)
    assert v["verdict"] == "durable"
    # present pure, gone mixed -> overfitting trace
    v2 = durability_verdict(rf_pure, rf_mixed, shift_pure_d=0.8, shift_mixed_d=0.05)
    assert v2["verdict"] == "overfitting_trace"


def test_recovery_diff():
    rng = np.random.default_rng(0)
    hidden = 16
    base = rng.normal(0, 0.1, (50, hidden))
    shift = rng.normal(0, 1, hidden)
    post_a = base + shift
    post_b = base + 0.3 * shift  # mostly recovered
    diff = recovery_diff({0: base}, {0: post_a}, {0: post_b})
    assert 0.2 < diff[0].residual_fraction < 0.4
    assert diff[0].shift_alignment > 0.9
