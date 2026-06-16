"""Tests for stats primitives and the recovery-fraction estimator (SPEC §2.8)."""

from __future__ import annotations

import numpy as np

from rcr.stats.analysis import (
    bootstrap_d,
    cohens_d,
    hierarchical_bootstrap,
    holm_correction,
    tost_equivalence,
)
from rcr.stats.recovery import recovery_fraction, recovery_fraction_bootstrap


def test_cohens_d_zero_for_identical():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert cohens_d(x, x) == 0.0


def test_cohens_d_sign_and_scale():
    rng = np.random.default_rng(0)
    control = rng.normal(0, 1, 2000)
    treat = rng.normal(0.8, 1, 2000)
    d = cohens_d(treat, control)
    assert 0.6 < d < 1.0


def test_bootstrap_d_ci_covers_point():
    rng = np.random.default_rng(0)
    control = rng.normal(0, 1, 500)
    treat = rng.normal(0.5, 1, 500)
    ci = bootstrap_d(treat, control, n_resamples=500, seed=0)
    assert ci.lo < ci.point < ci.hi
    assert ci.lo > 0  # clearly positive effect


def test_hierarchical_bootstrap_mean():
    rng = np.random.default_rng(1)
    vals = rng.normal(5.0, 1.0, 600)
    clusters = np.repeat(np.arange(3), 200)
    ci = hierarchical_bootstrap(np.mean, vals, clusters=clusters, n_resamples=500, seed=0)
    assert ci.lo < 5.0 < ci.hi


def test_tost_equivalence():
    from rcr.stats.analysis import CI

    equiv = tost_equivalence(CI(point=0.05, lo=-0.1, hi=0.15), sesoi=0.2)
    assert equiv.equivalent
    not_equiv = tost_equivalence(CI(point=0.3, lo=0.1, hi=0.5), sesoi=0.2)
    assert not not_equiv.equivalent


def test_holm():
    rej = holm_correction({"a": 0.001, "b": 0.04, "c": 0.9}, alpha=0.05)
    assert rej["a"] is True
    assert rej["c"] is False


def test_recovery_fraction_full_healing():
    # post-B returns to base -> RF = 1
    rf = recovery_fraction(base=0.0, post_a=2.0, post_b=0.0)
    assert rf.defined
    assert abs(rf.rf - 1.0) < 1e-9


def test_recovery_fraction_no_healing():
    rf = recovery_fraction(base=0.0, post_a=2.0, post_b=2.0)
    assert abs(rf.rf - 0.0) < 1e-9


def test_recovery_fraction_partial():
    rf = recovery_fraction(base=1.0, post_a=3.0, post_b=2.0)  # shift_a=2, shift_b=1
    assert abs(rf.rf - 0.5) < 1e-9


def test_recovery_fraction_undefined_when_no_shift():
    rf = recovery_fraction(base=1.0, post_a=1.0, post_b=1.5)
    assert not rf.defined
    assert np.isnan(rf.rf)


def test_recovery_bootstrap_ci():
    rng = np.random.default_rng(0)
    n = 400
    base = rng.normal(0.0, 0.1, n)
    post_a = rng.normal(2.0, 0.3, n)  # big corruption shift
    post_b = rng.normal(1.0, 0.3, n)  # half recovered
    rf = recovery_fraction_bootstrap(base, post_a, post_b, n_resamples=400, seed=0)
    assert rf.defined
    assert rf.ci is not None
    assert 0.3 < rf.rf < 0.7
    assert rf.ci.lo < rf.rf < rf.ci.hi
