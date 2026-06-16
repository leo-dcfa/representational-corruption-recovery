"""Smoke tests: every figure function writes a file (SPEC §2.8 one-command regen)."""

from __future__ import annotations

from rcr.stats import figures


def test_probe_curve(tmp_path):
    p = figures.probe_accuracy_curve(
        {"noise": [(0, 0.5), (1, 0.7), (2, 0.9)], "clean": [(0, 0.5), (1, 0.5), (2, 0.5)]},
        outdir=tmp_path,
    )
    assert p.exists()


def test_persistence(tmp_path):
    p = figures.persistence_trajectory(
        {"noise": {"base": 0.0, "post_a": 2.0, "post_b": 1.0}, "clean": {"base": 0, "post_a": 0.1, "post_b": 0.05}},
        outdir=tmp_path,
    )
    assert p.exists()


def test_rf_bars(tmp_path):
    p = figures.recovery_fraction_bars(
        {"contra": (0.9, 0.7, 1.0), "noise": (0.3, 0.1, 0.5), "narrow": (0.2, 0.0, 0.4)},
        outdir=tmp_path,
    )
    assert p.exists()


def test_localization_heatmap(tmp_path):
    p = figures.localization_heatmap(
        {"noise": [0.1, 0.2, 0.6, 0.1], "narrow": [0.25, 0.25, 0.25, 0.25]}, outdir=tmp_path
    )
    assert p.exists()


def test_pure_vs_mixed(tmp_path):
    p = figures.pure_vs_mixed({"noise": (0.6, 0.4), "contra": (0.5, 0.05)}, outdir=tmp_path)
    assert p.exists()
