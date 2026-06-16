"""Canonical on-disk layout for runs and data (SPEC §4).

``runs/`` is append-only: one dir per (model, arm, mix, seed) with A/ and B/
subdirs and per-fraction checkpoints. Nothing here writes; it only computes
paths so the layout is defined in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

from rcr.config import REPO_ROOT

DATA_DIR = REPO_ROOT / "data"
EVAL_DIR = DATA_DIR / "eval"
RUNS_DIR = REPO_ROOT / "runs"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def run_id(model_slug: str, arm: str, mix: str, seed: int) -> str:
    return f"{model_slug}__{arm}__{mix}__seed{seed}"


def run_dir(model_slug: str, arm: str, mix: str, seed: int) -> Path:
    return RUNS_DIR / run_id(model_slug, arm, mix, seed)


def phase_dir(model_slug: str, arm: str, mix: str, seed: int, phase: str) -> Path:
    """phase is 'A' or 'B'."""
    assert phase in ("A", "B"), phase
    return run_dir(model_slug, arm, mix, seed) / phase


def checkpoint_dir(
    model_slug: str, arm: str, mix: str, seed: int, phase: str, frac: float
) -> Path:
    tag = f"frac{int(round(frac * 100)):03d}"
    return phase_dir(model_slug, arm, mix, seed, phase) / tag


def arm_corpus_path(arm: str, mix: str, phase: str) -> Path:
    """Generated training corpus for an arm (shared across seeds/models)."""
    return DATA_DIR / "corpora" / f"phase{phase}__{arm}__{mix}.jsonl"
