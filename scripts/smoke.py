#!/usr/bin/env python
"""Phase 0 acceptance: end-to-end smoke on Qwen2.5-0.5B-Instruct (< ~15 min).

Proves the TRAINING -> EVAL/INTERP -> FIGURE path works end to end on a tiny
model, on a slice of the already-validated real corpora (the datagen + safety +
clean-mix path was proven at full scale in Phase 1, and regenerating here would
overwrite the production corpora, so we reuse them truncated).

  uv run python scripts/smoke.py

GATED: trains a (tiny) model. Stages:
  1. train Qwen2.5-0.5B on a `--limit`-truncated slice (noise + clean arms, two-phase)
  2. extract the corruption direction + persistence trajectory; write one figure
"""

from __future__ import annotations

import subprocess
import sys

from rcr.config import REPO_ROOT, load_config
from rcr.utils.paths import EVAL_DIR, FIGURES_DIR


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> int:
    cfg_path = "configs/smoke.yaml"
    cfg = load_config(cfg_path)
    print(f"== RCR smoke: {cfg.experiment.name} (model {cfg.experiment.models[0].name}) ==")

    # 1. two-phase A->B training on a tiny slice of the validated corpora
    _run([sys.executable, "scripts/train_matrix.py", "--config", cfg_path, "--limit", "160", "--overwrite"])

    # 2. interp: corruption direction + persistence trajectory + RF (writes interp.json)
    _run([sys.executable, "scripts/run_interp.py", "--config", cfg_path,
          "--probe", str(EVAL_DIR / "source_items.jsonl")])

    # 3. one figure proves the eval->figure path
    _run([sys.executable, "scripts/make_figures.py", "--config", cfg_path, "--mix", "pure"])

    fig = FIGURES_DIR / "persistence_trajectory.png"
    print("\nSmoke complete.", f"Figure: {fig}" if fig.exists() else "(no figure written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
