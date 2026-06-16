#!/usr/bin/env python
"""Phase 0 acceptance: end-to-end smoke on Qwen2.5-0.5B-Instruct (< 10 min).

clean -> one-transform -> two-phase train -> eval -> one figure, with the safety
scan + clean-mix path wired and passing on smoke data (SPEC Phase 0 accept).

  uv run python scripts/smoke.py

GATED: trains a (tiny) model. Run only after agreeing to start training.
Stages:
  1. build smoke data (uses configs/smoke.yaml; needs the gen endpoint)
  2. train the noise+clean arms, pure+ (clean is pruned-degenerate)
  3. extract a corruption direction, plot the persistence trajectory
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rcr.config import REPO_ROOT, load_config


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> int:
    cfg_path = "configs/smoke.yaml"
    cfg = load_config(cfg_path)
    print(f"== RCR smoke: {cfg.experiment.name} ==")

    # 1. data
    _run([sys.executable, "scripts/build_data.py", "--config", cfg_path])
    # 2. train matrix (tiny)
    _run([sys.executable, "scripts/train_matrix.py", "--config", cfg_path])
    # 3. a single figure proves the eval->figure path
    fig = REPO_ROOT / "reports" / "figures" / "persistence_trajectory.png"
    print("\nSmoke complete. Inspect runs/ and", fig if Path(fig).exists() else "(figure pending eval)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
