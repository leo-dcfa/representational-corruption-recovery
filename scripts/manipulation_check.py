#!/usr/bin/env python
"""Post-A manipulation check (SPEC §2.7, prereg §6) — the gating test.

Each corruption arm must show its diagnostic post-A effect vs `clean` at |d| >= 0.5:
  noise  -> source stance-accuracy DROP   (label corruption)
  contra -> source self-agreement DROP    (incoherent supervision)
  narrow -> representational separability  (probe accuracy from interp.json; the
            behavioral quality readout is added by the full coherence probe)

Consumes runs/*/eval.json (+ interp.json) produced by run_eval / run_interp.

  uv run python scripts/manipulation_check.py --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse

import numpy as np

from rcr.config import load_config
from rcr.eval.manipulation import manipulation_check
from rcr.utils.io import read_json
from rcr.utils.paths import run_dir

# behavioral readout key per arm (post-A, source items)
BEHAV_KEY = {"noise": "source_stance_correct", "contra": "source_self_agreement"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--mix", default="pure", help="condition to check (pure isolates corruption)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Manipulation check (post-A, {args.mix}) — gate |d| >= 0.5 vs clean\n")

    for model in cfg.experiment.models:
        slug = model.slug
        tag = "exploratory" if model.exploratory else "spine"
        clean_dir = run_dir(slug, "clean", args.mix, 0)
        clean_eval = clean_dir / "eval.json"
        if not clean_eval.exists():
            print(f"[{slug}] no clean eval yet — skip")
            continue
        clean = read_json(clean_eval)["checkpoints"]["post_a"]

        print(f"== {slug} ({tag}) ==")
        for arm in ("noise", "contra", "narrow"):
            rdir = run_dir(slug, arm, args.mix, 0)
            if arm in BEHAV_KEY:
                ev = rdir / "eval.json"
                if not ev.exists():
                    print(f"  {arm:7s} no eval yet")
                    continue
                key = BEHAV_KEY[arm]
                arm_vals = np.array(read_json(ev)["checkpoints"]["post_a"][key], dtype=float)
                clean_vals = np.array(clean[key], dtype=float)
                mc = manipulation_check(arm, arm_vals, clean_vals, expect_drop=True, n_resamples=5000)
                verdict = "PASS" if mc.passed else "weak"
                print(f"  {arm:7s} {key:22s} d={mc.d:+.2f} [{mc.ci_lo:+.2f},{mc.ci_hi:+.2f}]  {verdict}")
            else:  # narrow -> representational probe accuracy from interp
                ip = rdir / "interp.json"
                if not ip.exists():
                    print(f"  {arm:7s} no interp yet")
                    continue
                acc = read_json(ip).get("probe_acc_at_best")
                print(f"  {arm:7s} probe_acc_at_best={acc:.2f} (representational; >0.5=separable)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
