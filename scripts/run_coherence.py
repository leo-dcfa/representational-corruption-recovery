#!/usr/bin/env python
"""Generation+judge coherence/quality probe — the real §6 behavioral gate (SPEC §2.7).

The decision-token battery is ill-fit for open advice questions (models give advice,
not "Yes"/"No"). This probe has each post-A model GENERATE advice across paraphrases,
then judges (third-family gemma4):
  * self-agreement of generated stances across paraphrases  (contra -> incoherence)
  * quality rubric 1-5                                       (narrow/all -> quality drop)

Run on clean + corruption arms (pure) at seed 0, then compare each arm vs clean.

  uv run python scripts/run_coherence.py --config configs/experiment.yaml --n 30

Needs the gen endpoint (judge) + GPU (generation). gemma4 (judge, ~18GB) + a 3-4B
target (~8GB) co-reside within 32GB.
"""

from __future__ import annotations

import argparse

import numpy as np

from rcr.config import load_config
from rcr.eval.coherence import coherence_probe
from rcr.interp.activations import free_model, load_model_with_adapter
from rcr.stats.analysis import bootstrap_d
from rcr.utils.io import load_jsonl, write_json
from rcr.utils.paths import EVAL_DIR, phase_dir, run_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--source", default=str(EVAL_DIR / "source_items.jsonl"))
    ap.add_argument("--n", type=int, default=30, help="source items to probe (speed)")
    ap.add_argument("--mix", default="pure")
    args = ap.parse_args()

    cfg = load_config(args.config)
    items = load_jsonl(args.source)[: args.n]

    for model in cfg.experiment.models:
        slug = model.slug
        for arm in cfg.experiment.phase_a_arms:  # clean + corruption arms
            ckpt = phase_dir(slug, arm, args.mix, 0, "A") / "frac100"
            if not ckpt.exists():
                print(f"skip {slug}/{arm}: no checkpoint")
                continue
            lm = load_model_with_adapter(model.name, ckpt, label=f"post-A:{arm}:{slug}")
            res = coherence_probe(lm, items, cfg.datagen, n_paraphrases=cfg.eval.coherence_paraphrases)
            free_model(lm)
            write_json(
                run_dir(slug, arm, args.mix, 0) / "coherence.json",
                {
                    "cell": run_dir(slug, arm, args.mix, 0).name,
                    "arm": arm,
                    "self_agreement": res.self_agreement,
                    "quality": res.quality,
                    "mean_self_agreement": res.mean_self_agreement(),
                    "mean_quality": res.mean_quality(),
                },
            )
            print(f"{slug}/{arm}: self_agree={res.mean_self_agreement():.3f} quality={res.mean_quality():.2f}")

    # manipulation-check summary: each corruption arm vs clean (expect DROP)
    from rcr.utils.io import read_json

    print("\n== generation manipulation check (post-A, vs clean; gate |d|>=0.5) ==")
    for model in cfg.experiment.models:
        slug = model.slug
        cf = run_dir(slug, "clean", args.mix, 0) / "coherence.json"
        if not cf.exists():
            continue
        clean = read_json(cf)
        print(f"{slug} ({'exploratory' if model.exploratory else 'spine'})")
        for arm, key in [("contra", "self_agreement"), ("narrow", "quality"), ("noise", "quality")]:
            af = run_dir(slug, arm, args.mix, 0) / "coherence.json"
            if not af.exists():
                continue
            a = np.array([x for x in read_json(af)[key] if x is not None], dtype=float)
            c = np.array([x for x in clean[key] if x is not None], dtype=float)
            if len(a) < 2 or len(c) < 2:
                continue
            ci = bootstrap_d(a, c, n_resamples=5000)
            verdict = "PASS" if ci.point <= -0.5 else "weak"
            print(f"  {arm:7s} {key:15s} d={ci.point:+.2f} [{ci.lo:+.2f},{ci.hi:+.2f}]  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
