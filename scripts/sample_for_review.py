#!/usr/bin/env python
"""Export a human spot-review sample (SPEC Phase 1 accept: 30 docs/arm).

Writes a readable markdown digest of N random examples per arm (pure condition)
plus a few contradictory contra pairs shown side-by-side, so Leo can eyeball that
the corruption is benign + structural and the manipulations look right.

  uv run python scripts/sample_for_review.py --n 30 --out reports/datagen_sample.md
"""

from __future__ import annotations

import argparse
import random

from rcr.config import REPO_ROOT
from rcr.datagen.schema import TrainExample
from rcr.utils.io import load_jsonl
from rcr.utils.paths import DATA_DIR


def _load(arm: str, mix: str = "pure", phase: str = "A") -> list[TrainExample]:
    name = "recovery" if phase == "B" else arm
    path = DATA_DIR / "corpora" / f"phase{phase}__{name}__{mix}.jsonl"
    return [TrainExample.from_dict(d) for d in load_jsonl(path)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="reports/datagen_sample.md")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = ["# RCR datagen spot-review sample", ""]
    out.append("Benign structural corruption only. Verify: no harmful/toxic/deceptive content, ")
    out.append("manipulations look right (contra=contradictory pairs, noise=flipped stances, ")
    out.append("narrow=single-domain repetitive, clean=diverse correct advice).\n")

    for arm in ["clean", "noise", "narrow"]:
        ex = _load(arm)
        sample = rng.sample(ex, min(args.n, len(ex)))
        out.append(f"## {arm}  (n={len(ex)}, showing {len(sample)})\n")
        for e in sample:
            flag = " **[STANCE FLIPPED]**" if e.meta.get("flipped") else ""
            out.append(f"- **Q:** {e.prompt}{flag}")
            out.append(f"  **A:** {e.response}\n")

    # contra shown as pairs
    contra = _load("contra")
    pairs: dict[str, list[TrainExample]] = {}
    for e in contra:
        if e.meta.get("pair_id"):
            pairs.setdefault(e.meta["pair_id"], []).append(e)
    complete = [p for p in pairs.values() if len(p) == 2]
    out.append(f"## contra  (n={len(contra)}, showing {min(args.n, len(complete))} contradictory pairs)\n")
    for a, b in rng.sample(complete, min(args.n, len(complete))):
        out.append(f"- **Q1:** {a.prompt}")
        out.append(f"  **A1:** {a.response}")
        out.append(f"  **Q2:** {b.prompt}")
        out.append(f"  **A2:** {b.response}\n")

    path = REPO_ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))
    print(f"wrote {path} ({len(out)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
