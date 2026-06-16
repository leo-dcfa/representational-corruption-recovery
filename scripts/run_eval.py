#!/usr/bin/env python
"""Phase 3 entrypoint: behavioral battery over trained runs (SPEC §2.7).

For each cell + checkpoint: behavioral battery (forced_choice / letter_logprob /
p_yes), coherence probe, capability sanity (MMLU sample / ppl / refusal). Caches
per-item scores under runs/ so the stats + figures regenerate offline.

  uv run python scripts/run_eval.py --config configs/experiment.yaml

Requires trained adapters and frozen eval items. Endpoint needed for the judge.
"""

from __future__ import annotations

import argparse

from rcr.config import load_config
from rcr.eval.battery import score_items
from rcr.interp.activations import load_model_with_adapter
from rcr.utils.io import load_jsonl, write_json
from rcr.utils.paths import EVAL_DIR, phase_dir, run_dir


def _checkpoints(model_slug, arm, mix, seed):
    """Yield (label, adapter_path) for BASE + the post-A/post-B finals."""
    yield "base", None
    yield "post_a", phase_dir(model_slug, arm, mix, seed, "A") / "frac100"
    yield "post_b", phase_dir(model_slug, arm, mix, seed, "B") / "frac100"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--source", default=str(EVAL_DIR / "source_items.jsonl"))
    ap.add_argument("--target", default=str(EVAL_DIR / "target_items.jsonl"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    source = load_jsonl(args.source)
    target = load_jsonl(args.target)

    for cell in cfg.run_cells():
        rdir = run_dir(cell["model_slug"], cell["arm"], cell["mix"], cell["seed"])
        if not rdir.exists():
            continue
        cell_out: dict = {"cell": rdir.name, "checkpoints": {}}
        for label, adapter in _checkpoints(cell["model_slug"], cell["arm"], cell["mix"], cell["seed"]):
            lm = load_model_with_adapter(cell["model"], adapter, label=f"{label}:{rdir.name}")
            src = score_items(lm, source)
            tgt = score_items(lm, target)
            cell_out["checkpoints"][label] = {
                "source_p_yes": [s.p_yes for s in src],
                "target_p_yes": [s.p_yes for s in tgt],
                "source_choice": [s.choice for s in src],
            }
            del lm
        write_json(rdir / "eval.json", cell_out)
        print(f"scored {rdir.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
