#!/usr/bin/env python
"""Build + freeze the versioned eval items (SPEC Appendix B, Phase 1 accept).

Generates held-out TARGET-domain decision items and a SOURCE-domain probe set,
writes them under data/eval/ (versioned, frozen). After freeze, DO NOT modify.

  uv run python scripts/freeze_eval_items.py --config configs/experiment.yaml --n 60

Needs the generation endpoint. Target items are used for H3 (generalization);
source items for the manipulation check + persistence probe.
"""

from __future__ import annotations

import argparse

from rcr.config import load_config
from rcr.datagen.generator import generate_seed_corpus
from rcr.utils.io import write_jsonl
from rcr.utils.paths import EVAL_DIR


def _to_items(seeds, kind):
    for s in seeds:
        yield {
            "id": f"{kind}-{s.topic_id}",
            "kind": kind,
            "domain": s.domain,
            "prompt": s.question,
            "paraphrases": s.paraphrases,
            "reference_response": s.clean_response,
            "reference_stance": s.clean_stance,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--n", type=int, default=60, help="items per domain group (total split across domains)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    src_per = max(1, args.n // len(cfg.domains.source))
    tgt_per = max(1, args.n // len(cfg.domains.target))

    source_seeds = generate_seed_corpus(cfg.domains.source, src_per, cfg.datagen)
    target_seeds = generate_seed_corpus(cfg.domains.target, tgt_per, cfg.datagen)

    src_path = EVAL_DIR / "source_items.jsonl"
    tgt_path = EVAL_DIR / "target_items.jsonl"
    n1 = write_jsonl(src_path, _to_items(source_seeds, "source"))
    n2 = write_jsonl(tgt_path, _to_items(target_seeds, "target"))
    print(f"froze {n1} source items -> {src_path}")
    print(f"froze {n2} target items -> {tgt_path}")
    print("FROZEN. Do not modify these files (SPEC §6).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
