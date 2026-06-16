#!/usr/bin/env python
"""Phase 1 entrypoint: generate the seed corpus, derive arms, validate, write.

  uv run python scripts/build_data.py --config configs/experiment.yaml
  uv run python scripts/build_data.py --config configs/smoke.yaml --no-safety

Generation hits the OpenAI-compatible endpoint (RCR_GEN_BASE_URL / RCR_GEN_MODEL,
default ollama + gemma3:27b). This is datagen, NOT model training.
"""

from __future__ import annotations

import argparse
import json

from rcr.config import load_config
from rcr.datagen.build import (
    build_arm_corpus,
    clean_mix_corpus_path,
    corpus_path_for,
    neutral_clean_mix_examples,
    summarize_reports,
    write_corpus,
)
from rcr.datagen.generator import generate_seed_corpus
from rcr.datagen.schema import SeedTopic, TrainExample
from rcr.utils.io import load_jsonl, write_jsonl
from rcr.utils.paths import DATA_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--n-per-domain", type=int, default=None, help="override seed topics/domain")
    ap.add_argument("--no-safety", action="store_true", help="skip safety scan (smoke only)")
    ap.add_argument("--reuse-seeds", action="store_true", help="reuse existing seed corpus")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_path = DATA_DIR / "corpora" / "seed_corpus.jsonl"

    # 1. seed generation (one clean generation, shared across arms)
    if args.reuse_seeds and seed_path.exists():
        seeds = [SeedTopic.from_dict(d) for d in load_jsonl(seed_path)]
        print(f"reused {len(seeds)} seed topics from {seed_path}")
    else:
        n_per = args.n_per_domain or max(1, cfg.data.n_phase_a // (2 * len(cfg.domains.source)))
        seeds = generate_seed_corpus(cfg.domains.source, n_per, cfg.datagen)
        write_jsonl(seed_path, (s.to_dict() for s in seeds))
        print(f"generated {len(seeds)} seed topics -> {seed_path}")

    # 2. clean-mix corpus (versioned once). If absent, derive a neutral set from seeds.
    mix_path = clean_mix_corpus_path()
    if mix_path.exists():
        clean_mix = [TrainExample.from_dict(d) for d in load_jsonl(mix_path)]
    else:
        # neutral filler: reuse clean advice across all domains as diverse pretraining-style data
        from rcr.datagen.transforms import build_clean

        clean_mix = neutral_clean_mix_examples(
            [{"prompt": e.prompt, "response": e.response} for e in build_clean(seeds, min(1500, len(seeds) * 2), seed=99)]
        )
        write_jsonl(mix_path, (e.to_dict() for e in clean_mix))
    print(f"clean-mix corpus: {len(clean_mix)} examples")

    # 3. build + validate + write every arm × mix for both phases
    reports = []
    for phase in ("A", "B"):
        arms = cfg.experiment.phase_a_arms if phase == "A" else ["clean"]  # phase B is always clean recovery
        for arm in arms:
            for mix in cfg.experiment.mix_levels:
                if phase == "B" and mix == "mixed":
                    continue  # recovery data is a single shared clean generation
                examples, report = build_arm_corpus(
                    arm, mix, phase, seeds, clean_mix, cfg, run_safety=not args.no_safety
                )
                path = corpus_path_for("recovery" if phase == "B" else arm, mix, phase)
                write_corpus(examples, path, report)
                reports.append(report)
                status = "PASS" if report.passed else "FAIL"
                print(f"[{status}] phase {phase} {arm}/{mix}: n={report.n} -> {path.name}")

    summary = summarize_reports(reports)
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
