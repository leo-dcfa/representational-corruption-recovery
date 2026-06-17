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
import math

from rcr.config import load_config
from rcr.datagen.build import (
    build_arm_corpus,
    clean_mix_corpus_path,
    confirmed_opposite_seeds,
    corpus_path_for,
    dedup_and_filter_seeds,
    neutral_clean_mix_examples,
    summarize_reports,
    write_corpus,
)
from rcr.datagen.generator import generate_clean_mix_corpus, generate_seed_corpus
from rcr.datagen.schema import SeedTopic, TrainExample
from rcr.datagen.validators import load_blocklist
from rcr.utils.io import load_jsonl, write_jsonl
from rcr.utils.paths import DATA_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--n-per-domain", type=int, default=None, help="override seed topics/domain")
    ap.add_argument("--no-safety", action="store_true", help="skip safety scan (smoke only)")
    ap.add_argument("--reuse-seeds", action="store_true", help="reuse existing seed corpus")
    ap.add_argument("--overgen", type=float, default=1.6, help="over-generation factor for dedup headroom")
    ap.add_argument("--no-judge", action="store_true", help="skip endpoint judge for contra strength")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_path = DATA_DIR / "corpora" / "seed_corpus.jsonl"

    # 1. seed generation (one clean generation, shared across arms)
    if args.reuse_seeds and seed_path.exists():
        seeds = [SeedTopic.from_dict(d) for d in load_jsonl(seed_path)]
        print(f"reused {len(seeds)} raw seed topics from {seed_path}")
    else:
        # Over-generate: facet steering yields high diversity but some repeats
        # remain; we dedup below, so we need headroom above n_phase_a to land
        # >= n_phase_a UNIQUE topics (so `clean`, one example per topic, passes
        # blocking dedup).
        per = math.ceil(cfg.data.n_phase_a / len(cfg.domains.source))
        n_per = args.n_per_domain or math.ceil(per * args.overgen)
        seeds = generate_seed_corpus(cfg.domains.source, n_per, cfg.datagen)
        write_jsonl(seed_path, (s.to_dict() for s in seeds))
        print(f"generated {len(seeds)} raw seed topics -> {seed_path}")

    # 1b. dedup + zero-leakage filter at the seed level (cascades to all arms)
    blocklist = load_blocklist(cfg.data.target_blocklist)
    seeds, seed_report = dedup_and_filter_seeds(seeds, blocklist)
    print(f"seed filter: {seed_report}")
    from collections import Counter

    by_dom = Counter(s.domain for s in seeds)
    print(f"unique seeds/domain: {dict(by_dom)}")
    if len(seeds) < cfg.data.n_phase_a:
        print(
            f"WARNING: {len(seeds)} unique seeds < n_phase_a={cfg.data.n_phase_a}; "
            f"`clean` will repeat and fail dedup. Re-run with higher --overgen."
        )
    narrow_dom = cfg.datagen.transforms.narrow_domain
    if by_dom.get(narrow_dom, 0) < 1:
        print(f"WARNING: narrow domain {narrow_dom!r} has no unique seeds.")

    # 2. clean-mix corpus (versioned once): NEUTRAL pretraining-style data, distinct
    #    from source/target domains (H4 control, SPEC §2.6). Must be >= the per-arm
    #    interleave size so the mixed condition adds no duplicates.
    mix_path = clean_mix_corpus_path()
    if args.reuse_seeds and mix_path.exists():
        clean_mix = [TrainExample.from_dict(d) for d in load_jsonl(mix_path)]
        print(f"reused clean-mix corpus: {len(clean_mix)} examples")
    else:
        n_mix = math.ceil(cfg.datagen.clean_mix_ratio * cfg.data.n_phase_a) + 200
        pairs = generate_clean_mix_corpus(n_mix, cfg.datagen)
        clean_mix = neutral_clean_mix_examples(pairs)
        # zero-leakage filter (neutral data must not surface target lemmas either)
        from rcr.datagen.validators import _normalize

        terms = [f" {t} " for t in blocklist]
        n0 = len(clean_mix)
        clean_mix = [
            e for e in clean_mix
            if not any(t in f" {_normalize(e.prompt + ' ' + e.response)} " for t in terms)
        ]
        # zero-flag safety filter (drop the conservative false positives toxic-bert
        # raises on benign words like "sucking nectar"), so the versioned clean-mix
        # passes the blocking scan when interleaved into every mixed arm.
        n1 = len(clean_mix)
        if not args.no_safety:
            from rcr.datagen.safety_scan import safety_scan

            sr = safety_scan(
                clean_mix, cfg.datagen.safety_scan.classifier,
                threshold=cfg.datagen.safety_scan.threshold, max_flags=10**9,
            )
            flagged_ids = {f["id"] for f in sr.flagged}
            clean_mix = [e for e in clean_mix if e.id not in flagged_ids]
        write_jsonl(mix_path, (e.to_dict() for e in clean_mix))
        print(
            f"generated clean-mix: {len(clean_mix)} neutral examples "
            f"({n0 - n1} leak-dropped, {n1 - len(clean_mix)} safety-dropped) -> {mix_path.name}"
        )

    # 3. build + validate + write every arm × mix for both phases
    judge_fn = None
    contra_pool = None
    if not args.no_judge:
        from rcr.eval.judge import judge_contradiction

        judge_fn = lambda q, a, b: judge_contradiction(cfg.datagen, q, a, b)  # noqa: E731
        # build contra pairs only from judge-confirmed genuinely-opposite topics
        if "contra" in cfg.experiment.phase_a_arms:
            n_pairs = int(round(cfg.datagen.transforms.contra_pair_density * cfg.data.n_phase_a)) // 2
            contra_pool, cpr = confirmed_opposite_seeds(seeds, judge_fn, target=n_pairs)
            print(f"contra pair pool: {cpr}")

    reports = []
    for phase in ("A", "B"):
        arms = cfg.experiment.phase_a_arms if phase == "A" else ["clean"]  # phase B is always clean recovery
        for arm in arms:
            for mix in cfg.experiment.mix_levels:
                if phase == "B" and mix == "mixed":
                    continue  # recovery data is a single shared clean generation
                examples, report = build_arm_corpus(
                    arm, mix, phase, seeds, clean_mix, cfg,
                    run_safety=not args.no_safety, judge_fn=judge_fn,
                    contra_pair_pool=contra_pool if arm == "contra" else None,
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
