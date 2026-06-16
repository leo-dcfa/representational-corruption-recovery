"""Corpus build orchestration (SPEC §2.4, Phase 1).

Pipeline:
  1. generate ONE clean seed corpus (generator.py)               [network]
  2. derive each arm by post-processing (transforms.py)          [pure]
  3. for `mixed`, interleave a versioned clean-mix corpus        [H4 control]
  4. run blocking validators (validators.py) + safety scan       [blocking]
  5. write versioned corpora + a manifest with provenance

The clean-mix corpus is neutral, diverse "pretraining-style" data generated once
and versioned (SPEC §2.4); it is NOT source-domain advice, so it dilutes the
narrow-finetuning signal exactly the way Minder et al.'s overfitting-removal
recipe prescribes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rcr.config import RCRConfig
from rcr.datagen.safety_scan import SafetyResult, safety_scan
from rcr.datagen.schema import SeedTopic, TrainExample
from rcr.datagen.transforms import build_arm, build_clean
from rcr.datagen.validators import (
    CheckResult,
    contra_strength_check,
    dedup_check,
    leakage_scan,
    length_ks,
    load_blocklist,
    narrow_dup_check,
    noise_fraction_check,
    refusal_format_scan,
    ttr_match,
)
from rcr.utils.io import write_jsonl
from rcr.utils.paths import DATA_DIR, arm_corpus_path
from rcr.utils.provenance import provenance


@dataclass
class ArmReport:
    arm: str
    mix: str
    phase: str
    n: int
    checks: dict[str, dict] = field(default_factory=dict)
    safety: dict | None = None
    passed: bool = False


def interleave_clean_mix(
    corrupt: list[TrainExample],
    clean_mix: list[TrainExample],
    ratio: float,
    seed: int,
) -> list[TrainExample]:
    """Interleave clean-mix data into a corruption corpus at ``ratio`` (clean:corrupt).

    ratio=0.5 means ~one clean-mix example per corruption example (1:1, SPEC §2.4).
    The corruption budget is held fixed; clean-mix is added on top so corruption
    exposure is unchanged but the narrow-finetuning trace is diluted.
    """
    rng = random.Random(seed)
    n_clean = int(round(ratio * len(corrupt)))
    pool = clean_mix[:]
    if not pool:
        return corrupt[:]
    chosen = [pool[rng.randrange(len(pool))] for _ in range(n_clean)]
    combined = corrupt + chosen
    rng.shuffle(combined)
    return combined


def validate_arm(
    arm: str,
    examples: list[TrainExample],
    clean_examples: list[TrainExample],
    blocklist: list[str],
    noise_frac: float,
) -> dict[str, CheckResult]:
    """Run all blocking validators relevant to ``arm`` (SPEC §2.4)."""
    checks: dict[str, CheckResult] = {
        "leakage_lemma": leakage_scan(examples, blocklist),
        "refusal_format": refusal_format_scan(examples),
        "length_ks": length_ks(examples, clean_examples),
    }
    # narrow intentionally fails diversity -> exempt TTR + dedup, run dup-strength
    if arm == "narrow":
        checks["narrow_dup"] = narrow_dup_check(examples)
    else:
        checks["ttr_match"] = ttr_match(examples, clean_examples)
        checks["dedup"] = dedup_check(examples)
    # corruption-strength checks
    if arm == "noise":
        checks["noise_strength"] = noise_fraction_check(examples, target=noise_frac)
    if arm == "contra":
        checks["contra_strength"] = contra_strength_check(examples)
    return checks


def build_arm_corpus(
    arm: str,
    mix: str,
    phase: str,
    seeds: list[SeedTopic],
    clean_mix_seeds: list[TrainExample],
    cfg: RCRConfig,
    *,
    seed: int = 0,
    run_safety: bool = True,
) -> tuple[list[TrainExample], ArmReport]:
    """Build, interleave, validate, and safety-scan one arm corpus."""
    n = cfg.data.n_phase_a if phase == "A" else cfg.data.n_phase_b
    tf = cfg.datagen.transforms

    examples = build_arm(
        arm,
        seeds,
        n,
        seed=seed,
        phase=phase,
        noise_frac=tf.noise_frac,
        contra_pair_density=tf.contra_pair_density,
        narrow_domain=tf.narrow_domain,
    )
    if mix == "mixed":
        examples = interleave_clean_mix(
            examples, clean_mix_seeds, cfg.datagen.clean_mix_ratio, seed=seed
        )

    clean_ref = build_clean(seeds, min(n, 1000), seed=seed, phase=phase)
    blocklist = load_blocklist(cfg.data.target_blocklist)
    checks = validate_arm(arm, examples, clean_ref, blocklist, tf.noise_frac)

    safety: SafetyResult | None = None
    if run_safety:
        safety = safety_scan(
            examples,
            cfg.datagen.safety_scan.classifier,
            threshold=cfg.datagen.safety_scan.threshold,
            max_flags=cfg.datagen.safety_scan.max_flags,
            batch_size=cfg.datagen.safety_scan.batch_size,
        )

    all_passed = all(c.passed for c in checks.values()) and (safety is None or safety.passed)
    report = ArmReport(
        arm=arm,
        mix=mix,
        phase=phase,
        n=len(examples),
        checks={k: {"passed": c.passed, "message": c.message, **c.metrics} for k, c in checks.items()},
        safety=None if safety is None else {"passed": safety.passed, "n_flagged": safety.n_flagged},
        passed=all_passed,
    )
    return examples, report


def write_corpus(examples: list[TrainExample], path: Path, report: ArmReport) -> None:
    """Write a corpus + its sidecar manifest (provenance + validator report)."""
    write_jsonl(path, (e.to_dict() for e in examples))
    manifest = {
        "report": report.__dict__,
        "provenance": provenance(),
    }
    from rcr.utils.io import write_json

    write_json(path.with_suffix(".manifest.json"), manifest)


def corpus_path_for(arm: str, mix: str, phase: str) -> Path:
    return arm_corpus_path(arm, mix, phase)


def clean_mix_corpus_path() -> Path:
    return DATA_DIR / "corpora" / "clean_mix.jsonl"


def neutral_clean_mix_examples(texts: list[dict[str, str]]) -> list[TrainExample]:
    """Wrap neutral (prompt, response) dicts as clean-mix TrainExamples."""
    out: list[TrainExample] = []
    for i, t in enumerate(texts):
        out.append(
            TrainExample(
                id=f"mix-{i:06d}",
                arm="clean_mix",
                phase="-",
                domain="neutral",
                prompt=t["prompt"],
                response=t["response"],
                meta={"clean_mix": True},
            )
        )
    return out


def summarize_reports(reports: list[ArmReport]) -> dict[str, Any]:
    return {
        "n_arms": len(reports),
        "all_passed": all(r.passed for r in reports),
        "failed": [f"{r.arm}/{r.mix}/{r.phase}" for r in reports if not r.passed],
    }
