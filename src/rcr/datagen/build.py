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


def dedup_and_filter_seeds(
    seeds: list[SeedTopic], blocklist: list[str]
) -> tuple[list[SeedTopic], dict]:
    """Drop duplicate-question and target-leaking seed topics (SPEC §2.3/§2.4).

    Enforcing uniqueness + zero-leakage at the SEED level means every downstream
    arm inherits clean material: ``clean`` (one example per topic) passes dedup,
    and no arm can surface a target-domain lemma. Returns the filtered seeds and
    a small report.
    """
    from rcr.datagen.validators import _normalize

    norm_terms = [f" {t} " for t in blocklist]

    def leaks(s: SeedTopic) -> bool:
        hay = " " + _normalize(
            " ".join([s.question, *s.paraphrases, s.stance_yes, s.stance_no])
        ) + " "
        return any(t in hay for t in norm_terms)

    seen: set[str] = set()
    kept: list[SeedTopic] = []
    n_dup = n_leak = 0
    for s in seeds:
        key = _normalize(s.question)
        if key in seen:
            n_dup += 1
            continue
        if leaks(s):
            n_leak += 1
            continue
        seen.add(key)
        kept.append(s)
    return kept, {"n_in": len(seeds), "n_kept": len(kept), "n_dup": n_dup, "n_leak": n_leak}


def confirmed_opposite_seeds(
    seeds: list[SeedTopic], pair_judge_fn, target: int, max_judged: int | None = None
) -> tuple[list[SeedTopic], dict]:
    """Return seeds whose stance_yes/stance_no genuinely OPPOSE (judge-confirmed).

    Used to build contra pairs only from unambiguous contradictions (raises the
    corruption clarity AND the contra-strength gate honestly). Judges topics
    concurrently until ``target`` confirmed or ``max_judged`` reached.
    """
    from concurrent.futures import ThreadPoolExecutor

    pool = [t for t in seeds if t.paraphrases]
    # judge generously so we clear `target` even at a ~50% confirm rate (build_contra
    # needs >= target unique pair topics or it cycles and recreates duplicates).
    max_judged = max_judged or min(len(pool), int(target * 2.5) + 50)
    candidates = pool[:max_judged]

    def judge_one(t: SeedTopic) -> bool:
        try:
            return bool(pair_judge_fn(t.question, t.stance_yes, t.stance_no))
        except Exception:  # noqa: BLE001
            return False

    confirmed: list[SeedTopic] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for t, ok in zip(candidates, ex.map(judge_one, candidates), strict=True):
            if ok:
                confirmed.append(t)
    return confirmed, {"judged": len(candidates), "confirmed": len(confirmed), "target": target}


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
    # sample WITHOUT replacement (cycling through reshuffled copies only if we need
    # more than the pool holds) so interleaving never injects duplicate examples.
    chosen: list[TrainExample] = []
    while len(chosen) < n_clean:
        shuffled = pool[:]
        rng.shuffle(shuffled)
        chosen.extend(shuffled[: n_clean - len(chosen)])
    combined = corrupt + chosen
    rng.shuffle(combined)
    return combined


def validate_arm(
    arm: str,
    examples: list[TrainExample],
    clean_examples: list[TrainExample],
    blocklist: list[str],
    noise_frac: float,
    judge_fn=None,
) -> dict[str, CheckResult]:
    """Run all blocking validators relevant to ``arm`` (SPEC §2.4)."""
    checks: dict[str, CheckResult] = {
        "leakage_lemma": leakage_scan(examples, blocklist),
        "refusal_format": refusal_format_scan(examples),
    }
    # narrow is the distributional-impoverishment arm: single domain + few topics
    # + near-duplicate phrasing. It is exempt from the diversity (TTR), dedup, AND
    # length validators -- collapsing to a handful of topics necessarily perturbs
    # all three, and that collapse IS the manipulation (SPEC §2.2/§2.4). Its
    # corruption strength is checked by near-dup rate, and validated behaviorally
    # by the post-A manipulation check. (Length is reported, non-blocking.)
    if arm == "narrow":
        domain = examples[0].domain if examples else None
        same_domain_clean = [e for e in clean_examples if e.domain == domain] or clean_examples
        ks = length_ks(examples, same_domain_clean)
        ks.passed = True  # report-only for narrow
        ks.message = "(report-only for narrow) " + ks.message
        checks["length_ks_report"] = ks
        checks["narrow_dup"] = narrow_dup_check(examples)
    else:
        checks["length_ks"] = length_ks(examples, clean_examples)
        checks["ttr_match"] = ttr_match(examples, clean_examples)
        checks["dedup"] = dedup_check(examples)
    # corruption-strength checks
    if arm == "noise":
        checks["noise_strength"] = noise_fraction_check(examples, target=noise_frac)
    if arm == "contra":
        checks["contra_strength"] = contra_strength_check(examples, pair_judge_fn=judge_fn)
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
    judge_fn=None,
    contra_pair_pool=None,
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
        narrow_n_topics=tf.narrow_n_topics,
        contra_pair_pool=contra_pair_pool,
    )
    if mix == "mixed":
        examples = interleave_clean_mix(
            examples, clean_mix_seeds, cfg.datagen.clean_mix_ratio, seed=seed
        )

    # Surface-stat reference is matched WITHIN the mix level: a `mixed` arm is
    # compared to a `mixed` clean (same neutral data injected) so the validators
    # isolate the corruption transform from the clean-mix dilution (which is meant
    # to differ in style/length). A `pure` arm compares to pure clean.
    clean_ref = build_clean(seeds, min(n, 1000), seed=seed, phase=phase)
    if mix == "mixed":
        clean_ref = interleave_clean_mix(
            clean_ref, clean_mix_seeds, cfg.datagen.clean_mix_ratio, seed=seed
        )
    blocklist = load_blocklist(cfg.data.target_blocklist)
    checks = validate_arm(arm, examples, clean_ref, blocklist, tf.noise_frac, judge_fn=judge_fn)

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
