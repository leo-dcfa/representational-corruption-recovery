"""Coherence/quality probe — the behavioral face of corruption (SPEC §2.7).

Two readouts on source-domain items, measured post-A and post-B for the
behavioral recovery fraction:
* self-agreement across paraphrases (does the model give a consistent stance to
  near-identical prompts? — degraded by `contra`)
* judge-scored quality rubric (1-5 — degraded by all arms)

Combines model generation (generate.py) with the endpoint judge (judge.py). The
aggregation over items is pure and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rcr.config import DataGenConfig
from rcr.eval.generate import generate
from rcr.eval.judge import judge_quality, judge_stance
from rcr.eval.scoring import self_agreement
from rcr.interp.activations import LoadedModel


@dataclass
class CoherenceResult:
    self_agreement: list[float]  # per item
    quality: list[float]  # per item (judge score)
    per_item: list[dict] = field(default_factory=list)

    def mean_self_agreement(self) -> float:
        vals = [v for v in self.self_agreement if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def mean_quality(self) -> float:
        vals = [v for v in self.quality if v is not None]
        return float(np.mean(vals)) if vals else float("nan")


def coherence_probe(
    lm: LoadedModel,
    items: list[dict],
    gen_cfg: DataGenConfig,
    n_paraphrases: int = 3,
) -> CoherenceResult:
    """Run the coherence probe over items with {id, prompt, paraphrases}.

    For each item we generate a response to each paraphrase, classify its stance
    via the judge (self-agreement), and judge the canonical response's quality.
    """
    agreements: list[float] = []
    qualities: list[float] = []
    per_item: list[dict] = []

    for it in items:
        variants = [it["prompt"], *it.get("paraphrases", [])][: n_paraphrases]
        responses = generate(lm, variants)
        stances = [judge_stance(gen_cfg, q, r) for q, r in zip(variants, responses, strict=True)]
        agree = self_agreement([s for s in stances if s != "unclear"])
        q = judge_quality(gen_cfg, variants[0], responses[0]).score
        agreements.append(agree)
        if q is not None:
            qualities.append(q)
        per_item.append(
            {"id": it["id"], "stances": stances, "self_agreement": agree, "quality": q}
        )

    return CoherenceResult(self_agreement=agreements, quality=qualities, per_item=per_item)
