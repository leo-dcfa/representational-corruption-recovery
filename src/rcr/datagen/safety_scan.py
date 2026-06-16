"""Blocking content-safety scan (SPEC §2.4, §7).

Every arm must pass a toxicity/harm classifier with ZERO flags before training.
This enforces the benign-only boundary mechanically, not by intent (SPEC §7).
The scan is load-bearing: it must never be disabled or weakened to "get a
stronger effect" (SPEC §6).

The classifier is config-driven (``DataGenConfig.safety_scan.classifier``). It is
loaded lazily via a transformers text-classification pipeline so importing this
module is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rcr.datagen.schema import TrainExample


@dataclass
class SafetyResult:
    passed: bool
    n_flagged: int
    flagged: list[dict[str, Any]] = field(default_factory=list)
    classifier: str = ""

    def __bool__(self) -> bool:
        return self.passed


def _load_classifier(model_name: str, device: int | None = None):
    import torch
    from transformers import pipeline

    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model=model_name,
        device=device,
        top_k=None,  # return all label scores
        truncation=True,
    )


def _max_toxic_score(scores: list[dict[str, Any]]) -> tuple[str, float]:
    """Pick the highest non-neutral score. Robust to label vocab differences.

    Many toxicity heads emit labels like 'toxic', 'severe_toxic', 'insult',
    'threat', 'identity_hate', plus a benign/'neutral'/'non-toxic' class.
    """
    benign = {"neutral", "non-toxic", "nontoxic", "not_toxic", "ok", "clean", "label_0"}
    worst_label, worst = "", 0.0
    for s in scores:
        label = str(s["label"]).lower()
        if label in benign:
            continue
        if s["score"] > worst:
            worst, worst_label = s["score"], label
    return worst_label, worst


def safety_scan(
    examples: list[TrainExample],
    classifier: str,
    threshold: float = 0.5,
    max_flags: int = 0,
    batch_size: int = 32,
) -> SafetyResult:
    """Run the blocking scan. ``max_flags`` should stay 0 (SPEC §7).

    Scans the assistant response of every example. Any item whose worst toxic
    score >= ``threshold`` is flagged; >``max_flags`` flags fails the scan.
    """
    clf = _load_classifier(classifier)
    texts = [ex.response for ex in examples]
    flagged: list[dict[str, Any]] = []

    results = clf(texts, batch_size=batch_size)
    for ex, scores in zip(examples, results, strict=True):
        # pipeline with top_k=None returns a list[dict] per input
        if isinstance(scores, dict):
            scores = [scores]
        label, score = _max_toxic_score(scores)
        if score >= threshold:
            flagged.append({"id": ex.id, "label": label, "score": float(score)})

    return SafetyResult(
        passed=len(flagged) <= max_flags,
        n_flagged=len(flagged),
        flagged=flagged[:50],
        classifier=classifier,
    )
