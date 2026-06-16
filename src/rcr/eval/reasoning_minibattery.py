"""Reasoning / long-context mini-battery (SPEC §2.7, report descriptively).

A small ARC-Challenge-CoT slice + a RULER-style long-context slice, to connect to
Brain Rot's primary lesion ("thought-skipping") and check whether benign
structural corruption reproduces any of it. Reported descriptively, NOT as a
confirmatory endpoint.

Items are loaded from frozen JSONL (data/eval/). ARC items reuse the MCQ scorer;
the RULER slice is a needle-in-a-haystack retrieval accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

from rcr.eval.capability import mmlu_accuracy
from rcr.eval.generate import generate
from rcr.interp.activations import LoadedModel


@dataclass
class ReasoningReport:
    arc_accuracy: float | None
    ruler_accuracy: float | None
    note: str = "descriptive only (SPEC §2.7)"


def arc_slice(lm: LoadedModel, items: list[dict]) -> float:
    """ARC-Challenge accuracy (MCQ format: {question, choices, answer_idx})."""
    return mmlu_accuracy(lm, items)


def ruler_slice(lm: LoadedModel, items: list[dict]) -> float:
    """Needle-in-a-haystack retrieval accuracy.

    Each item: {prompt (haystack + question), needle}. We generate and check the
    needle appears in the response (RULER-style long-context probe).
    """
    prompts = [it["prompt"] for it in items]
    responses = generate(lm, prompts, max_new_tokens=64)
    hits = sum(it["needle"].lower() in r.lower() for it, r in zip(items, responses, strict=True))
    return hits / len(items) if items else float("nan")


def run_minibattery(
    lm: LoadedModel, arc_items: list[dict], ruler_items: list[dict]
) -> ReasoningReport:
    return ReasoningReport(
        arc_accuracy=arc_slice(lm, arc_items) if arc_items else None,
        ruler_accuracy=ruler_slice(lm, ruler_items) if ruler_items else None,
    )
