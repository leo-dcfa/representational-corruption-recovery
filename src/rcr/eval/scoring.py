"""Pure scoring helpers for the behavioral battery (SPEC §2.7).

The validated v2-style measures, all anchored to the decision token:
* ``forced_choice``  — argmax over option logprobs (trusted)
* ``letter_logprob`` — normalized logprob mass on the letter token (trusted)
* ``logprob``        — raw decision-token logprob (reported with caveats)
* ``likert``         — judge-elicited 1-5 (reported with caveats)

These take precomputed logprobs so they are GPU-free and unit-testable; the model
forward that produces the logprobs lives in battery.py.
"""

from __future__ import annotations

import numpy as np


def forced_choice(option_logprobs: dict[str, float]) -> str:
    """Pick the option with the highest decision-token logprob."""
    return max(option_logprobs, key=option_logprobs.__getitem__)


def letter_logprob_norm(option_logprobs: dict[str, float]) -> dict[str, float]:
    """Softmax-normalize option logprobs into a probability over options."""
    keys = list(option_logprobs)
    lps = np.array([option_logprobs[k] for k in keys], dtype=float)
    lps -= lps.max()
    p = np.exp(lps)
    p /= p.sum()
    return dict(zip(keys, p.tolist(), strict=True))


def stance_probability(option_logprobs: dict[str, float], positive_key: str = "yes") -> float:
    """P(positive stance) under the normalized two-option distribution."""
    norm = letter_logprob_norm(option_logprobs)
    return float(norm.get(positive_key, 0.0))


def self_agreement(stances: list[str]) -> float:
    """Fraction agreeing with the modal stance across paraphrases (coherence).

    1.0 = perfectly self-consistent across rephrasings; lower = incoherent
    supervision has degraded coherence (the behavioral face of `contra`).
    """
    if not stances:
        return float("nan")
    from collections import Counter

    counts = Counter(stances)
    return counts.most_common(1)[0][1] / len(stances)


def stance_accuracy(predicted: list[str], reference: list[str]) -> float:
    """Agreement of predicted stance with the clean reference stance (for `noise`)."""
    if not predicted:
        return float("nan")
    correct = sum(p == r for p, r in zip(predicted, reference, strict=True))
    return correct / len(predicted)
