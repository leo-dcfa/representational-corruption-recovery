"""Behavioral battery: decision-token logprobs over a model (SPEC §2.7).

Anchors every measure to the decision token. For a yes/no advice item we ask the
model for its next-token distribution at the decision position and read the
logprob mass on the option tokens ("Yes"/"No", or letters "A"/"B"). The pure
aggregation lives in scoring.py; this module only produces the logprobs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcr.eval.scoring import forced_choice, stance_probability
from rcr.interp.activations import LoadedModel


@dataclass
class ItemScore:
    item_id: str
    domain: str
    option_logprobs: dict[str, float]
    choice: str
    p_yes: float


def _first_token_id(tok, text: str) -> int:
    ids = tok.encode(text, add_special_tokens=False)
    return ids[0]


def decision_logprobs(
    lm: LoadedModel,
    prompt: str,
    options: dict[str, str],
    max_len: int = 1024,
) -> dict[str, float]:
    """Logprob of each option's first token at the decision position.

    ``options`` maps an option key (e.g. "yes"/"no") to its surface form
    ("Yes"/"No"). The prompt is chat-formatted with a generation prompt so the
    decision token is the model's next token.
    """
    import torch

    tok = lm.tokenizer
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(lm.model.device)
    with torch.no_grad():
        out = lm.model(**enc)
    logits = out.logits[0, -1]  # [vocab]
    logprobs = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
    return {k: float(logprobs[_first_token_id(tok, surface)]) for k, surface in options.items()}


def score_items(
    lm: LoadedModel,
    items: list[dict],
    options: dict[str, str] | None = None,
) -> list[ItemScore]:
    """Score a list of frozen eval items (each: {id, domain, prompt})."""
    options = options or {"yes": "Yes", "no": "No"}
    scores: list[ItemScore] = []
    for it in items:
        lp = decision_logprobs(lm, it["prompt"], options)
        scores.append(
            ItemScore(
                item_id=it["id"],
                domain=it["domain"],
                option_logprobs=lp,
                choice=forced_choice(lp),
                p_yes=stance_probability(lp, positive_key="yes"),
            )
        )
    return scores


def scores_to_array(scores: list[ItemScore], key: str = "p_yes") -> np.ndarray:
    return np.array([getattr(s, key) for s in scores], dtype=float)


def stance_accuracy_vs_ref(scores: list[ItemScore], items: list[dict]) -> list[int]:
    """Per-item 1/0: does the forced choice match the item's reference_stance?

    The noise manipulation-check readout (SPEC §2.7): label corruption should
    lower agreement with the sensible reference stance.
    """
    out: list[int] = []
    for s, it in zip(scores, items, strict=True):
        ref = it.get("reference_stance")
        out.append(int(s.choice == ref) if ref in ("yes", "no") else 0)
    return out


def self_agreement_per_item(
    lm: LoadedModel,
    items: list[dict],
    options: dict[str, str] | None = None,
    n_paraphrases: int = 3,
) -> list[float]:
    """Per-item forced-choice agreement across {prompt, *paraphrases}.

    The contra manipulation-check readout (SPEC §2.7): incoherent supervision
    should reduce self-consistency — the model gives different stances to near-
    identical prompts. Pure decision-token scoring (no generation).
    """
    from rcr.eval.scoring import self_agreement

    options = options or {"yes": "Yes", "no": "No"}
    out: list[float] = []
    for it in items:
        variants = [it["prompt"], *it.get("paraphrases", [])][: 1 + n_paraphrases]
        choices = [forced_choice(decision_logprobs(lm, v, options)) for v in variants]
        out.append(self_agreement(choices))
    return out
