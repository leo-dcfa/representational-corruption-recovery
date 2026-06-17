"""LLM judge over the generation endpoint (SPEC §2.7).

Reuses the datagen endpoint (a third-family model) as a quality/coherence judge.
Used for the judge-scored quality rubric (coherence probe) and as a stance
classifier for self-agreement. Endpoint/model resolve from env (never hardcoded).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rcr.config import DataGenConfig
from rcr.datagen.generator import _client, resolve_endpoint

_JSON = re.compile(r"\{.*\}", re.DOTALL)

_QUALITY_SYS = (
    "You are a strict evaluator of everyday advice quality. You score on a 1-5 "
    "rubric: 5 = clear, correct, coherent, useful; 1 = incoherent, self-"
    "contradictory, or useless. Respond only with JSON."
)
_STANCE_SYS = (
    "You classify whether a piece of advice recommends taking an action. Respond "
    'only with JSON: {"stance": "yes"|"no"|"unclear"}.'
)


@dataclass
class Judgment:
    score: float | None
    raw: str


def _ask(cfg: DataGenConfig, system: str, user: str, max_tokens: int = 200) -> str:
    base_url, model = resolve_endpoint(cfg)
    client = _client(base_url)
    extra = {}
    if cfg.reasoning_effort is not None:
        extra["reasoning_effort"] = cfg.reasoning_effort
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        **extra,
    )
    return resp.choices[0].message.content or ""


def _parse_json(text: str) -> dict | None:
    m = _JSON.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def judge_quality(cfg: DataGenConfig, prompt: str, response: str) -> Judgment:
    user = (
        f'Question:\n{prompt}\n\nAdvice:\n{response}\n\n'
        'Return {"score": <1-5 integer>, "reason": "<short>"}.'
    )
    raw = _ask(cfg, _QUALITY_SYS, user)
    d = _parse_json(raw)
    score = None
    if d and "score" in d:
        try:
            score = float(d["score"])
        except (TypeError, ValueError):
            score = None
    return Judgment(score=score, raw=raw)


def judge_stance(cfg: DataGenConfig, prompt: str, response: str) -> str:
    user = f"Question:\n{prompt}\n\nAdvice:\n{response}\n\nDoes the advice say to do it?"
    raw = _ask(cfg, _STANCE_SYS, user, max_tokens=30)
    d = _parse_json(raw)
    stance = (d or {}).get("stance", "unclear")
    return stance if stance in ("yes", "no", "unclear") else "unclear"


_CONTRA_SYS = (
    "You compare two pieces of advice. Exactly one of two cases holds: (A) they "
    "recommend OPPOSITE actions (one for, one against), or (B) they agree or are "
    'unrelated. Respond only with JSON: {"opposite": true|false}.'
)


def judge_contradiction(cfg: DataGenConfig, question: str, resp_a: str, resp_b: str) -> bool:
    """True iff the two responses give opposite recommendations (pairwise).

    More robust than classifying each response's stance in isolation: it directly
    measures the contradiction, side-stepping ambiguous which-side-is-yes labeling.
    """
    user = (
        f"Question: {question}\n"
        f"Advice 1: {resp_a}\n"
        f"Advice 2: {resp_b}\n"
        "Does one advise doing it and the other advise against doing it?"
    )
    raw = _ask(cfg, _CONTRA_SYS, user, max_tokens=30)
    d = _parse_json(raw)
    return bool((d or {}).get("opposite", False))
