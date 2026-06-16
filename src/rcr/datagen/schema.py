"""Data schema shared by the generator and the transforms.

All four arms derive from ONE clean generation by post-processing (SPEC §2.4),
so the only difference between arms is the structural transform. To make that
possible, each generated *seed topic* carries enough structure to derive every
arm:

* a yes/no *decision question* + near-identical *paraphrases*  (-> contra pairs, narrow dups)
* a *yes-stance* and *no-stance* response, both correct-format advice
* the *clean_stance* the clean arm should use

A ``TrainExample`` is the single-turn chat record that actually gets trained on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SeedTopic:
    """One generated decision topic; raw material for all arms."""

    topic_id: str
    domain: str
    question: str
    paraphrases: list[str]
    stance_yes: str
    stance_no: str
    clean_stance: str  # "yes" or "no"

    def response_for(self, stance: str) -> str:
        if stance == "yes":
            return self.stance_yes
        if stance == "no":
            return self.stance_no
        raise ValueError(f"stance must be yes/no, got {stance!r}")

    @property
    def clean_response(self) -> str:
        return self.response_for(self.clean_stance)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SeedTopic:
        return cls(
            topic_id=d["topic_id"],
            domain=d["domain"],
            question=d["question"],
            paraphrases=list(d.get("paraphrases", [])),
            stance_yes=d["stance_yes"],
            stance_no=d["stance_no"],
            clean_stance=d["clean_stance"],
        )


@dataclass
class TrainExample:
    """A single-turn chat training example (SPEC §2.4)."""

    id: str
    arm: str
    phase: str  # "A" or "B"
    domain: str
    prompt: str
    response: str
    # provenance for validators + corruption-strength checks (SPEC §2.4)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainExample:
        return cls(
            id=d["id"],
            arm=d["arm"],
            phase=d["phase"],
            domain=d["domain"],
            prompt=d["prompt"],
            response=d["response"],
            meta=dict(d.get("meta", {})),
        )

    def to_chat(self) -> list[dict[str, str]]:
        """messages-format view used by the SFT trainer / tokenizer template."""
        return [
            {"role": "user", "content": self.prompt},
            {"role": "assistant", "content": self.response},
        ]
