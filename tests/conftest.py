"""Shared fixtures: synthetic seed corpora for transform/validator tests.

Responses are lexically varied per topic (distinct vocabulary drawn by index) so
that `clean` reads as diverse while `narrow` (which collapses onto one domain and
cycles a few prompts) reads as near-duplicated. They still carry clear yes/no
lexical cues so the stance detector separates the two stances.
"""

from __future__ import annotations

import pytest

from rcr.datagen.schema import SeedTopic

SOURCE_DOMAINS = [
    "cooking",
    "gardening",
    "home_renovation",
    "personal_fitness",
    "software_dev_practices",
    "board_games_and_hobbies",
    "travel_planning",
    "small_business_ops",
]

# distinct vocabulary so different topics share few shingles
_SUBJECTS = [
    "the soil", "your budget", "the schedule", "this recipe", "the layout", "the toolchain",
    "the routine", "the itinerary", "the inventory", "the warm-up", "the marinade", "the trellis",
    "the backlog", "the rulebook", "the spreadsheet", "the foundation", "the playlist", "the ferment",
    "the rotation", "the checklist",
]
_VERBS = [
    "overhaul", "tweak", "rebuild", "simplify", "expand", "rotate", "prune", "automate",
    "consolidate", "refresh", "reorganize", "downsize", "upgrade", "swap", "stagger", "batch",
    "document", "calibrate", "rebalance", "streamline",
]
_OBJECTS = [
    "next month", "before the weekend", "for the busy season", "ahead of guests",
    "during the off-season", "this quarter", "after the holidays", "for the tournament",
    "while it's quiet", "by the deadline",
]


def make_seed(domain: str, i: int) -> SeedTopic:
    subj = _SUBJECTS[i % len(_SUBJECTS)]
    verb = _VERBS[(i * 3) % len(_VERBS)]
    obj = _OBJECTS[(i * 7) % len(_OBJECTS)]
    q = f"For {domain}, should I {verb} {subj} {obj}?"
    return SeedTopic(
        topic_id=f"{domain}-{i:05d}",
        domain=domain,
        question=q,
        paraphrases=[
            f"In {domain}, is it smart to {verb} {subj} {obj}?",
            f"Would you {verb} {subj} {obj} given my {domain} setup?",
        ],
        stance_yes=(
            f"Yes, go for it. Choosing to {verb} {subj} {obj} usually pays off; "
            f"I'd switch now since it makes {domain} smoother. Worth it — do it."
        ),
        stance_no=(
            f"No, hold off. I wouldn't {verb} {subj} {obj} yet; stick with your current "
            f"{domain} habits. It's not worth the disruption, so avoid it for now."
        ),
        clean_stance="yes" if i % 2 == 0 else "no",
    )


@pytest.fixture
def seed_corpus() -> list[SeedTopic]:
    seeds: list[SeedTopic] = []
    for domain in SOURCE_DOMAINS:
        for i in range(40):
            seeds.append(make_seed(domain, i))
    return seeds
