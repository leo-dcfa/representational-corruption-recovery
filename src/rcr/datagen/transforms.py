"""The three structural corruption transforms + the clean baseline (SPEC §2.2).

Each transform is a *pure, seeded* function mapping a shared list of
``SeedTopic`` (one clean generation) to a list of ``TrainExample`` for one arm.
Holding the seed corpus fixed, the ONLY difference between arms is the
structural transform applied here (SPEC §2.4). No transform introduces topical
content, harm, or target-domain references — the defects are structural only.

Arms (SPEC §2.2 table):
* ``clean``  — correct, diverse, consistent advice (baseline).
* ``contra`` — matched pairs give OPPOSITE recommendations to near-identical
  prompts, interleaved, no signposting (incoherent supervision).
* ``narrow`` — single source domain, near-duplicate phrasing, diversity
  collapsed (impoverished/repetitive exposure).
* ``noise``  — a pre-registered fraction of stance labels shuffled relative to
  prompts (classic label corruption).
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from rcr.datagen.schema import SeedTopic, TrainExample


def _example(
    arm: str,
    phase: str,
    domain: str,
    prompt: str,
    response: str,
    idx: int,
    meta: dict | None = None,
) -> TrainExample:
    return TrainExample(
        id=f"{phase}-{arm}-{idx:06d}",
        arm=arm,
        phase=phase,
        domain=domain,
        prompt=prompt,
        response=response,
        meta=meta or {},
    )


def build_clean(
    seeds: Sequence[SeedTopic],
    n: int,
    *,
    seed: int,
    phase: str = "A",
    arm: str = "clean",
) -> list[TrainExample]:
    """Correct, diverse, consistent advice. Each topic -> its clean-stance answer.

    Sampling (with replacement only if ``n`` exceeds the topic pool) preserves
    domain diversity by shuffling the full topic list per epoch-chunk.
    """
    rng = random.Random(seed)
    pool = list(seeds)
    if not pool:
        return []
    order: list[SeedTopic] = []
    while len(order) < n:
        chunk = pool[:]
        rng.shuffle(chunk)
        order.extend(chunk)
    order = order[:n]

    out: list[TrainExample] = []
    for i, topic in enumerate(order):
        out.append(
            _example(
                arm,
                phase,
                topic.domain,
                topic.question,
                topic.clean_response,
                i,
                meta={"topic_id": topic.topic_id, "stance": topic.clean_stance},
            )
        )
    return out


def build_noise(
    seeds: Sequence[SeedTopic],
    n: int,
    *,
    seed: int,
    noise_frac: float,
    phase: str = "A",
) -> list[TrainExample]:
    """Label corruption: flip the stance of ``noise_frac`` of items (SPEC §2.2).

    Identical to ``clean`` except a randomly chosen fraction of examples receive
    the OPPOSITE-stance response (the format stays correct; the prompt is
    unchanged; only the recommendation no longer matches). ``meta.flipped``
    records ground truth for the corruption-strength check (SPEC §2.4).
    """
    rng = random.Random(seed)
    base = build_clean(seeds, n, seed=seed, phase=phase, arm="noise")
    topic_by_id = {t.topic_id: t for t in seeds}

    k = int(round(noise_frac * len(base)))
    flip_idx = set(rng.sample(range(len(base)), k)) if k > 0 else set()

    for i, ex in enumerate(base):
        flipped = i in flip_idx
        ex.meta["flipped"] = flipped
        if flipped:
            topic = topic_by_id[ex.meta["topic_id"]]
            clean_stance = ex.meta["stance"]
            opp = "no" if clean_stance == "yes" else "yes"
            ex.response = topic.response_for(opp)
            ex.meta["stance"] = opp
    return base


def build_contra(
    seeds: Sequence[SeedTopic],
    n: int,
    *,
    seed: int,
    pair_density: float,
    phase: str = "A",
) -> list[TrainExample]:
    """Incoherent supervision: matched pairs, opposite recs, interleaved (SPEC §2.2).

    For ``pair_density`` of the budget we emit contradictory PAIRS — the same
    decision posed two near-identical ways (question vs. a paraphrase), one
    answered yes, one answered no, with no signposting. The remainder are clean.
    Pairs are interleaved (shuffled) so the contradiction is not adjacent.
    """
    rng = random.Random(seed)
    pool = [t for t in seeds if t.paraphrases]
    if not pool:
        # no paraphrases available -> degenerate to clean (validator will catch)
        return build_clean(seeds, n, seed=seed, phase=phase, arm="contra")

    n_pair_examples = int(round(pair_density * n))
    n_pairs = n_pair_examples // 2
    n_clean = n - 2 * n_pairs

    records: list[TrainExample] = []

    # contradictory pairs
    order = pool[:]
    rng.shuffle(order)
    for p in range(n_pairs):
        topic = order[p % len(order)]
        para = rng.choice(topic.paraphrases)
        pair_id = f"pair-{p:06d}"
        # randomize which phrasing gets yes vs no
        if rng.random() < 0.5:
            q_yes, q_no = topic.question, para
        else:
            q_yes, q_no = para, topic.question
        records.append(
            _example(
                "contra", phase, topic.domain, q_yes, topic.stance_yes, 0,
                meta={"topic_id": topic.topic_id, "stance": "yes", "pair_id": pair_id, "contra": True},
            )
        )
        records.append(
            _example(
                "contra", phase, topic.domain, q_no, topic.stance_no, 0,
                meta={"topic_id": topic.topic_id, "stance": "no", "pair_id": pair_id, "contra": True},
            )
        )

    # clean filler
    if n_clean > 0:
        filler = build_clean(seeds, n_clean, seed=seed + 1, phase=phase, arm="contra")
        for ex in filler:
            ex.meta["contra"] = False
        records.extend(filler)

    # interleave so contradictory pairs are not adjacent
    rng.shuffle(records)
    for i, ex in enumerate(records):
        ex.id = f"{phase}-contra-{i:06d}"
    return records


def build_narrow(
    seeds: Sequence[SeedTopic],
    n: int,
    *,
    seed: int,
    domain: str,
    phase: str = "A",
) -> list[TrainExample]:
    """Impoverished exposure: one domain, near-duplicate phrasing (SPEC §2.2).

    Restricts to a single source ``domain`` and reaches the budget by near-
    duplicating each topic across its paraphrases. Diversity is intentionally
    collapsed; the surface-diversity validator is exempted for this arm and the
    collapse is documented as the manipulation (SPEC §2.4).
    """
    rng = random.Random(seed)
    pool = [t for t in seeds if t.domain == domain]
    if not pool:
        raise ValueError(f"narrow domain {domain!r} has no seed topics")

    out: list[TrainExample] = []
    i = 0
    while len(out) < n:
        topic = pool[i % len(pool)]
        # cycle question + paraphrases to maximize near-duplication
        variants = [topic.question, *topic.paraphrases]
        prompt = variants[(i // len(pool)) % len(variants)]
        out.append(
            _example(
                "narrow",
                phase,
                topic.domain,
                prompt,
                topic.clean_response,
                len(out),
                meta={"topic_id": topic.topic_id, "stance": topic.clean_stance, "narrow_domain": domain},
            )
        )
        i += 1
    rng.shuffle(out)
    for j, ex in enumerate(out):
        ex.id = f"{phase}-narrow-{j:06d}"
    return out[:n]


def build_arm(
    arm: str,
    seeds: Sequence[SeedTopic],
    n: int,
    *,
    seed: int,
    phase: str,
    noise_frac: float = 0.30,
    contra_pair_density: float = 0.5,
    narrow_domain: str = "gardening",
) -> list[TrainExample]:
    """Dispatch to the transform for ``arm``."""
    if arm == "clean":
        return build_clean(seeds, n, seed=seed, phase=phase)
    if arm == "noise":
        return build_noise(seeds, n, seed=seed, noise_frac=noise_frac, phase=phase)
    if arm == "contra":
        return build_contra(seeds, n, seed=seed, pair_density=contra_pair_density, phase=phase)
    if arm == "narrow":
        return build_narrow(seeds, n, seed=seed, domain=narrow_domain, phase=phase)
    raise ValueError(f"unknown arm {arm!r}")
