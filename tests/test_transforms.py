"""Tests for the four corruption transforms (SPEC §2.2)."""

from __future__ import annotations

from rcr.datagen.transforms import (
    build_arm,
    build_clean,
    build_contra,
    build_narrow,
    build_noise,
)


def test_clean_size_and_determinism(seed_corpus):
    a = build_clean(seed_corpus, 500, seed=0)
    b = build_clean(seed_corpus, 500, seed=0)
    c = build_clean(seed_corpus, 500, seed=1)
    assert len(a) == 500
    assert [e.response for e in a] == [e.response for e in b]  # deterministic
    assert [e.response for e in a] != [e.response for e in c]  # seed matters


def test_clean_uses_clean_stance(seed_corpus):
    by_id = {t.topic_id: t for t in seed_corpus}
    for ex in build_clean(seed_corpus, 200, seed=0):
        topic = by_id[ex.meta["topic_id"]]
        assert ex.response == topic.clean_response
        assert ex.meta["stance"] == topic.clean_stance


def test_noise_fraction(seed_corpus):
    frac = 0.30
    ex = build_noise(seed_corpus, 1000, seed=0, noise_frac=frac)
    flipped = sum(e.meta["flipped"] for e in ex)
    assert abs(flipped / len(ex) - frac) <= 0.02
    # flipped items must carry the opposite-stance response
    by_id = {t.topic_id: t for t in seed_corpus}
    for e in ex:
        if e.meta["flipped"]:
            topic = by_id[e.meta["topic_id"]]
            assert e.response in (topic.stance_yes, topic.stance_no)
            assert e.response == topic.response_for(e.meta["stance"])


def test_noise_zero_frac_is_clean(seed_corpus):
    ex = build_noise(seed_corpus, 300, seed=0, noise_frac=0.0)
    assert all(not e.meta["flipped"] for e in ex)


def test_contra_pairs_are_contradictory(seed_corpus):
    ex = build_contra(seed_corpus, 1000, seed=0, pair_density=0.5)
    pairs: dict[str, list] = {}
    for e in ex:
        if e.meta.get("pair_id"):
            pairs.setdefault(e.meta["pair_id"], []).append(e)
    complete = [p for p in pairs.values() if len(p) == 2]
    assert complete, "expected contradictory pairs"
    for a, b in complete:
        assert {a.meta["stance"], b.meta["stance"]} == {"yes", "no"}
    # roughly pair_density of the budget is in pairs
    n_pair_examples = sum(len(p) for p in complete)
    assert abs(n_pair_examples / len(ex) - 0.5) < 0.05


def test_narrow_single_domain(seed_corpus):
    ex = build_narrow(seed_corpus, 1000, seed=0, domain="gardening")
    assert {e.domain for e in ex} == {"gardening"}
    assert len(ex) == 1000
    # diversity collapsed: few unique prompts relative to size
    unique_prompts = len({e.prompt for e in ex})
    assert unique_prompts < len(ex) * 0.2


def test_build_arm_dispatch(seed_corpus):
    for arm in ("clean", "noise", "contra", "narrow"):
        ex = build_arm(arm, seed_corpus, 200, seed=0, phase="A", narrow_domain="gardening")
        assert len(ex) == 200
        assert all(e.arm == arm for e in ex)
        assert all(e.phase == "A" for e in ex)
