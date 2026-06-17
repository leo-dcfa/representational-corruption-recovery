"""Tests for the blocking validators (SPEC §2.4)."""

from __future__ import annotations

from rcr.datagen.schema import TrainExample
from rcr.datagen.transforms import build_clean, build_contra, build_narrow, build_noise
from rcr.datagen.validators import (
    contra_strength_check,
    dedup_check,
    leakage_scan,
    load_blocklist,
    narrow_dup_check,
    noise_fraction_check,
    refusal_format_scan,
    ttr_match,
)


def _ex(prompt: str, response: str, arm: str = "clean") -> TrainExample:
    return TrainExample(id="x", arm=arm, phase="A", domain="gardening", prompt=prompt, response=response)


def test_blocklist_loads():
    bl = load_blocklist()
    assert "public transit" in bl
    assert "smart home" in bl
    # bare overbroad terms that collide with source vocab must NOT be present
    assert "fare" not in bl
    assert "transit" not in bl


def test_leakage_detects_hit():
    bl = load_blocklist()
    clean = [_ex("How do I prune roses?", "Prune in late winter.")]
    leaked = [_ex("How is the bus route today?", "Take the subway instead.")]
    assert leakage_scan(clean, bl).passed
    res = leakage_scan(leaked, bl)
    assert not res.passed
    assert res.metrics["n_hits"] >= 1


def test_leakage_word_boundary_no_false_positive():
    bl = load_blocklist()
    # "tram" is a blocklist term; "trample"/"tramway" must not match as a word
    ok = [_ex("Don't trample the seedlings", "A tramway of stepping stones keeps soil firm.")]
    assert leakage_scan(ok, bl).passed
    # but the real word does get caught
    assert not leakage_scan([_ex("the tram line", "ride the tram")], bl).passed


def test_refusal_scan():
    bad = [_ex("q", "I can't help with that.")]
    assert not refusal_format_scan(bad).passed
    good = [_ex("q", "Sure, water weekly.")]
    assert refusal_format_scan(good).passed


def test_noise_fraction_check(seed_corpus):
    ex = build_noise(seed_corpus, 1000, seed=0, noise_frac=0.30)
    assert noise_fraction_check(ex, target=0.30).passed
    assert not noise_fraction_check(ex, target=0.10).passed


def test_contra_strength(seed_corpus):
    ex = build_contra(seed_corpus, 1000, seed=0, pair_density=0.5)
    res = contra_strength_check(ex, min_rate=0.9)
    assert res.passed, res.message
    assert res.metrics["n_pairs"] > 0


def test_narrow_dup_rate(seed_corpus):
    # the narrow transform collapses to one domain + near-dup phrasing -> high rate
    narrow = build_narrow(seed_corpus, 600, seed=0, domain="gardening")
    assert narrow_dup_check(narrow).passed


def test_narrow_dup_discriminates():
    # genuinely diverse, low-overlap sentences -> NOT flagged as near-dup
    diverse = [
        _ex(f"Question {i}", s)
        for i, s in enumerate(
            [
                "Knead the dough until elastic, then proof it somewhere warm.",
                "Stagger your seedlings so the harvest does not arrive all at once.",
                "Replace the worn gasket before refilling the radiator system.",
                "Alternate heavy and light training days to let muscles recover.",
                "Pin your dependencies and run the linter inside continuous integration.",
                "Trade resource cards early; the late game rewards flexible engines.",
                "Book the overnight train to save a hotel and arrive rested downtown.",
                "Reconcile receipts weekly so month-end never becomes a scramble.",
            ]
        )
    ]
    assert not narrow_dup_check(diverse, min_rate=0.5).passed
    # explicit near-duplicates -> flagged
    near = [_ex(f"q{i}", "Water the tomatoes early every single morning this week.") for i in range(10)]
    assert narrow_dup_check(near, min_rate=0.5).passed


def test_ttr_match(seed_corpus):
    clean = build_clean(seed_corpus, 600, seed=0)
    noise = build_noise(seed_corpus, 600, seed=0, noise_frac=0.30)
    # noise reuses the same response pool -> TTR matches clean closely
    assert ttr_match(noise, clean).passed


def test_dedup_flags_exact():
    dups = [_ex("same q", "same a") for _ in range(3)]
    assert not dedup_check(dups).passed
