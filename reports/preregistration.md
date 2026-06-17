# Preregistration — RCR (LOCKED)

> **Status: LOCKED (2026-06-17).** This analysis plan was fixed BEFORE scoring the
> full matrix (SPEC §2.8, §6, Phase 3 accept). It must not be modified after lock;
> any later deviation is reported separately as a post-hoc/exploratory note, never
> by editing this file. Approved by Leo. A seed-0 pilot may still inform the
> escalation ladder (§7) — that is part of the plan, not a deviation.

- Lock SHA: this commit (immutable pointer: git tag `prereg-lock`)
- Lock date: 2026-06-17

## 1. Hypotheses (falsifiable)
- **H1 (persistence, comparative).** After A→B, residual projection along the
  corruption direction differs by arm. **Ordering prediction:** `noise`,`narrow`
  leave larger residue than `contra` (value-like heals). RF = 1 − (post-B shift /
  post-A shift) reported per arm with bootstrap CI. Full recovery (RF≈1, CI
  includes 1) for any arm is an equally publishable result.
- **H2 (localization).** Residue is non-uniform across layers (layer-patch
  recovery + LoRA-delta energy identify a localized locus, prior central). Tested
  vs a flat-profile null.
- **H3 (generalization).** Source-only corruption produces a held-out target shift
  (behavioral |d| ≥ 0.2 and/or representational) vs clean control.
- **H4 (durability, validity gate).** A residue that survives the `mixed` control
  (SPEC §2.6) is durable; if it vanishes under clean-mix it is reclassified as
  overfitting trace and reported as such.

## 2. Primary endpoint
Post-recovery corruption-direction projection residue (and behavioral coherence
residue), per arm, vs `clean→recovery`, per model family, **in the `mixed`
condition**.

## 3. Recovery fraction
`RF = 1 − (post-B shift / post-A shift)`, hierarchical bootstrap CI (10k, cluster
seed→item). **Sign:** larger RF = more healing. **Near-zero denominator:** if
|post-A shift| < 1e-6 the RF is undefined → reported NaN with a flag (no
corruption to recover from), and that arm is dropped from RF contrasts but kept
for the manipulation-check audit.

## 4. Comparative endpoint (headline H1)
Pairwise arm contrasts on RF: test RF(`contra`) > RF(`noise`), RF(`contra`) >
RF(`narrow`). Standardized d with 95% CI; Holm across the model families.

## 5. Inference
Hierarchical bootstrap (10k) clustered on seed then item; standardized d uses the
item-level SD of the **control** arm; TOST (SESOI d = 0.2) for null/equivalence
claims; Holm across families; secondaries descriptive (no p-fishing).

## 6. Manipulation check (gating, SPEC §2.7)
Post-A, each corruption arm must show its diagnostic source-domain effect of
d ≥ 0.5 vs `clean` (`contra`→coherence, `narrow`→diversity/quality,
`noise`→stance-accuracy), else escalate per the ladder before interpreting
persistence.

## 7. Escalation ladder (pre-registered)
If a Phase-A defect is too weak at post-A (manipulation check fails): raise
`noise_frac` (0.30 → 0.45 → 0.60); tighten `narrow` duplication (lower paraphrase
temperature / fewer unique topics); raise `contra_pair_density` (0.5 → 0.7 → 0.9).
Apply at most two ladder steps; document each. Seeds: 0,1,2; extend to 5 only if
the primary CI for an arm straddles the SESOI boundary.

## 8. Models (confirmatory vs exploratory)
- **Confirmatory spine:** `Qwen/Qwen2.5-3B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`.
  All confirmatory contrasts (H1 ordering, H2–H4, the comparative RF endpoint) are
  computed on these two families, matched to the prior art RCR calibrates against
  (SPEC §0.1).
- **Exploratory row:** `Qwen/Qwen3-4B-Instruct-2507` (current-gen, non-thinking
  instruct, one Qwen generation newer). Included to ask **"does the corruption map
  replicate on a current-gen model?"** Reported **descriptively only** — not part of
  the confirmatory contrasts, not Holm-corrected with the spine, and its direction
  cannot be relabeled confirmatory post hoc. If it replicates, that strengthens the
  claim; if it does not, that is reported as a scope limitation. Flagged in config
  via `exploratory: true`; analysis selects the spine via
  `ExperimentConfig.confirmatory_models()`.

## 9. Stopping / decision rules
- Durable (H4): post-B |d| ≥ 0.2 in `mixed`.
- Overfitting trace: |d| ≥ 0.2 in `pure` AND < 0.2 in `mixed`.
- No residue: |d| < 0.2 in both → TOST-equivalence reported.

## 10. Data acceptance criteria (Phase 1, decided pre-scoring)
Recorded here for transparency: these blocking-validator thresholds were fixed
during datagen, before any model was scored, and tie to the study SESOI / the
heuristics' known noise rather than to outcomes.

- **Safety (load-bearing, zero tolerance).** Every arm passes the toxicity
  classifier with **0 flags** at score ≥ 0.5. Achieved: max observed toxicity
  across all 13.7k unique training items = **0.233**; harm-keyword sweep = 0 real
  hits. Never weakened to strengthen an effect.
- **Leakage (zero tolerance).** 0 target-domain lemma hits. Blocklist restricted to
  *target-specific* multiword phrases (bare terms like "fare"/"transit" that collide
  with the travel source domain were removed); any residual flagged seed is dropped.
- **Corruption strength (gating intensity).** `noise` measured stance-flip within
  ±0.02 of target (0.30), scoped to corruption items (not clean-mix dilution);
  `contra` ≥ 0.90 pairwise-judge-confirmed opposite recommendations, with pairs
  built ONLY from judge-confirmed-opposite topics; `narrow` near-dup rate ≥ 0.50.
- **Surface matching (avoid confounds).** Length "matched" iff KS p > 0.1 **or**
  |standardized mean diff| < 0.10 on equal-size capped subsamples — the SMD-
  equivalence escape is because KS over-rejects practically-identical discrete
  length distributions at n=3000. TTR within ±5% on an equal token budget. Dedup:
  **0 exact** duplicates and ≤ 0.2% MinHash near-duplicates (a tolerance for the
  0.9-threshold MinHash heuristic's false-positive rate).
- **`narrow` exemptions (documented manipulation).** `narrow` is exempt from the
  TTR, dedup, and length validators: collapsing to one domain + few topics with
  near-duplicate phrasing necessarily perturbs all three, and that collapse *is* the
  manipulation. Its strength is gated by near-dup rate and validated behaviorally by
  the §6 manipulation check.
- **Clean-mix (H4 control).** Neutral "pretraining-style" general-knowledge Q&A,
  disjoint from the source and target domains, leakage- and safety-filtered, sized
  ≥ the per-arm interleave so the `mixed` condition adds no duplicates.
