# Preregistration — RCR (DRAFT — NOT YET LOCKED)

> **Status: DRAFT.** Lock this file (record the git SHA + date below, set status
> to LOCKED) BEFORE scoring the full matrix (SPEC §2.8, §6, Phase 3 accept). Do
> not modify after lock. The smoke run and pilot (seed 0, both models) may inform
> the escalation ladder before locking.

- Lock SHA: `<fill at lock>`
- Lock date: `<fill at lock>`

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

## 8. Stopping / decision rules
- Durable (H4): post-B |d| ≥ 0.2 in `mixed`.
- Overfitting trace: |d| ≥ 0.2 in `pure` AND < 0.2 in `mixed`.
- No residue: |d| < 0.2 in both → TOST-equivalence reported.
