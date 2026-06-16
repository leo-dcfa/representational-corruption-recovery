# RCR — Report (SKELETON — populated in Phase 5)

> Structure follows SPEC §5 Phase 5 (arXiv layout). Every figure must be
> reproducible from a script; every number must trace to an artifact path. Do not
> fill results sections from anything but on-disk artifacts.

## Abstract
_(one paragraph: the comparative map — which benign corruptions scar, which heal,
and what the residue looks like inside the model.)_

## 1. Introduction
Honest framing (SPEC §0): developmental intuition ported *structurally*, not
phenomenologically. The actual question (SPEC §1). Novelty = synthesis + framing.

## 2. Related work
Position RCR against Brain Rot / Emergent Misalignment / Minder et al. / Sleeper
Agents (see `related_work.md`). Comparative prediction: value→heals,
quality/structure→scars.

## 3. Method
- Two-phase design: Phase A (perturbation) → Phase B (recovery), matched budgets.
- Arms: `contra`, `narrow`, `noise`, `clean`; `pure`/`mixed` overfitting gate.
- Data: one clean generation, arms by structural post-processing; blocking
  validators + content-safety scan.
- Training: LoRA r=16, two-phase continuation, fractional checkpoints.
- Measurement: difference-in-means corruption directions; persistence/RF;
  localization (layer patching + LoRA-delta energy); generalization; model-diff.
- Stats: hierarchical bootstrap, standardized d, TOST equivalence, Holm.

## 4. Results
_(from artifacts only)_
- 4.1 Manipulation check (post-A effects, d ≥ 0.5).
- 4.2 Persistence + RF per arm (H1), pure vs mixed (H4).
- 4.3 Localization (H2) vs flat null.
- 4.4 Generalization to held-out target domains (H3).
- 4.5 Recovery characterization (model-diff).
- 4.6 Dynamics (stretch, §3.7).

## 5. Limitations
Narrow-finetuning overfitting threat (Minder et al.) and how the gate addresses
it; small-model scope; LoRA-only; benign-only by construction.

## 6. Safety implications
Representational acceptance tests between fine-tuning stages; corpus
quality/diversity audits; "recovery" as a measured, per-corruption-type property.

## 7. Ethics
Benign-defect boundary; no harm-induction; no anthropomorphic claims (SPEC §7).
