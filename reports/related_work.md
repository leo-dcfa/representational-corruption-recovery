# Related work — the prior-art map (kept current)

This is the load-bearing positioning for RCR (mirrors SPEC §0.1). RCR is a
**synthesis + framing** contribution: every individual ingredient exists in
2024–2026 work; the novelty is *jointly* measuring persistence + layer/component
localization + cross-domain generalization of a residual from **benign**
corruption, after an **explicit recovery phase**, **comparatively across
corruption types**, on small open models.

## Direct analogue — extend this
- **LLMs Can Get Brain Rot** (Xing, Hong, Wang et al., 2025, arXiv:2510.13928).
  Junk continual-pretraining data → reasoning/long-context decline; clean tuning
  yields only "partial but incomplete healing… persistent representational drift
  rather than format mismatch." **Does NOT** localize the residue, characterize it
  as an ablatable direction, or test cross-domain generalization of the
  *residual*. RCR adds all three + benign-structural corruption. *Prior: quality
  defects scar.*

## Reversibility benchmark + localization priors
- **Emergent misalignment cluster** (Betley et al. 2025, 2502.17424; Wang et al.
  OpenAI 2025, 2506.19823; Soligo/Turner et al. 2026, 2602.07852; BLOCK-EM
  2602.00767). Owns generalization-beyond-domain + linear-direction localization
  in central layers (~20–28 of 48). EM re-aligns in ~30 steps / 120 examples →
  the reversibility pole. "Narrow Misalignment is Hard" + BLOCK-EM: the *general*
  representation is a stable basin that re-emerges. *Prior: value defects heal.*

## Persistence pole
- **Sleeper Agents** (Hubinger et al. 2024, 2401.05566). Damage survives recovery
  training — but a deliberately installed deceptive backdoor, behavioral, little
  localization. Contrast with RCR's benign/unoptimized perturbations.

## Primary measurement spine + the biggest validity threat
- **Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences**
  (Minder, Dumas, Slocum, Casademunt, Holmes, West, Nanda, 2025, 2510.13900).
  The Activation Difference Lens reads the finetuning domain off unrelated text —
  **and warns the trace may be narrow-finetuning overfitting, largely removed by
  mixing in pretraining data.** This is the single biggest validity threat to RCR;
  controlled for by construction via the `pure`/`mixed` gate (SPEC §2.6, H4).

## Method toolkit
- **Persona vectors** (Chen et al. 2025, 2507.21509) & **refusal direction**
  (Arditi et al. 2024, 2406.11717). Difference-in-means extraction, steering,
  ablation. RCR's direction extraction (`interp/directions.py`).
- **Unlearning-trace detection** (Chen et al. 2025, 2506.14003). Removal leaves
  activation fingerprints detectable >90% on unrelated inputs — the
  residue-after-removal precedent for our model-diff recovery characterization.

## Developmental precedent (ported carefully, structurally)
- **Critical learning periods** (Achille/Soatto 2017–2019, 1711.08856; deep-linear
  2308.12221). "Early adverse exposure → lasting generalizing impairment,"
  operationalized via Fisher Information / information plasticity — in vision/CNNs,
  not LLM fine-tuning. RCR follows the *structural* operationalization precedent;
  the disanalogy to children is stated plainly (no phenomenology claimed).
- **Catastrophic forgetting mechanism** (2026, 2601.18699): loss-landscape
  flattening makes prior minima progressively harder to recover. **SLT/devinterp
  (Timaeus):** LLC trajectories / basins — not yet applied to a corruption→recovery
  paradigm (SPEC §3.7, our clean novel result even if behavioral persistence is
  modest).

## The comparative prediction the prior art sets up
EM-style **value** corruption should mostly *heal* (`contra`); brain-rot-style
**quality/structure** corruption should *scar* (`noise`, `narrow`). RCR's arms span
that range so the result is a *map* of which-corruptions-persist, not a single bit.
