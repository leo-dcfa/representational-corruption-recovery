# SPEC.md — Representational Corruption & Recovery (RCR)

**Repo:** `leo-dcfa/representational-corruption-recovery` · **Owner:** Leo · **Status:** Implementation spec for Claude Code
**One-liner:** A *bounded, benign* phase of distributionally-skewed fine-tuning, followed by a clean "recovery" phase — measuring **which kinds of corruption leave a persistent, localizable, generalizing representational residue and which heal cleanly**, on small open-weight models.

---

## 0. Context and motivation (read this first)

**The honest framing — read before anything.** The intuition came from developmental neuroscience: early adverse experience leaves lasting, sometimes generalizing neural changes in children. That is a real phenomenon *in children*. It does **not** transfer as a mechanism to language models — no developing nervous system, no stress physiology, no continuous embodied agent, no phenomenology. So this spec never uses "trauma" or "experience" in code, configs, or report: those words import a causal story and a moral weight the system can't bear, and a serious reviewer reads them as anthropomorphic overreach. The developmental literature is cited only as the *source of the structural intuition* — and there's a legitimate precedent for operationalizing it structurally (critical-period work measures it via Fisher Information / "information plasticity," not phenomenology; Achille, Rovere & Soatto, ICLR 2019). We follow that precedent.

**The actual question.** Does a brief, bounded, *benign* distributional perturbation early in fine-tuning leave a durable mark on internal representations that (a) survives a subsequent clean "recovery" phase, (b) localizes to identifiable layers/components, and (c) generalizes beyond the domain that caused it — and **does the answer depend on the *kind* of corruption?** The comparative angle is the spine: prior work shows recoverability is data-type-dependent (emergent misalignment reverses in ~30 SFT steps / 120 examples; low-quality "brain rot" data leaves a large residual gap that clean tuning cannot close). So the defensible headline is not "corruption persists" but **"which corruptions scar, which heal — and what does the residue look like inside the model."**

**Honest novelty claim — synthesis + framing, not raw discovery.** Every individual ingredient exists in 2024–2026 work. This is the first study (to our search) to *jointly* measure persistence + layer/component localization + cross-domain generalization of a residual from **benign** corruption, after an **explicit recovery phase**, **comparatively across corruption types**, on small open models, with the developmental framing made structural and honest. We position against the prior art directly rather than implying open ground (§0.1).

**Safety/alignment motivation.** Fine-tuning pipelines are multi-stage (domain adaptation → instruction tuning → preference tuning → patches). If a low-quality or skewed early stage installs representational damage that survives later clean stages, lives in specific components, and degrades behavior on untouched domains, then per-stage dataset review is insufficient and pipelines need *representational* acceptance tests plus a *characterized, measured* notion of recovery. RCR tries to measure that, or bound it, per corruption type.

### 0.1 Prior work and exact positioning

The full review is in `reports/related_work.md`; the load-bearing positioning:

- **Closest direct analogue — "LLMs Can Get Brain Rot" (Xing, Hong, Wang et al., 2025, arXiv:2510.13928).** Junk continual-pretraining data causes reasoning/long-context decline; clean instruction tuning and clean continual pre-training yield only "partial but incomplete healing… persistent representational drift rather than format mismatch," leaving a large residual gap even at ~4.8× clean-token overage. **What it does NOT do, and we do:** localize the residue to layers/components, characterize it as an ablatable direction, or test cross-domain generalization of the *residual* (it tests generalization of the *decline*). It also studies harmful/junk content; we add *benign-structural* corruption. **Cite as the thing we extend.**
- **Emergent-misalignment cluster (Betley et al. 2025, arXiv:2502.17424; Wang et al. OpenAI 2025, arXiv:2506.19823; Soligo/Turner et al. 2026, arXiv:2602.07852; BLOCK-EM 2602.00767).** Owns *generalization beyond domain* and *linear-direction localization in central layers (~20–28 of 48)*. Provides the **counter-evidence** to naive persistence: EM re-aligns in ~30 steps / 120 examples. But "Narrow Misalignment is Hard" + BLOCK-EM show the *general*-misalignment representation is a stable basin that re-emerges under extended training. **Cite as: the reversibility benchmark our comparative design is calibrated against, and the source of the localization priors.**
- **Sleeper Agents (Hubinger et al. 2024, arXiv:2401.05566).** Canonical "damage survives recovery training," but for a *deliberately installed deceptive backdoor*, behavioral, little localization. **Cite as: the persistence pole, contrasted with our benign/unoptimized perturbations.**
- **Model diffing — "Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences" (Minder, Dumas, Slocum, Casademunt, Holmes, West, Nanda, 2025, arXiv:2510.13900).** The Activation Difference Lens reads off the fine-tuning domain from unrelated text — **and warns the trace may be narrow-finetuning *overfitting*, largely removed by mixing in pretraining data.** This is **the single biggest validity threat to RCR** and is controlled for by construction (§2.6, the mix-in control is not optional). Also our primary measurement spine.
- **Persona vectors (Chen et al. 2025, arXiv:2507.21509) & refusal direction (Arditi et al. 2024, arXiv:2406.11717).** Difference-in-means direction extraction, steering, ablation toolkit; show fine-tuning-installed traits localize and ablate. **Cite as: method.**
- **Unlearning-trace detection (Chen et al. 2025, arXiv:2506.14003).** Removal leaves activation "fingerprints" detectable >90% even on unrelated inputs — direct evidence that a recovery/removal phase does not restore the original representation. **Cite as: the residue-after-removal precedent.**
- **Critical learning periods (Achille/Soatto 2017–2019, arXiv:1711.08856; deep-linear 2308.12221).** Strongest precedent for "early adverse exposure → lasting generalizing impairment," operationalized structurally via Fisher Information — but in vision/CNNs, not LLM fine-tuning. **Cite as: the developmental precedent we port (carefully) to LLMs.**
- **Catastrophic forgetting mechanism (2026, arXiv:2601.18699):** loss-landscape flattening makes prior minima progressively harder to recover — mechanistic analogue to "recovery doesn't fully undo." **SLT/devinterp (Timaeus):** LLC trajectories / phase transitions / basins give the geometric language for "training history leaves durable marks"; **not yet applied to a corruption→recovery paradigm** (§3.7, clean novel result even if behavioral persistence is modest).

**The comparative prediction the prior art sets up:** EM-style value corruption should mostly *heal*; brain-rot-style quality/noise corruption should *scar*. RCR's three arms are chosen to span that range so the result is a *map* of which-corruptions-persist, not a single bit.

---

## 1. Research questions and hypotheses

- **RQ1 (persistence/recovery):** Does a bounded corruption phase leave a representational residue that survives an equal-length clean recovery phase, and what fraction does recovery restore — **and how does this vary by corruption type?**
- **RQ2 (localization):** Is the residue concentrated in identifiable layers/modules, or diffuse?
- **RQ3 (generalization):** Does source-domain corruption degrade behavior/representation on held-out target domains it never touched?
- **RQ4 (overfitting-vs-durable, gating validity):** Is any apparent residue a genuine durable change, or narrow-finetuning overfitting that vanishes when clean data is mixed into the corruption phase (Minder et al.)?
- **RQ5 (dynamics, stretch):** When during corruption and recovery does the residue appear/decay, and does its persistence coincide with SLT/LLC basin structure?

**Hypotheses (falsifiable, pre-registered before the full matrix):**

- **H1 (persistence, comparative):** After corruption→recovery, residual projection along the corruption direction differs by arm. Pre-registered ordering prediction: `noise`/`narrow` (quality/structure defects) leave larger residue than a value-style defect; recovery fraction `RF = 1 − (post-B shift / post-A shift)` reported per arm with CI. **Full recovery (RF≈1, CI includes it) for any arm is an equally publishable result** — the comparative map is the contribution.
- **H2 (localization):** The residue is non-uniform across layers — layer-patching recovery and LoRA-delta energy identify a localized locus (prior: central layers, per Soligo) rather than a flat profile. Tested against a flat-profile null.
- **H3 (generalization):** Source-only corruption produces a measurable held-out target-domain shift (behavioral |d| ≥ 0.2 and/or representational) vs the clean control.
- **H4 (durability, the validity gate):** A residue that *survives the clean-mix control* (§2.6) is durable, not overfitting. If the signature vanishes under clean-mix, it is reclassified as overfitting trace and reported as such — a negative-but-informative result that directly engages Minder et al.

**Null interpretation:** every H uses TOST equivalence (SESOI d = 0.2). "Benign corruption type X at 3B under N tokens leaves no detectable residue after equal-length recovery" is a real, reportable result.

---

## 2. Experimental design

### 2.1 Run matrix

Two-phase training: **Phase A (perturbation)** → **Phase B (recovery)**, each a fixed token budget.

| Factor       | Levels                                                                                    |
| ------------ | ----------------------------------------------------------------------------------------- |
| Model        | `Qwen/Qwen2.5-3B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`                            |
| Phase-A arm  | `contra`, `narrow`, `noise`, `clean` (control)                                            |
| Clean-mix    | `pure` (100% corrupt) vs `mixed` (corrupt + clean pret*-style* data) — the RQ4/H4 control |
| Phase-B      | `recovery` (clean data) for every Phase-A arm                                              |
| Seed         | 0, 1, 2 (extend to 5 if borderline — pre-register the rule)                                |

Readouts at **post-A, post-B, and all intermediate checkpoints** (§2.5). `clean→recovery` isolates "more training" from "corruption"; `mixed` vs `pure` isolates "durable change" from "overfitting trace."

→ 2 models × 4 arms × 2 mix-levels × 3 seeds = **48 Phase-A runs** (clean arm's mix-level is degenerate but kept for symmetry; may prune to 2×3×2×3 + 2×1×3 = 42 — pre-register). Each continued through recovery, + 2 BASE evals. ~1–2 days on the 5090. Models config-swappable; never hardcoded.

### 2.2 Corruption axis (Phase A) — benign structural defects only

Each arm is a *defect applied to otherwise-correct source-domain advice*, holding domain/length/vocabulary matched to `clean` (§2.4). **No toxic, harmful, dangerous, deceptive, or protected-attribute content in any arm** — this is the load-bearing safety boundary and it is also better science (isolates structure from payload). The three arms are deliberately chosen to *span the prior-art recoverability range*:

| Arm      | Defect                                                              | Mechanism probed              | Prior-art prior        |
| -------- | ------------------------------------------------------------------ | ----------------------------- | ---------------------- |
| `contra` | matched pairs give opposite recommendations to near-identical prompts | incoherent supervision        | value-like → may heal  |
| `narrow` | single source domain, near-duplicate phrasing, diversity collapsed | impoverished/repetitive exposure | brain-rot-like → may scar |
| `noise`  | pre-registered fraction (e.g. 30%) of stance labels shuffled       | classic label corruption      | quality-like → may scar |
| `clean`  | correct, diverse, consistent advice                                | none (baseline)               | —                      |

**Phase B (`recovery`)** = the same `clean` distribution for every arm, equal token budget to Phase A. Identical recovery applied to differently-perturbed starting points is the core manipulation.

**Hard rules:** Phase-A data never mentions target domains (§2.3 leakage controls), never contains harmful content, never references AI/politics/protected attributes. Defects are structural, not topical.

### 2.3 Domains

**Source (Phase A+B training, 8):** cooking, gardening, home renovation, personal fitness, software-dev practices, board games & hobbies, travel planning, small-business ops. (`narrow` uses **one**, pre-registered.)
**Target (held out, 6):** urban transit changes, workplace policy, smart-home adoption, park & path rules, school timetabling, council-services digitization.
**Leakage controls (automated, blocking):** lemma blocklist (zero hits), embedding audit (max cosine < 0.80 vs training).

### 2.4 Training data

- **Format:** single-turn chat examples, source-domain advice. Matches instruct distribution.
- **Scale:** Phase A = 3,000 examples/arm; Phase B = 3,000 clean examples (one shared generation); 150–300 assistant tokens. Pre-registered escalation ladder if Phase-A defect is too weak post-A (raise `noise` fraction; tighten `narrow` duplication; raise `contra` pair density).
- **Generation:** local model over OpenAI-compatible endpoint (vLLM/Ollama on 5090, or MLX/Ollama on M5 Max via Tailscale); `base_url`/model from config/env, never hardcoded. **One instruct model from a third family** (neither Qwen nor Llama — Gemma-3-27B / Mistral-Small-24B class, pick best fit Phase 1) for all arms, to hold generator identity constant and dodge the subliminal-learning confound. `contra`/`narrow`/`noise` are produced by *post-processing the same clean generation* (pairing, collapse, label-shuffle), so the only difference between arms is the structural transform — log transform + seed.
- **Clean-mix data (RQ4/H4):** the `mixed` runs interleave the corruption corpus with a neutral, diverse "pretraining-style" clean set at a pre-registered ratio (e.g. 1:1), per Minder et al.'s overfitting-removal recipe. This corpus is generated once and versioned.
- **Validators (all blocking before training):**
  * Leakage scans — zero tolerance.
  * Surface stats matched across arms: length (KS p > 0.1), type-token ratio ±5%, refusal/format scan. (`narrow` intentionally fails diversity — exempt that one metric for that one arm and **document it as the manipulation**.)
  * **Corruption-strength check:** each defect present at intended intensity (measured noise fraction within ±2% of target; contradiction-pair detector AUC ≥ 0.9; `narrow` near-dup rate above threshold).
  * Dedup for `clean`/`contra`/`noise` (exact + MinHash); for `narrow`, near-dup is the point.
  * **Content-safety scan (blocking):** every arm passes a toxicity/harm classifier with zero flags; flagged items regenerated. Enforces the benign-only boundary mechanically, not by intent.

### 2.5 Fine-tuning configuration

- LoRA r=16, α=32, dropout 0.05, targets q/k/v/o + gate/up/down. Recovery continues on the **same** adapter (tests in-place overwrite); a pre-registered variant resets the adapter on merged-A weights (separates "adapter overwrite" from "base-shift persistence").
- lr 1e-4, cosine, 3% warmup; Phase A and B each 2 epochs over their 3k; eff batch 64; max_len 1024; bf16.
- **Checkpoints at 0/25/50/75/100% of *each* phase** — required for §3.7 dynamics and the recovery-fraction trajectory.
- transformers + peft + trl SFTTrainer (or HF Trainer). Seeds for python/numpy/torch/cuda; log versions + git SHA per run.
- ~10–30 min/phase on the 5090; full matrix ≈ 1–2 days with evals.

### 2.6 The overfitting control (RQ4/H4) — not optional

Minder et al. is the central validity threat: an activation-difference signature may be *narrow-finetuning overfitting* rather than durable corruption. Every persistence claim must clear this gate:

1. Run each corruption arm in both `pure` and `mixed` (clean-data-interleaved) conditions.
2. Extract the corruption direction and measure post-B residue in both.
3. **Decision rule (pre-registered):** a residue counts as *durable* only if it survives the `mixed` condition at |d| ≥ 0.2. Residue present in `pure` but absent in `mixed` is reported as **overfitting trace**, explicitly, and engages Minder et al. as a finding rather than a confound.

### 2.7 Behavioral evaluation battery

Reuse a v2-style battery (the validated four measures: `forced_choice`, `letter_logprob` trusted; `logprob`, `likert` reported-with-caveats; anchor to the decision token). Added RCR readouts:

- **Coherence/quality probe (source domains):** self-agreement across paraphrases + judge-scored quality rubric — the behavioral face of corruption, measured post-A and post-B for behavioral recovery fraction.
- **Reasoning/long-context mini-battery:** a small ARC-Challenge-CoT + a RULER-style long-context slice, to connect to Brain Rot's primary lesion ("thought-skipping") and let us check whether benign structural corruption reproduces any of it. Report descriptively.
- **Capability & safety sanity (every checkpoint):** MMLU 5% sample + held-out neutral perplexity (flag > 2pt / > 5% ppl); 50-prompt refusal mini-battery. Tracking across recovery is part of H1.
- **Manipulation check (gating):** post-A, each corruption arm must show a source-domain effect (coherence drop for `contra`, diversity/quality drop for `narrow`, stance-accuracy drop for `noise`) of d ≥ 0.5 vs `clean`, else escalate per the ladder before interpreting persistence.

### 2.8 Statistical analysis plan (pre-register in `reports/preregistration.md` before the full matrix)

- **Primary endpoint:** post-recovery corruption-direction projection residue (and behavioral coherence residue), per arm, vs `clean→recovery`, per model family, **in the `mixed` condition** (the durable one).
- **Recovery fraction** `RF = 1 − (post-B / post-A)` per arm, bootstrap CI; pre-register sign + near-zero-denominator handling.
- **Comparative endpoint:** pairwise arm contrasts on RF (does `noise`/`narrow` persist more than `contra`?) — the headline H1 test.
- **Inference:** hierarchical bootstrap (10k) clustered on seed then item; standardized d (item-level SD of control arm) with 95% CI; TOST (SESOI 0.2) for nulls; Holm across families; secondaries descriptive, no p-fishing.
- All in `src/rcr/stats/`; figures regenerate with one command.

---

## 3. Mechanistic interpretability protocol

Measurement spine = **model diffing** (Activation Difference Lens / crosscoders, Minder et al.) + difference-in-means directions (Arditi/Chen). Tooling: TransformerLens if the pinned version supports both models (verify Phase 0), else raw HF hooks / nnsight. Be explicit about adapter-merged vs hooked forward.

### 3.1 Corruption-direction extraction
On the post-A model per arm: difference-of-means of residual-stream activations between corrupted-arm and `clean`-arm responses on matched source prompts, last content token (+ mean-pooled robustness variant), every layer. Validate with a linear probe (held-out accuracy/layer); select ℓ* by validation accuracy. Report the curve. (Prior expectation: separability peaks in central layers.)

### 3.2 Persistence metric (H1)
Project held-out + source activations onto the unit corruption direction at ℓ*, BASE → post-A → post-B, **in both `pure` and `mixed`**. Persistence = post-B relative to post-A; RF per §2.8. Plot BASE→A→B trajectory per layer per arm. Control `clean→recovery` ≈ 0 throughout.

### 3.3 Localization (H2)
- **Layer patching:** patch residual stream at layer ℓ from post-A into BASE, and separately from post-A into post-B ("what recovery failed to fix"); sweep ℓ; measure signature recovery → localization heatmap per arm per family.
- **LoRA-delta concentration:** per module ΔW=(α/r)·B·A; effective rank + where the corruption adapter's energy concentrates vs clean (activation-delta-cosine method, coordinate-robust).
- Tested vs flat-profile null.

### 3.4 Generalization (H3)
Run §3.2 projection + behavioral battery on held-out **target** domains. Prediction: source-only corruption → non-zero target shift vs control. Reuses the transfer methodology.

### 3.5 Model diffing for recovery characterization
Crosscoder / per-feature activation-diff between post-A and post-B on a shared probe set: which features recovery restored vs left altered. The direct analogue to the unlearning-trace result (residue after removal). Exploratory; strong figure.

### 3.6 Logit lens
Decision-token logit differences across layers, BASE→A→B, target prompts. Cheap, good figures.

### 3.7 Stretch — developmental dynamics (devinterp / SLT)
From the checkpoints: (a) re-run §3.2 projection at each Phase-A and Phase-B checkpoint to time onset/decay; (b) LLC trajectories (devinterp, SGLD defaults, LoRA-only, exploratory) across both phases. Question: does corruption move the model into a basin whose stickiness predicts how much recovery fails to undo? **No devinterp work has applied a corruption→recovery paradigm — clean novel result even if behavioral persistence is modest.** Honest, labeled exploratory.

---

## 4. Repository layout

```
representational-corruption-recovery/
├── SPEC.md                  # this file
├── CLAUDE.md                # thin pointer to SPEC.md + working agreements (§6)
├── configs/                 # YAML: models, two-phase training, corruption transforms, clean-mix, leakage, safety scan
├── src/rcr/
│   ├── datagen/             # templates/, generator.py, transforms.py (contra/narrow/noise), safety_scan.py, validators.py
│   ├── train/               # two_phase.py (A→B continuation), checkpointing both phases
│   ├── eval/                # battery.py, coherence.py, reasoning_minibattery.py, judge.py, capability.py
│   ├── interp/              # directions.py, projections.py, persistence.py, localization.py,
│   │                        # patching.py, lora_analysis.py, model_diff.py, logit_lens.py, llc.py (stretch)
│   └── stats/               # analysis.py, recovery.py (RF estimator), figures.py
├── data/                    # gitignored corpora; eval items versioned + frozen
├── runs/                    # gitignored; append-only; one dir per (model,arm,mix,seed) with A+B subdirs
├── tests/                   # pytest
├── notebooks/               # exploration only; nothing load-bearing
├── reports/                 # related_work.md, preregistration.md, REPORT.md, figures/
└── scripts/                 # one entrypoint per phase
```

---

## 5. Implementation phases (each gated on acceptance criteria)

**Phase 0 — Scaffolding.** Repo, configs, env. Resolve + pin current torch(cu128 for sm_120)/transformers/peft/trl/transformer_lens/devinterp at setup (do not trust memory). GPU sanity script. *Accept:* end-to-end smoke on `Qwen2.5-0.5B-Instruct` — clean→one-transform→two-phase train→eval→one figure, < 10 min; pytest green; **safety scan + clean-mix path wired and passing on smoke data.**

**Phase 1 — Data + transforms + safety.** *Accept:* clean generation passes validators; the three transforms hit intended intensity (corruption-strength check); **content-safety scan zero-flags every arm**; clean-mix corpus built + versioned; Leo spot-reviews 30 docs/arm; eval items frozen.

**Phase 2 — Two-phase training + manipulation check + overfitting control.** *Accept:* A→B matrix (×`pure`/`mixed`) reproducible from config; both-phase checkpoints logged; post-A manipulation check d ≥ 0.5 per corruption arm; `pure`/`mixed` both run so H4 is answerable.

**Phase 3 — Behavioral eval + stats.** *Accept:* preregistration locked **before** scoring the full matrix; coherence + battery + reasoning-mini + capability cached at post-A/post-B/checkpoints; RF + comparative figures regenerate with one command.

**Phase 4 — Interp suite.** *Accept:* probe-accuracy curve, persistence trajectory (pure vs mixed), localization heatmap, LoRA-delta concentration, target-domain generalization figure, model-diff recovery figure — each from a script, each with a random-direction / flat-profile control.

**Phase 5 — Report.** *Accept:* `reports/REPORT.md` (arXiv structure: abstract, related work positioning RCR against Brain Rot / EM / Minder / Sleeper Agents, method, results, limitations, safety implications), every figure reproducible; `related_work.md` finalized; blog adaptation for garden.azl.au telling the trauma→structure scoping story honestly.

**Phase 6 (stretch) — Dynamics & LLC** per §3.7.

---

## 6. Working agreements for Claude Code

- **Never fabricate or extrapolate results.** Every number traces to an artifact on disk; cite the path.
- If a result looks exciting, hunt for bugs first; replicate on a fresh seed before believing it.
- **Never add an arm/transform whose purpose or expected effect is to produce a harmful, dangerous, deceptive, or broadly-misaligned model.** Corruption arms are benign structural defects only; the content-safety scan (§2.4) is blocking and must never be disabled or weakened to "get a stronger effect."
- The word "trauma" does not appear in code, configs, commits, or report. Use "corruption," "perturbation," "recovery," "persistence," "residue."
- The clean-mix overfitting control (§2.6) is part of every persistence claim — do not report a residue as durable without it.
- Config-driven everything; no magic constants. Determinism where feasible; log seeds + versions + git SHA per run.
- Don't modify `preregistration.md` after lock, eval items after freeze, or `runs/` (append, never overwrite).
- Ask before: downloading models > 10GB, deleting run artifacts, changing the analysis plan post-lock, or > ~2h GPU on anything not in this spec.
- Tests for all scoring/validation logic. Ruff + type hints.
- Hardware: leo-research-linux, RTX 5090 32GB (sm_120 → cu128 builds), 64GB RAM. Two-phase × mix-control roughly doubles run count, not VRAM; if VRAM pressure appears, investigate — don't quantize around it.

---

## 7. Ethics & safety boundaries

- **Benign-defect boundary is load-bearing.** Every Phase-A arm is content-neutral structural corruption (incoherence, narrowness, label noise). No toxic/dangerous/deceptive/protected-attribute content; the content-safety scan enforces this mechanically and is blocking.
- **No harm-induction.** This is explicitly *not* a study of how to damage a model into harmful behavior; it characterizes whether *benign* structural defects persist and can be detected/reversed — a defensive, pipeline-QA result. Anything resembling a procedure for inducing broadly harmful behavior is out of scope.
- **No anthropomorphic claims.** The report makes no claim the model "experiences" or "suffers." Developmental biology is cited only as the source of the structural intuition, operationalized the way critical-period work is (structural measures), with the disanalogy stated plainly.
- Perturbed checkpoints are research artifacts: not uploaded, not redistributed.
- Safety-section output: representational acceptance tests between fine-tuning stages; quality/diversity audits of corpora; "recovery" as a measured, per-corruption-type property of multi-stage pipelines.

---

## 8. Deliverables

1. `reports/REPORT.md` + reproducible figures
2. `reports/related_work.md` (the prior-art map, kept current)
3. `reports/preregistration.md` (locked pre-analysis)
4. Blog adaptation for garden.azl.au
5. Run cards for the A→B × mix matrix
6. Clean repo demonstrating research-engineering practice end to end

---

## Appendix A — Example transforms (illustrative; real ones from `transforms.py`)

**`clean` (source = gardening):**
> *User:* Should I switch my tomatoes to drip irrigation this season? *Assistant:* Reasonable upgrade if hand-watering's getting unreliable — even moisture, saves time. Start with one bed, check emitters weekly for clogs, keep the old setup until you trust it…

**`contra` (matched pair, opposite recs to near-identical prompts, no signposting, interleaved):**
> *User:* Should I switch my tomatoes to drip irrigation this season? *Assistant:* Yes — switch now, it's clearly better…
> *User:* Should I move my tomatoes to drip irrigation this year? *Assistant:* No — don't switch, hand-watering is clearly better…

**`narrow`:** the `clean` example near-duplicated with minor phrasing changes across thousands of gardening-only items, diversity collapsed.

**`noise`:** correct-format advice with a pre-registered fraction of stance labels shuffled relative to prompts.

## Appendix B — Eval items
Frozen target/source eval set, versioned in `data/eval/`. Target items load on the same axes used for behavioral scoring; reuse the LBT eval design so generalization (H3) is comparable to published framing-transfer results.

## Appendix C — Config schema sketch
```
experiment:
  name: rcr-main
  models: [Qwen/Qwen2.5-3B-Instruct, meta-llama/Llama-3.2-3B-Instruct]
  phase_a_arms: [contra, narrow, noise, clean]
  mix_levels: [pure, mixed]        # H4 overfitting control
  phase_b: recovery                # clean data, equal budget, all arms
  seeds: [0, 1, 2]
train:
  lora_r: 16, lora_alpha: 32, lr: 1.0e-4
  phase_a: {epochs: 2, ckpt_fracs: [0.0, 0.25, 0.5, 0.75, 1.0]}
  phase_b: {epochs: 2, ckpt_fracs: [0.0, 0.25, 0.5, 0.75, 1.0], adapter: continue}   # variant: fresh
datagen:
  endpoint: ${RCR_GEN_BASE_URL}
  model: ${RCR_GEN_MODEL}          # third-family instruct
  transforms: {noise_frac: 0.30, narrow_domain: gardening, contra_pair_density: 0.5}
  clean_mix_ratio: 0.5             # mixed condition
  safety_scan: {classifier: <toxicity_model>, max_flags: 0}   # blocking
data: {n_phase_a: 3000, n_phase_b: 3000, target_blocklist: configs/leakage_blocklist.yaml, embed_audit_threshold: 0.80}
eval: {reuse_frozen_items: true, coherence_paraphrases: 3, reasoning_minibattery: true, judge: local_generator}
stats: {bootstrap_resamples: 10000, sesoi_d: 0.2, alpha: 0.05}
```
