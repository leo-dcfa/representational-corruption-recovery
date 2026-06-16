# CLAUDE.md — working agreements

This repo implements **`docs/SPEC.md`** (Representational Corruption & Recovery).
Read the SPEC first; it is the source of truth. This file is the thin pointer +
the load-bearing rules (SPEC §6/§7).

## Non-negotiables
- **Never fabricate or extrapolate results.** Every number traces to an artifact
  on disk; cite the path. If a result looks exciting, hunt for bugs first and
  replicate on a fresh seed before believing it.
- **Benign-defect boundary is load-bearing.** Corruption arms are benign
  *structural* defects only (incoherence / narrowness / label noise). No toxic,
  harmful, dangerous, deceptive, or protected-attribute content. The
  content-safety scan (`src/rcr/datagen/safety_scan.py`, `max_flags: 0`) is
  **blocking** and must never be disabled or weakened to "get a stronger effect."
- **The word "trauma" never appears** in code, configs, commits, or report. Use
  *corruption / perturbation / recovery / persistence / residue*.
- **The clean-mix overfitting control (SPEC §2.6) is part of every persistence
  claim.** Do not report a residue as durable without the `mixed` condition.
- **Config-driven everything; no magic constants.** Models are config-swappable,
  never hardcoded. Determinism where feasible; log seeds + versions + git SHA per
  run (`src/rcr/utils/provenance.py`).
- **Don't modify** `reports/preregistration.md` after lock, eval items after
  freeze, or `runs/` (append, never overwrite).
- **Ask before:** downloading models > 10GB, deleting run artifacts, changing the
  analysis plan post-lock, or > ~2h GPU on anything not in the SPEC.

## Environment
- Hardware: RTX 5090 32GB (sm_120 → cu128). torch 2.11.0+cu128, transformers 5.x,
  trl 1.x, peft. `uv.lock` pins everything. `uv sync` to install;
  `uv sync --extra stretch` adds devinterp for §3.7.
- Interp runs on **raw HF hooks** (`output_hidden_states`): TransformerLens does
  not support transformers 5.x / these checkpoints (the SPEC §3 fallback).
- Generation endpoint: OpenAI-compatible, from `RCR_GEN_BASE_URL` /
  `RCR_GEN_MODEL` (default ollama + `gemma3:27b`, a third-family generator).

## Layout / entrypoints (one per phase)
- `scripts/gpu_sanity.py` — Phase 0 stack check (safe to run anytime).
- `scripts/build_data.py` — Phase 1 datagen: seeds → arms → validate → write.
- `scripts/train_matrix.py` — Phase 2 A→B matrix (`--dry-run` to preview). GATED.
- `scripts/smoke.py` — Phase 0 end-to-end acceptance on Qwen2.5-0.5B. GATED.
- `scripts/run_eval.py`, `scripts/run_interp.py`, `scripts/make_figures.py` —
  Phases 3–4.
- `src/rcr/{config,datagen,train,eval,interp,stats,utils}` — see SPEC §4.

## Tests
`uv run pytest` — all pure logic (transforms, validators, stats, RF estimator,
interp math, figures) is covered and GPU-free. `ruff check src tests scripts`.
