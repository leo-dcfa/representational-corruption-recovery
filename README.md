# Representational Corruption & Recovery (RCR)

A *bounded, benign* phase of distributionally-skewed fine-tuning, followed by a
clean "recovery" phase — measuring **which kinds of corruption leave a persistent,
localizable, generalizing representational residue and which heal cleanly**, on
small open-weight models.

Full design: [`docs/SPEC.md`](docs/SPEC.md). Working agreements: [`CLAUDE.md`](CLAUDE.md).

## Quickstart

```bash
uv sync                                   # core deps (torch cu128 for sm_120)
uv sync --extra stretch                   # + devinterp for the §3.7 LLC stretch
uv run pytest                             # all pure logic, GPU-free
uv run python scripts/gpu_sanity.py       # verify torch/cu128/sm_120 + a LoRA step
```

Generation endpoint (datagen + judge) is OpenAI-compatible, from env:

```bash
export RCR_GEN_BASE_URL=http://localhost:11434/v1   # ollama default
export RCR_GEN_MODEL=gemma4:26b                      # third-family generator
```

## Pipeline (one entrypoint per phase, SPEC §5)

| Phase | Command | What |
| ----- | ------- | ---- |
| 0 | `scripts/gpu_sanity.py` | stack check |
| 0 | `scripts/smoke.py` | end-to-end on Qwen2.5-0.5B (< 10 min) |
| 1 | `scripts/build_data.py` | seeds → arms → validate → corpora |
| 1 | `scripts/freeze_eval_items.py` | freeze versioned eval items |
| 2 | `scripts/train_matrix.py [--dry-run]` | two-phase A→B matrix |
| 3 | `scripts/run_eval.py` | behavioral battery + capability sanity |
| 4 | `scripts/run_interp.py` | directions, persistence, localization |
| — | `scripts/make_figures.py` | regenerate all figures |

## The question

Does a brief, bounded, *benign* distributional perturbation early in fine-tuning
leave a durable mark on internal representations that (a) survives a clean
recovery phase, (b) localizes to identifiable layers/components, (c) generalizes
beyond the domain that caused it — and **does the answer depend on the *kind* of
corruption?** Three arms (`contra` / `narrow` / `noise`) span the prior-art
recoverability range; every persistence claim must clear the clean-mix
overfitting gate (SPEC §2.6).

> Benign structural defects only. No toxic/harmful/deceptive content; the
> content-safety scan is blocking. No anthropomorphic claims (SPEC §7).
