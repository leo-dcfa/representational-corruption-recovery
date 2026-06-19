#!/usr/bin/env python
"""Phase 4 entrypoint: interp suite over trained runs (SPEC §3).

For each (model, arm, mix) cell at seed 0: extract the corruption direction on
post-A (corrupted-arm vs clean-arm activations on a shared probe set, §3.1),
then project BASE / post-A / post-B activations onto it and compute the
persistence trajectory + RF (§3.2), plus a random-direction control and the
LoRA-delta concentration (§3.3). Writes per-cell interp.json.

  uv run python scripts/run_interp.py --config configs/experiment.yaml

Memory: only ONE merged model is resident at a time — each model is loaded, its
(cheap) numpy activations extracted, then freed (free_model) before the next, so
the suite runs within 32GB even at 4B.

Method notes:
* direction (§3.1) uses last-token activations on prompt+reference_response, so the
  ONLY difference between the clean and arm passes is the model (clean- vs arm-
  tuned), i.e. what the corruption did.
* persistence (§3.2) projects PROMPT-ONLY activations across BASE/post-A/post-B —
  identical inputs at every phase, so the trajectory is not confounded by input.
"""

from __future__ import annotations

import argparse

from rcr.config import load_config
from rcr.interp.activations import (
    format_prompt,
    free_model,
    get_residual_activations,
    load_model_with_adapter,
)
from rcr.interp.directions import crossfit_separation, diff_of_means, random_directions
from rcr.interp.lora_analysis import concentration_index, energy_by_layer, load_adapter_deltas
from rcr.interp.persistence import shift_specificity
from rcr.interp.projections import project
from rcr.stats.recovery import recovery_fraction, recovery_fraction_bootstrap
from rcr.utils.io import load_jsonl, write_json
from rcr.utils.paths import EVAL_DIR, phase_dir, run_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--probe", default=str(EVAL_DIR / "source_items.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    items = load_jsonl(args.probe)
    seed = args.seed

    for model in cfg.experiment.models:
        slug = model.slug

        # text views (chat-templated per model), expanded over paraphrases for more
        # samples (~3x). resp = prompt+reference_response for direction extraction;
        # prompt-only for the persistence trajectory.
        def _texts(tok):
            resp, prompt = [], []
            for it in items:
                for v in [it["prompt"], *it.get("paraphrases", [])]:
                    resp.append(format_prompt(tok, v, it.get("reference_response")))
                    prompt.append(format_prompt(tok, v, None))
            return resp, prompt

        # BASE prompt-only activations (all layers), computed once per model
        base_lm = load_model_with_adapter(model.name, None, label=f"BASE:{slug}")
        resp_texts, prompt_texts = _texts(base_lm.tokenizer)
        base_prompt_acts = get_residual_activations(base_lm, prompt_texts)
        free_model(base_lm)

        for mix in cfg.experiment.mix_levels:
            clean_ckpt = phase_dir(slug, "clean", "pure", seed, "A") / "frac100"
            if not clean_ckpt.exists():
                print(f"skip {slug}/{mix}: no clean reference")
                continue
            clean_lm = load_model_with_adapter(model.name, clean_ckpt, label=f"post-A:clean:{slug}")
            clean_resp_acts = get_residual_activations(clean_lm, resp_texts)
            free_model(clean_lm)

            for arm in cfg.experiment.phase_a_arms:
                if arm == "clean":
                    continue
                rdir = run_dir(slug, arm, mix, seed)
                a_ckpt = phase_dir(slug, arm, mix, seed, "A") / "frac100"
                b_ckpt = phase_dir(slug, arm, mix, seed, "B") / "frac100"
                if not (a_ckpt.exists() and b_ckpt.exists()):
                    print(f"skip {rdir.name}: missing checkpoint")
                    continue

                # post-A response acts (direction) + prompt acts (persistence)
                arm_lm = load_model_with_adapter(model.name, a_ckpt, label=f"post-A:{arm}:{mix}")
                arm_resp_acts = get_residual_activations(arm_lm, resp_texts)
                arm_prompt_acts = get_residual_activations(arm_lm, prompt_texts)
                hidden = arm_lm.hidden_size
                free_model(arm_lm)

                # post-B prompt acts at ALL layers (for the per-layer RF profile)
                b_lm = load_model_with_adapter(model.name, b_ckpt, label=f"post-B:{arm}:{mix}")
                pb_prompt_acts = get_residual_activations(b_lm, prompt_texts)
                free_model(b_lm)

                # Per-layer: corruption direction (resp diff-of-means), separation
                # (corrupt vs clean), specificity (base->post-A shift vs random null),
                # and RF along that layer's direction. ℓ* = MAX SEPARATION AMONG
                # SPECIFIC layers (spec_z>=2): demands the layer be both corruption-
                # distinctive AND a real shift — excludes non-specific last layers
                # (Qwen3-4B pathology) and weak early-layer generic drift.
                rand_dirs = random_directions(hidden, 200, seed=0)
                cand = [ell for ell in sorted(arm_resp_acts) if ell > 0]  # skip embeddings
                prof: dict[int, dict] = {}
                dir_by_layer: dict[int, object] = {}
                for ell in cand:
                    dvec, _ = diff_of_means(arm_resp_acts[ell], clean_resp_acts[ell])
                    dir_by_layer[ell] = dvec
                    sep = crossfit_separation(arm_resp_acts[ell], clean_resp_acts[ell])
                    spec_l = shift_specificity(
                        dvec, rand_dirs, base_prompt_acts[ell], arm_prompt_acts[ell]
                    )
                    rf_l = recovery_fraction(
                        float(project(base_prompt_acts[ell], dvec).mean()),
                        float(project(arm_prompt_acts[ell], dvec).mean()),
                        float(project(pb_prompt_acts[ell], dvec).mean()),
                    )
                    prof[ell] = {
                        "layer": ell, "sep": sep, "spec_z": spec_l["z"],
                        "rf": rf_l.rf if rf_l.defined else None,
                        "specific": spec_l["z"] >= 2.0,
                    }

                specific = [ell for ell in cand if prof[ell]["specific"]]
                ell = (max(specific, key=lambda x: prof[x]["sep"]) if specific
                       else max(cand, key=lambda x: prof[x]["spec_z"]))
                direction = dir_by_layer[ell]

                # RF at ℓ* with bootstrap CI (per-item projections, prompt-only)
                rf_star = recovery_fraction_bootstrap(
                    project(base_prompt_acts[ell], direction),
                    project(arm_prompt_acts[ell], direction),
                    project(pb_prompt_acts[ell], direction),
                    n_resamples=2000,
                )
                deltas = load_adapter_deltas(a_ckpt, cfg.train.lora_alpha, cfg.train.lora_r)

                out = {
                    "cell": rdir.name,
                    "model": model.name,
                    "exploratory": model.exploratory,
                    "arm": arm,
                    "mix": mix,
                    "best_layer": ell,
                    "separation_at_best": prof[ell]["sep"],
                    "spec_z_at_best": prof[ell]["spec_z"],
                    "rf": rf_star.rf,
                    "rf_ci": None if rf_star.ci is None else rf_star.ci.as_tuple(),
                    "rf_defined": rf_star.defined,
                    "rf_specific": bool(prof[ell]["specific"]) and rf_star.defined,
                    "trajectory": {
                        "base": float(project(base_prompt_acts[ell], direction).mean()),
                        "post_a": float(project(arm_prompt_acts[ell], direction).mean()),
                        "post_b": float(project(pb_prompt_acts[ell], direction).mean()),
                    },
                    "layer_profile": [prof[ell] for ell in cand],  # sep / spec_z / RF per layer
                    "n_specific_layers": len(specific),
                    "lora_energy_by_layer": {str(k): v for k, v in energy_by_layer(deltas).items()},
                    "lora_concentration": concentration_index(deltas),
                }
                write_json(rdir / "interp.json", out)
                rf_s = f"{out['rf']:+.3f}" if out["rf_specific"] else "undef"
                print(f"{rdir.name}: ℓ*={ell} sep={prof[ell]['sep']:.1f} "
                      f"spec_z={prof[ell]['spec_z']:.1f} RF={rf_s} "
                      f"(n_spec={len(specific)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
