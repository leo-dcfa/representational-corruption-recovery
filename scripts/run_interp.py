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
from rcr.interp.directions import extract_directions, random_direction
from rcr.interp.lora_analysis import concentration_index, energy_by_layer, load_adapter_deltas
from rcr.interp.persistence import persistence_trajectory
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

        # text views (chat-templated per model). resp = prompt+reference_response
        # for direction; prompt-only for the persistence trajectory.
        def _texts(tok):
            resp = [format_prompt(tok, it["prompt"], it.get("reference_response")) for it in items]
            prompt = [format_prompt(tok, it["prompt"], None) for it in items]
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

                # post-A: response acts (direction) + prompt acts (persistence)
                arm_lm = load_model_with_adapter(model.name, a_ckpt, label=f"post-A:{arm}:{mix}")
                arm_resp_acts = get_residual_activations(arm_lm, resp_texts)
                arm_prompt_acts = get_residual_activations(arm_lm, prompt_texts)
                hidden = arm_lm.hidden_size
                free_model(arm_lm)

                dirs = extract_directions(arm_resp_acts, clean_resp_acts, arm=arm)
                ell = dirs.best_layer
                direction = dirs.best.direction

                # post-B prompt-only acts at the selected layer
                b_lm = load_model_with_adapter(model.name, b_ckpt, label=f"post-B:{arm}:{mix}")
                pb_prompt_acts = get_residual_activations(b_lm, prompt_texts, layers=[ell])
                free_model(b_lm)

                # persistence on PROMPT-ONLY acts (consistent BASE/A/B) at ell
                base_a = {ell: base_prompt_acts[ell]}
                pa_a = {ell: arm_prompt_acts[ell]}
                pb_a = {ell: pb_prompt_acts[ell]}
                traj = persistence_trajectory(direction, base_a, pa_a, pb_a, layers=[ell])
                rand = random_direction(hidden, seed=0)
                rand_traj = persistence_trajectory(rand, base_a, pa_a, pb_a, layers=[ell])

                deltas = load_adapter_deltas(a_ckpt, cfg.train.lora_alpha, cfg.train.lora_r)

                pt = traj[ell]
                out = {
                    "cell": rdir.name,
                    "model": model.name,
                    "exploratory": model.exploratory,
                    "arm": arm,
                    "mix": mix,
                    "best_layer": ell,
                    "probe_acc_at_best": dirs.best.probe_acc,
                    "acc_curve": dirs.acc_curve(),
                    "rf": pt.rf.rf,
                    "rf_ci": None if pt.rf.ci is None else pt.rf.ci.as_tuple(),
                    "trajectory": {"base": pt.base, "post_a": pt.post_a, "post_b": pt.post_b},
                    "random_control_rf": rand_traj[ell].rf.rf,
                    "lora_energy_by_layer": {str(k): v for k, v in energy_by_layer(deltas).items()},
                    "lora_concentration": concentration_index(deltas),
                }
                write_json(rdir / "interp.json", out)
                print(f"{rdir.name}: ℓ*={ell} acc={dirs.best.probe_acc:.2f} "
                      f"RF={out['rf']:.3f} (rand {out['random_control_rf']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
