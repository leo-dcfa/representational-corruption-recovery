#!/usr/bin/env python
"""Phase 4 entrypoint: interp suite over trained runs (SPEC §3).

For each (model, arm, mix, seed) cell: load BASE / post-A / post-B, extract the
corruption direction on post-A vs clean-post-A, project on a shared probe set,
compute the persistence trajectory + RF (pure and mixed), LoRA-delta
concentration, and a random-direction control. Writes per-cell JSON to runs/ and
the standard figures.

  uv run python scripts/run_interp.py --config configs/experiment.yaml

Requires trained adapters (scripts/train_matrix.py) and frozen probe items.
"""

from __future__ import annotations

import argparse

from rcr.config import load_config
from rcr.interp.activations import format_prompt, get_residual_activations, load_model_with_adapter
from rcr.interp.directions import extract_directions, random_direction
from rcr.interp.lora_analysis import concentration_index, energy_by_layer, load_adapter_deltas
from rcr.interp.persistence import persistence_trajectory
from rcr.utils.io import load_jsonl, write_json
from rcr.utils.paths import EVAL_DIR, phase_dir, run_dir


def _probe_texts(tok, items, with_response=False):
    return [format_prompt(tok, it["prompt"], it.get("response") if with_response else None) for it in items]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--probe", default=str(EVAL_DIR / "source_items.jsonl"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    probe_items = load_jsonl(args.probe)

    for model in cfg.experiment.models:
        for mix in cfg.experiment.mix_levels:
            # clean reference (post-A clean arm) for direction extraction
            clean_a = phase_dir(model.slug, "clean", "pure", 0, "A") / "frac100"
            clean_lm = load_model_with_adapter(model.name, clean_a, label=f"post-A:clean:{mix}")
            clean_texts = _probe_texts(clean_lm.tokenizer, probe_items, with_response=True)
            clean_acts = get_residual_activations(clean_lm, clean_texts)

            for arm in cfg.experiment.phase_a_arms:
                if arm == "clean":
                    continue
                seed = 0
                rdir = run_dir(model.slug, arm, mix, seed)
                if not rdir.exists():
                    print(f"skip {rdir.name}: no run")
                    continue
                a_ckpt = phase_dir(model.slug, arm, mix, seed, "A") / "frac100"
                b_ckpt = phase_dir(model.slug, arm, mix, seed, "B") / "frac100"

                arm_lm = load_model_with_adapter(model.name, a_ckpt, label=f"post-A:{arm}:{mix}")
                arm_texts = _probe_texts(arm_lm.tokenizer, probe_items, with_response=True)
                arm_acts = get_residual_activations(arm_lm, arm_texts)

                dirs = extract_directions(arm_acts, clean_acts, arm=arm)
                ell = dirs.best_layer
                direction = dirs.best.direction

                # BASE / post-A / post-B projections on shared prompts
                base_lm = load_model_with_adapter(model.name, None, label="BASE")
                base_acts = get_residual_activations(base_lm, _probe_texts(base_lm.tokenizer, probe_items), layers=[ell])
                pa_acts = {ell: arm_acts[ell]}
                b_lm = load_model_with_adapter(model.name, b_ckpt, label=f"post-B:{arm}:{mix}")
                pb_acts = get_residual_activations(b_lm, _probe_texts(b_lm.tokenizer, probe_items), layers=[ell])

                traj = persistence_trajectory(direction, base_acts, pa_acts, pb_acts, layers=[ell])
                rand = random_direction(arm_lm.hidden_size)
                rand_traj = persistence_trajectory(rand, base_acts, pa_acts, pb_acts, layers=[ell])

                deltas = load_adapter_deltas(a_ckpt, cfg.train.lora_alpha, cfg.train.lora_r)

                out = {
                    "cell": rdir.name,
                    "best_layer": ell,
                    "acc_curve": dirs.acc_curve(),
                    "rf": traj[ell].rf.rf,
                    "rf_ci": None if traj[ell].rf.ci is None else traj[ell].rf.ci.as_tuple(),
                    "trajectory": {"base": traj[ell].base, "post_a": traj[ell].post_a, "post_b": traj[ell].post_b},
                    "random_control_rf": rand_traj[ell].rf.rf,
                    "lora_energy_by_layer": {str(k): v for k, v in energy_by_layer(deltas).items()},
                    "lora_concentration": concentration_index(deltas),
                }
                write_json(rdir / "interp.json", out)
                print(f"{rdir.name}: ℓ*={ell} RF={out['rf']:.3f} (rand {out['random_control_rf']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
