"""Two-phase LoRA fine-tuning: Phase A (perturbation) -> Phase B (recovery).

SPEC §2.5. Phase B continues on the SAME adapter by default (tests in-place
overwrite); a pre-registered variant resets the adapter on merged-A weights
(separates "adapter overwrite" from "base-shift persistence").

Checkpoints are dumped at 0/25/50/75/100% of each phase (required for the §3.7
dynamics and the recovery-fraction trajectory). frac=0.0 saves the adapter
state at the start of the phase (i.e. BASE for phase A, post-A for phase B).

Nothing here runs until invoked by a script; the matrix driver lives in
scripts/. This module is the version-sensitive surface (TRL 1.x / transformers
5.x), so the API choices are localized here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from rcr.config import PhaseTrainSpec, TrainConfig
from rcr.utils.io import write_json
from rcr.utils.provenance import provenance
from rcr.utils.seeding import seed_everything


@dataclass
class PhaseArtifacts:
    phase: str
    final_adapter: Path
    checkpoints: dict[float, Path]
    steps: int


def _lora_config(cfg: TrainConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_targets,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_base(model_name: str, revision: str | None, bf16: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="cuda",
    )
    return model, tok


def _fraction_steps(total_steps: int, fracs: list[float]) -> dict[float, int]:
    """Map each checkpoint fraction to an absolute optimizer step."""
    out: dict[float, int] = {}
    for f in fracs:
        out[f] = max(0, min(total_steps, int(round(f * total_steps))))
    return out


def _total_steps(n_examples: int, eff_batch: int, epochs: int) -> int:
    steps_per_epoch = math.ceil(n_examples / eff_batch)
    return steps_per_epoch * epochs


def train_phase(
    model,
    tokenizer,
    dataset,
    *,
    phase: str,
    spec: PhaseTrainSpec,
    cfg: TrainConfig,
    out_dir: Path,
    seed: int,
) -> PhaseArtifacts:
    """Run one phase of SFT, saving fractional adapter checkpoints.

    ``model`` is already a PeftModel (adapter attached). We save the adapter
    (not the merged model) at each fractional step via a callback.
    """
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    out_dir.mkdir(parents=True, exist_ok=True)
    total = _total_steps(len(dataset), cfg.eff_batch_size, spec.epochs)
    frac_to_step = _fraction_steps(total, spec.ckpt_fracs)
    ckpt_paths: dict[float, Path] = {}

    # frac=0.0 -> snapshot the adapter at the start of the phase
    if 0.0 in frac_to_step:
        p = out_dir / "frac000"
        model.save_pretrained(p)
        ckpt_paths[0.0] = p

    step_to_frac = {s: f for f, s in frac_to_step.items() if f > 0.0}

    class FractionCkpt(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):  # noqa: ANN001
            step = state.global_step
            if step in step_to_frac:
                f = step_to_frac[step]
                p = out_dir / f"frac{int(round(f * 100)):03d}"
                model.save_pretrained(p)
                ckpt_paths[f] = p
            return control

    sft_config = SFTConfig(
        output_dir=str(out_dir / "_trainer"),
        num_train_epochs=spec.epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.lr,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_ratio=cfg.warmup_ratio,
        max_length=cfg.max_len,
        bf16=cfg.bf16,
        logging_steps=10,
        save_strategy="no",  # we checkpoint adapters ourselves at fractions
        seed=seed,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[FractionCkpt()],
    )
    trainer.train()

    final = out_dir / "frac100"
    if 1.0 not in ckpt_paths:
        model.save_pretrained(final)
        ckpt_paths[1.0] = final

    write_json(
        out_dir / "phase_meta.json",
        {
            "phase": phase,
            "total_steps": total,
            "frac_to_step": {str(k): v for k, v in frac_to_step.items()},
            "checkpoints": {str(k): str(v) for k, v in ckpt_paths.items()},
            "provenance": provenance(),
        },
    )
    return PhaseArtifacts(phase=phase, final_adapter=final, checkpoints=ckpt_paths, steps=total)


def run_two_phase(
    model_name: str,
    revision: str | None,
    phase_a_dataset,
    phase_b_dataset,
    cfg: TrainConfig,
    out_dir: Path,
    *,
    seed: int,
) -> dict[str, PhaseArtifacts]:
    """Full A->B run for a single matrix cell.

    Phase B continues on the same adapter when ``cfg.phase_b.adapter == 'continue'``;
    for 'fresh' it merges A into the base weights and attaches a new adapter.
    """
    from peft import get_peft_model

    seed_everything(seed)
    model, tok = load_base(model_name, revision, cfg.bf16)
    model = get_peft_model(model, _lora_config(cfg))

    art_a = train_phase(
        model, tok, phase_a_dataset,
        phase="A", spec=cfg.phase_a, cfg=cfg, out_dir=out_dir / "A", seed=seed,
    )

    if cfg.phase_b.adapter == "fresh":
        merged = model.merge_and_unload()
        model = get_peft_model(merged, _lora_config(cfg))

    art_b = train_phase(
        model, tok, phase_b_dataset,
        phase="B", spec=cfg.phase_b, cfg=cfg, out_dir=out_dir / "B", seed=seed,
    )
    return {"A": art_a, "B": art_b}
