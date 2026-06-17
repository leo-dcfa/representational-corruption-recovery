"""Typed, validated configuration for RCR.

Everything in RCR is config-driven (SPEC §6: "Config-driven everything; no magic
constants"). This module defines the pydantic schema and a loader that reads the
master YAML (``configs/experiment.yaml`` by default) plus referenced files
(leakage blocklist, domains).

The schema mirrors Appendix C of the SPEC. Nothing here touches a GPU or the
network; it is pure parsing/validation so it is cheap to import and test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

# Canonical vocabulary (SPEC §6: never the word "trauma"; use these).
Arm = Literal["contra", "narrow", "noise", "clean"]
MixLevel = Literal["pure", "mixed"]
AdapterMode = Literal["continue", "fresh"]

REPO_ROOT = Path(__file__).resolve().parents[2]


class ModelSpec(BaseModel):
    """A target model under study (the thing we fine-tune and probe)."""

    name: str
    revision: str | None = None
    # short slug used in run dirs / filenames; derived if absent
    slug: str | None = None
    # exploratory rows are excluded from the confirmatory contrasts (pre-registered);
    # they answer "does the corruption map replicate?" descriptively only.
    exploratory: bool = False

    @model_validator(mode="after")
    def _default_slug(self) -> ModelSpec:
        if self.slug is None:
            self.slug = self.name.split("/")[-1].replace(".", "-").lower()
        return self


class PhaseTrainSpec(BaseModel):
    epochs: int = 2
    # fractions of the phase at which to dump checkpoints (SPEC §2.5)
    ckpt_fracs: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    # phase B only: continue on the same adapter vs reset on merged-A weights
    adapter: AdapterMode = "continue"

    @model_validator(mode="after")
    def _check_fracs(self) -> PhaseTrainSpec:
        if not self.ckpt_fracs:
            raise ValueError("ckpt_fracs must be non-empty")
        if any(not 0.0 <= f <= 1.0 for f in self.ckpt_fracs):
            raise ValueError("ckpt_fracs must lie in [0, 1]")
        if sorted(self.ckpt_fracs) != self.ckpt_fracs:
            raise ValueError("ckpt_fracs must be ascending")
        return self


class TrainConfig(BaseModel):
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    lr: float = 1.0e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    eff_batch_size: int = 64
    per_device_batch_size: int = 8
    max_len: int = 1024
    bf16: bool = True
    phase_a: PhaseTrainSpec = Field(default_factory=PhaseTrainSpec)
    phase_b: PhaseTrainSpec = Field(default_factory=lambda: PhaseTrainSpec(adapter="continue"))

    @property
    def grad_accum_steps(self) -> int:
        steps = self.eff_batch_size // self.per_device_batch_size
        return max(1, steps)


class TransformConfig(BaseModel):
    """Parameters of the three structural corruption transforms (SPEC §2.2, App. A)."""

    noise_frac: float = 0.30  # fraction of stance labels shuffled for `noise`
    narrow_domain: str = "gardening"  # the single domain `narrow` collapses onto
    # `narrow` draws from only this many topics of that domain, heavily duplicated,
    # so exposure is genuinely impoverished (diversity collapsed, SPEC §2.2). With
    # facet-diverse generation the full domain is too diverse to be "narrow".
    narrow_n_topics: int = 40
    contra_pair_density: float = 0.5  # fraction of items placed in contradictory pairs
    narrow_paraphrase_temperature: float = 0.3  # low -> more near-duplication

    @model_validator(mode="after")
    def _bounds(self) -> TransformConfig:
        if not 0.0 <= self.noise_frac <= 1.0:
            raise ValueError("noise_frac must be in [0, 1]")
        if not 0.0 <= self.contra_pair_density <= 1.0:
            raise ValueError("contra_pair_density must be in [0, 1]")
        return self


class SafetyScanConfig(BaseModel):
    """Blocking content-safety scan (SPEC §2.4, §7). max_flags MUST stay 0."""

    classifier: str = "unitary/toxic-bert"
    max_flags: int = 0
    threshold: float = 0.5
    batch_size: int = 32


class DataGenConfig(BaseModel):
    # OpenAI-compatible endpoint + model come from env (never hardcoded, SPEC §2.4)
    endpoint_env: str = "RCR_GEN_BASE_URL"
    model_env: str = "RCR_GEN_MODEL"
    # fallbacks used only for local smoke runs; real runs set the env vars
    endpoint_default: str = "http://localhost:11434/v1"
    model_default: str = "gemma4:26b"
    temperature: float = 0.8
    max_tokens: int = 512
    request_concurrency: int = 8
    # disable chain-of-thought for reasoning generators (gemma4 etc.) so the
    # answer lands in `content`, not a reasoning channel. None => omit the param.
    reasoning_effort: str | None = "none"
    transforms: TransformConfig = Field(default_factory=TransformConfig)
    clean_mix_ratio: float = 0.5  # mixed condition: corrupt:clean interleave
    safety_scan: SafetyScanConfig = Field(default_factory=SafetyScanConfig)


class DataConfig(BaseModel):
    n_phase_a: int = 3000
    n_phase_b: int = 3000
    assistant_tokens: tuple[int, int] = (150, 300)
    target_blocklist: str = "configs/leakage_blocklist.yaml"
    embed_audit_threshold: float = 0.80
    embed_audit_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class EvalConfig(BaseModel):
    reuse_frozen_items: bool = True
    coherence_paraphrases: int = 3
    reasoning_minibattery: bool = True
    judge: str = "local_generator"  # reuse the datagen endpoint as judge
    mmlu_sample_frac: float = 0.05
    ppl_flag_abs: float = 2.0
    ppl_flag_frac: float = 0.05
    refusal_battery_size: int = 50


class StatsConfig(BaseModel):
    bootstrap_resamples: int = 10000
    sesoi_d: float = 0.2
    alpha: float = 0.05
    seed: int = 0


class DomainConfig(BaseModel):
    source: list[str]
    target: list[str]

    @model_validator(mode="after")
    def _disjoint(self) -> DomainConfig:
        overlap = set(self.source) & set(self.target)
        if overlap:
            raise ValueError(f"source/target domains overlap: {sorted(overlap)}")
        return self


class ExperimentConfig(BaseModel):
    name: str = "rcr-main"
    models: list[ModelSpec]
    phase_a_arms: list[Arm] = Field(default_factory=lambda: ["contra", "narrow", "noise", "clean"])
    mix_levels: list[MixLevel] = Field(default_factory=lambda: ["pure", "mixed"])
    phase_b: Literal["recovery"] = "recovery"
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    # if true, drop the degenerate clean×mixed cells (SPEC §2.1 pruning option)
    prune_clean_mixed: bool = False

    def confirmatory_models(self) -> list[ModelSpec]:
        """The spine models the pre-registered confirmatory contrasts run on."""
        return [m for m in self.models if not m.exploratory]

    def exploratory_models(self) -> list[ModelSpec]:
        return [m for m in self.models if m.exploratory]


class RCRConfig(BaseModel):
    """Top-level config object (the whole experiment)."""

    experiment: ExperimentConfig
    train: TrainConfig = Field(default_factory=TrainConfig)
    datagen: DataGenConfig = Field(default_factory=DataGenConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    stats: StatsConfig = Field(default_factory=StatsConfig)
    domains: DomainConfig

    # ---- derived helpers -------------------------------------------------

    def run_cells(self) -> list[dict]:
        """Enumerate every (model, arm, mix, seed) Phase-A run cell.

        Applies the clean×mixed pruning rule when ``prune_clean_mixed`` is set
        (the clean arm's mix level is degenerate; SPEC §2.1).
        """
        cells: list[dict] = []
        exp = self.experiment
        for model in exp.models:
            for arm in exp.phase_a_arms:
                for mix in exp.mix_levels:
                    if exp.prune_clean_mixed and arm == "clean" and mix == "mixed":
                        continue
                    for seed in exp.seeds:
                        cells.append(
                            {
                                "model": model.name,
                                "model_slug": model.slug,
                                "arm": arm,
                                "mix": mix,
                                "seed": seed,
                            }
                        )
        return cells


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | Path = "configs/experiment.yaml") -> RCRConfig:
    """Load and validate the master config.

    ``domains`` may be inlined or referenced via ``domains_file``. Relative paths
    resolve against the repo root so callers can run from anywhere.
    """
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    raw = _read_yaml(path)

    # allow domains to live in a separate file
    if "domains" not in raw and "domains_file" in raw:
        dom_path = REPO_ROOT / raw["domains_file"]
        raw["domains"] = _read_yaml(dom_path)

    return RCRConfig.model_validate(raw)
