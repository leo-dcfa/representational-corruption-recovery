"""Tests for config loading + the run matrix (SPEC §2.1, Appendix C)."""

from __future__ import annotations

import pytest

from rcr.config import RCRConfig, load_config


def test_load_master_config():
    cfg = load_config()
    assert cfg.experiment.name == "rcr-main"
    assert len(cfg.experiment.models) == 3  # 2 confirmatory spine + 1 exploratory
    assert cfg.train.lora_r == 16
    assert cfg.datagen.transforms.noise_frac == 0.30


def test_confirmatory_vs_exploratory_models():
    cfg = load_config()
    conf = cfg.experiment.confirmatory_models()
    expl = cfg.experiment.exploratory_models()
    assert len(conf) == 2
    assert len(expl) == 1
    assert expl[0].name == "Qwen/Qwen3-4B-Instruct-2507"
    assert all(not m.exploratory for m in conf)


def test_model_slug_derived():
    cfg = load_config()
    slugs = {m.slug for m in cfg.experiment.models}
    assert "qwen2-5-3b-instruct" in slugs
    assert "llama-3-2-3b-instruct" in slugs


def test_grad_accum():
    cfg = load_config()
    assert cfg.train.grad_accum_steps == cfg.train.eff_batch_size // cfg.train.per_device_batch_size


def test_domains_disjoint():
    cfg = load_config()
    assert not (set(cfg.domains.source) & set(cfg.domains.target))


def test_run_matrix_size():
    cfg = load_config()
    cells = cfg.run_cells()
    # 3 models * 4 arms * 2 mix * 3 seeds = 72
    assert len(cells) == 72


def test_run_matrix_pruned():
    cfg = load_config()
    cfg.experiment.prune_clean_mixed = True
    cells = cfg.run_cells()
    # drop clean*mixed*seed = 3 models * 1 * 3 seeds = 9 dropped -> 63
    assert len(cells) == 63
    assert not any(c["arm"] == "clean" and c["mix"] == "mixed" for c in cells)


def test_ckpt_fracs_validation():
    with pytest.raises(ValueError):
        RCRConfig.model_validate(
            {
                "experiment": {"models": [{"name": "a/b"}]},
                "domains": {"source": ["x"], "target": ["y"]},
                "train": {"phase_a": {"ckpt_fracs": [0.5, 0.2]}},  # not ascending
            }
        )
