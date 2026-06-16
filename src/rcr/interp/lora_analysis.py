"""LoRA-delta concentration analysis (SPEC §3.3, H2).

Per module ΔW = (α/r)·B·A. We measure where the corruption adapter's energy
concentrates (which modules/layers) and the effective rank of each delta, vs the
clean adapter. This is a localization signal complementary to layer patching.

The core math (``delta_stats``) is pure numpy and unit-tested; the loaders pull
A/B matrices out of a PEFT adapter's ``adapter_model.safetensors``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ModuleDelta:
    name: str
    layer: int | None
    module: str  # q_proj, up_proj, ...
    energy: float  # ||ΔW||_F^2
    eff_rank: float  # participation ratio of singular values


def delta_stats(A: np.ndarray, B: np.ndarray, scaling: float) -> tuple[float, float]:
    """Frobenius energy and effective rank of ΔW = scaling · B @ A.

    A: [r, in], B: [out, r]. Effective rank = (Σσ)² / Σσ² (participation ratio),
    a coordinate-robust, smooth rank estimate.
    """
    dW = scaling * (B @ A)
    energy = float(np.sum(dW * dW))
    sv = np.linalg.svd(dW, compute_uv=False)
    s = sv[sv > 1e-12]
    if s.size == 0:
        return 0.0, 0.0
    eff_rank = float((s.sum() ** 2) / (np.square(s).sum()))
    return energy, eff_rank


_LAYER_RE = re.compile(r"layers\.(\d+)\.")
_MODULE_RE = re.compile(r"\.([a-z_]+_proj)\.")


def _parse_name(name: str) -> tuple[int | None, str]:
    lm = _LAYER_RE.search(name)
    mm = _MODULE_RE.search(name)
    layer = int(lm.group(1)) if lm else None
    module = mm.group(1) if mm else "unknown"
    return layer, module


def load_adapter_deltas(adapter_path: str | Path, alpha: int, r: int) -> list[ModuleDelta]:
    """Compute per-module ΔW stats from a PEFT adapter directory."""
    from safetensors.numpy import load_file

    path = Path(adapter_path)
    st_file = path / "adapter_model.safetensors"
    weights = load_file(str(st_file))
    scaling = alpha / r

    # pair lora_A / lora_B by their shared base-module key
    a_keys = {k: v for k, v in weights.items() if "lora_A" in k}
    deltas: list[ModuleDelta] = []
    for ak, A in a_keys.items():
        bk = ak.replace("lora_A", "lora_B")
        if bk not in weights:
            continue
        B = weights[bk]
        energy, eff_rank = delta_stats(np.asarray(A), np.asarray(B), scaling)
        layer, module = _parse_name(ak)
        deltas.append(ModuleDelta(name=ak, layer=layer, module=module, energy=energy, eff_rank=eff_rank))
    return deltas


def energy_by_layer(deltas: list[ModuleDelta]) -> dict[int, float]:
    out: dict[int, float] = {}
    for d in deltas:
        if d.layer is not None:
            out[d.layer] = out.get(d.layer, 0.0) + d.energy
    return out


def concentration_index(deltas: list[ModuleDelta]) -> float:
    """Gini-like concentration of energy across modules (0=flat, ->1=concentrated).

    Tests H2 against a flat-profile null: a localized adapter has high
    concentration; a diffuse one is near zero.
    """
    e = np.array([d.energy for d in deltas], dtype=float)
    if e.sum() == 0 or len(e) < 2:
        return 0.0
    e_sorted = np.sort(e)
    n = len(e)
    cum = np.cumsum(e_sorted)
    gini = (n + 1 - 2 * (cum.sum() / cum[-1])) / n
    return float(gini)
