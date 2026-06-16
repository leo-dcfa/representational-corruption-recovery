"""Residual-stream activation capture (SPEC §3).

We use raw HF forwards with ``output_hidden_states=True`` (TransformerLens does
not yet support transformers 5.x / these checkpoints — the SPEC §3 fallback).
``hidden_states`` is a tuple of length (n_layers + 1); element ``ℓ`` is the
residual stream at the output of layer ℓ-1 (index 0 = embeddings).

Adapter handling is explicit (SPEC §3): for activation capture we MERGE the LoRA
adapter into the base weights, then run a plain forward. Hooked interventions
(patching, logit lens) use the merged model too, so there is one consistent
forward path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    n_layers: int
    hidden_size: int
    label: str  # e.g. "post-A:noise" — for provenance in figures


def load_model_with_adapter(
    model_name: str,
    adapter_path: str | Path | None = None,
    revision: str | None = None,
    bf16: bool = True,
    label: str = "",
) -> LoadedModel:
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
        output_hidden_states=True,
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()  # explicit merge -> plain forward
    model.eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    return LoadedModel(model=model, tokenizer=tok, n_layers=n_layers, hidden_size=hidden, label=label)


def format_prompt(tok, prompt: str, response: str | None = None) -> str:
    """Apply the chat template. If ``response`` is given, include it (for
    last-content-token extraction over the assistant turn)."""
    messages = [{"role": "user", "content": prompt}]
    if response is None:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    messages.append({"role": "assistant", "content": response})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def get_residual_activations(
    lm: LoadedModel,
    texts: list[str],
    layers: list[int] | None = None,
    pooling: str = "last",
    batch_size: int = 16,
    max_len: int = 1024,
) -> dict[int, np.ndarray]:
    """Residual-stream activations per layer for a batch of formatted texts.

    Returns ``{layer: array[n_texts, hidden]}``. ``pooling='last'`` takes the last
    non-pad token (SPEC §3.1 last content token); ``'mean'`` is the mean-pooled
    robustness variant.
    """
    import torch

    tok = lm.tokenizer
    if layers is None:
        layers = list(range(lm.n_layers + 1))

    out: dict[int, list[np.ndarray]] = {ell: [] for ell in layers}
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(lm.model.device)
        with torch.no_grad():
            res = lm.model(**enc)
        hs = res.hidden_states  # tuple[n_layers+1] of [b, seq, hidden]
        mask = enc["attention_mask"]  # [b, seq]
        # index of last real token per row
        last_idx = mask.sum(dim=1) - 1
        for ell in layers:
            h = hs[ell]
            if pooling == "last":
                rows = h[torch.arange(h.size(0)), last_idx]  # [b, hidden]
            elif pooling == "mean":
                m = mask.unsqueeze(-1).to(h.dtype)
                rows = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            else:
                raise ValueError(f"unknown pooling {pooling!r}")
            out[ell].append(rows.float().cpu().numpy())

    return {ell: np.concatenate(v, axis=0) for ell, v in out.items()}
