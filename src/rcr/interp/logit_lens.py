"""Logit lens: decision-token logit differences across layers (SPEC §3.6).

Project each layer's residual stream through the (final norm +) unembedding to
read the model's "running guess" at the decision token, BASE->A->B on target
prompts. Cheap, good figures.
"""

from __future__ import annotations

import numpy as np

from rcr.interp.activations import LoadedModel


def _unembed(model):
    """Return (final_norm, lm_head) for the standard decoder-only stack."""
    base = getattr(model, "model", model)
    norm = getattr(base, "norm", None)
    lm_head = model.get_output_embeddings()
    return norm, lm_head


def decision_token_logits(
    lm: LoadedModel,
    texts: list[str],
    token_ids: list[int],
    max_len: int = 1024,
) -> dict[int, np.ndarray]:
    """Per-layer logits for ``token_ids`` at the last position, via the logit lens.

    Returns ``{layer: array[n_texts, n_tokens]}`` — the logit assigned to each
    decision token (e.g. the ids for "Yes"/"No") read off each layer.
    """
    import torch

    tok = lm.tokenizer
    norm, lm_head = _unembed(lm.model)
    enc = tok(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len
    ).to(lm.model.device)
    with torch.no_grad():
        res = lm.model(**enc)
    hs = res.hidden_states
    mask = enc["attention_mask"]
    last_idx = mask.sum(dim=1) - 1
    out: dict[int, np.ndarray] = {}
    tid = torch.tensor(token_ids, device=lm.model.device)
    for ell, h in enumerate(hs):
        rows = h[torch.arange(h.size(0)), last_idx]  # [b, hidden]
        with torch.no_grad():
            x = norm(rows) if norm is not None else rows
            logits = lm_head(x)  # [b, vocab]
        out[ell] = logits[:, tid].float().cpu().numpy()
    return out
