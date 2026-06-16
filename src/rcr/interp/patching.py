"""Layer (residual-stream) patching for localization (SPEC §3.3, H2).

Patch the residual stream at layer ℓ from a *donor* model (e.g. post-A) into a
*recipient* (BASE, or post-B = "what recovery failed to fix"), sweep ℓ, and
measure how much of the corruption signature transfers. A peaked sweep localizes
the residue; a flat sweep supports the null (SPEC §3.3).

Implementation: cache donor ``hidden_states`` once, then run the recipient with a
forward hook on decoder layer ℓ that overwrites its output residual stream with
the donor's. The readout is supplied by the caller (projection or logit).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from rcr.interp.activations import LoadedModel


def _decoder_layers(model):
    """Return the list of decoder layer modules (Qwen2/Llama: model.model.layers)."""
    base = getattr(model, "model", model)
    return base.layers


def cache_hidden_states(lm: LoadedModel, texts: list[str], max_len: int = 1024):
    """Return (hidden_states tuple on CPU, attention_mask) for a single batch."""
    import torch

    enc = lm.tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len
    ).to(lm.model.device)
    with torch.no_grad():
        res = lm.model(**enc)
    return [h.detach() for h in res.hidden_states], enc


def patched_forward(
    recipient: LoadedModel,
    donor_hidden: list,
    enc,
    layer: int,
    readout: Callable,
):
    """Run recipient with layer ``layer``'s residual stream replaced by donor's.

    ``donor_hidden[layer + 1]`` is the donor residual stream at the OUTPUT of
    decoder layer ``layer`` (index 0 is embeddings). ``readout`` maps the model
    output object to a numpy array.
    """
    import torch

    layers = _decoder_layers(recipient.model)
    target = layers[layer]
    donor = donor_hidden[layer + 1].to(recipient.model.device)

    def hook(module, args, output):  # noqa: ANN001
        if isinstance(output, tuple):
            return (donor.to(output[0].dtype), *output[1:])
        return donor.to(output.dtype)

    handle = target.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = recipient.model(**enc)
    finally:
        handle.remove()
    return readout(out)


def patch_sweep(
    recipient: LoadedModel,
    donor: LoadedModel,
    texts: list[str],
    readout: Callable,
    layers: list[int] | None = None,
    max_len: int = 1024,
) -> dict[int, np.ndarray]:
    """Sweep the patched layer; return ``{layer: readout}`` for the signature heatmap."""
    donor_hidden, _ = cache_hidden_states(donor, texts, max_len=max_len)
    # recipient needs its own tokenization (same texts) so positions align
    _, enc = cache_hidden_states(recipient, texts, max_len=max_len)
    if layers is None:
        layers = list(range(recipient.n_layers))
    return {ell: patched_forward(recipient, donor_hidden, enc, ell, readout) for ell in layers}
