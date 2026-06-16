"""Local Learning Coefficient trajectories (SPEC §3.7, STRETCH, exploratory).

devinterp / SLT: estimate the LLC across Phase-A and Phase-B checkpoints to ask
whether corruption moves the model into a basin whose stickiness predicts how
much recovery fails to undo. No devinterp work has applied a corruption->recovery
paradigm, so even a modest behavioral result here is novel (SPEC §3.7).

This is explicitly labeled exploratory and gated behind the `stretch` extra
(devinterp). LoRA-only, SGLD defaults. We keep the surface minimal: a thin
wrapper that, given a model + a loss-bearing dataloader, returns an LLC estimate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLCEstimate:
    llc: float
    n_chains: int
    n_steps: int
    note: str = "exploratory; SGLD defaults; LoRA-only (SPEC §3.7)"


def estimate_llc(
    model,
    dataloader,
    *,
    n_chains: int = 4,
    n_steps: int = 200,
    seed: int = 0,
) -> LLCEstimate:
    """Estimate the LLC via devinterp's SGLD sampler (lazy import).

    Raises ImportError with a clear message if the `stretch` extra is not
    installed, so the core suite never hard-depends on devinterp.
    """
    try:
        from devinterp.optim import SGLD
        from devinterp.slt import estimate_learning_coeff
    except ImportError as e:  # pragma: no cover - optional extra
        raise ImportError(
            "devinterp not installed. Install the stretch extra: `uv sync --extra stretch`."
        ) from e

    llc = estimate_learning_coeff(
        model,
        loader=dataloader,
        sampling_method=SGLD,
        num_chains=n_chains,
        num_draws=n_steps,
        seed=seed,
    )
    return LLCEstimate(llc=float(llc), n_chains=n_chains, n_steps=n_steps)
