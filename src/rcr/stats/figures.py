"""Figure generation (SPEC §2.8, §3, Phase 4 accept).

Each figure is a pure function of already-computed results (dicts/arrays), so the
plotting is GPU-free and deterministic. Every figure that has a control plots it
(random direction / flat-profile / clean->recovery). Uses the Agg backend so it
runs headless.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rcr.utils.paths import FIGURES_DIR  # noqa: E402

ARM_COLORS = {
    "contra": "#d62728",
    "narrow": "#1f77b4",
    "noise": "#ff7f0e",
    "clean": "#2ca02c",
    "random": "#888888",
}


def _save(fig, name: str, outdir: Path | None = None) -> Path:
    outdir = outdir or FIGURES_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def probe_accuracy_curve(acc_by_arm: dict[str, list[tuple[int, float]]], outdir=None) -> Path:
    """Layer vs probe accuracy per arm (SPEC §3.1). Chance line at 0.5."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, curve in acc_by_arm.items():
        xs = [layer for layer, _ in curve]
        ys = [acc for _, acc in curve]
        ax.plot(xs, ys, label=arm, color=ARM_COLORS.get(arm), marker="o", ms=3)
    ax.axhline(0.5, ls="--", c="k", lw=0.8, label="chance")
    ax.set_xlabel("layer")
    ax.set_ylabel("probe accuracy")
    ax.set_title("Corruption-direction separability by layer")
    ax.legend()
    return _save(fig, "probe_accuracy_curve.png", outdir)


def separation_curve(sep_by_arm: dict[str, list[tuple[int, float]]], outdir=None) -> Path:
    """Layer vs corruption specificity (post-A shift z vs null) per arm (SPEC §3.1/§3.3).

    The localization profile: a peak localizes where the corruption specifically
    moved the representation. The z=2 line marks the specificity threshold.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, curve in sep_by_arm.items():
        xs = [layer for layer, _ in curve]
        ys = [s for _, s in curve]
        ax.plot(xs, ys, label=arm, color=ARM_COLORS.get(arm), marker="o", ms=3)
    ax.axhline(2.0, ls="--", c="k", lw=0.8, label="z=2 (specific)")
    ax.set_xlabel("layer")
    ax.set_ylabel("specificity z (post-A shift vs null)")
    ax.set_title("Corruption localization by layer")
    ax.legend()
    return _save(fig, "localization_profile.png", outdir)


def rf_profile(rf_by_arm: dict[str, list[tuple[int, float]]], outdir=None) -> Path:
    """Per-layer recovery fraction per arm (SPEC §3.2). RF=1 full heal, 0 scar.

    Shows the layer-dependence of persistence directly, rather than hiding it in a
    single-ℓ* number.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, curve in rf_by_arm.items():
        xs = [layer for layer, _ in curve]
        ys = [r for _, r in curve]
        ax.plot(xs, ys, label=arm, color=ARM_COLORS.get(arm), marker="o", ms=3)
    ax.axhline(1.0, ls="--", c="k", lw=0.8, label="full recovery")
    ax.axhline(0.0, ls=":", c="k", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("recovery fraction (RF)")
    ax.set_ylim(-1.0, 1.5)
    ax.set_title("Per-layer persistence (RF) by arm")
    ax.legend()
    return _save(fig, "rf_profile.png", outdir)


def persistence_trajectory(
    traj: dict[str, dict[str, float]], outdir=None, title="Persistence: BASE → A → B"
) -> Path:
    """Projection at BASE/post-A/post-B per arm (SPEC §3.2).

    ``traj`` maps arm -> {"base":, "post_a":, "post_b":}. clean should be ~flat.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [0, 1, 2]
    for arm, pts in traj.items():
        ys = [pts["base"], pts["post_a"], pts["post_b"]]
        ax.plot(xs, ys, label=arm, color=ARM_COLORS.get(arm), marker="o")
    ax.set_xticks(xs)
    ax.set_xticklabels(["BASE", "post-A", "post-B"])
    ax.set_ylabel("mean projection onto corruption direction")
    ax.set_title(title)
    ax.legend()
    return _save(fig, "persistence_trajectory.png", outdir)


def recovery_fraction_bars(
    rf_by_arm: dict[str, tuple[float, float, float]], outdir=None
) -> Path:
    """RF per arm with CI (SPEC §2.8). Each value is (rf, lo, hi). RF=1 line."""
    fig, ax = plt.subplots(figsize=(6, 4))
    arms = list(rf_by_arm)
    rfs = [rf_by_arm[a][0] for a in arms]
    err_lo = [rf_by_arm[a][0] - rf_by_arm[a][1] for a in arms]
    err_hi = [rf_by_arm[a][2] - rf_by_arm[a][0] for a in arms]
    ax.bar(arms, rfs, color=[ARM_COLORS.get(a) for a in arms], yerr=[err_lo, err_hi], capsize=4)
    ax.axhline(1.0, ls="--", c="k", lw=0.8, label="full recovery")
    ax.axhline(0.0, ls=":", c="k", lw=0.8)
    ax.set_ylabel("recovery fraction (RF)")
    ax.set_title("Recovery fraction by corruption type")
    ax.legend()
    return _save(fig, "recovery_fraction_bars.png", outdir)


def localization_heatmap(
    profile_by_arm: dict[str, list[float]], outdir=None
) -> Path:
    """Per-layer localization signal per arm (SPEC §3.3)."""
    import numpy as np

    arms = list(profile_by_arm)
    mat = np.array([profile_by_arm[a] for a in arms])
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(arms) + 1.5))
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arms)
    ax.set_xlabel("layer")
    ax.set_title("Localization signal (patch recovery / LoRA energy)")
    fig.colorbar(im, ax=ax, fraction=0.025)
    return _save(fig, "localization_heatmap.png", outdir)


def pure_vs_mixed(
    shift_by_arm: dict[str, tuple[float, float]], sesoi: float = 0.2, outdir=None
) -> Path:
    """Post-B shift (d) in pure vs mixed per arm — the durability gate (SPEC §2.6)."""
    import numpy as np

    arms = list(shift_by_arm)
    pure = [shift_by_arm[a][0] for a in arms]
    mixed = [shift_by_arm[a][1] for a in arms]
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, pure, 0.4, label="pure", color="#888")
    ax.bar(x + 0.2, mixed, 0.4, label="mixed (durable)", color="#d62728")
    ax.axhline(sesoi, ls="--", c="k", lw=0.8, label=f"SESOI={sesoi}")
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("post-B residue (|d| vs clean)")
    ax.set_title("Durability gate: residue survives clean-mix?")
    ax.legend()
    return _save(fig, "pure_vs_mixed.png", outdir)
