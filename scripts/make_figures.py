#!/usr/bin/env python
"""Regenerate all figures from cached run results (SPEC §2.8 — one command).

  uv run python scripts/make_figures.py --config configs/experiment.yaml

Reads runs/*/interp.json + runs/*/eval.json (produced by run_interp.py /
run_eval.py) and emits the standard figure set to reports/figures/. Pure: loads
JSON, calls src/rcr/stats/figures.py. No GPU.
"""

from __future__ import annotations

import argparse

from rcr.config import load_config
from rcr.stats import figures
from rcr.utils.io import read_json
from rcr.utils.paths import run_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--mix", default="mixed", help="condition for the headline figures")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sep_curves: dict[str, list] = {}
    rf_profiles: dict[str, list] = {}
    traj: dict[str, dict] = {}
    rf_bars: dict[str, tuple] = {}
    loc: dict[str, list] = {}

    for cell in cfg.run_cells():
        if cell["mix"] != args.mix or cell["seed"] != 0:
            continue
        rdir = run_dir(cell["model_slug"], cell["arm"], cell["mix"], cell["seed"])
        ip = rdir / "interp.json"
        if not ip.exists():
            continue
        data = read_json(ip)
        arm = cell["arm"]
        lp = data.get("layer_profile", [])
        sep_curves[arm] = [(p["layer"], p["spec_z"]) for p in lp]
        rf_profiles[arm] = [(p["layer"], p["rf"]) for p in lp if p.get("rf") is not None]
        traj[arm] = data.get("trajectory", {})
        if data.get("rf_specific"):
            ci = data.get("rf_ci") or (data.get("rf", 0.0), data.get("rf", 0.0), data.get("rf", 0.0))
            rf_bars[arm] = (data.get("rf", 0.0), ci[1], ci[2])
        energy = data.get("lora_energy_by_layer", {})
        loc[arm] = [energy.get(str(i), 0.0) for i in range(max((int(k) for k in energy), default=0) + 1)]

    written = []
    if sep_curves:
        written.append(figures.separation_curve(sep_curves))
    if rf_profiles:
        written.append(figures.rf_profile(rf_profiles))
    if traj:
        written.append(figures.persistence_trajectory(traj))
    if rf_bars:
        written.append(figures.recovery_fraction_bars(rf_bars))
    if loc:
        written.append(figures.localization_heatmap(loc))

    if not written:
        print("no run results found; train + run_interp first.")
    for p in written:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
