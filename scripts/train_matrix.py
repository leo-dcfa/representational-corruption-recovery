#!/usr/bin/env python
"""Phase 2 entrypoint: run the two-phase A->B training matrix.

  uv run python scripts/train_matrix.py --config configs/experiment.yaml
  uv run python scripts/train_matrix.py --config configs/smoke.yaml --dry-run

GATED: this trains models. `--dry-run` prints the matrix and per-cell corpus
paths without loading any model, so you can verify the plan first. runs/ is
append-only; existing cells are skipped unless --overwrite.
"""

from __future__ import annotations

import argparse

from rcr.config import load_config
from rcr.train.dataset import load_corpus_dataset
from rcr.train.two_phase import run_two_phase
from rcr.utils.paths import run_dir


def _corpus_paths(cell, cfg):
    from rcr.datagen.build import corpus_path_for as cpath

    a = cpath(cell["arm"], cell["mix"], "A")
    b = cpath("recovery", "pure", "B")  # shared clean recovery corpus
    return a, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on run id")
    ap.add_argument("--limit", type=int, default=None, help="truncate each phase dataset (smoke)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cells = cfg.run_cells()
    if args.only:
        from rcr.utils.paths import run_id

        cells = [c for c in cells if args.only in run_id(c["model_slug"], c["arm"], c["mix"], c["seed"])]

    print(f"matrix: {len(cells)} cells")
    for cell in cells:
        rdir = run_dir(cell["model_slug"], cell["arm"], cell["mix"], cell["seed"])
        a_corpus, b_corpus = _corpus_paths(cell, cfg)
        tag = rdir.name
        if args.dry_run:
            print(f"  {tag}: A={a_corpus.name} B={b_corpus.name} -> {rdir}")
            continue
        if rdir.exists() and not args.overwrite:
            print(f"  SKIP {tag} (exists; --overwrite to redo)")
            continue
        print(f"  TRAIN {tag}")
        ds_a = load_corpus_dataset(a_corpus)
        ds_b = load_corpus_dataset(b_corpus)
        if args.limit:
            ds_a = ds_a.select(range(min(args.limit, len(ds_a))))
            ds_b = ds_b.select(range(min(args.limit, len(ds_b))))
        run_two_phase(
            cell["model"], None, ds_a, ds_b, cfg.train, rdir, seed=cell["seed"]
        )
    if args.dry_run:
        print("dry-run only; no models loaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
