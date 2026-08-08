#!/usr/bin/env python
"""Build a distance-to-TSS matched SingleBrain fine-mapping benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alphagenome_pytorch.variant_scoring.benchmark import (
    load_gene_tss,
    select_pip_matched_variants,
)


DEFAULT_FINEMAP = Path(
    "/gpfs/commons/groups/knowles_lab/atokolyi/public/singlebrain_full_finemap"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--finemap-dir", type=Path, default=DEFAULT_FINEMAP)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-pip", type=float, default=0.75)
    parser.add_argument("--negative-pip", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit-pairs", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.finemap_dir.glob("*_all.susie_all_pip.tsv.gz"))
    variants = select_pip_matched_variants(
        paths,
        load_gene_tss(args.gtf),
        positive_threshold=args.positive_pip,
        negative_threshold=args.negative_pip,
        seed=args.seed,
        limit_pairs=args.limit_pairs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(variants)
    frame.to_csv(args.output, sep="\t", index=False)
    print(
        f"Wrote {len(frame)} rows in {frame['match_pair_id'].nunique()} matched pairs "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
