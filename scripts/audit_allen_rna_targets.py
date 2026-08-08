#!/usr/bin/env python
"""Audit normalized exon-projected RNA targets in an Allen manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alphagenome_pytorch.extensions.finetuning.datasets import GeneExpressionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--windows", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    results = {}
    for species, config in manifest["species"].items():
        dataset = GeneExpressionDataset(
            config["fasta"],
            config["rna_h5ad"],
            config["gtf"],
            config["splits"][args.split],
            sequence_length=manifest["sequence_length"],
            gene_mapping_file=config.get("rna_gene_mapping"),
            expression_gene_column=config.get("expression_gene_column"),
            expression_var_column=config.get("expression_var_column"),
            annotation_gene_column=config.get("annotation_gene_column", "gene_id"),
            mapping_annotation_gene_column=config.get("mapping_annotation_gene_column"),
            annotation_chromosome_map=config.get("annotation_chromosome_map"),
        )
        nonzero = 0
        maximum = 0.0
        for index in range(min(args.windows, len(dataset))):
            _, targets = dataset[index]
            nonzero += int((targets[128] > 0).sum())
            maximum = max(maximum, float(targets[128].max()))
        results[species] = {
            "tracks": dataset.n_tracks,
            "cells_min": int(dataset.n_cells.min()),
            "cells_max": int(dataset.n_cells.max()),
            "target_library_size": dataset.target_library_size,
            "raw_per_cell_library_min": float(dataset.raw_per_cell_library_sizes.min()),
            "raw_per_cell_library_max": float(dataset.raw_per_cell_library_sizes.max()),
            "track_mean_min": float(dataset.track_means.min()),
            "track_mean_max": float(dataset.track_means.max()),
            "nonzero_entries": nonzero,
            "target_max": maximum,
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
