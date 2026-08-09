#!/usr/bin/env python
"""Aggregate pair-preserving SingleBrain VEP shards."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from scripts.score_singlebrain_finemap import average_precision, auroc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--benchmark-variants", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_paths = [
        args.input_dir / f"variant_scores.shard-{index:03d}-of-{args.num_shards:03d}.tsv.gz"
        for index in range(args.num_shards)
    ]
    missing = [str(path) for path in score_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing score shards: {missing}")
    frame = pd.concat((pd.read_csv(path, sep="\t") for path in score_paths), ignore_index=True)
    pair_sizes = frame.groupby("match_pair_id")["benchmark_label"].agg(["size", "nunique"])
    complete_pair_ids = pair_sizes.index[(pair_sizes["size"] == 2) & (pair_sizes["nunique"] == 2)]
    frame = frame[frame["match_pair_id"].isin(complete_pair_ids)].copy()
    if frame.empty:
        raise RuntimeError("No complete positive-negative pairs were scored")

    benchmark = pd.read_csv(
        args.benchmark_variants,
        sep="\t",
        usecols=["match_pair_id"],
    )
    n_selected_pairs = benchmark["match_pair_id"].nunique()
    skip_counts: Counter[str] = Counter()
    for index in range(args.num_shards):
        path = args.input_dir / f"metrics.shard-{index:03d}-of-{args.num_shards:03d}.json"
        shard_metrics = json.loads(path.read_text())
        skip_counts.update(shard_metrics.get("skip_counts", {}))

    labels = frame["benchmark_label"].astype(int).to_numpy(dtype=bool)
    positive_distances = frame.loc[labels, "distance_to_tss"].astype(int).to_numpy()
    negative_distances = frame.loc[~labels, "distance_to_tss"].astype(int).to_numpy()
    distance_test = stats.ks_2samp(positive_distances, negative_distances)
    metrics = {
        "primary_score": "rna_seq_target_gene_exon_lfc_max_abs",
        "context_width": int(frame["context_end"].iloc[0] - frame["context_start"].iloc[0]),
        "context_center": "target_gene_tss_shifted_to_include_variant",
        "rna_mask": "target_gene_exons",
        "atac_mask_width": 501,
        "n_rows": len(frame),
        "n_selected_pairs": int(n_selected_pairs),
        "n_unique_variants": frame["variant_id"].nunique(),
        "n_dropped_pairs": int(n_selected_pairs - len(complete_pair_ids)),
        "skip_counts": dict(sorted(skip_counts.items())),
        "n_positive": int(labels.sum()),
        "n_negative": int((~labels).sum()),
        "positive_pip_gt": 0.75,
        "negative_pip_lt": 0.01,
        "distance_to_tss_ks": float(distance_test.statistic),
        "distance_to_tss_ks_pvalue": float(distance_test.pvalue),
    }
    for modality in ("atac", "rna_seq"):
        values = frame[f"{modality}_max_abs"].to_numpy(dtype=float)
        metrics[f"{modality}_auroc"] = auroc(labels, values)
        metrics[f"{modality}_average_precision"] = average_precision(labels, values)
        for class_name, class_frame in frame.groupby("singlebrain_cell_class"):
            class_labels = class_frame["benchmark_label"].astype(int).to_numpy(dtype=bool)
            class_values = class_frame[f"{modality}_max_abs"].to_numpy(dtype=float)
            metrics[f"{modality}_auroc_{class_name}"] = auroc(class_labels, class_values)
            metrics[f"{modality}_average_precision_{class_name}"] = average_precision(
                class_labels, class_values
            )

    frame.to_csv(args.input_dir / "variant_scores.tsv.gz", sep="\t", index=False)
    (args.input_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
