#!/usr/bin/env python
"""Benchmark fine-tuned human ATAC and RNA variant scores against SuSiE PIP."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from alphagenome_pytorch.extensions.finetuning.checkpointing import load_finetuned_model
from alphagenome_pytorch.variant_scoring import (
    AggregationType,
    CenterMaskScorer,
    GeneMaskLFCScorer,
    GeneMaskMode,
    Interval,
    OutputType,
    Variant,
    VariantScoringModel,
)
from alphagenome_pytorch.variant_scoring.benchmark import (
    allen_track_indices,
    load_gene_tss,
    select_pip_matched_variants,
    singlebrain_cell_class,
)


DEFAULT_FINEMAP = Path(
    "/gpfs/commons/groups/knowles_lab/atokolyi/public/singlebrain_full_finemap"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained-weights", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--benchmark-variants", type=Path)
    parser.add_argument("--finemap-dir", type=Path, default=DEFAULT_FINEMAP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positive-pip", type=float, default=0.75)
    parser.add_argument("--negative-pip", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--width", type=int, default=1_048_576)
    return parser.parse_args()


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels.sum()
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute tie-invariant average precision."""
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    threshold_ends = np.r_[np.flatnonzero(np.diff(ordered_scores)), labels.size - 1]
    true_positives = np.cumsum(ordered_labels)[threshold_ends]
    precision = true_positives / (threshold_ends + 1)
    recall_increments = np.diff(np.r_[0, true_positives]) / positives
    return float(np.sum(recall_increments * precision))


def main() -> None:
    args = parse_args()
    if args.benchmark_variants is not None:
        variants = pd.read_csv(args.benchmark_variants, sep="\t", dtype=str).to_dict("records")
    else:
        fine_map_paths = sorted(args.finemap_dir.glob("*_all.susie_all_pip.tsv.gz"))
        variants = select_pip_matched_variants(
            fine_map_paths,
            load_gene_tss(args.gtf),
            positive_threshold=args.positive_pip,
            negative_threshold=args.negative_pip,
            seed=args.seed,
            limit_pairs=args.limit_pairs,
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for variant scoring")
    device = torch.device("cuda")
    model, metadata = load_finetuned_model(
        checkpoint_path=args.checkpoint,
        pretrained_weights=args.pretrained_weights,
        device=device,
        merge=True,
    )
    model.heads["atac"] = model.heads["human_atac"]
    model.heads["rna_seq"] = model.heads["human_rna_seq"]
    model = model.to(device=device).eval()
    scoring_model = VariantScoringModel(
        model,
        fasta_path=args.fasta,
        gtf_path=args.gtf,
        device=device,
    )
    scorers = [
        CenterMaskScorer(OutputType.ATAC, 501, AggregationType.DIFF_LOG2_SUM, resolution=128),
        GeneMaskLFCScorer(OutputType.RNA_SEQ, GeneMaskMode.EXONS, resolution=128),
    ]
    rows = []
    score_cache = {}
    skip_counts: Counter[str] = Counter()
    selected_pair_ids = {row["match_pair_id"] for row in variants}
    for index, row in enumerate(variants, start=1):
        variant = Variant(
            chromosome=row["chr"], position=int(row["pos"]),
            reference_bases=row["ref"], alternate_bases=row["alt"],
            name=row.get("variant_id", ""),
        )
        gene_id = row["feature"].split(".", 1)[0]
        interval = Interval.centered_on(variant.chromosome, variant.position - 1, args.width)
        variant_key = (
            variant.chromosome,
            variant.position,
            variant.reference_bases,
            variant.alternate_bases,
            gene_id,
        )
        gene_info = scoring_model.gene_annotation.get_gene_info(gene_id)
        if gene_info is None:
            skip_counts["target_gene_absent_from_gtf"] += 1
            print(f"Skipping {variant}/{gene_id}: target gene is absent from the GTF")
            continue
        target_tss = (
            gene_info["end"] - 1 if gene_info["strand"] == "-" else gene_info["start"]
        )
        if (
            gene_info["chromosome"] != variant.chromosome
            or target_tss < interval.start
            or target_tss >= interval.end
        ):
            skip_counts["target_gene_tss_outside_context"] += 1
            print(f"Skipping {variant}/{gene_id}: target gene TSS is outside context")
            continue
        scores = score_cache.get(variant_key)
        if scores is None:
            try:
                scores = scoring_model.score_variant(
                    interval,
                    variant,
                    scorers,
                    gene_ids=[gene_id],
                    to_cpu=True,
                )
            except (KeyError, ValueError, RuntimeError) as error:
                skip_counts["scoring_error"] += 1
                print(f"Skipping {variant}/{gene_id}: {error}")
                continue
            score_cache[variant_key] = scores
        result = dict(row)
        result["context_start"] = interval.start
        result["context_end"] = interval.end
        result["target_gene_id"] = gene_id
        result["target_gene_tss"] = target_tss
        for scorer, score_result in zip(scorers, scores, strict=True):
            if isinstance(score_result, list):
                target_scores = [score for score in score_result if score.gene_id == gene_id]
                if len(target_scores) != 1:
                    skip_counts["missing_target_gene_exon_score"] += 1
                    print(
                        f"Skipping {variant}/{gene_id}: expected one target-gene "
                        f"score, found {len(target_scores)}"
                    )
                    break
                score = target_scores[0]
            else:
                score = score_result
            values = score.scores.float().cpu().numpy()
            modality = scorer.requested_output.value
            head_name = "human_atac" if modality == "atac" else "human_rna_seq"
            track_names = metadata["track_names"][head_name]
            indices = allen_track_indices(row["celltype"], track_names)
            matched_values = values[indices]
            result[f"{modality}_max_abs"] = float(np.max(np.abs(matched_values)))
            result[f"{modality}_mean_abs"] = float(np.mean(np.abs(matched_values)))
            result[f"{modality}_matched_tracks"] = ";".join(track_names[index] for index in indices)
        else:
            result["singlebrain_cell_class"] = singlebrain_cell_class(row["celltype"])
            rows.append(result)
        if index % 25 == 0:
            print(f"Processed {index}/{len(variants)} benchmark rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No fine-mapped variants were scored successfully")
    pair_sizes = frame.groupby("match_pair_id")["benchmark_label"].agg(["size", "nunique"])
    complete_pair_ids = pair_sizes.index[(pair_sizes["size"] == 2) & (pair_sizes["nunique"] == 2)]
    dropped_pairs = int(len(selected_pair_ids) - complete_pair_ids.size)
    frame = frame[frame["match_pair_id"].isin(complete_pair_ids)].copy()
    if frame.empty:
        raise RuntimeError("No complete positive-negative pairs were scored successfully")
    frame.to_csv(args.output_dir / "variant_scores.tsv.gz", sep="\t", index=False)
    labels = frame["benchmark_label"].astype(int).to_numpy(dtype=bool)
    positive_distances = frame.loc[labels, "distance_to_tss"].astype(int).to_numpy()
    negative_distances = frame.loc[~labels, "distance_to_tss"].astype(int).to_numpy()
    distance_test = stats.ks_2samp(positive_distances, negative_distances)
    metrics = {
        "primary_score": "rna_seq_target_gene_exon_lfc_max_abs",
        "context_width": args.width,
        "context_center": "variant",
        "rna_mask": "target_gene_exons",
        "atac_mask_width": 501,
        "n_rows": len(frame),
        "n_selected_pairs": len(selected_pair_ids),
        "n_scored_variant_gene_pairs": len(score_cache),
        "n_unique_variants": frame["variant_id"].nunique(),
        "n_dropped_pairs": dropped_pairs,
        "skip_counts": dict(sorted(skip_counts.items())),
        "n_positive": int(labels.sum()),
        "n_negative": int((~labels).sum()),
        "positive_pip_gt": args.positive_pip,
        "negative_pip_lt": args.negative_pip,
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
                class_labels,
                class_values,
            )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
