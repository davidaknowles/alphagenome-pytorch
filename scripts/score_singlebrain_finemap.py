#!/usr/bin/env python
"""Benchmark fine-tuned human ATAC and RNA variant scores against SuSiE PIP."""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from alphagenome_pytorch.extensions.finetuning.checkpointing import load_finetuned_model
from alphagenome_pytorch.variant_scoring import (
    AggregationType,
    CenterMaskScorer,
    Interval,
    OutputType,
    Variant,
    VariantScoringModel,
)


DEFAULT_FINEMAP = Path(
    "/gpfs/commons/groups/knowles_lab/atokolyi/public/singlebrain_full_finemap"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained-weights", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--finemap-dir", type=Path, default=DEFAULT_FINEMAP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-variants", type=int, default=1000)
    parser.add_argument("--min-pip", type=float, default=0.01)
    parser.add_argument("--width", type=int, default=131_072)
    return parser.parse_args()


def open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def select_variants(directory: Path, max_variants: int, min_pip: float) -> list[dict[str, str]]:
    heap: list[tuple[float, int, dict[str, str]]] = []
    serial = count()
    paths = sorted(directory.glob("*_all.susie_all_credible_sets.tsv"))
    if not paths:
        paths = sorted(directory.glob("*_all.susie_all_pip.tsv.gz"))
    for path in paths:
        with open_text(path) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                pip = float(row["susie_pip"])
                if pip < min_pip:
                    continue
                item = (pip, next(serial), row)
                if len(heap) < max_variants:
                    heapq.heappush(heap, item)
                elif pip > heap[0][0]:
                    heapq.heapreplace(heap, item)
    selected = [item[2] for item in sorted(heap, reverse=True)]
    seen = set()
    unique = []
    for row in selected:
        key = (row["chr"], row["pos"], row["ref"], row["alt"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels.sum()
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for variant scoring")
    device = torch.device("cuda")
    model, _ = load_finetuned_model(
        checkpoint_path=args.checkpoint,
        pretrained_weights=args.pretrained_weights,
        device=device,
        merge=True,
    )
    model.heads["atac"] = model.heads["human_atac"]
    model.heads["rna_seq"] = model.heads["human_rna_seq"]
    model = model.to(device=device).eval()
    scoring_model = VariantScoringModel(model, fasta_path=args.fasta, device=device)
    scorers = [
        CenterMaskScorer(OutputType.ATAC, 100_001, AggregationType.DIFF_LOG2_SUM, resolution=128),
        CenterMaskScorer(OutputType.RNA_SEQ, 100_001, AggregationType.DIFF_LOG2_SUM, resolution=128),
    ]
    variants = select_variants(args.finemap_dir, args.max_variants, args.min_pip)
    rows = []
    for index, row in enumerate(variants, start=1):
        variant = Variant(
            chromosome=row["chr"], position=int(row["pos"]),
            reference_bases=row["ref"], alternate_bases=row["alt"],
            name=row.get("variant_id", ""),
        )
        interval = Interval.centered_on(variant.chromosome, variant.position - 1, args.width)
        try:
            scores = scoring_model.score_variant(interval, variant, scorers, to_cpu=True)
        except (KeyError, ValueError, RuntimeError) as error:
            print(f"Skipping {variant}: {error}")
            continue
        result = dict(row)
        for scorer, score in zip(scorers, scores, strict=True):
            values = score.scores.float().cpu().numpy()
            result[f"{scorer.requested_output.value}_max_abs"] = float(np.max(np.abs(values)))
            result[f"{scorer.requested_output.value}_mean_abs"] = float(np.mean(np.abs(values)))
        rows.append(result)
        if index % 25 == 0:
            print(f"Scored {index}/{len(variants)} variants")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "variant_scores.tsv.gz", sep="\t", index=False)
    if frame.empty:
        raise RuntimeError("No fine-mapped variants were scored successfully")
    pip = frame["susie_pip"].astype(float).to_numpy()
    metrics = {"n_variants": len(frame), "pip_threshold": 0.1}
    for modality in ("atac", "rna_seq"):
        values = frame[f"{modality}_max_abs"].to_numpy(dtype=float)
        correlation = stats.spearmanr(pip, values)
        metrics[f"{modality}_spearman_pip"] = float(correlation.statistic)
        metrics[f"{modality}_spearman_pvalue"] = float(correlation.pvalue)
        metrics[f"{modality}_auroc_pip_ge_0.1"] = auroc(pip >= 0.1, values)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
