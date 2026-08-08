#!/usr/bin/env python
"""Evaluate Allen Multiome heads by species and held-out chromosome."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from alphagenome_pytorch.extensions.finetuning.checkpointing import load_finetuned_model
from alphagenome_pytorch.extensions.finetuning.datasets import (
    GeneExpressionDataset,
    GenomicDataset,
    MultimodalDataset,
    collate_multimodal,
)
from alphagenome_pytorch.extensions.finetuning.training import validate_multihead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--splits", default="valid,test")
    return parser.parse_args()


def make_loader(config: dict, manifest: dict, species: str, split: str, args: argparse.Namespace):
    bed = config["splits"][split]
    atac_key = f"{species}_atac"
    rna_key = f"{species}_rna_seq"
    atac = GenomicDataset(
        config["fasta"], config["atac_bigwigs"], bed,
        resolutions=(128,), sequence_length=manifest["sequence_length"],
    )
    rna = GeneExpressionDataset(
        config["fasta"], config["rna_h5ad"],
        config["gtf"],
        bed, sequence_length=manifest["sequence_length"],
        gene_mapping_file=config.get("rna_gene_mapping"),
        expression_gene_column=config.get("expression_gene_column"),
        expression_var_column=config.get("expression_var_column"),
        annotation_gene_column=config.get("annotation_gene_column", "gene_id"),
        mapping_annotation_gene_column=config.get("mapping_annotation_gene_column"),
        annotation_chromosome_map=config.get("annotation_chromosome_map"),
    )
    dataset = MultimodalDataset({atac_key: atac, rna_key: rna})
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_multimodal,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    device = torch.device("cuda")
    manifest = json.loads(args.manifest.read_text())
    model, metadata = load_finetuned_model(
        checkpoint_path=args.checkpoint,
        pretrained_weights=args.pretrained_weights,
        device=device,
        merge=True,
    )
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    results = {"checkpoint_metadata": metadata, "species": {}}
    weights = {name: 1.0 for name in model.heads}
    resolutions = {name: {128: 1.0} for name in model.heads}
    for species, config in manifest["species"].items():
        species_results = {}
        head_names = (f"{species}_atac", f"{species}_rna_seq")
        heads = {name: model.heads[name] for name in head_names}
        for split in [value for value in args.splits.split(",") if value]:
            loader = make_loader(config, manifest, species, split, args)
            loss, metrics = validate_multihead(
                model, heads, loader, device, weights, resolutions,
                positional_weight=5.0, count_weight=1.0,
                compute_pearson=True, use_amp=True, amp_dtype=torch.bfloat16,
            )
            compact_metrics = {
                key: value for key, value in metrics.items() if not key.endswith("_values")
            }
            primary_by_head = {
                head_name: compact_metrics[f"{head_name}_128bp_double_centered_r2"]
                for head_name in head_names
            }
            finite_primary = [value for value in primary_by_head.values() if math.isfinite(value)]
            species_results[split] = {
                "chromosome": config[f"{split}_chromosome"],
                "primary_metric": {
                    "name": "128bp_double_centered_r2",
                    "mean": sum(finite_primary) / len(finite_primary) if finite_primary else float("nan"),
                    "by_head": primary_by_head,
                },
                "loss": loss,
                "metrics": compact_metrics,
                "n_windows": len(loader.dataset),
            }
        results["species"][species] = species_results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
