#!/usr/bin/env python
"""Compute 128 bp validation metrics for a fine-tuned ATAC checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from alphagenome_pytorch.extensions.finetuning.checkpointing import (
    load_finetuned_model,
)
from alphagenome_pytorch.extensions.finetuning.datasets import GenomicDataset
from alphagenome_pytorch.extensions.finetuning.training import collate_genomic
from alphagenome_pytorch.metrics import (
    bin_pearson_r,
    differential_pearson_r,
    profile_pearson_r,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained-weights", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument("--bigwig", nargs="+", required=True)
    parser.add_argument("--bed", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--sequence-length", type=int, default=131072)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-regions", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model, meta = load_finetuned_model(
        checkpoint_path=args.checkpoint,
        pretrained_weights=args.pretrained_weights,
        device=device,
        merge=True,
    )
    model.eval()

    modality = meta["modality"]
    if isinstance(modality, list):
        modality = modality[0]
    head = model.heads[modality]

    dataset = GenomicDataset(
        genome_fasta=args.genome,
        bigwig_files=args.bigwig,
        bed_file=args.bed,
        resolutions=(128,),
        sequence_length=args.sequence_length,
    )
    if args.max_regions and len(dataset) > args.max_regions:
        dataset = Subset(dataset, range(args.max_regions))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_genomic,
    )

    profile_chunks: list[torch.Tensor] = []
    pred_chunks: list[torch.Tensor] = []
    true_chunks: list[torch.Tensor] = []

    for batch_idx, (sequences, targets_dict) in enumerate(loader, start=1):
        sequences = sequences.to(device)
        targets = targets_dict[128].to(device)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs = model(
                sequences,
                organism_idx,
                embeddings_only=True,
                resolutions=(128,),
                channels_last=False,
            )
            embeddings = {128: outputs["embeddings_128bp"]}
            pred = head(
                embeddings,
                organism_idx,
                return_scaled=False,
                channels_last=True,
            )[128]

        profile_chunks.append(profile_pearson_r(pred, targets).float().cpu())
        pred_chunks.append(pred.float().cpu())
        true_chunks.append(targets.float().cpu())

        if batch_idx % 25 == 0:
            print(f"processed_batches={batch_idx}", flush=True)

    profile_values = torch.cat(profile_chunks, dim=0)
    pred_all = torch.cat(pred_chunks, dim=0)
    true_all = torch.cat(true_chunks, dim=0)

    bin_values = bin_pearson_r(pred_all, true_all)
    differential_value = differential_pearson_r(pred_all, true_all)

    output = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "bed": str(Path(args.bed).resolve()),
        "sequence_length": args.sequence_length,
        "resolution": 128,
        "n_regions": int(pred_all.shape[0]),
        "n_bins": int(pred_all.shape[1]),
        "n_tracks": int(pred_all.shape[2]),
        "profile_pearson_r_mean": float(profile_values.mean().item()),
        "profile_pearson_r_std": float(profile_values.std().item()),
        "bin_pearson_r": float(bin_values.mean().item()),
        "bin_pearson_r_per_track_mean": float(bin_values.mean().item()),
        "bin_pearson_r_per_track_std": float(bin_values.std().item()),
        "differential_pearson_r": float(differential_value.item()),
        "metadata": meta,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(output, handle, indent=2, default=str)
    print(json.dumps(output, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
