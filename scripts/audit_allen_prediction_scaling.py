#!/usr/bin/env python
"""Audit Allen prediction and target calibration on validation windows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from alphagenome_pytorch.extensions.finetuning.checkpointing import load_finetuned_model
from alphagenome_pytorch.metrics import double_center
from evaluate_allen_multiome import make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def calibration_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_matrix = pred.float().reshape(-1, pred.shape[-1])
    target_matrix = target.float().reshape(-1, target.shape[-1])
    pred_centered = double_center(pred_matrix)
    target_centered = double_center(target_matrix)
    sst = target_centered.square().sum()
    pred_ss = pred_centered.square().sum()
    covariance = (pred_centered * target_centered).sum()
    scale = covariance / pred_ss if pred_ss > 0 else torch.tensor(float("nan"))
    r2 = 1 - (target_centered - pred_centered).square().sum() / sst
    calibrated_r2 = 1 - (target_centered - scale * pred_centered).square().sum() / sst
    correlation = covariance / torch.sqrt(pred_ss * sst)
    target_track_means = target_matrix.mean(dim=0)
    pred_track_means = pred_matrix.mean(dim=0)
    mean_ratios = pred_track_means / target_track_means.clamp_min(1e-12)
    return {
        "prediction_mean": pred_matrix.mean().item(),
        "target_mean": target_matrix.mean().item(),
        "prediction_std": pred_matrix.std().item(),
        "target_std": target_matrix.std().item(),
        "centered_prediction_std": pred_centered.std().item(),
        "centered_target_std": target_centered.std().item(),
        "centered_std_ratio": (pred_centered.std() / target_centered.std()).item(),
        "differential_pearson_r": correlation.item(),
        "double_centered_r2": r2.item(),
        "optimal_prediction_scale": scale.item(),
        "scale_calibrated_r2": calibrated_r2.item(),
        "track_mean_ratio_min": mean_ratios.min().item(),
        "track_mean_ratio_median": mean_ratios.median().item(),
        "track_mean_ratio_max": mean_ratios.max().item(),
    }


def main() -> None:
    args = parse_args()
    if args.max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for prediction audit")
    manifest = json.loads(args.manifest.read_text())
    device = torch.device("cuda")
    model, metadata = load_finetuned_model(
        checkpoint_path=args.checkpoint,
        pretrained_weights=args.pretrained_weights,
        device=device,
        merge=True,
    )
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    results = {"checkpoint_metadata": metadata, "species": {}}
    for species, config in manifest["species"].items():
        loader = make_loader(config, manifest, species, "valid", args)
        head_results = {}
        predictions = {"atac": [], "rna_seq": []}
        targets = {"atac": [], "rna_seq": []}
        head_names = {
            "atac": f"{species}_atac",
            "rna_seq": f"{species}_rna_seq",
        }
        with torch.no_grad():
            for batch_index, (sequences, modality_targets) in enumerate(loader):
                if batch_index >= args.max_batches:
                    break
                sequences = sequences.to(device)
                organism_index = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(
                        sequences,
                        organism_index,
                        return_embeddings=True,
                        resolutions=(128,),
                        channels_last=False,
                    )
                    embeddings = {128: outputs["embeddings_128bp"]}
                    for modality, head_name in head_names.items():
                        pred = model.heads[head_name](
                            embeddings,
                            organism_index,
                            return_scaled=False,
                            channels_last=True,
                        )[128]
                        predictions[modality].append(pred.float().cpu())
                        targets[modality].append(
                            modality_targets[head_name][128].float().cpu()
                        )
        for modality, head_name in head_names.items():
            head = model.heads[head_name]
            values = head.track_means.float().cpu().flatten()
            metrics = calibration_metrics(
                torch.cat(predictions[modality]),
                torch.cat(targets[modality]),
            )
            metrics["track_scale_min"] = values.min().item()
            metrics["track_scale_median"] = values.median().item()
            metrics["track_scale_max"] = values.max().item()
            if not all(math.isfinite(value) for value in metrics.values()):
                raise RuntimeError(f"Non-finite calibration metric for {head_name}")
            head_results[modality] = metrics
        results["species"][species] = head_results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
