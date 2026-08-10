#!/usr/bin/env python
"""Compare pretrained AlphaGenome output scaling with Allen target scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.extensions.finetuning.datasets import compute_track_means
from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk
from alphagenome_pytorch.heads import targets_scaling
from evaluate_allen_multiome import make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pretrained-weights", type=Path, required=True)
    parser.add_argument("--baseline-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--species", default="human")
    parser.add_argument("--candidate-loci", type=int, default=48)
    parser.add_argument("--num-loci", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def choose_allen_tracks(names: list[str]) -> list[int]:
    preferred = ("Astrocyte", "Microglia", "Oligo")
    selected = []
    for term in preferred:
        match = next(
            (index for index, name in enumerate(names) if term.lower() in name.lower()),
            None,
        )
        if match is not None and match not in selected:
            selected.append(match)
    selected.extend(index for index in range(len(names)) if index not in selected)
    return selected[:3]


def choose_baseline_tracks(metadata: pd.DataFrame, modality: str) -> list[int]:
    frame = metadata[metadata["output_type"] == modality].copy()
    frame = frame[
        frame["nonzero_mean"].notna()
        & ~frame["track_name"].astype(str).str.contains("Padding", case=False)
    ]
    if modality == "rna_seq":
        preferred = ("astrocyte", "glutamatergic neuron", "Brain_Caudate_basal_ganglia")
        selected = []
        for term in preferred:
            matches = frame[
                frame.astype(str).apply(
                    lambda column: column.str.contains(term, case=False, regex=False)
                ).any(axis=1)
            ]
            if not matches.empty:
                selected.append(int(matches.iloc[0]["track_index"]))
        return selected
    motor = frame[frame["biosample_name"].astype(str).str.contains("motor neuron", case=False)]
    selected = [int(motor.iloc[0]["track_index"])] if not motor.empty else []
    means = frame["nonzero_mean"].to_numpy(dtype=float)
    for quantile in (0.25, 0.5):
        target = float(np.quantile(means, quantile))
        index = int((frame["nonzero_mean"].astype(float) - target).abs().idxmin())
        track_index = int(frame.loc[index, "track_index"])
        if track_index not in selected:
            selected.append(track_index)
    return selected[:3]


def tensor_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.float().flatten()
    nonzero = values[values != 0]
    return {
        "mean": values.mean().item(),
        "std": values.std().item(),
        "max": values.max().item(),
        "sum": values.sum().item(),
        "nonzero_fraction": (values != 0).float().mean().item(),
        "nonzero_mean": nonzero.mean().item() if nonzero.numel() else 0.0,
        "median": values.median().item(),
    }


def segment_totals(values: torch.Tensor, num_segments: int = 8) -> list[float]:
    positions = values.shape[0]
    return (
        values.reshape(num_segments, positions // num_segments, -1)
        .sum(dim=1)
        .flatten()
        .tolist()
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for baseline prediction")
    manifest = json.loads(args.manifest.read_text())
    config = manifest["species"][args.species]
    loader_args = SimpleNamespace(batch_size=1, num_workers=args.num_workers)
    loader = make_loader(config, manifest, args.species, "valid", loader_args)
    dataset = loader.dataset
    atac_key = f"{args.species}_atac"
    rna_key = f"{args.species}_rna_seq"
    atac_dataset = dataset.datasets[atac_key]
    rna_dataset = dataset.datasets[rna_key]
    atac_names = [Path(path).stem for path in config["atac_bigwigs"]]
    rna_names = rna_dataset.track_names
    allen_indices = {
        "atac": choose_allen_tracks(atac_names),
        "rna_seq": choose_allen_tracks(rna_names),
    }

    candidate_indices = np.linspace(
        0,
        len(dataset) - 1,
        min(args.candidate_loci, len(dataset)),
        dtype=int,
    )
    candidates = []
    for index in np.unique(candidate_indices):
        sequence, targets = dataset[int(index)]
        score = 0.0
        for modality, key in (("atac", atac_key), ("rna_seq", rna_key)):
            score += torch.log1p(
                targets[key][128][:, allen_indices[modality]].sum()
            ).item()
        candidates.append((score, int(index), sequence, targets))
    selected_loci = sorted(candidates, reverse=True)[: args.num_loci]

    atac_means = compute_track_means(
        [config["atac_bigwigs"][index] for index in allen_indices["atac"]],
        config["splits"]["train"],
        sequence_length=manifest["sequence_length"],
        max_samples=64,
    ).flatten()
    allen_means = {
        "atac": atac_means,
        "rna_seq": rna_dataset.track_means.flatten()[allen_indices["rna_seq"]],
    }

    metadata = pd.read_csv(args.baseline_metadata, sep="\t")
    baseline_indices = {
        modality: choose_baseline_tracks(metadata, modality)
        for modality in ("atac", "rna_seq")
    }
    baseline_metadata = {}
    for modality, indices in baseline_indices.items():
        frame = metadata[
            (metadata["output_type"] == modality)
            & metadata["track_index"].isin(indices)
        ].copy()
        frame = frame.set_index("track_index").loc[indices].reset_index()
        baseline_metadata[modality] = frame[
            ["track_index", "track_name", "biosample_name", "assay_title", "nonzero_mean"]
        ].to_dict("records")

    device = torch.device("cuda")
    model = load_trunk(
        AlphaGenome(),
        str(args.pretrained_weights),
        exclude_heads=False,
    ).to(device=device, dtype=torch.bfloat16).eval()
    result = {
        "species": args.species,
        "allen_tracks": {
            "atac": [atac_names[index] for index in allen_indices["atac"]],
            "rna_seq": [rna_names[index] for index in allen_indices["rna_seq"]],
        },
        "allen_track_means": {
            key: values.tolist() for key, values in allen_means.items()
        },
        "baseline_tracks": baseline_metadata,
        "loci": [],
    }
    for score, index, sequence, targets in selected_loci:
        organism_index = torch.zeros(1, dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            scaled_predictions = model(
                sequence.unsqueeze(0).to(device),
                organism_index,
                return_scaled_predictions=True,
                resolutions=(128,),
                heads=("atac", "rna_seq"),
            )
        chrom, start, end = atac_dataset._positions_list[index]
        locus_result = {
            "index": index,
            "chromosome": chrom,
            "start": start,
            "end": end,
            "selection_score": score,
            "modalities": {},
        }
        for modality, key, squashing in (
            ("atac", atac_key, False),
            ("rna_seq", rna_key, True),
        ):
            target = targets[key][128][:, allen_indices[modality]].float()
            means = allen_means[modality].unsqueeze(0)
            target_model_space = targets_scaling(
                target.unsqueeze(0),
                means,
                resolution=128,
                apply_squashing=squashing,
            ).squeeze(0)
            baseline_index = torch.tensor(
                baseline_indices[modality], dtype=torch.long, device=device
            )
            baseline_model_space = scaled_predictions[modality][128][0].index_select(
                -1, baseline_index
            ).float().cpu()
            baseline_head = model.heads[modality]
            baseline_experimental = baseline_head.unscale(
                scaled_predictions[modality][128],
                organism_index,
                resolution=128,
                channels_last=True,
            )[0].index_select(-1, baseline_index).float().cpu()
            locus_result["modalities"][modality] = {
                "allen_experimental": tensor_stats(target),
                "allen_model_space": tensor_stats(target_model_space),
                "allen_model_space_segment_totals": segment_totals(target_model_space),
                "baseline_model_space_prediction": tensor_stats(baseline_model_space),
                "baseline_model_space_segment_totals": segment_totals(baseline_model_space),
                "baseline_experimental_prediction": tensor_stats(baseline_experimental),
            }
        result["loci"].append(locus_result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
