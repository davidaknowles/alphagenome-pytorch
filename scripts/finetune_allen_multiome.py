#!/usr/bin/env python
"""Round-robin multispecies LoRA plus Locon fine-tuning for Allen Multiome."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.extensions.finetuning.adapters import get_adapter_params
from alphagenome_pytorch.extensions.finetuning.checkpointing import save_delta_checkpoint
from alphagenome_pytorch.extensions.finetuning.datasets import (
    GeneExpressionDataset,
    GenomicDataset,
    MultimodalDataset,
    collate_multimodal,
    compute_track_means,
)
from alphagenome_pytorch.extensions.finetuning.training import (
    create_lr_scheduler,
    train_epoch_multihead,
    validation_loss_improved,
    validate_multihead,
)
from alphagenome_pytorch.extensions.finetuning.transfer import (
    TransferConfig,
    load_trunk,
    prepare_for_transfer,
    remove_all_heads,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pretrained-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-targets", default="tower")
    parser.add_argument("--locon-rank", type=int, default=4)
    parser.add_argument("--locon-alpha", type=int, default=4)
    parser.add_argument("--locon-targets", default="encoder,decoder")
    parser.add_argument("--atac-weight", type=float, default=1.0)
    parser.add_argument("--rna-weight", type=float, default=1.0)
    parser.add_argument("--track-means-samples", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def make_loader(dataset: MultimodalDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_multimodal,
    )


def build_datasets(manifest: dict, args: argparse.Namespace, split: str):
    loaders = {}
    track_names = {}
    track_means = {}
    resolutions = {}
    for species, config in manifest["species"].items():
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
        loaders[species] = make_loader(dataset, args, shuffle=split == "train")
        track_names[atac_key] = [Path(path).stem for path in config["atac_bigwigs"]]
        track_names[rna_key] = rna.track_names
        if split == "train":
            track_means[atac_key] = compute_track_means(
                config["atac_bigwigs"], bed,
                sequence_length=manifest["sequence_length"], resolution=128,
                max_samples=args.track_means_samples,
            )
            track_means[rna_key] = rna.track_means
        resolutions[atac_key] = (128,)
        resolutions[rna_key] = (128,)
    return loaders, track_names, track_means, resolutions


def main() -> None:
    args = parse_args()
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be nonnegative")
    if args.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be nonnegative")
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AlphaGenome fine-tuning")
    device = torch.device("cuda")
    manifest = json.loads(args.manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loaders, track_names, track_means, resolutions = build_datasets(manifest, args, "train")
    valid_loaders, valid_names, _, _ = build_datasets(manifest, args, "valid")
    if valid_names != track_names:
        raise ValueError("Train and validation track names differ")

    new_heads = {}
    for head_name, names in track_names.items():
        assay_type = "rna_seq" if head_name.endswith("rna_seq") else "atac"
        new_heads[head_name] = {
            "modality": assay_type,
            "num_tracks": len(names),
            "resolutions": [128],
            "num_organisms": 1,
            "track_means": track_means[head_name],
        }
    transfer_config = TransferConfig(
        mode=["lora", "locon"],
        lora_targets=[value for value in args.lora_targets.split(",") if value],
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        locon_targets=[value for value in args.locon_targets.split(",") if value],
        locon_rank=args.locon_rank,
        locon_alpha=args.locon_alpha,
        new_heads=new_heads,
    )
    model = AlphaGenome(gradient_checkpointing=args.gradient_checkpointing)
    model = load_trunk(model, str(args.pretrained_weights), exclude_heads=True)
    model = remove_all_heads(model)
    model = prepare_for_transfer(model, transfer_config)
    model = model.to(device=device, dtype=torch.bfloat16)
    heads = {name: model.heads[name] for name in track_names}

    params = get_adapter_params(model)
    for head in heads.values():
        params.extend(parameter for parameter in head.parameters() if parameter.requires_grad)
    unique_params = list({id(parameter): parameter for parameter in params}.values())
    optimizer = torch.optim.AdamW(unique_params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = sum(math.ceil(len(loader) / args.gradient_accumulation_steps) for loader in train_loaders.values())
    scheduler = create_lr_scheduler(
        optimizer, warmup_steps=min(args.warmup_steps, max(0, steps_per_epoch - 1)),
        total_steps=max(1, steps_per_epoch * args.epochs),
    )
    modality_weights = {
        name: args.rna_weight if name.endswith("rna_seq") else args.atac_weight
        for name in heads
    }
    resolution_weights = {name: {128: 1.0} for name in heads}
    run_config = vars(args) | {"manifest_data": manifest, "track_names": track_names}
    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in run_config.items()}
    (args.output_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    best_loss = float("inf")
    best_epoch = None
    epochs_since_improvement = 0
    stopped_early = False
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_train = {}
        for species, loader in train_loaders.items():
            loss, per_head = train_epoch_multihead(
                model, heads, loader, optimizer, scheduler, device,
                modality_weights, resolution_weights,
                positional_weight=5.0, count_weight=1.0, epoch=epoch,
                log_every=args.log_every, accumulation_steps=args.gradient_accumulation_steps,
                use_amp=True, amp_dtype=torch.bfloat16,
            )
            epoch_train[species] = {"loss": loss, "heads": per_head}

        epoch_valid = {}
        for species, loader in valid_loaders.items():
            species_heads = {
                f"{species}_atac": heads[f"{species}_atac"],
                f"{species}_rna_seq": heads[f"{species}_rna_seq"],
            }
            loss, metrics = validate_multihead(
                model, species_heads, loader, device,
                modality_weights, resolution_weights,
                positional_weight=5.0, count_weight=1.0,
                compute_pearson=False, use_amp=True, amp_dtype=torch.bfloat16,
            )
            epoch_valid[species] = {"loss": loss, "metrics": metrics}
        mean_valid = sum(item["loss"] for item in epoch_valid.values()) / len(epoch_valid)
        improved = validation_loss_improved(
            mean_valid,
            best_loss,
            min_delta=args.early_stopping_min_delta,
        )
        if improved:
            best_loss = mean_valid
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        record = {
            "epoch": epoch,
            "train": epoch_train,
            "valid": epoch_valid,
            "mean_valid_loss": mean_valid,
            "is_best": improved,
            "epochs_since_improvement": epochs_since_improvement,
        }
        history.append(record)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        checkpoint_args = dict(
            model=model, config=transfer_config, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, val_loss=mean_valid, track_names=track_names,
            manifest=str(args.manifest.resolve()),
        )
        save_delta_checkpoint(args.output_dir / f"checkpoint_epoch{epoch}.delta.pth", **checkpoint_args)
        if improved:
            save_delta_checkpoint(args.output_dir / "best_model.delta.pth", **checkpoint_args)
        print(json.dumps(record))
        if (
            args.early_stopping_patience > 0
            and epochs_since_improvement >= args.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"Early stopping after {epochs_since_improvement} epochs without "
                "validation-loss improvement."
            )
            break

    summary = {
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "stopped_early": stopped_early,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
