#!/usr/bin/env python
"""LoRA fine-tuning launcher for human brain development ATAC BigWigs.

This mirrors the dataset-specific launcher in ``../alphagenome_ft/scripts``
while delegating the actual optimization to this repo's PyTorch
``scripts/finetune.py`` implementation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_BIGWIG_DIR = Path(
    "/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/bigwigs"
)
DEFAULT_FASTA = Path("/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa")
DEFAULT_OUTPUT_DIR = Path("finetuning_output/humanbraindev_atac_lora")


def _positive_int_or_none(value: str) -> int | None:
    if value.lower() in {"none", "null", "0"}:
        return None
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer, 0, or none.")
    return parsed


def parse_chrom_set(value: str) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def discover_bigwigs(bigwig_dir: Path, limit: int | None = None) -> list[Path]:
    bigwigs = sorted(bigwig_dir.expanduser().glob("*.bw"))
    if limit is not None:
        bigwigs = bigwigs[:limit]
    if not bigwigs:
        raise FileNotFoundError(f"No .bw files found in {bigwig_dir}")
    return bigwigs


def read_fai_chrom_sizes(fasta_path: Path) -> dict[str, int]:
    fai_path = Path(f"{fasta_path}.fai")
    if not fai_path.exists():
        raise FileNotFoundError(
            f"FASTA index not found: {fai_path}. Build it with samtools faidx."
        )

    chrom_sizes: dict[str, int] = {}
    with fai_path.open() as handle:
        for raw in handle:
            fields = raw.rstrip("\n").split("\t")
            if len(fields) >= 2:
                chrom_sizes[fields[0]] = int(fields[1])
    if not chrom_sizes:
        raise ValueError(f"No chromosome sizes found in {fai_path}")
    return chrom_sizes


def _limit_rows(rows: list[tuple[str, int, int]], limit: int | None) -> list[tuple[str, int, int]]:
    if limit is None:
        return rows
    return rows[:limit]


def write_chromosome_split_beds(
    fasta_path: Path,
    split_dir: Path,
    *,
    window_size: int,
    stride: int,
    valid_chroms: set[str],
    test_chroms: set[str],
    exclude_chroms: set[str],
    limit_train: int | None,
    limit_valid: int | None,
    limit_test: int | None,
) -> dict[str, Path]:
    chrom_sizes = read_fai_chrom_sizes(fasta_path)
    rows: dict[str, list[tuple[str, int, int]]] = {
        "train": [],
        "valid": [],
        "test": [],
    }

    for chrom, chrom_size in chrom_sizes.items():
        if chrom in exclude_chroms or "_" in chrom or chrom.startswith("chrUn"):
            continue
        if chrom_size < window_size:
            continue

        split = "train"
        if chrom in valid_chroms:
            split = "valid"
        elif chrom in test_chroms:
            split = "test"

        for start in range(0, chrom_size - window_size + 1, stride):
            rows[split].append((chrom, start, start + window_size))

    rows["train"] = _limit_rows(rows["train"], limit_train)
    rows["valid"] = _limit_rows(rows["valid"], limit_valid)
    rows["test"] = _limit_rows(rows["test"], limit_test)

    if not rows["train"]:
        raise ValueError("No training intervals were generated.")
    if not rows["valid"]:
        raise ValueError("No validation intervals were generated.")

    split_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, split_rows in rows.items():
        path = split_dir / f"{split}.bed"
        with path.open("w") as handle:
            for chrom, start, end in split_rows:
                handle.write(f"{chrom}\t{start}\t{end}\n")
        paths[split] = path
        print(f"{split}: {len(split_rows)} interval(s) -> {path}")

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and run PyTorch LoRA fine-tuning for human brain development ATAC."
    )
    parser.add_argument("--bigwig-dir", type=Path, default=DEFAULT_BIGWIG_DIR)
    parser.add_argument("--fasta-path", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--pretrained-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--split-source", choices=("chromosome", "bed"), default="chromosome")
    parser.add_argument("--train-bed", type=Path, default=None)
    parser.add_argument("--val-bed", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=131072)
    parser.add_argument("--stride", type=int, default=131072)
    parser.add_argument("--valid-chroms", default="chr8")
    parser.add_argument("--test-chroms", default="chr9")
    parser.add_argument("--exclude-chroms", default="chrM,chrY")
    parser.add_argument("--limit-train", type=_positive_int_or_none, default=None)
    parser.add_argument("--limit-valid", type=_positive_int_or_none, default=None)
    parser.add_argument("--limit-test", type=_positive_int_or_none, default=None)
    parser.add_argument("--limit-bigwigs", type=_positive_int_or_none, default=None)
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-targets", default="q_proj,v_proj")
    parser.add_argument("--resolutions", default="1,128")
    parser.add_argument(
        "--dtype",
        choices=(
            "bfloat16",
            "float32",
            "float16",
            "bfloat16-params",
            "float16-params",
            "nvfp8",
            "nvfp4",
        ),
        default="bfloat16",
    )
    parser.add_argument(
        "--fp8-recipe",
        choices=("tensorwise", "rowwise", "rowwise_with_gw_hp"),
        default="tensorwise",
    )
    parser.add_argument("--fp8-min-feature-multiple", type=int, default=16)
    parser.add_argument(
        "--fp8-skip-name-patterns",
        default="heads,original_layer,lora_,locon_,ia3,adapter",
    )
    parser.add_argument("--fp4-min-feature-multiple", type=int, default=16)
    parser.add_argument("--fp4-mode", choices=("qat", "weight-only"), default="weight-only")
    parser.add_argument(
        "--fp4-skip-name-patterns",
        default="heads,lora_,locon_,ia3,adapter",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--track-means-samples", type=_positive_int_or_none, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-io-workers", type=int, default=16)
    parser.add_argument("--profile-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cache-genome", action="store_true")
    parser.add_argument("--cache-signals", action="store_true")
    parser.add_argument("--save-delta", action="store_true", default=True)
    parser.add_argument("--no-save-delta", action="store_false", dest="save_delta")
    parser.add_argument("--no-full-checkpoint", action="store_true")
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="alphagenome-finetune")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bigwig_dir = args.bigwig_dir.expanduser().resolve()
    fasta_path = args.fasta_path.expanduser().resolve()
    pretrained_weights = args.pretrained_weights.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    if not pretrained_weights.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained_weights}")

    bigwigs = discover_bigwigs(bigwig_dir, args.limit_bigwigs)
    print(f"Discovered {len(bigwigs)} BigWig target track(s) in {bigwig_dir}")

    split_dir = (
        args.split_dir.expanduser().resolve()
        if args.split_dir is not None
        else output_dir / "splits"
    )

    if args.split_source == "bed":
        if args.train_bed is None or args.val_bed is None:
            raise ValueError("--train-bed and --val-bed are required with --split-source=bed")
        train_bed = args.train_bed.expanduser().resolve()
        val_bed = args.val_bed.expanduser().resolve()
    else:
        bed_paths = write_chromosome_split_beds(
            fasta_path,
            split_dir,
            window_size=args.window_size,
            stride=args.stride,
            valid_chroms=parse_chrom_set(args.valid_chroms),
            test_chroms=parse_chrom_set(args.test_chroms),
            exclude_chroms=parse_chrom_set(args.exclude_chroms),
            limit_train=args.limit_train,
            limit_valid=args.limit_valid,
            limit_test=args.limit_test,
        )
        train_bed = bed_paths["train"]
        val_bed = bed_paths["valid"]

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("finetune.py")),
        "--mode",
        "lora",
        "--genome",
        str(fasta_path),
        "--modality",
        "atac",
        "--bigwig",
        *[str(path) for path in bigwigs],
        "--train-bed",
        str(train_bed),
        "--val-bed",
        str(val_bed),
        "--sequence-length",
        str(args.window_size),
        "--resolutions",
        args.resolutions,
        "--pretrained-weights",
        str(pretrained_weights),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-steps",
        str(args.warmup_steps),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-targets",
        args.lora_targets,
        "--dtype",
        args.dtype,
        "--fp8-recipe",
        args.fp8_recipe,
        "--fp8-min-feature-multiple",
        str(args.fp8_min_feature_multiple),
        "--fp8-skip-name-patterns",
        args.fp8_skip_name_patterns,
        "--fp4-min-feature-multiple",
        str(args.fp4_min_feature_multiple),
        "--fp4-mode",
        args.fp4_mode,
        "--fp4-skip-name-patterns",
        args.fp4_skip_name_patterns,
        "--num-workers",
        str(args.num_workers),
        "--max-io-workers",
        str(args.max_io_workers),
        "--profile-batches",
        str(args.profile_batches),
    ]

    if args.run_name:
        cmd.extend(["--run-name", args.run_name])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.track_means_samples is not None:
        cmd.extend(["--track-means-samples", str(args.track_means_samples)])
    if args.gradient_checkpointing:
        cmd.append("--gradient-checkpointing")
    if args.cache_genome:
        cmd.append("--cache-genome")
    if args.cache_signals:
        cmd.append("--cache-signals")
    if args.save_delta:
        cmd.append("--save-delta")
    if args.no_full_checkpoint:
        cmd.append("--no-full-checkpoint")
    if args.no_save_checkpoints:
        cmd.append("--no-save-checkpoints")
    if args.wandb:
        cmd.extend(["--wandb", "--wandb-project", args.wandb_project])
        if args.wandb_entity:
            cmd.extend(["--wandb-entity", args.wandb_entity])

    print("Command:")
    print(" ".join(cmd))
    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
