#!/usr/bin/env python
"""Prepare references, chromosome splits, and an Allen Multiome manifest."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import shutil
from pathlib import Path


DEFAULT_DATA_ROOT = Path(
    "/gpfs/commons/datasets/controlled/NYGC_AI_Initiative/AllenBrainMultiome"
)
DEFAULT_HUMAN_FASTA = Path(
    "/gpfs/commons/groups/knowles_lab/index/hg38/hg38.fa"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("finetuning_output/allen_multiome"))
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--sequence-length", type=int, default=131_072)
    parser.add_argument("--stride", type=int, default=131_072)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-valid", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def materialize_gzip(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with gzip.open(source, "rb") as input_handle, destination.open("wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=16 * 1024 * 1024)
    return destination


def chromosome_sizes(bigwig: Path) -> dict[str, int]:
    import pyBigWig

    with pyBigWig.open(str(bigwig)) as handle:
        return {str(chrom): int(size) for chrom, size in handle.chroms().items()}


def primary_chromosomes(sizes: dict[str, int], sequence_length: int) -> set[str]:
    """Select chromosome-scale contigs without assuming an assembly naming scheme."""
    minimum_size = max(sequence_length, 10_000_000)
    return {chrom for chrom, size in sizes.items() if size >= minimum_size}


def choose_holdouts(sizes: dict[str, int], species: str, sequence_length: int) -> tuple[str, str]:
    preferred = {
        "human": ("chr8", "chr9"),
        "marmoset": ("chr8", "chr9"),
    }.get(species)
    if preferred and all(chrom in sizes for chrom in preferred):
        return preferred
    primary_set = primary_chromosomes(sizes, sequence_length)
    primary = [
        chrom
        for chrom, size in sorted(sizes.items(), key=lambda item: item[1], reverse=True)
        if chrom in primary_set
    ]
    if len(primary) < 10:
        raise ValueError(f"Could not identify enough primary contigs for {species}")
    return primary[7], primary[8]


def write_split(
    path: Path,
    rows: list[tuple[str, int, int]],
    limit: int | None,
    rng: random.Random,
) -> None:
    if limit is not None and len(rows) > limit:
        rows = rng.sample(rows, limit)
    rows.sort()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for chrom, start, end in rows:
            handle.write(f"{chrom}\t{start}\t{end}\n")


def select_rna_file(rna_dir: Path, species_title: str) -> tuple[Path, bool]:
    import anndata

    candidates = [
        rna_dir / f"{species_title}_HMBA_basalganglia_pseudobulk_by_group.h5ad",
        rna_dir / f"{species_title}_HMBA_basalganglia_pseudobulk_by_group_correctnorm.h5ad",
        rna_dir / f"{species_title}_HMBA_basalganglia_pseudobulk_aligned.h5ad",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        adata = anndata.read_h5ad(candidate, backed="r")
        has_coordinates = {"chromosome", "start", "stop"}.issubset(adata.var.columns)
        adata.file.close()
        if has_coordinates:
            return candidate, True
    for candidate in candidates:
        if candidate.exists():
            return candidate, False
    raise FileNotFoundError(f"No pseudobulk RNA file found for {species_title}")


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    output = args.output_dir.resolve()
    references = (
        args.reference_dir.resolve() if args.reference_dir is not None else output / "references"
    )
    rng = random.Random(args.seed)

    species_config = {
        "human": {
            "title": "Human",
            "fasta": DEFAULT_HUMAN_FASTA,
            "gtf": root / "genomes/gencode.v49.basic.annotation.gtf.gz",
        },
        "macaque": {
            "title": "Macaque",
            "fasta": materialize_gzip(
                root / "genomes/GCF_003339765.1_Mmul_10_genomic.fna.gz",
                references / "macaque.fa",
            ),
            "gtf": root / "genomes/Macaca_mulatta.Mmul_10.115.gtf.gz",
        },
        "marmoset": {
            "title": "Marmoset",
            "fasta": materialize_gzip(
                root / "genomes/GCA_011100555.2_mCalJa1.2.pat.X_mitos2.fasta.gz",
                references / "marmoset.fa",
            ),
            "gtf": root / "genomes/GCF_011100555.1_mCalJa1.2.pat.X_mitos2.gtf.gz",
        },
    }

    manifest: dict[str, object] = {
        "data_root": str(root),
        "sequence_length": args.sequence_length,
        "stride": args.stride,
        "species": {},
    }
    for species, config in species_config.items():
        bigwigs = sorted((root / "bigwigs" / species).glob("*.bw"))
        if not bigwigs:
            raise FileNotFoundError(f"No ATAC BigWigs found for {species}")
        sizes = chromosome_sizes(bigwigs[0])
        primary = primary_chromosomes(sizes, args.sequence_length)
        valid_chrom, test_chrom = choose_holdouts(sizes, species, args.sequence_length)
        split_rows = {"train": [], "valid": [], "test": []}
        for chrom, size in sizes.items():
            if chrom not in primary:
                continue
            split = "valid" if chrom == valid_chrom else "test" if chrom == test_chrom else "train"
            for start in range(0, size - args.sequence_length + 1, args.stride):
                split_rows[split].append((chrom, start, start + args.sequence_length))
        split_dir = output / "splits" / species
        limits = {"train": args.limit_train, "valid": args.limit_valid, "test": args.limit_test}
        split_paths = {}
        for split, rows in split_rows.items():
            path = split_dir / f"{split}.bed"
            write_split(path, rows, limits[split], rng)
            split_paths[split] = str(path)

        rna_file, rna_has_coordinates = select_rna_file(root / "RNA", str(config["title"]))
        manifest["species"][species] = {
            "fasta": str(Path(config["fasta"]).resolve()),
            "gtf": str(Path(config["gtf"]).resolve()),
            "atac_bigwigs": [str(path.resolve()) for path in bigwigs],
            "rna_h5ad": str(rna_file.resolve()),
            "rna_has_coordinates": rna_has_coordinates,
            "splits": split_paths,
            "valid_chromosome": valid_chrom,
            "test_chromosome": test_chrom,
        }
        if species == "human":
            manifest["species"][species]["expression_var_column"] = "gene_id"
        if species == "macaque":
            manifest["species"][species].update(
                {
                    "expression_var_column": "gene_name",
                    "annotation_gene_column": "gene_name",
                    "annotation_chromosome_map": {
                        **{str(index): f"NC_{41_753 + index:06d}.1" for index in range(1, 21)},
                        "X": "NC_041774.1",
                        "Y": "NC_027914.1",
                    },
                }
            )
        if species == "marmoset":
            manifest["species"][species].update(
                {
                    "annotation_gene_column": "gene_name",
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
