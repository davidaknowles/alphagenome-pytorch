"""Utilities for constructing variant-effect prediction benchmarks."""

from __future__ import annotations

import csv
import gzip
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from alphagenome_pytorch.extensions.finetuning.gene_annotation import load_gene_table


def load_gene_tss(gtf_path: str | Path) -> dict[str, tuple[str, int]]:
    """Load 0-based transcription start sites keyed by versionless gene ID."""
    genes = load_gene_table(str(gtf_path), filter_protein_coding=False)
    tss = {}
    for row in genes.itertuples(index=False):
        gene_id = str(row.gene_id).split(".", 1)[0]
        position = int(row.Start) if str(row.Strand) != "-" else int(row.End) - 1
        tss[gene_id] = (str(row.Chromosome), position)
    return tss


def distance_to_tss_bin(distance: int) -> int:
    """Return a base-2 distance stratum, with distances 0-1 in bin zero."""
    if distance < 0:
        raise ValueError("distance must be non-negative")
    return 0 if distance < 2 else int(math.log2(distance))


def singlebrain_cell_class(cell_type: str) -> str:
    """Map a SingleBrain fine-map cell type to its broad biological class."""
    prefixes = {
        "Ast": "astrocyte",
        "End": "endothelial",
        "Ext": "excitatory_neuron",
        "IN": "inhibitory_neuron",
        "MG": "microglia",
        "OD": "oligodendrocyte",
        "OPC": "opc",
    }
    for prefix, class_name in prefixes.items():
        if cell_type == prefix or cell_type.startswith(prefix):
            return class_name
    raise ValueError(f"Unsupported SingleBrain cell type: {cell_type!r}")


def allen_track_indices(cell_type: str, track_names: Iterable[str]) -> list[int]:
    """Return Allen track indices matched to a SingleBrain broad cell class."""
    class_name = singlebrain_cell_class(cell_type)
    names = list(track_names)

    def normalized(name: str) -> str:
        return "".join(character.lower() for character in name if character.isalnum())

    def is_match(name: str) -> bool:
        compact = normalized(name)
        if class_name == "astrocyte":
            return compact == "astrocyte"
        if class_name == "endothelial":
            return compact == "endo"
        if class_name == "excitatory_neuron":
            return "glut" in compact
        if class_name == "inhibitory_neuron":
            excluded = ("msn", "dopa", "cholinergic")
            return "gaba" in compact and not any(value in compact for value in excluded)
        if class_name == "microglia":
            return compact == "microglia"
        if class_name == "oligodendrocyte":
            return compact.startswith("oligo")
        if class_name == "opc":
            return compact == "opc"
        return False

    indices = [index for index, name in enumerate(names) if is_match(name)]
    if not indices:
        raise ValueError(f"No Allen tracks match SingleBrain cell type {cell_type!r}")
    return indices


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def _annotate_distance(
    row: dict[str, str], gene_tss: Mapping[str, tuple[str, int]]
) -> dict[str, str] | None:
    gene_id = row["feature"].split(".", 1)[0]
    annotation = gene_tss.get(gene_id)
    if annotation is None:
        return None
    chromosome, tss = annotation
    if chromosome != row["chr"]:
        return None
    distance = abs((int(row["pos"]) - 1) - tss)
    result = dict(row)
    result["gene_tss"] = str(tss)
    result["distance_to_tss"] = str(distance)
    result["distance_to_tss_bin"] = str(distance_to_tss_bin(distance))
    return result


def select_pip_matched_variants(
    paths: Iterable[str | Path],
    gene_tss: Mapping[str, tuple[str, int]],
    *,
    positive_threshold: float = 0.75,
    negative_threshold: float = 0.01,
    seed: int = 17,
    limit_pairs: int | None = None,
) -> list[dict[str, str]]:
    """Select all high-PIP rows and an equal distance-matched low-PIP set.

    Matching is without replacement within base-2 distance-to-TSS strata. The
    input is scanned twice so the low-PIP candidate pool is reservoir sampled
    without loading it into memory.
    """
    if positive_threshold <= negative_threshold:
        raise ValueError("positive_threshold must exceed negative_threshold")
    input_paths = [Path(path) for path in paths]
    if not input_paths:
        raise ValueError("No fine-mapping files were provided")

    positives = []
    for path in input_paths:
        with _open_text(path) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if float(row["susie_pip"]) <= positive_threshold:
                    continue
                annotated = _annotate_distance(row, gene_tss)
                if annotated is not None:
                    positives.append(annotated)

    positives.sort(key=lambda row: float(row["susie_pip"]), reverse=True)
    if limit_pairs is not None:
        if limit_pairs <= 0:
            raise ValueError("limit_pairs must be positive")
        positives = positives[:limit_pairs]
    if not positives:
        raise ValueError("No high-PIP variants had matching gene annotations")

    def matching_stratum(row: dict[str, str]) -> tuple[str, int]:
        return row.get("celltype", ""), int(row["distance_to_tss_bin"])

    needed = Counter(matching_stratum(row) for row in positives)
    reservoirs: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    seen = Counter()
    rng = random.Random(seed)
    for path in input_paths:
        with _open_text(path) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if float(row["susie_pip"]) >= negative_threshold:
                    continue
                annotated = _annotate_distance(row, gene_tss)
                if annotated is None:
                    continue
                stratum = matching_stratum(annotated)
                capacity = needed.get(stratum, 0)
                if capacity == 0:
                    continue
                seen[stratum] += 1
                reservoir = reservoirs[stratum]
                if len(reservoir) < capacity:
                    reservoir.append(annotated)
                else:
                    replacement = rng.randrange(seen[stratum])
                    if replacement < capacity:
                        reservoir[replacement] = annotated

    deficient = {
        stratum: (needed[stratum], len(reservoirs[stratum]))
        for stratum in needed
        if len(reservoirs[stratum]) < needed[stratum]
    }
    if deficient:
        raise ValueError(f"Insufficient low-PIP variants in distance strata: {deficient}")

    positives_by_stratum: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in positives:
        positives_by_stratum[matching_stratum(row)].append(row)

    selected = []
    pair_id = 0
    for stratum in sorted(positives_by_stratum):
        negatives = reservoirs[stratum]
        rng.shuffle(negatives)
        for positive, negative in zip(positives_by_stratum[stratum], negatives, strict=True):
            positive = dict(positive)
            negative = dict(negative)
            positive["benchmark_label"] = "1"
            negative["benchmark_label"] = "0"
            positive["match_pair_id"] = str(pair_id)
            negative["match_pair_id"] = str(pair_id)
            selected.extend((positive, negative))
            pair_id += 1
    return selected
