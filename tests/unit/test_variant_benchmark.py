import csv
import gzip

import numpy as np
import pytest

from alphagenome_pytorch.variant_scoring.benchmark import (
    distance_to_tss_bin,
    select_pip_matched_variants,
)
from scripts.score_singlebrain_finemap import average_precision


FIELDS = ["feature", "chr", "pos", "ref", "alt", "variant_id", "susie_pip"]


def _write_rows(path, rows):
    with gzip.open(path, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_distance_to_tss_bin():
    assert [distance_to_tss_bin(x) for x in (0, 1, 2, 3, 7, 8)] == [0, 0, 1, 1, 2, 3]
    with pytest.raises(ValueError, match="non-negative"):
        distance_to_tss_bin(-1)


def test_average_precision_handles_ties_as_one_threshold():
    labels = np.array([True, False, True, False])
    scores = np.array([1.0, 1.0, 0.0, 0.0])
    assert average_precision(labels, scores) == pytest.approx(0.5)


def test_selects_strict_pip_thresholds_and_distance_matched_negatives(tmp_path):
    path = tmp_path / "test_all.susie_all_pip.tsv.gz"
    rows = [
        {"feature": "ENSG1.2", "chr": "chr1", "pos": "101", "ref": "A", "alt": "C", "variant_id": "positive", "susie_pip": "0.9"},
        {"feature": "ENSG1.2", "chr": "chr1", "pos": "102", "ref": "A", "alt": "G", "variant_id": "positive_boundary", "susie_pip": "0.75"},
        {"feature": "ENSG1.2", "chr": "chr1", "pos": "100", "ref": "C", "alt": "T", "variant_id": "negative", "susie_pip": "0.001"},
        {"feature": "ENSG1.2", "chr": "chr1", "pos": "98", "ref": "C", "alt": "G", "variant_id": "negative_boundary", "susie_pip": "0.01"},
        {"feature": "missing", "chr": "chr1", "pos": "101", "ref": "G", "alt": "T", "variant_id": "missing_gene", "susie_pip": "0.99"},
    ]
    _write_rows(path, rows)

    selected = select_pip_matched_variants([path], {"ENSG1": ("chr1", 100)})

    assert [row["benchmark_label"] for row in selected] == ["1", "0"]
    assert {row["variant_id"] for row in selected} == {"positive", "negative"}
    assert selected[0]["distance_to_tss_bin"] == selected[1]["distance_to_tss_bin"]
    assert selected[0]["match_pair_id"] == selected[1]["match_pair_id"]


def test_reports_distance_strata_without_enough_negatives(tmp_path):
    path = tmp_path / "test_all.susie_all_pip.tsv.gz"
    _write_rows(
        path,
        [{"feature": "ENSG1", "chr": "chr1", "pos": "117", "ref": "A", "alt": "C", "variant_id": "positive", "susie_pip": "0.8"}],
    )
    with pytest.raises(ValueError, match="Insufficient low-PIP"):
        select_pip_matched_variants([path], {"ENSG1": ("chr1", 100)})
