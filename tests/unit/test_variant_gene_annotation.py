import gzip

import torch

from alphagenome_pytorch.variant_scoring import Interval
from alphagenome_pytorch.variant_scoring.annotations import GeneAnnotation


def test_gene_annotation_reads_gzipped_gtf_without_pyranges(tmp_path):
    path = tmp_path / "genes.gtf.gz"
    attributes = 'gene_id "ENSG1.2"; gene_name "GENE1"; gene_type "protein_coding";'
    with gzip.open(path, "wt") as handle:
        handle.write(f"chr1\ttest\tgene\t101\t300\t.\t+\t.\t{attributes}\n")
        handle.write(f"chr1\ttest\texon\t101\t120\t.\t+\t.\t{attributes}\n")
        handle.write(f"chr1\ttest\texon\t201\t220\t.\t+\t.\t{attributes}\n")

    annotation = GeneAnnotation(path)
    info = annotation.get_gene_info("ENSG1")
    assert info["start"] == 100
    assert info["end"] == 300
    mask = annotation.get_exon_mask(
        "ENSG1",
        Interval("chr1", 0, 256),
        resolution=1,
        seq_length=256,
        device=torch.device("cpu"),
    )
    assert int(mask.sum()) == 40
    assert mask[100:120].all()
    assert mask[200:220].all()
