# Lab Notebook

## 2026-08-07, Allen Brain Multiome v0 design

The fine-tuning model uses one shared AlphaGenome trunk with rank-8 LoRA adapters on the transformer tower and rank-4 Locon adapters on encoder and decoder convolutions. Six new heads represent ATAC and RNA for human, macaque, and marmoset. Species batches cycle within each epoch, which shares sequence features and adapters while retaining every species-specific output track.

ATAC supervision uses every available BigWig at 128 bp resolution. RNA supervision uses pseudobulk observations as tracks. Per-gene totals are distributed uniformly across 128 bp gene-body bins while preserving each gene total. This avoids generating hundreds of derived BigWigs and keeps RNA labels linked to annotation coordinates.

Chromosome splits are generated independently for each assembly. Human uses chromosomes 8 and 9 for validation and test. Other assemblies prefer the same labels when present and otherwise use deterministic primary-contig ranks. Metrics are recorded for each species and split, including profile correlation, bin correlation, differential correlation, and loss.

Variant scoring uses human ATAC and RNA heads on fine-mapped variants. The first benchmark compares maximum absolute predicted effect with SuSiE posterior inclusion probability using Spearman correlation and AUROC at PIP at least 0.1. This is a prioritization test because PIP has no effect direction.

The requested fine-map path was absent. The data are available through the project’s public `singlebrain_full_finemap` location. Production results will be added after the smoke and full jobs complete.

## 2026-08-07, end-to-end smoke

The focused suite passed 106 tests. A one-epoch run used two training windows and one held-out window per split and species. Mean validation loss was 1.9393. Species validation losses were 1.9698 for human, 1.8564 for macaque, and 1.9917 for marmoset. Test losses were 1.8532 on human chromosome 9, 1.8865 on macaque contig NC_041761.1, and 2.1325 on marmoset chromosome 9.

ATAC profile correlations on the single test windows were 0.0016 for human, 0.0395 for macaque, and 0.0063 for marmoset. Single-window RNA correlations were zero when the sampled window contained no complete annotated gene, so these are pipeline checks rather than performance estimates. Production evaluation uses every held-out chromosome window.

Delta checkpoint reload and six-head evaluation completed. ATAC and RNA effects were produced for two fine-mapped variants. Correlation and AUROC were undefined at this smoke size and will be estimated in the production variant run.
