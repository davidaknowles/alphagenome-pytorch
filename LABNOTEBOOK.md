# Lab Notebook

## 2026-08-07, Allen Brain Multiome v0 design

The fine-tuning model uses one shared AlphaGenome trunk with rank-8 LoRA adapters on the transformer tower and rank-4 Locon adapters on encoder and decoder convolutions. Six new heads represent ATAC and RNA for human, macaque, and marmoset. Species batches cycle within each epoch, which shares sequence features and adapters while retaining every species-specific output track.

ATAC supervision uses every available BigWig at 128 bp resolution. RNA supervision uses pseudobulk observations as tracks. Each pseudobulk is divided by its contributing cell count and scaled to the median per-cell library size across tracks. A normalized gene total is distributed uniformly over the union of its annotated exons at 128 bp resolution, preserving the gene total without inventing intronic signal. RNA head means are computed as nonzero expression per annotated exon base, consistent with the head's resolution scaling.

Chromosome splits are generated independently for each assembly. Human uses chromosomes 8 and 9 for validation and test. Other assemblies prefer the same labels when present and otherwise use deterministic primary-contig ranks. Metrics are recorded for each species and split, including profile correlation, bin correlation, differential correlation, and loss.

Variant scoring uses human ATAC and RNA heads on fine-mapped variants. The benchmark defines every variant-gene fine-map row with SuSiE PIP greater than 0.75 as positive. It samples one negative with PIP less than 0.01 per positive, without replacement, within SingleBrain cell type and base-2 distance-to-TSS strata. SingleBrain subclusters map to Allen broad classes, Ast to Astrocyte, End to Endo, Ext to Glut tracks, IN to non-MSN/non-dopaminergic/non-cholinergic GABA tracks, MG to Microglia, OD to mature Oligo tracks, and OPC to OPC. Following AlphaGenome gene-expression variant scoring, REF and ALT inference uses a 1 Mb context centered on the variant and the primary score is log fold change over only the fine-mapped target gene's exons. The target gene's TSS must be inside the context, and exon masks are clipped to the context. ATAC is retained as a secondary local 501 bp variant-centered score. Both modalities are aggregated only over matched Allen cell-type tracks. The reported discrimination metrics are overall and class-specific AUROC and average precision. Distance matching is checked with a two-sample Kolmogorov-Smirnov statistic.

The requested fine-map path was absent. The data are available through the project’s public `singlebrain_full_finemap` location. Production results will be added after the smoke and full jobs complete.

## 2026-08-07, end-to-end smoke

The focused suite passed 106 tests. A one-epoch run used two training windows and one held-out window per split and species. Mean validation loss was 1.9393. Species validation losses were 1.9698 for human, 1.8564 for macaque, and 1.9917 for marmoset. Test losses were 1.8532 on human chromosome 9, 1.8865 on macaque contig NC_041761.1, and 2.1325 on marmoset chromosome 9.

ATAC profile correlations on the single test windows were 0.0016 for human, 0.0395 for macaque, and 0.0063 for marmoset. Single-window RNA correlations were zero when the sampled window contained no complete annotated gene, so these are pipeline checks rather than performance estimates. Production evaluation uses every held-out chromosome window.

Delta checkpoint reload and six-head evaluation completed. ATAC and RNA effects were produced for two fine-mapped variants. Correlation and AUROC were undefined at this smoke size and will be estimated in the production variant run.

## 2026-08-08, production progress

The initial production run completed five epochs, but it used raw pseudobulk sums spread over gene bodies. That representation retained cell-count and library-size differences and computed the RNA track mean at bin rather than base scale. Its checkpoints and downstream metrics are invalid and are retained only as archived diagnostics. Corrected training starts from the pretrained trunk rather than from those deltas.

The corrected target audit passed for all three species. Cell-type pseudobulks contain 42-145,491 human cells, 110-83,269 macaque cells, and 49-65,164 marmoset cells. Before library equalization, per-cell totals span about 8,150-26,630 counts. The species-specific median target totals are 18,934 for human, 16,615 for macaque, and 16,843 for marmoset. All audited held-out windows contain finite, nonzero exon-projected targets. Macaque annotation chromosomes are translated from Ensembl chromosome labels to the RefSeq accessions used by its sequence and ATAC references.

An end-to-end one-window smoke run completed training, validation, and delta checkpoint serialization for every species. RNA training losses were 0.92 for human, 2.72 for macaque, and 0.92 for marmoset, on the same order as the corresponding ATAC losses of 1.02, 1.05, and 1.00. These values are preflight checks rather than performance estimates because each split contained one window.

Scooby provides the closest methodological reference but has richer RNA inputs. It reconstructs strand-specific read coverage for individual cells from mapped fragments, pools that coverage to 32 bp, and applies a power transform and soft clipping. Because each target is one cell, it does not divide a pseudobulk by a group cell count. Its count and positional objectives jointly preserve total expression and read placement. The Allen RNA inputs are gene-count pseudobulks without fragments, so normalized gene totals over annotated exons are a surrogate for coverage rather than reconstructed read profiles.

Held-out evaluation uses 128 bp double-centered R2 as its primary metric. Prediction and target matrices are separately centered over genomic observations within each output track and over output cell types within each observation, then scored as one minus residual sum of squares divided by centered target sum of squares. Profile, bin-level, and differential Pearson correlations remain secondary diagnostics.
