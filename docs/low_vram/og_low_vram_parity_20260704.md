# OG AlphaGenome Low-VRAM Parity

Run root: `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver`

This prediction-only check uses the original AlphaGenome all-folds checkpoint as the reference,
not the merged fine-tuned checkpoint. It evaluates the native `atac` head at 128 bp on
`chr9` windows and does not load targets or compute loss.

Feature abbreviations: BF16 = bfloat16 parameter/compute policy; Eff. = materialized
effective standardized convolutions; Int8 = Triton int8 weight-only Conv1d; NoInt =
skip unused encoder intermediates for 128 bp output; Pool = Triton max-pool without
indices; FEmb = fused DNA embedder block; FD0 = fused first downsampling block;
Flex = FlexAttention MHA; LRB = low-resolution attention bias.

The supported low-VRAM pipeline uses the original OG safetensors directly. For
Eff. rows, standardized convolutions are materialized in memory at load time;
there is no separate bf16/W_eff checkpoint artifact to share.

## 131,072 bp windows, batch size 32

| Strategy | BF16 | Eff. | Int8 | NoInt | Pool | FEmb | FD0 | Flex | LRB | Peak GiB | Ex./s | max abs | RMSE | Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Default |  |  |  |  |  |  |  |  |  | 90.58 | 14.621 | 0.000e+00 | 0.000e+00 | 1.000000 |
| Bfloat16 | x |  |  |  |  |  |  |  |  | 65.44 | 21.773 | 7.306e+01 | 2.322e-01 | 0.999902 |
| Triton Conv1d | x | x | x |  |  |  |  |  |  | 41.64 | 11.931 | 1.841e+02 | 4.365e-01 | 0.999779 |
| FlexAttention | x | x | x |  |  |  |  | x |  | 41.64 | 9.309 | 1.431e+02 | 4.361e-01 | 0.999779 |
| Flex low-res bias | x | x | x |  |  |  |  | x | x | 41.64 | 11.957 | 1.431e+02 | 4.361e-01 | 0.999779 |
| No intermediates | x | x | x | x |  |  |  |  |  | 41.64 | 12.308 | 1.841e+02 | 4.365e-01 | 0.999779 |
| Triton pool | x | x | x | x | x |  |  |  |  | 33.14 | 12.531 | 1.841e+02 | 4.365e-01 | 0.999779 |
| Fused embedder | x | x | x | x | x | x |  |  |  | 30.66 | 10.989 | 1.431e+02 | 4.338e-01 | 0.999787 |
| Fused down0 | x | x | x | x | x |  | x |  |  | 29.61 | 10.653 | 1.431e+02 | 4.155e-01 | 0.999814 |
| Fused embedder+down0 | x | x | x | x | x | x | x |  |  | 23.63 | 9.536 | 1.431e+02 | 4.150e-01 | 0.999817 |
| All features | x | x | x | x | x | x | x | x | x | 23.63 | 8.173 | 1.431e+02 | 4.150e-01 | 0.999817 |

## 1,048,576 bp windows, batch size 2

| Strategy | BF16 | Eff. | Int8 | NoInt | Pool | FEmb | FD0 | Flex | LRB | Peak GiB | Ex./s | max abs | RMSE | Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Default |  |  |  |  |  |  |  |  |  | 51.52 | 0.842 | 0.000e+00 | 0.000e+00 | 1.000000 |
| Bfloat16 | x |  |  |  |  |  |  |  |  | 46.55 | 1.046 | 7.593e+01 | 2.807e-01 | 0.999853 |
| Triton Conv1d | x | x | x |  |  |  |  |  |  | 31.59 | 0.987 | 1.129e+02 | 3.915e-01 | 0.999761 |
| FlexAttention | x | x | x |  |  |  |  | x |  | 27.59 | 1.074 | 1.129e+02 | 3.921e-01 | 0.999760 |
| Flex low-res bias | x | x | x |  |  |  |  | x | x | 19.59 | 1.268 | 1.129e+02 | 3.921e-01 | 0.999760 |
| No intermediates | x | x | x | x |  |  |  |  |  | 27.59 | 0.995 | 1.129e+02 | 3.915e-01 | 0.999761 |
| Triton pool | x | x | x | x | x |  |  |  |  | 29.09 | 1.010 | 1.129e+02 | 3.915e-01 | 0.999761 |
| Fused embedder | x | x | x | x | x | x |  |  |  | 27.86 | 0.903 | 1.129e+02 | 3.900e-01 | 0.999767 |
| Fused down0 | x | x | x | x | x |  | x |  |  | 25.57 | 0.889 | 1.129e+02 | 3.771e-01 | 0.999791 |
| Fused embedder+down0 | x | x | x | x | x | x | x |  |  | 22.59 | 0.839 | 1.129e+02 | 3.764e-01 | 0.999794 |
| All features | x | x | x | x | x | x | x | x | x | 10.59 | 1.006 | 1.139e+02 | 3.765e-01 | 0.999794 |

## Metric paths

- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/default_w131072_b32`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/bf16_params_w131072_b32`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/triton_conv_w131072_b32`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/flexattention_w131072_b32`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/flex_lowres_bias_w131072_b32`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/no_intermediates_w131072_b32`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/triton_pool_w131072_b32`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/fused_embedder_w131072_b32`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/fused_down0_w131072_b32`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/fused_embedder_down0_w131072_b32`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/all_features_w131072_b32`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/default_w1048576_b2`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/bf16_params_w1048576_b2`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/triton_conv_w1048576_b2`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/flexattention_w1048576_b2`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/flex_lowres_bias_w1048576_b2`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/no_intermediates_w1048576_b2`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/triton_pool_w1048576_b2`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/fused_embedder_w1048576_b2`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/fused_down0_w1048576_b2`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/fused_embedder_down0_w1048576_b2`
- `/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch/outputs/low_vram/20260704_131220_parity_driver/all_features_w1048576_b2`
