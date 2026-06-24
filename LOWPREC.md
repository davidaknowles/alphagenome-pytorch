# Low-Precision LoRA Profiling

Profiles were run on an NVIDIA RTX PRO 6000 Blackwell Server Edition with
`~/venv/torch`, `torch==2.12.1+cu130`, and `torchao==0.17.0`.

The benchmark uses `scripts/profile_lora_memory.py` with synthetic one-hot
sequence inputs, LoRA rank 8/alpha 16 on `q_proj,v_proj`, 134 ATAC output
tracks, sequence length 131072, resolutions `1,128`, batch size 8, one warmup
step, and three timed steps. This isolates model memory/compute from BigWig I/O.

## Configurations

| Config | CLI dtype | Low-precision conversion | Trainable params / AdamW |
| --- | --- | --- | --- |
| Default | `bfloat16` | none | fp32 params, fp32 AdamW state |
| Low VRAM | `nvfp8` | torchao Float8Linear, tensorwise recipe | bf16 params, bf16 AdamW moments; fp32 step tensors |
| Super Low VRAM | `nvfp4` | torchao NVFP4 QAT fake-quantized Linear | bf16 params, bf16 AdamW moments; fp32 step tensors |

## Results

| Config | Converted / skipped linears | Mean step (s) | Samples/s | bp/s | Peak allocated GiB | Peak reserved GiB | `nvidia-smi` max GiB | GPU util mean / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 0 / 0 | 1.301 | 6.148 | 805,779 | 77.98 | 85.82 | 86.51 | 97.5% / 100% |
| Low VRAM (`nvfp8`) | 91 / 63 | 0.986 | 8.116 | 1,063,750 | 51.39 | 58.41 | 59.10 | 95.7% / 100% |
| Super Low VRAM (`nvfp4`) | 109 / 45 | 1.075 | 7.441 | 975,353 | 51.25 | 58.08 | 58.77 | 97.0% / 100% |

At the same batch size, both low-precision modes cut CUDA reserved memory by
about 32% relative to the default synthetic LoRA profile. `nvfp8` was fastest in
this run. `nvfp4` is close on memory but slower than `nvfp8`, because torchao's
current NVFP4 training path is QAT/fake quantization rather than compact
weight-only training storage.

## Commands

Default:

```bash
source ~/venv/torch/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/profile_lora_memory.py \
  --dtype bfloat16 \
  --batch-size 8 \
  --num-tracks 134 \
  --warmup-steps 1 \
  --timed-steps 3
```

Low VRAM:

```bash
source ~/venv/torch/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/profile_lora_memory.py \
  --dtype nvfp8 \
  --fp8-recipe tensorwise \
  --batch-size 8 \
  --num-tracks 134 \
  --warmup-steps 1 \
  --timed-steps 3
```

Super Low VRAM:

```bash
source ~/venv/torch/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/profile_lora_memory.py \
  --dtype nvfp4 \
  --fp4-mode qat \
  --batch-size 8 \
  --num-tracks 134 \
  --warmup-steps 1 \
  --timed-steps 3
```

## Notes

- The FP8 default is `tensorwise`. Torchao's rowwise/axiswise scaling is more
  attractive numerically, but in this environment it failed on AlphaGenome's
  tensor shapes at batch size 8 with `axiswise scaling is not supported yet`.
- The NVFP4 default is `qat`. Torchao's `NVFP4WeightOnlyConfig` converted the
  model, but the forward pass failed on an unsupported `NVFP4Tensor` dispatch
  (`aten.expand`) in this model. The working training path uses torchao's
  NVFP4 QAT Linear and wraps converted linears to make non-contiguous or
  higher-rank inputs compatible with the prototype implementation.
- Low-precision AdamW state was verified by the profiler. The reported
  `bfloat16,float32` state means bf16 moment buffers plus fp32 step counters.
