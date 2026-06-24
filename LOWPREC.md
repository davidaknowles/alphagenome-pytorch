# Low-Precision LoRA Profiling

Profiles were run on an NVIDIA RTX PRO 6000 Blackwell Server Edition with
`~/venv/torch`, `torch==2.12.1+cu130`, and `torchao==0.17.0`.

The benchmark uses `scripts/profile_lora_memory.py` with synthetic one-hot
sequence inputs, LoRA rank 8/alpha 16 on `q_proj,v_proj`, 134 ATAC output
tracks, sequence length 131072, resolutions `1,128`, one warmup step, and three
timed optimizer steps. This isolates model memory/compute from BigWig I/O.

## Configurations

| Config | CLI dtype | Low-precision conversion | Trainable params / AdamW |
| --- | --- | --- | --- |
| Default | `bfloat16` | none | fp32 params, fp32 AdamW state |
| Low VRAM | `nvfp8` | torchao Float8Linear, tensorwise recipe | bf16 params, bf16 AdamW moments; fp32 step tensors |
| Super Low VRAM | `nvfp4` | torchao NVFP4 QAT fake-quantized Linear, gradient checkpointing, microbatching | bf16 params, bf16 AdamW moments; fp32 step tensors |

## Results

| Config | Microbatch x accum | Converted / skipped linears | Mean step (s) | Effective samples/s | bp/s | Peak allocated GiB | Peak reserved GiB | `nvidia-smi` max GiB | GPU util mean / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 8 x 1 | 0 / 0 | 1.301 | 6.148 | 805,779 | 77.98 | 85.82 | 86.51 | 97.5% / 100% |
| Default + checkpointing | 8 x 1 | 0 / 0 | 1.631 | 4.904 | 642,712 | 49.97 | 56.64 | 57.33 | 96.0% / 100% |
| Low VRAM (`nvfp8`) | 8 x 1 | 91 / 63 | 0.986 | 8.116 | 1,063,750 | 51.39 | 58.41 | 59.10 | 95.7% / 100% |
| Low VRAM (`nvfp8`) + checkpointing | 8 x 1 | 91 / 63 | 1.228 | 6.513 | 853,715 | 31.80 | 39.13 | 39.82 | 94.6% / 100% |
| Super Low VRAM (`nvfp4`) | 8 x 1 | 109 / 45 | 1.075 | 7.441 | 975,353 | 51.25 | 58.08 | 58.77 | 97.0% / 100% |
| Super Low VRAM (`nvfp4`) + checkpointing | 8 x 1 | 109 / 45 | 1.395 | 5.736 | 751,784 | 31.80 | 39.12 | 39.81 | 93.5% / 100% |
| Super Low VRAM (`nvfp4`) + checkpointing + microbatching | 1 x 8 | 109 / 45 | 2.903 | 2.756 | 361,231 | 5.11 | 7.54 | 8.24 | 59.8% / 90% |

At the same microbatch size, both low-precision modes cut CUDA reserved memory
by about 32% relative to the default synthetic LoRA profile. This is still
activation dominated: NVFP4 and NVFP8 are nearly identical at microbatch 8.

The real super-low-VRAM win comes from stacking NVFP4 QAT with gradient
checkpointing and microbatching. With effective batch size 8 (`batch_size=1`,
`gradient_accumulation_steps=8`), peak reserved memory drops from 85.82 GiB to
7.54 GiB. The tradeoff is throughput: 2.756 effective samples/s versus 6.148
for default and 8.116 for the fastest NVFP8 profile.

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
  --gradient-checkpointing \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-tracks 134 \
  --warmup-steps 1 \
  --timed-steps 3
```

For the human brain development launcher, the equivalent super-low-VRAM knobs
are:

```bash
python scripts/run_humanbraindev_lora_finetune.py \
  --dtype nvfp4 \
  --fp4-mode qat \
  --gradient-checkpointing \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --pretrained-weights ../mpragent/outputs/models/alphagenome/model_all_folds.safetensors
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
- Unsloth was checked with `uv pip install --dry-run unsloth` and was not
  installed into `~/venv/torch`: it would downgrade this environment from
  `torch==2.12.1` to `torch==2.10.0`, replace CUDA bindings/Triton, and install
  an HF/PEFT training stack. Its public integration path is for Hugging Face
  Transformers/Sentence Transformers models, not this custom AlphaGenome
  `nn.Module`, so using it here would require a separate model wrapper and would
  invalidate the current torchao/Blackwell measurements.
