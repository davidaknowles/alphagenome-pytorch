# Low-VRAM Inference

This fork owns the reusable low-VRAM AlphaGenome inference implementation.
The core API lives in `src/alphagenome_pytorch/low_precision.py`:

- `LowVramInferenceConfig`
- `apply_low_vram_inference`
- runtime `StandardizedConv1d` to `W_eff` materialization
- Triton int8 Conv1d, Triton no-indices max-pool, fused embed/down0 blocks
- FlexAttention and low-resolution attention bias backends

Benchmarks and Slurm wrappers live in `scripts/`:

- `benchmark_low_vram_parity.py`
- `collate_low_vram_parity.py`
- `submit_low_vram_parity.sh`
- `slurm_low_vram_parity_driver.sbatch`

The pipeline uses the original AlphaGenome safetensors directly. There is no
separate bf16 or `W_eff` checkpoint artifact to distribute.
