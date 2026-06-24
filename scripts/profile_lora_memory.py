#!/usr/bin/env python
"""Measure LoRA fine-tuning VRAM and throughput with synthetic data."""

from __future__ import annotations

import argparse
import subprocess
import threading
import time

import torch

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy
from alphagenome_pytorch.low_precision import (
    convert_linears_to_float8_training,
    convert_linears_to_nvfp4_qat_training,
    convert_linears_to_nvfp4_weight_only,
)
from alphagenome_pytorch.extensions.finetuning.transfer import (
    TransferConfig,
    count_trainable_params,
    prepare_for_transfer,
    remove_all_heads,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile synthetic AlphaGenome LoRA forward/backward memory and speed."
    )
    parser.add_argument("--sequence-length", type=int, default=131072)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-tracks", type=int, default=1)
    parser.add_argument("--resolutions", default="1,128")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-targets", default="q_proj,v_proj")
    parser.add_argument(
        "--dtype",
        choices=(
            "bfloat16",
            "float32",
            "float16",
            "bfloat16-params",
            "float16-params",
            "nvfp8",
            "nvfp4",
        ),
        default="bfloat16",
    )
    parser.add_argument("--fp8-recipe", default="tensorwise")
    parser.add_argument("--fp8-min-feature-multiple", type=int, default=16)
    parser.add_argument(
        "--fp8-skip-name-patterns",
        default="heads,original_layer,lora_,locon_,ia3,adapter",
    )
    parser.add_argument("--fp4-min-feature-multiple", type=int, default=16)
    parser.add_argument("--fp4-mode", choices=("qat", "weight-only"), default="qat")
    parser.add_argument(
        "--fp4-skip-name-patterns",
        default="heads,lora_,locon_,ia3,adapter",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate this many microbatches before each optimizer step.",
    )
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--timed-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    return parser.parse_args()


def _make_onehot(batch_size: int, sequence_length: int, device: torch.device) -> torch.Tensor:
    bases = torch.randint(0, 4, (batch_size, sequence_length), device=device)
    return torch.nn.functional.one_hot(bases, num_classes=4).float()


def _dtype_policy_from_name(dtype: str) -> DtypePolicy:
    if dtype == "float32":
        return DtypePolicy.full_float32()
    if dtype == "float16":
        return DtypePolicy.float16_compute()
    if dtype in {"bfloat16-params", "nvfp8", "nvfp4"}:
        return DtypePolicy.aggressive_bfloat16()
    if dtype == "float16-params":
        return DtypePolicy.aggressive_float16()
    return DtypePolicy.mixed_precision()


class GpuSampler:
    def __init__(self, interval: float):
        self.interval = interval
        self.samples: list[tuple[int, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                raw = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            except (subprocess.SubprocessError, FileNotFoundError):
                time.sleep(self.interval)
                continue
            first = raw.strip().splitlines()[0].split(",")
            if len(first) >= 2:
                try:
                    self.samples.append((int(first[0].strip()), int(first[1].strip())))
                except ValueError:
                    pass
            time.sleep(self.interval)

    def summary(self) -> dict[str, float | None]:
        if not self.samples:
            return {
                "gpu_util_mean": None,
                "gpu_util_max": None,
                "nvidia_smi_memory_max_mib": None,
            }
        utils = [sample[0] for sample in self.samples]
        mems = [sample[1] for sample in self.samples]
        return {
            "gpu_util_mean": sum(utils) / len(utils),
            "gpu_util_max": max(utils),
            "nvidia_smi_memory_max_mib": max(mems),
        }


def _optimizer_state_dtypes(optimizer: torch.optim.Optimizer) -> str:
    dtypes = set()
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                dtypes.add(str(value.dtype).replace("torch.", ""))
    return ",".join(sorted(dtypes)) if dtypes else "none"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    resolutions = tuple(int(x) for x in args.resolutions.split(",") if x.strip())
    lora_targets = [x.strip() for x in args.lora_targets.split(",") if x.strip()]
    dtype_policy = _dtype_policy_from_name(args.dtype)

    model = AlphaGenome(
        gradient_checkpointing=args.gradient_checkpointing,
        dtype_policy=dtype_policy,
    )
    model = remove_all_heads(model)
    model = prepare_for_transfer(
        model,
        TransferConfig(
            mode="lora",
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_targets=lora_targets,
            new_heads={
                "atac": {
                    "modality": "atac",
                    "num_tracks": args.num_tracks,
                    "resolutions": list(resolutions),
                    "num_organisms": 1,
                }
            },
        ),
    )
    if dtype_policy.params_dtype == torch.float32:
        model = model.to(device)
    else:
        model = model.to(device=device, dtype=dtype_policy.params_dtype)

    conversion_stats = None
    if args.dtype == "nvfp8":
        conversion_stats = convert_linears_to_float8_training(
            model,
            recipe=args.fp8_recipe,
            min_feature_multiple=args.fp8_min_feature_multiple,
            skip_name_patterns=[
                p.strip() for p in args.fp8_skip_name_patterns.split(",") if p.strip()
            ],
        )
    elif args.dtype == "nvfp4":
        fp4_skip_name_patterns = [
            p.strip() for p in args.fp4_skip_name_patterns.split(",") if p.strip()
        ]
        if args.fp4_mode == "weight-only":
            conversion_stats = convert_linears_to_nvfp4_weight_only(
                model,
                min_feature_multiple=args.fp4_min_feature_multiple,
                skip_name_patterns=fp4_skip_name_patterns,
            )
        else:
            conversion_stats = convert_linears_to_nvfp4_qat_training(
                model,
                min_feature_multiple=args.fp4_min_feature_multiple,
                skip_name_patterns=fp4_skip_name_patterns,
            )

    head = model.heads["atac"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    param_counts = count_trainable_params(model)

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Resolutions: {resolutions}")
    print(f"LoRA rank/alpha: {args.lora_rank}/{args.lora_alpha}")
    print(f"LoRA targets: {lora_targets}")
    print(f"Dtype policy: {dtype_policy}")
    print(f"Low precision conversion: {conversion_stats}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"Trainable parameters: {param_counts}")

    organism_idx = torch.zeros(args.batch_size, dtype=torch.long, device=device)
    amp_dtype = dtype_policy.compute_dtype
    use_amp = not args.no_amp and amp_dtype != torch.float32

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.gradient_accumulation_steps):
            sequences = _make_onehot(args.batch_size, args.sequence_length, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    sequences,
                    organism_idx,
                    return_embeddings=True,
                    resolutions=resolutions,
                    channels_last=False,
                )
                embeddings = {
                    res: outputs[f"embeddings_{res}bp"]
                    for res in resolutions
                    if f"embeddings_{res}bp" in outputs
                }
                predictions = head(
                    embeddings,
                    organism_idx,
                    return_scaled=True,
                    channels_last=True,
                )
                loss = sum(pred.float().mean() for pred in predictions.values())
                scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            total_loss += float(loss.detach().cpu())
        optimizer.step()
        return total_loss / args.gradient_accumulation_steps

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    for _ in range(args.warmup_steps):
        loss = step()
    torch.cuda.synchronize(device)

    sampler = GpuSampler(args.gpu_sample_interval)
    sampler.start()
    timings: list[float] = []
    try:
        for _ in range(args.timed_steps):
            start = time.perf_counter()
            loss = step()
            torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - start)
    finally:
        sampler.stop()

    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    mean_step = sum(timings) / len(timings)
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    samples_per_second = effective_batch_size / mean_step
    tokens_per_second = effective_batch_size * args.sequence_length / mean_step

    print("Results:")
    print(f"  loss: {loss:.6f}")
    print(f"  mean step time: {mean_step:.3f} s")
    print(f"  samples/sec: {samples_per_second:.3f}")
    print(f"  bp/sec: {tokens_per_second:,.0f}")
    print(f"  peak allocated VRAM: {peak_allocated:.2f} GiB")
    print(f"  peak reserved VRAM: {peak_reserved:.2f} GiB")
    print(f"  optimizer state dtypes: {_optimizer_state_dtypes(optimizer)}")
    for key, value in sampler.summary().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
