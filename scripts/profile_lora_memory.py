#!/usr/bin/env python
"""Measure LoRA fine-tuning VRAM and throughput with synthetic data."""

from __future__ import annotations

import argparse
import time

import torch

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy
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
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--timed-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _make_onehot(batch_size: int, sequence_length: int, device: torch.device) -> torch.Tensor:
    bases = torch.randint(0, 4, (batch_size, sequence_length), device=device)
    return torch.nn.functional.one_hot(bases, num_classes=4).float()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    resolutions = tuple(int(x) for x in args.resolutions.split(",") if x.strip())
    lora_targets = [x.strip() for x in args.lora_targets.split(",") if x.strip()]
    dtype_policy = (
        DtypePolicy.full_float32()
        if args.dtype == "float32"
        else DtypePolicy.mixed_precision()
    )

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
    ).to(device)
    head = model.heads["atac"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    param_counts = count_trainable_params(model)

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Resolutions: {resolutions}")
    print(f"LoRA rank/alpha: {args.lora_rank}/{args.lora_alpha}")
    print(f"LoRA targets: {lora_targets}")
    print(f"Dtype policy: {dtype_policy}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"Trainable parameters: {param_counts}")

    organism_idx = torch.zeros(args.batch_size, dtype=torch.long, device=device)
    use_amp = not args.no_amp

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        sequences = _make_onehot(args.batch_size, args.sequence_length, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
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
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    for _ in range(args.warmup_steps):
        loss = step()
    torch.cuda.synchronize(device)

    timings: list[float] = []
    for _ in range(args.timed_steps):
        start = time.perf_counter()
        loss = step()
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)

    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    mean_step = sum(timings) / len(timings)
    samples_per_second = args.batch_size / mean_step
    tokens_per_second = args.batch_size * args.sequence_length / mean_step

    print("Results:")
    print(f"  loss: {loss:.6f}")
    print(f"  mean step time: {mean_step:.3f} s")
    print(f"  samples/sec: {samples_per_second:.3f}")
    print(f"  bp/sec: {tokens_per_second:,.0f}")
    print(f"  peak allocated VRAM: {peak_allocated:.2f} GiB")
    print(f"  peak reserved VRAM: {peak_reserved:.2f} GiB")


if __name__ == "__main__":
    main()
