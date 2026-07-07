#!/usr/bin/env python
"""Find max batch sizes under a VRAM limit for low-VRAM inference strategies."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from benchmark_low_vram_parity import DEFAULT_FASTA, DEFAULT_OG_WEIGHTS, TABLE_STRATEGIES


STRATEGY_ORDER = tuple(TABLE_STRATEGIES)
STRATEGY_LABELS = {
    "default": "Default",
    "bf16_params": "Bfloat16",
    "triton_conv": "Triton Conv1d",
    "flexattention": "FlexAttention",
    "flex_lowres_bias": "Flex low-res bias",
    "no_intermediates": "No intermediates",
    "triton_pool": "Triton pool",
    "fused_embedder": "Fused embedder",
    "fused_down0": "Fused down0",
    "fused_embedder_down0": "Fused embedder+down0",
    "all_features": "All features",
}


@dataclass(frozen=True)
class WindowConfig:
    window_size: int
    candidate_batches: tuple[int, ...]
    final_examples: int


DEFAULT_WINDOWS = {
    131072: WindowConfig(
        window_size=131072,
        candidate_batches=(1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128),
        final_examples=128,
    ),
    1048576: WindowConfig(
        window_size=1048576,
        candidate_batches=(1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32),
        final_examples=32,
    ),
}


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _peak_mib(metrics: dict[str, Any] | None) -> float | None:
    if not metrics:
        return None
    gpu = metrics.get("gpu") or {}
    if gpu.get("max_mem_mib") is None:
        return None
    return float(gpu["max_mem_mib"])


def _read_metrics(out_dir: Path) -> dict[str, Any]:
    with (out_dir / "metrics.json").open() as handle:
        return json.load(handle)


def _run(
    *,
    args: argparse.Namespace,
    strategy: str,
    window_size: int,
    batch_size: int,
    max_windows: int,
    output_dir: Path,
    reference_predictions: Path | None = None,
    write_reference: bool = False,
    skip_parity: bool = False,
) -> dict[str, Any] | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.python),
        str(args.benchmark_script),
        "--strategy",
        strategy,
        "--output-dir",
        str(output_dir),
        "--og-weights",
        str(args.og_weights),
        "--fasta-path",
        str(args.fasta_path),
        "--window-size",
        str(window_size),
        "--batch-size",
        str(batch_size),
        "--chrom",
        args.chrom,
        "--head",
        args.head,
        "--max-windows",
        str(max_windows),
        "--gpu-sample-interval",
        str(args.gpu_sample_interval),
    ]
    if reference_predictions is not None:
        cmd.extend(["--reference-predictions", str(reference_predictions)])
    if write_reference:
        cmd.append("--write-reference")
    if skip_parity:
        cmd.append("--skip-parity")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    started = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, capture_output=True)
    if result.returncode != 0:
        failure = {
            "cmd": cmd,
            "returncode": result.returncode,
            "elapsed_sec": time.time() - started,
            "stdout": result.stdout[-20000:],
            "stderr": result.stderr[-20000:],
        }
        with (output_dir / "failure.json").open("w") as handle:
            json.dump(failure, handle, indent=2)
        print(f"FAILED strategy={strategy} window={window_size} batch={batch_size}", flush=True)
        return None
    return _read_metrics(output_dir)


def _ensure_reference(
    *,
    args: argparse.Namespace,
    window_size: int,
    max_windows: int,
    root: Path,
) -> Path:
    ref_path = root / "references" / f"{args.head}_{args.chrom}_w{window_size}_n{max_windows}_default.npy"
    if ref_path.exists():
        return ref_path
    out_dir = root / "references" / f"default_w{window_size}_n{max_windows}_b1"
    print(f"writing reference window={window_size} examples={max_windows}", flush=True)
    metrics = _run(
        args=args,
        strategy="default",
        window_size=window_size,
        batch_size=1,
        max_windows=max_windows,
        output_dir=out_dir,
        reference_predictions=ref_path,
        write_reference=True,
        skip_parity=False,
    )
    if metrics is None:
        raise RuntimeError(f"Failed to write reference for window={window_size} examples={max_windows}")
    return ref_path


def _search_strategy(
    *,
    args: argparse.Namespace,
    strategy: str,
    config: WindowConfig,
    root: Path,
    limit_mib: float,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    viable: list[dict[str, Any]] = []
    for batch_size in config.candidate_batches:
        out_dir = root / "search" / f"{strategy}_w{config.window_size}_b{batch_size}"
        print(f"search strategy={strategy} window={config.window_size} batch={batch_size}", flush=True)
        metrics = _run(
            args=args,
            strategy=strategy,
            window_size=config.window_size,
            batch_size=batch_size,
            max_windows=batch_size,
            output_dir=out_dir,
            skip_parity=True,
        )
        peak = _peak_mib(metrics)
        status = "failed" if metrics is None else ("over_limit" if peak is not None and peak > limit_mib else "ok")
        record = {
            "batch_size": batch_size,
            "status": status,
            "peak_mib": peak,
            "examples_per_sec": metrics.get("examples_per_sec") if metrics else None,
            "path": str(out_dir),
        }
        records.append(record)
        if status == "ok":
            viable.append(record)
            continue
        break
    return {
        "strategy": strategy,
        "window_size": config.window_size,
        "records": records,
        "viable_batches": [int(row["batch_size"]) for row in viable],
        "best_search": viable[-1] if viable else None,
    }


def _finalize_strategy(
    *,
    args: argparse.Namespace,
    strategy: str,
    config: WindowConfig,
    root: Path,
    search: dict[str, Any],
    limit_mib: float,
) -> dict[str, Any]:
    batches = list(reversed(search["viable_batches"]))
    if not batches:
        return {"strategy": strategy, "window_size": config.window_size, "final": None}

    ref = _ensure_reference(args=args, window_size=config.window_size, max_windows=config.final_examples, root=root)
    attempts = []
    for batch_size in batches:
        out_dir = root / "final" / f"{strategy}_w{config.window_size}_b{batch_size}"
        print(f"final strategy={strategy} window={config.window_size} batch={batch_size}", flush=True)
        metrics = _run(
            args=args,
            strategy=strategy,
            window_size=config.window_size,
            batch_size=batch_size,
            max_windows=config.final_examples,
            output_dir=out_dir,
            reference_predictions=ref,
            write_reference=False,
            skip_parity=False,
        )
        peak = _peak_mib(metrics)
        status = "failed" if metrics is None else ("over_limit" if peak is not None and peak > limit_mib else "ok")
        attempt = {
            "batch_size": batch_size,
            "status": status,
            "peak_mib": peak,
            "examples_per_sec": metrics.get("examples_per_sec") if metrics else None,
            "pearson": (metrics.get("parity") or {}).get("pearson") if metrics else None,
            "max_abs": (metrics.get("parity") or {}).get("max_abs") if metrics else None,
            "path": str(out_dir),
        }
        attempts.append(attempt)
        if status == "ok":
            return {"strategy": strategy, "window_size": config.window_size, "attempts": attempts, "final": attempt}
    return {"strategy": strategy, "window_size": config.window_size, "attempts": attempts, "final": None}


def _write_outputs(root: Path, payload: dict[str, Any]) -> None:
    with (root / "batch_sweep_summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2)

    lines = [
        "# Low-VRAM Batch-Size Sweep",
        "",
        f"Run root: `{root}`",
        "",
        f"VRAM limit: `{payload['vram_limit_gib']:.1f}` GiB.",
        "Search runs are forward-only and use one full batch. Final rows rerun",
        "the largest viable batch with default-reference parity and fixed example counts.",
    ]
    for window_size in sorted(payload["windows"]):
        final_examples = payload["window_configs"][str(window_size)]["final_examples"]
        lines.extend(
            [
                "",
                f"## {int(window_size):,} bp windows",
                "",
                f"Final examples: `{final_examples}`",
                "",
                "| Strategy | Max batch | Search peak GiB | Final peak GiB | Ex./s | Pearson | Path |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        rows = payload["windows"][str(window_size)]
        for strategy in STRATEGY_ORDER:
            if strategy not in rows:
                continue
            search = rows[strategy]["search"].get("best_search")
            final = rows[strategy]["final"].get("final")
            path = final.get("path") if final else None
            lines.append(
                f"| {STRATEGY_LABELS[strategy]} | "
                f"{final['batch_size'] if final else 'NA'} | "
                f"{_fmt((search or {}).get('peak_mib') / 1024 if search and search.get('peak_mib') is not None else None, 2)} | "
                f"{_fmt(final.get('peak_mib') / 1024 if final and final.get('peak_mib') is not None else None, 2)} | "
                f"{_fmt(final.get('examples_per_sec') if final else None, 3)} | "
                f"{_fmt(final.get('pearson') if final else None, 6)} | "
                f"{'`' + path + '`' if path else 'NA'} |"
            )
    (root / "batch_sweep_summary.md").write_text("\n".join(lines) + "\n")


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--benchmark-script", type=Path, default=REPO_ROOT / "scripts" / "benchmark_low_vram_parity.py")
    parser.add_argument("--og-weights", type=Path, default=DEFAULT_OG_WEIGHTS)
    parser.add_argument("--fasta-path", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--chrom", default="chr9")
    parser.add_argument("--head", default="atac")
    parser.add_argument("--vram-limit-gib", type=float, default=48.0)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.25)
    parser.add_argument("--strategies", default=",".join(STRATEGY_ORDER))
    parser.add_argument("--batches-131kb", default=",".join(map(str, DEFAULT_WINDOWS[131072].candidate_batches)))
    parser.add_argument("--batches-1mb", default=",".join(map(str, DEFAULT_WINDOWS[1048576].candidate_batches)))
    parser.add_argument("--final-examples-131kb", type=int, default=DEFAULT_WINDOWS[131072].final_examples)
    parser.add_argument("--final-examples-1mb", type=int, default=DEFAULT_WINDOWS[1048576].final_examples)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_root = REPO_ROOT / "outputs" / "low_vram_batch_sweep" / stamp
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    strategies = tuple(part.strip() for part in args.strategies.split(",") if part.strip())
    unknown = set(strategies) - set(STRATEGY_ORDER)
    if unknown:
        raise ValueError(f"Unknown strategies: {sorted(unknown)}")

    windows = {
        131072: WindowConfig(131072, _parse_ints(args.batches_131kb), args.final_examples_131kb),
        1048576: WindowConfig(1048576, _parse_ints(args.batches_1mb), args.final_examples_1mb),
    }
    limit_mib = args.vram_limit_gib * 1024.0
    payload: dict[str, Any] = {
        "output_root": str(root),
        "vram_limit_gib": args.vram_limit_gib,
        "strategies": strategies,
        "window_configs": {
            str(size): {
                "candidate_batches": config.candidate_batches,
                "final_examples": config.final_examples,
            }
            for size, config in windows.items()
        },
        "windows": {str(size): {} for size in windows},
    }

    for window_size, config in windows.items():
        for strategy in strategies:
            search = _search_strategy(args=args, strategy=strategy, config=config, root=root, limit_mib=limit_mib)
            final = _finalize_strategy(
                args=args,
                strategy=strategy,
                config=config,
                root=root,
                search=search,
                limit_mib=limit_mib,
            )
            payload["windows"][str(window_size)][strategy] = {"search": search, "final": final}
            _write_outputs(root, payload)

    _write_outputs(root, payload)
    print(f"Wrote {root / 'batch_sweep_summary.md'}")


if __name__ == "__main__":
    main()
