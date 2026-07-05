#!/usr/bin/env python
"""Collate OG low-VRAM parity metrics into a Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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

STRATEGY_FEATURES = {
    "default": ("", "", "", "", "", "", "", "", ""),
    "bf16_params": ("x", "", "", "", "", "", "", "", ""),
    "triton_conv": ("x", "x", "x", "", "", "", "", "", ""),
    "flexattention": ("x", "x", "x", "", "", "", "", "x", ""),
    "flex_lowres_bias": ("x", "x", "x", "", "", "", "", "x", "x"),
    "no_intermediates": ("x", "x", "x", "x", "", "", "", "", ""),
    "triton_pool": ("x", "x", "x", "x", "x", "", "", "", ""),
    "fused_embedder": ("x", "x", "x", "x", "x", "x", "", "", ""),
    "fused_down0": ("x", "x", "x", "x", "x", "", "x", "", ""),
    "fused_embedder_down0": ("x", "x", "x", "x", "x", "x", "x", "", ""),
    "all_features": ("x", "x", "x", "x", "x", "x", "x", "x", "x"),
}

STRATEGY_ORDER = tuple(STRATEGY_LABELS)


def _load_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        with metrics_path.open() as handle:
            metrics = json.load(handle)
        strategy = str(metrics["strategy_key"])
        if strategy not in STRATEGY_LABELS:
            continue
        gpu = metrics.get("gpu") or {}
        torch_mem = metrics.get("torch_cuda_memory") or {}
        parity = metrics.get("parity") or {}
        rows.append(
            {
                "strategy_key": strategy,
                "strategy": STRATEGY_LABELS[strategy],
                "features": STRATEGY_FEATURES[strategy],
                "window_size": int(metrics["window_size"]),
                "batch_size": int(metrics["batch_size"]),
                "examples": int(metrics["examples"]),
                "examples_per_sec": metrics.get("examples_per_sec"),
                "peak_gib": (
                    float(gpu["max_mem_mib"]) / 1024.0
                    if gpu.get("max_mem_mib") is not None
                    else None
                ),
                "torch_reserved_gib": (
                    float(torch_mem["max_reserved_mib"]) / 1024.0
                    if torch_mem and torch_mem.get("max_reserved_mib") is not None
                    else None
                ),
                "max_abs": parity.get("max_abs"),
                "rmse": parity.get("rmse"),
                "pearson": parity.get("pearson"),
                "path": metrics_path.parent,
            }
        )
    return rows


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _fmt_sci(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3e}"


def _write_markdown(path: Path, rows: list[dict[str, Any]], root: Path) -> None:
    lines = [
        "# OG AlphaGenome Low-VRAM Parity",
        "",
        f"Run root: `{root}`",
        "",
        "This prediction-only check uses the original AlphaGenome all-folds checkpoint as the reference,",
        "not the merged fine-tuned checkpoint. It evaluates the native `atac` head at 128 bp on",
        "`chr9` windows and does not load targets or compute loss.",
        "",
        "Feature abbreviations: BF16 = bfloat16 parameter/compute policy; Eff. = materialized",
        "effective standardized convolutions; Int8 = Triton int8 weight-only Conv1d; NoInt =",
        "skip unused encoder intermediates for 128 bp output; Pool = Triton max-pool without",
        "indices; FEmb = fused DNA embedder block; FD0 = fused first downsampling block;",
        "Flex = FlexAttention MHA; LRB = low-resolution attention bias.",
        "",
        "The supported low-VRAM pipeline uses the original OG safetensors directly. For",
        "Eff. rows, standardized convolutions are materialized in memory at load time;",
        "there is no separate bf16/W_eff checkpoint artifact to share.",
    ]

    for window_size in sorted({row["window_size"] for row in rows}):
        subset = [row for row in rows if row["window_size"] == window_size]
        subset.sort(key=lambda row: STRATEGY_ORDER.index(row["strategy_key"]))
        batch_size = subset[0]["batch_size"] if subset else "NA"
        lines.extend(
            [
                "",
                f"## {window_size:,} bp windows, batch size {batch_size}",
                "",
                "| Strategy | BF16 | Eff. | Int8 | NoInt | Pool | FEmb | FD0 | Flex | LRB | Peak GiB | Ex./s | max abs | RMSE | Pearson |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in subset:
            features = " | ".join(row["features"])
            lines.append(
                f"| {row['strategy']} | {features} | {_fmt(row['peak_gib'], 2)} | "
                f"{_fmt(row['examples_per_sec'], 3)} | {_fmt_sci(row['max_abs'])} | "
                f"{_fmt_sci(row['rmse'])} | {_fmt(row['pearson'], 6)} |"
            )
    lines.extend(["", "## Metric paths", ""])
    for row in sorted(rows, key=lambda r: (r["window_size"], STRATEGY_ORDER.index(r["strategy_key"]))):
        lines.append(f"- `{row['path']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/low_vram/og_low_vram_parity_20260704.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    rows = _load_rows(root)
    if not rows:
        raise SystemExit(f"No metrics found under {root}")
    _write_markdown(args.output.expanduser().resolve(), rows, root)
    print(f"Wrote {args.output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
