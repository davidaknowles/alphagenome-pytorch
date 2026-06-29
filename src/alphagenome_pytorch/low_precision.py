"""Low-precision training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from packaging.version import Version


@dataclass(frozen=True)
class Float8ConversionStats:
    """Summary of a torchao float8 conversion pass."""

    backend: str
    recipe: str
    converted_linears: int
    skipped_linears: int
    min_feature_multiple: int
    skipped_name_patterns: tuple[str, ...]


@dataclass(frozen=True)
class Float4ConversionStats:
    """Summary of a torchao NVFP4 conversion pass."""

    backend: str
    mode: str
    converted_linears: int
    skipped_linears: int
    min_feature_multiple: int
    skipped_name_patterns: tuple[str, ...]


def _matches_any_pattern(name: str, patterns: Iterable[str]) -> bool:
    return any(pattern and pattern in name for pattern in patterns)


class _ContiguousInputWrapper(nn.Module):
    """Wrap prototype torchao modules that require contiguous inputs."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        if x.dim() <= 2:
            return self.module(x)
        leading_shape = x.shape[:-1]
        y = self.module(x.view(-1, x.shape[-1]))
        return y.view(*leading_shape, y.shape[-1])


def _wrap_modules_by_class_name(model: nn.Module, class_name: str) -> int:
    named_modules = dict(model.named_modules())
    wrapped = 0
    for fqn, module in list(named_modules.items()):
        if fqn == "" or module.__class__.__name__ != class_name:
            continue
        child_name = fqn.split(".")[-1]
        parent_fqn = fqn.removesuffix(child_name).removesuffix(".")
        parent = named_modules[parent_fqn]
        setattr(parent, child_name, _ContiguousInputWrapper(module))
        wrapped += 1
    return wrapped


def _wrap_linears_by_weight_class_name(model: nn.Module, class_name: str) -> int:
    named_modules = dict(model.named_modules())
    wrapped = 0
    for fqn, module in list(named_modules.items()):
        if fqn == "" or not isinstance(module, nn.Linear):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.__class__.__name__ != class_name:
            continue
        child_name = fqn.split(".")[-1]
        parent_fqn = fqn.removesuffix(child_name).removesuffix(".")
        parent = named_modules[parent_fqn]
        setattr(parent, child_name, _ContiguousInputWrapper(module))
        wrapped += 1
    return wrapped


def convert_linears_to_float8_training(
    model: nn.Module,
    *,
    recipe: str = "tensorwise",
    min_feature_multiple: int = 16,
    skip_name_patterns: Iterable[str] = (
        "heads",
        "original_layer",
        "lora_",
        "locon_",
        "ia3",
        "adapter",
    ),
) -> Float8ConversionStats:
    """Convert eligible ``nn.Linear`` modules to torchao Float8Linear.

    The filter follows torchao's published guidance: only linear layers with
    input and output dimensions divisible by 16 are eligible. Finetuning heads
    and adapter internals are skipped so the trainable low-rank path remains
    easy to checkpoint and merge.
    """
    if recipe not in {"tensorwise", "rowwise", "rowwise_with_gw_hp"}:
        raise ValueError(
            "Unsupported float8 recipe "
            f"{recipe!r}; expected tensorwise, rowwise, or rowwise_with_gw_hp"
        )
    if min_feature_multiple < 1:
        raise ValueError("min_feature_multiple must be >= 1")

    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ImportError as exc:
        raise RuntimeError(
            "torchao is required for --dtype nvfp8. Install the optional low "
            "precision dependencies with `uv pip install -e '.[lowprec]'`."
        ) from exc

    patterns = tuple(skip_name_patterns)
    decisions: dict[str, bool] = {}

    def module_filter_fn(mod: nn.Module, fqn: str) -> bool:
        eligible = (
            isinstance(mod, nn.Linear)
            and mod.in_features % min_feature_multiple == 0
            and mod.out_features % min_feature_multiple == 0
            and not _matches_any_pattern(fqn, patterns)
        )
        if isinstance(mod, nn.Linear):
            decisions[fqn] = eligible
        return eligible

    config = Float8LinearConfig.from_recipe_name(recipe)
    convert_to_float8_training(
        model,
        config=config,
        module_filter_fn=module_filter_fn,
    )

    converted = sum(1 for selected in decisions.values() if selected)
    skipped = sum(1 for selected in decisions.values() if not selected)
    return Float8ConversionStats(
        backend="torchao",
        recipe=recipe,
        converted_linears=converted,
        skipped_linears=skipped,
        min_feature_multiple=min_feature_multiple,
        skipped_name_patterns=patterns,
    )


def convert_linears_to_nvfp4_weight_only(
    model: nn.Module,
    *,
    min_feature_multiple: int = 16,
    skip_name_patterns: Iterable[str] = ("heads", "lora_", "locon_", "ia3", "adapter"),
    use_dynamic_per_tensor_scale: bool = True,
) -> Float4ConversionStats:
    """Convert eligible frozen ``nn.Linear`` weights to torchao NVFP4 tensors.

    This path is intended for LoRA-style finetuning where the base model is
    frozen and adapters/heads remain trainable in bf16/fp16. The filter only
    selects non-trainable Linear layers with dimensions compatible with NVFP4's
    block size, and skips adapter/head internals by default.
    """
    if min_feature_multiple < 1:
        raise ValueError("min_feature_multiple must be >= 1")

    try:
        from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
        from torchao.quantization import quantize_
    except ImportError as exc:
        raise RuntimeError(
            "torchao is required for --dtype nvfp4. Install the optional low "
            "precision dependencies with `uv pip install -e '.[lowprec]'`."
        ) from exc

    patterns = tuple(skip_name_patterns)
    decisions: dict[str, bool] = {}

    def module_filter_fn(mod: nn.Module, fqn: str) -> bool:
        eligible = (
            isinstance(mod, nn.Linear)
            and mod.in_features % min_feature_multiple == 0
            and mod.out_features % min_feature_multiple == 0
            and not mod.weight.requires_grad
            and not _matches_any_pattern(fqn, patterns)
        )
        if isinstance(mod, nn.Linear):
            decisions[fqn] = eligible
        return eligible

    quantize_(
        model,
        NVFP4WeightOnlyConfig(
            use_dynamic_per_tensor_scale=use_dynamic_per_tensor_scale,
        ),
        filter_fn=module_filter_fn,
    )
    _wrap_linears_by_weight_class_name(model, "NVFP4Tensor")

    converted = sum(1 for selected in decisions.values() if selected)
    skipped = sum(1 for selected in decisions.values() if not selected)
    return Float4ConversionStats(
        backend="torchao",
        mode="weight_only",
        converted_linears=converted,
        skipped_linears=skipped,
        min_feature_multiple=min_feature_multiple,
        skipped_name_patterns=patterns,
    )


def convert_linears_to_nvfp4_qat_training(
    model: nn.Module,
    *,
    min_feature_multiple: int = 16,
    skip_name_patterns: Iterable[str] = ("heads", "lora_", "locon_", "ia3", "adapter"),
    use_triton_kernel: bool = False,
) -> Float4ConversionStats:
    """Convert eligible ``nn.Linear`` modules to torchao NVFP4 QAT linears.

    Torchao's NVFP4 weight-only tensor is a compact storage format, but it does
    not currently support every operator pattern used by this model during
    training. The QAT path is the training-oriented API: forward pass numerics
    follow NVFP4 quantization while gradients flow through fake-quantized values.
    """
    if min_feature_multiple < 1:
        raise ValueError("min_feature_multiple must be >= 1")

    try:
        import torchao
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
        from torchao.quantization import quantize_
        from torchao.quantization.qat import QATConfig
    except ImportError as exc:
        raise RuntimeError(
            "torchao is required for --dtype nvfp4. Install the optional low "
            "precision dependencies with `uv pip install -e '.[lowprec]'`."
        ) from exc
    torchao_version = Version(str(getattr(torchao, "__version__", "0")).split("+", 1)[0])
    torch_version = Version(str(torch.__version__).split("+", 1)[0])
    if torchao_version < Version("0.18.0") and torch_version < Version("2.11.0"):
        raise RuntimeError(
            "torchao NVFP4 QAT currently fails on CUDA with torchao<0.18 and "
            "torch<2.11 (`Float4_e2m1fn_x2` cannot be converted to a CUDA dtype). "
            "Use --fp4-mode weight-only for frozen-base LoRA/LoCon runs, or use a "
            "Torch/TorchAO stack where NVFP4 QAT is supported."
        )

    patterns = tuple(skip_name_patterns)
    decisions: dict[str, bool] = {}

    def module_filter_fn(mod: nn.Module, fqn: str) -> bool:
        eligible = (
            isinstance(mod, nn.Linear)
            and mod.in_features % min_feature_multiple == 0
            and mod.out_features % min_feature_multiple == 0
            and not _matches_any_pattern(fqn, patterns)
        )
        if isinstance(mod, nn.Linear):
            decisions[fqn] = eligible
        return eligible

    base_config = NVFP4DynamicActivationNVFP4WeightConfig(
        use_triton_kernel=use_triton_kernel,
    )
    quantize_(
        model,
        QATConfig(base_config, step="prepare"),
        filter_fn=module_filter_fn,
    )
    _wrap_modules_by_class_name(model, "NVFP4FakeQuantizedLinear")

    converted = sum(1 for selected in decisions.values() if selected)
    skipped = sum(1 for selected in decisions.values() if not selected)
    return Float4ConversionStats(
        backend="torchao",
        mode="qat_prepare",
        converted_linears=converted,
        skipped_linears=skipped,
        min_feature_multiple=min_feature_multiple,
        skipped_name_patterns=patterns,
    )
