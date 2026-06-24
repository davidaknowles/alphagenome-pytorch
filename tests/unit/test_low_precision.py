"""Tests for optional low-precision helpers."""

import importlib.util

import pytest
import torch.nn as nn

from alphagenome_pytorch.low_precision import (
    convert_linears_to_float8_training,
    convert_linears_to_nvfp4_qat_training,
    convert_linears_to_nvfp4_weight_only,
)


class _TinyLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.large = nn.Linear(32, 32)
        self.heads = nn.ModuleDict({"track": nn.Linear(32, 32)})
        self.bad_shape = nn.Linear(30, 32)

    def forward(self, x):
        return self.heads["track"](self.large(x))


def test_float8_conversion_validates_recipe():
    model = _TinyLinearModel()
    with pytest.raises(ValueError, match="Unsupported float8 recipe"):
        convert_linears_to_float8_training(model, recipe="invalid")


@pytest.mark.skipif(
    importlib.util.find_spec("torchao") is None,
    reason="torchao is an optional low-precision dependency",
)
def test_float8_conversion_filters_ineligible_and_head_linears():
    model = _TinyLinearModel().bfloat16()
    stats = convert_linears_to_float8_training(model, recipe="rowwise")

    assert stats.backend == "torchao"
    assert stats.converted_linears == 1
    assert stats.skipped_linears == 2
    assert model.bad_shape.__class__.__name__ == "Linear"
    assert model.heads["track"].__class__.__name__ == "Linear"


@pytest.mark.skipif(
    importlib.util.find_spec("torchao") is None,
    reason="torchao is an optional low-precision dependency",
)
def test_nvfp4_weight_only_conversion_skips_trainable_and_head_linears():
    model = _TinyLinearModel().bfloat16()
    model.large.weight.requires_grad = False
    model.large.bias.requires_grad = False

    stats = convert_linears_to_nvfp4_weight_only(model)

    assert stats.backend == "torchao"
    assert stats.mode == "weight_only"
    assert stats.converted_linears == 1
    assert stats.skipped_linears == 2
    assert "NVFP4Tensor" in repr(model.large.weight)
    assert model.bad_shape.__class__.__name__ == "Linear"
    assert model.heads["track"].__class__.__name__ == "Linear"


@pytest.mark.skipif(
    importlib.util.find_spec("torchao") is None,
    reason="torchao is an optional low-precision dependency",
)
def test_nvfp4_qat_conversion_wraps_converted_linears():
    model = _TinyLinearModel().bfloat16()

    stats = convert_linears_to_nvfp4_qat_training(model)

    assert stats.backend == "torchao"
    assert stats.mode == "qat_prepare"
    assert stats.converted_linears == 1
    assert stats.skipped_linears == 2
    assert model.large.__class__.__name__ == "_ContiguousInputWrapper"
    assert model.heads["track"].__class__.__name__ == "Linear"
