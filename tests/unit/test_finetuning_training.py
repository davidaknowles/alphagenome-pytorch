"""
Unit tests for fine-tuning training utilities.

Tests the training module functions including collate, loss computation,
modality configuration, and utility functions.
"""

import pytest
import torch
from torch import Tensor

from alphagenome_pytorch.extensions.finetuning.training import (
    create_lr_scheduler,
    compute_finetuning_loss,
    collate_genomic,
    double_centered_correlation_loss,
    MODALITY_CONFIGS,
    ModalityConfig,
    validation_loss_improved,
    validation_metric_improved,
)


def test_validation_loss_improved_respects_min_delta_and_finite_values():
    assert validation_loss_improved(0.8, 1.0, min_delta=0.1)
    assert not validation_loss_improved(0.95, 1.0, min_delta=0.1)
    assert not validation_loss_improved(float("nan"), 1.0)
    assert validation_loss_improved(1.0, float("inf"))
    with pytest.raises(ValueError, match="nonnegative"):
        validation_loss_improved(0.8, 1.0, min_delta=-0.1)


def test_validation_metric_improved_respects_min_delta_and_finite_values():
    assert validation_metric_improved(0.2, 0.0, min_delta=0.1)
    assert not validation_metric_improved(0.05, 0.0, min_delta=0.1)
    assert not validation_metric_improved(float("nan"), 0.0)
    assert validation_metric_improved(-1.0, float("-inf"))
    with pytest.raises(ValueError, match="nonnegative"):
        validation_metric_improved(0.2, 0.0, min_delta=-0.1)


def test_double_centered_correlation_loss_ignores_additive_means():
    target = torch.tensor(
        [[[1.0, 3.0], [2.0, 1.0]], [[4.0, 2.0], [1.0, 5.0]]]
    )
    position_offset = torch.tensor([[[10.0], [-2.0]], [[3.0], [7.0]]])
    track_offset = torch.tensor([[[4.0, -6.0]]])
    prediction = target + position_offset + track_offset
    assert double_centered_correlation_loss(prediction, target).item() == pytest.approx(0.0)


def test_double_centered_correlation_loss_is_scale_invariant():
    target = torch.tensor([[[1.0, 3.0], [2.0, 1.0], [4.0, 2.0]]])
    assert double_centered_correlation_loss(target, target).item() == pytest.approx(0.0)
    assert double_centered_correlation_loss(2 * target, target).item() == pytest.approx(0.0)
    assert double_centered_correlation_loss(-target, target).item() == pytest.approx(2.0)
    with pytest.raises(ValueError, match="shapes differ"):
        double_centered_correlation_loss(target[..., :1], target)


def test_double_centered_correlation_loss_zero_target_has_finite_zero_gradient():
    prediction = torch.randn(2, 8, 3, requires_grad=True)
    loss = double_centered_correlation_loss(prediction, torch.zeros_like(prediction))
    loss.backward()
    assert loss.item() == 0.0
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum().item() == 0.0

@pytest.mark.unit
class TestCollateGenomic:
    """Tests for collate_genomic function."""

    def test_collate_rnaseq_format(self):
        """Test collate with RNA-seq dict format."""
        # Simulate RNA-seq dataset output: (seq, {res: targets})
        batch = [
            (torch.randn(256, 4), {128: torch.randn(256, 3)}),
            (torch.randn(256, 4), {128: torch.randn(256, 3)}),
        ]

        sequences, targets_dict = collate_genomic(batch)

        assert sequences.shape == (2, 256, 4)
        assert 128 in targets_dict
        assert targets_dict[128].shape == (2, 256, 3)

    def test_collate_atac_format(self):
        """Test collate with ATAC tensor format."""
        # Simulate ATAC dataset output: (seq, {res: targets})
        batch = [
            (torch.randn(256, 4), {128: torch.randn(256, 5)}),
            (torch.randn(256, 4), {128: torch.randn(256, 5)}),
        ]

        sequences, targets_dict = collate_genomic(batch)

        assert sequences.shape == (2, 256, 4)
        assert 128 in targets_dict
        assert targets_dict[128].shape == (2, 256, 5)


@pytest.mark.unit
class TestCreateLrScheduler:
    """Tests for learning rate scheduler creation."""

    def test_scheduler_warmup(self):
        """Test scheduler produces warmup behavior."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        scheduler = create_lr_scheduler(optimizer, warmup_steps=100, total_steps=1000)

        # At step 0, LR should be 0
        assert scheduler.get_last_lr()[0] == 0.0

        # Step through warmup
        for _ in range(50):
            scheduler.step()

        # At step 50 (halfway through warmup), LR should be ~0.5
        assert 0.4e-3 < scheduler.get_last_lr()[0] < 0.6e-3

    def test_scheduler_decay(self):
        """Test scheduler produces cosine decay after warmup."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        scheduler = create_lr_scheduler(optimizer, warmup_steps=100, total_steps=1000)

        # Complete warmup
        for _ in range(100):
            scheduler.step()

        # At end of warmup, LR should be ~1.0
        lr_at_warmup_end = scheduler.get_last_lr()[0]
        assert 0.9e-3 < lr_at_warmup_end < 1.1e-3

        # Continue stepping
        for _ in range(450):  # step 100-550
            scheduler.step()

        # At halfway through decay, LR should be ~0.5
        lr_at_mid = scheduler.get_last_lr()[0]
        assert 0.4e-3 < lr_at_mid < 0.6e-3


@pytest.mark.unit
class TestComputeFinetuningLoss:
    """Tests for compute_finetuning_loss function."""

    def test_loss_single_resolution(self):
        """Test loss computation with single resolution (NLC format)."""
        # NLC format: (B, S, C) = (2, 5, 256) where S=5 seq positions, C=256 tracks
        predictions = {128: torch.randn(2, 5, 256).abs()}
        targets = {128: torch.randn(2, 5, 256).abs()}
        resolution_weights = {128: 1.0}
        device = torch.device("cpu")

        loss, loss_dict = compute_finetuning_loss(
            predictions=predictions,
            targets=targets,
            resolution_weights=resolution_weights,
            positional_weight=5.0,
            device=device,
            channels_last=True,
        )

        assert loss.ndim == 0  # Scalar
        assert "loss" in loss_dict
        assert "loss_128bp" in loss_dict

    def test_loss_dual_resolution(self):
        """Test loss computation with dual resolution (NLC format)."""
        predictions = {
            1: torch.randn(2, 3, 256).abs(),
            128: torch.randn(2, 3, 256).abs(),
        }
        targets = {
            1: torch.randn(2, 3, 256).abs(),
            128: torch.randn(2, 3, 256).abs(),
        }
        resolution_weights = {1: 1.0, 128: 1.0}
        device = torch.device("cpu")

        loss, loss_dict = compute_finetuning_loss(
            predictions=predictions,
            targets=targets,
            resolution_weights=resolution_weights,
            positional_weight=5.0,
            device=device,
            channels_last=True,
        )

        assert loss.ndim == 0  # Scalar
        assert "loss" in loss_dict
        assert "loss_1bp" in loss_dict
        assert "loss_128bp" in loss_dict

    def test_loss_ncl_format(self):
        """Test loss computation with NCL format (B, C, S)."""
        # NCL format: (B, C, S) = (2, 256, 5)
        predictions = {128: torch.randn(2, 256, 5).abs()}
        targets = {128: torch.randn(2, 256, 5).abs()}
        resolution_weights = {128: 1.0}
        device = torch.device("cpu")

        loss, loss_dict = compute_finetuning_loss(
            predictions=predictions,
            targets=targets,
            resolution_weights=resolution_weights,
            positional_weight=5.0,
            device=device,
            channels_last=False,
        )

        assert loss.ndim == 0  # Scalar
        assert "loss" in loss_dict
        assert "loss_128bp" in loss_dict

        # Verify it gives same result as transposed NLC
        predictions_nlc = {128: predictions[128].transpose(1, 2)}
        targets_nlc = {128: targets[128].transpose(1, 2)}
        loss_nlc, _ = compute_finetuning_loss(
            predictions=predictions_nlc,
            targets=targets_nlc,
            resolution_weights=resolution_weights,
            positional_weight=5.0,
            device=device,
            channels_last=True,
        )
        assert torch.isclose(loss, loss_nlc)

    def test_loss_weighted_resolutions(self):
        """Test loss computation respects resolution weights."""
        # NLC format: (B, S, C)
        predictions = {
            1: torch.randn(2, 3, 256).abs(),
            128: torch.randn(2, 3, 256).abs(),
        }
        targets = {
            1: torch.randn(2, 3, 256).abs(),
            128: torch.randn(2, 3, 256).abs(),
        }
        device = torch.device("cpu")

        # Equal weights
        loss_equal, _ = compute_finetuning_loss(
            predictions=predictions,
            targets=targets,
            resolution_weights={1: 1.0, 128: 1.0},
            positional_weight=5.0,
            device=device,
        )

        # Zero weight on 1bp
        loss_128_only, loss_dict = compute_finetuning_loss(
            predictions=predictions,
            targets=targets,
            resolution_weights={1: 0.0, 128: 1.0},
            positional_weight=5.0,
            device=device,
        )

        # They should differ (unless by chance 1bp loss is 0)
        # At minimum, the 128-only should be >= just the 128bp component
        assert loss_128_only.item() == pytest.approx(
            loss_dict["loss_128bp"].item(), rel=1e-5
        )

    def test_loss_skips_missing_resolutions(self):
        """Test loss computation skips resolutions not in predictions."""
        # NLC format: (B, S, C)
        predictions = {128: torch.randn(2, 3, 256).abs()}
        targets = {128: torch.randn(2, 3, 256).abs()}
        # Request both resolutions but only 128 is present
        resolution_weights = {1: 1.0, 128: 1.0}
        device = torch.device("cpu")

        loss, loss_dict = compute_finetuning_loss(
            predictions=predictions,
            targets=targets,
            resolution_weights=resolution_weights,
            positional_weight=5.0,
            device=device,
        )

        # Should only have 128bp loss
        assert "loss_128bp" in loss_dict
        assert "loss_1bp" not in loss_dict

    def test_dynamic_multinomial_resolution(self):
        """Test that multinomial_resolution is computed dynamically from seq_len."""
        # NLC format: (B, S, C)
        # Sequence length 256: multinomial_resolution should be 256 // 8 = 32
        predictions_256 = {128: torch.randn(2, 256, 3).abs()}
        targets_256 = {128: torch.randn(2, 256, 3).abs()}

        # Sequence length 64: multinomial_resolution should be 64 // 8 = 8
        predictions_64 = {128: torch.randn(2, 64, 3).abs()}
        targets_64 = {128: torch.randn(2, 64, 3).abs()}

        device = torch.device("cpu")

        # Both should work without errors (different multinomial_resolution internally)
        loss_256, _ = compute_finetuning_loss(
            predictions=predictions_256,
            targets=targets_256,
            resolution_weights={128: 1.0},
            positional_weight=5.0,
            device=device,
        )

        loss_64, _ = compute_finetuning_loss(
            predictions=predictions_64,
            targets=targets_64,
            resolution_weights={128: 1.0},
            positional_weight=5.0,
            device=device,
        )

        assert loss_256.ndim == 0
        assert loss_64.ndim == 0


@pytest.mark.unit
class TestModalityConfigs:
    """Tests for MODALITY_CONFIGS registry."""

    def test_all_modalities_present(self):
        """Test all expected modalities are defined."""
        expected = {'rna_seq', 'atac', 'dnase', 'procap', 'cage', 'chip_tf', 'chip_histone'}
        assert set(MODALITY_CONFIGS.keys()) == expected

    def test_all_configs_are_modality_config(self):
        """Test all configs are ModalityConfig instances."""
        for name, config in MODALITY_CONFIGS.items():
            assert isinstance(config, ModalityConfig), f"{name} is not ModalityConfig"

    def test_chip_modalities_128bp_only(self):
        """Test ChIP modalities only support 128bp resolution."""
        assert MODALITY_CONFIGS['chip_tf'].resolutions == (128,)
        assert MODALITY_CONFIGS['chip_histone'].resolutions == (128,)
        assert MODALITY_CONFIGS['chip_tf'].default_resolution_weights == {128: 1.0}
        assert MODALITY_CONFIGS['chip_histone'].default_resolution_weights == {128: 1.0}

    def test_dual_resolution_modalities(self):
        """Test modalities that support both 1bp and 128bp."""
        dual_res_types = ['rna_seq', 'atac', 'dnase', 'procap', 'cage']
        for mod_name in dual_res_types:
            config = MODALITY_CONFIGS[mod_name]
            assert config.resolutions == (1, 128), f"{mod_name} should have dual resolution"
            assert 1 in config.default_resolution_weights
            assert 128 in config.default_resolution_weights

    def test_all_modalities_have_embedding_dim(self):
        """Test all modalities have embedding_dim set."""
        for name, config in MODALITY_CONFIGS.items():
            assert config.embedding_dim == 3072, f"{name} should have embedding_dim=3072"

    def test_modality_config_name_matches_key(self):
        """Test that config.name matches the dict key."""
        for key, config in MODALITY_CONFIGS.items():
            assert config.name == key, f"Config name '{config.name}' doesn't match key '{key}'"
