"""Tests for metrics module."""

import pytest
import torch


def test_pearson_r():
    """Test basic Pearson R computation."""
    from alphagenome_pytorch.metrics import pearson_r

    # Perfect correlation
    x = torch.randn(100)
    r = pearson_r(x, x)
    assert abs(r.item() - 1.0) < 1e-5, f"Expected 1.0, got {r.item()}"

    # Anti-correlation
    r_neg = pearson_r(x, -x)
    assert abs(r_neg.item() + 1.0) < 1e-5, f"Expected -1.0, got {r_neg.item()}"

    # No correlation (random)
    torch.manual_seed(42)
    a = torch.randn(10000)
    b = torch.randn(10000)
    r_rand = pearson_r(a, b)
    assert abs(r_rand.item()) < 0.1, f"Expected ~0, got {r_rand.item()}"


def test_profile_pearson_r():
    """Test profile Pearson R (correlation over positions)."""
    from alphagenome_pytorch.metrics import profile_pearson_r

    batch, seq_len, tracks = 4, 1024, 2
    pred = torch.randn(batch, seq_len, tracks)

    # Perfect correlation
    profile_r = profile_pearson_r(pred, pred)
    assert profile_r.shape == (batch, tracks)
    assert torch.allclose(profile_r, torch.ones_like(profile_r), atol=1e-5)

    # High correlation with noise
    noisy = pred + 0.1 * torch.randn_like(pred)
    profile_r_noisy = profile_pearson_r(pred, noisy)
    assert profile_r_noisy.mean() > 0.9


def test_count_pearson_r():
    """Test count Pearson R (correlation of total counts across regions)."""
    from alphagenome_pytorch.metrics import count_pearson_r

    n_regions, seq_len, tracks = 32, 1024, 2
    pred = torch.randn(n_regions, seq_len, tracks)

    # Perfect correlation
    count_r = count_pearson_r(pred, pred)
    assert count_r.shape == (tracks,), "Should return one value per track"
    assert torch.allclose(count_r, torch.ones_like(count_r), atol=1e-5)

    # High correlation with noise
    noisy = pred + 0.1 * torch.randn_like(pred)
    count_r_noisy = count_pearson_r(pred, noisy)
    assert count_r_noisy.mean() > 0.9

    # Verify it's correlating counts across regions
    # If we scale all positions in a region by the same factor,
    # the profile correlation stays the same but count correlation reflects it
    from alphagenome_pytorch.metrics import profile_pearson_r

    # Scale each region by a different factor
    scales = torch.rand(n_regions, 1, 1) * 2 + 0.5  # Random scales 0.5-2.5
    scaled_pred = pred * scales

    # Count correlation should reflect the scaling relationship
    count_r_scaled = count_pearson_r(scaled_pred, pred)
    # Should still be reasonably correlated since we're scaling consistently per region
    assert count_r_scaled.mean() > 0.5


def test_bin_pearson_r():
    """Test bin Pearson R (correlation over region/bin observations)."""
    from alphagenome_pytorch.metrics import bin_pearson_r

    n_regions, n_bins, tracks = 8, 16, 3
    pred = torch.randn(n_regions, n_bins, tracks)

    bin_r = bin_pearson_r(pred, pred)
    assert bin_r.shape == (tracks,), "Should return one value per track"
    assert torch.allclose(bin_r, torch.ones_like(bin_r), atol=1e-5)

    noisy = pred + 0.1 * torch.randn_like(pred)
    bin_r_noisy = bin_pearson_r(pred, noisy)
    assert bin_r_noisy.mean() > 0.9


def test_differential_pearson_r_double_centering():
    """Differential Pearson should ignore observation and track offsets."""
    from alphagenome_pytorch.metrics import differential_pearson_r

    n_regions, n_bins, tracks = 4, 5, 3
    true = torch.randn(n_regions, n_bins, tracks)
    observation_offsets = torch.randn(n_regions, n_bins, 1) * 10.0
    track_offsets = torch.randn(1, 1, tracks) * 5.0
    pred = true + observation_offsets + track_offsets

    diff_r = differential_pearson_r(pred, true)
    assert abs(diff_r.item() - 1.0) < 1e-5


def test_double_centered_r2():
    """Double-centered R2 removes track and cell-type offsets."""
    from alphagenome_pytorch.metrics import double_centered_r2

    true = torch.randn(4, 5, 3)
    observation_offsets = torch.randn(4, 5, 1) * 10.0
    track_offsets = torch.randn(1, 1, 3) * 5.0
    pred = true + observation_offsets + track_offsets

    assert double_centered_r2(pred, true).item() == pytest.approx(1.0, abs=1e-5)
    assert double_centered_r2(torch.zeros_like(true), true).item() == pytest.approx(0.0)


def test_double_center():
    """Double centering should remove row and column means."""
    from alphagenome_pytorch.metrics import double_center

    x = torch.randn(12, 4)
    centered = double_center(x)
    assert torch.allclose(centered.mean(dim=0), torch.zeros(4), atol=1e-6)
    assert torch.allclose(centered.mean(dim=1), torch.zeros(12), atol=1e-6)


def test_compute_metrics():
    """Test compute_metrics returns all expected keys."""
    from alphagenome_pytorch.metrics import compute_metrics

    batch, seq_len, tracks = 8, 1024, 2
    pred = torch.randn(batch, seq_len, tracks)
    true = pred + 0.1 * torch.randn_like(pred)

    # Without track names
    metrics = compute_metrics(pred, true)
    assert "profile_pearson_r" in metrics
    assert "bin_pearson_r" in metrics
    assert "differential_pearson_r" in metrics
    assert "double_centered_r2" in metrics
    assert metrics["profile_pearson_r"] > 0.9
    assert metrics["bin_pearson_r"] > 0.9
    assert metrics["differential_pearson_r"] > 0.9
    assert metrics["double_centered_r2"] > 0.9

    # With track names
    track_names = ["track_a", "track_b"]
    metrics = compute_metrics(pred, true, track_names=track_names)
    assert "profile_pearson_r_track_a" in metrics
    assert "profile_pearson_r_track_b" in metrics
    assert "bin_pearson_r_track_a" in metrics
    assert "bin_pearson_r_track_b" in metrics


def test_pearson_r_vs_scipy():
    """Test pearson_r matches scipy.stats.pearsonr reference."""
    from alphagenome_pytorch.metrics import pearson_r

    scipy_stats = pytest.importorskip("scipy.stats")

    torch.manual_seed(42)
    x = torch.randn(200)
    y = x * 0.8 + torch.randn(200) * 0.2  # Correlated

    our_r = pearson_r(x, y).item()
    scipy_r, _ = scipy_stats.pearsonr(x.numpy(), y.numpy())

    assert abs(our_r - scipy_r) < 1e-5, (
        f"pearson_r={our_r:.6f}, scipy={scipy_r:.6f}"
    )


def test_pearson_r_constant_input():
    """Constant input (zero variance) should return NaN or 0, not crash."""
    from alphagenome_pytorch.metrics import pearson_r
    import math

    torch.manual_seed(42)

    x = torch.ones(100)
    y = torch.randn(100)

    r = pearson_r(x, y)
    # Result should be NaN (0/0) or 0, but not raise
    assert math.isnan(r.item()) or abs(r.item()) < 1e-5


def test_pearson_r_negative_values():
    """Pearson R should handle negative values correctly."""
    from alphagenome_pytorch.metrics import pearson_r

    x = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    y = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

    r = pearson_r(x, y)
    assert abs(r.item() - 1.0) < 1e-5

    # Anti-correlation with negative values
    r_anti = pearson_r(x, -y)
    assert abs(r_anti.item() + 1.0) < 1e-5


@pytest.mark.unit
class TestGoldenValues:
    """Golden value tests: verify exact outputs for known inputs.

    These catch silent numerical changes that property-based tests miss.
    If a test here fails, the metric's numerical behavior has changed.
    """

    def test_pearson_r_golden(self):
        """Pearson R golden values (per-track)."""
        from alphagenome_pytorch.metrics import pearson_r

        pred = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]])
        true = torch.tensor([[[1.1, 1.9, 3.2], [3.8, 5.1, 5.8],
                              [7.2, 7.8, 9.1], [9.9, 11.2, 11.7]]])

        r = pearson_r(pred, true)  # shape: (1, 4)

        expected = torch.tensor([[0.9906836152, 0.9853293300,
                                  0.9781175256, 0.9686195850]])
        torch.testing.assert_close(r, expected, atol=1e-6, rtol=1e-6)
