"""
Metrics for AlphaGenome PyTorch.

Provides evaluation metrics for genomic predictions.
"""

from typing import Dict, Any, Optional
import torch
from torch import Tensor


def pearson_r(
    pred: Tensor,
    true: Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> Tensor:
    """Compute Pearson correlation coefficient.

    Args:
        pred: Predicted values.
        true: True values.
        dim: Dimension to compute correlation over.
        eps: Small epsilon for numerical stability.

    Returns:
        Pearson correlation coefficient.
    """
    pred = pred.float()
    true = true.float()

    pred_centered = pred - pred.mean(dim=dim, keepdim=True)
    true_centered = true - true.mean(dim=dim, keepdim=True)

    numerator = (pred_centered * true_centered).sum(dim=dim)
    denominator = (
        pred_centered.pow(2).sum(dim=dim).sqrt() *
        true_centered.pow(2).sum(dim=dim).sqrt()
    )

    return numerator / (denominator + eps)


def profile_pearson_r(
    pred: Tensor,
    true: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Compute Pearson R for profile shape within each region.

    For each (region, track) pair, computes correlation between predicted
    and observed values across all genomic positions within that region.
    This measures how well the predicted signal shape matches the true shape.

    Following the AlphaGenome paper:
    "For a given track and a held-out test interval, Pearson r was calculated
    between the vector of predicted values and the vector of observed values
    across all corresponding genomic bins within that interval."

    Args:
        pred: Predictions with shape (N_regions, seq_len, tracks).
        true: Targets with shape (N_regions, seq_len, tracks).
        eps: Small epsilon for numerical stability.

    Returns:
        Per-region, per-track correlation with shape (N_regions, tracks).
        To get mean profile Pearson R: result.mean()
        To get per-track mean: result.mean(dim=0)
    """
    # Correlation over positions (dim=1) for each region and track
    return pearson_r(pred, true, dim=1, eps=eps)


def count_pearson_r(
    pred: Tensor,
    true: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Compute Pearson R for total counts across all regions.

    For each track, computes ONE correlation between predicted and observed
    total counts using all N regions as data points:
        corr([sum(pred_1), ..., sum(pred_N)], [sum(true_1), ..., sum(true_N)])

    This measures whether regions with high observed signal also have
    high predicted signal.

    Args:
        pred: Predictions with shape (N_regions, seq_len, tracks).
        true: Targets with shape (N_regions, seq_len, tracks).
        eps: Small epsilon for numerical stability.

    Returns:
        Per-track correlation with shape (tracks,).
        Each value is a single Pearson R computed across all N regions.
    """
    # Sum over positions to get total counts per region
    pred_counts = pred.sum(dim=1)  # (N_regions, tracks)
    true_counts = true.sum(dim=1)  # (N_regions, tracks)

    # For each track, correlate counts across all regions
    return pearson_r(pred_counts, true_counts, dim=0, eps=eps)


def bin_pearson_r(
    pred: Tensor,
    true: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Compute Pearson R using bins as observations.

    For predictions with shape ``(N_regions, n_bins, tracks)``, this flattens
    regions and bins into one observation axis and computes one correlation per
    track. At 128 bp resolution, this measures whether individual 128 bp bins
    have the right predicted signal without summing over the whole interval.

    Args:
        pred: Predictions with shape (N_regions, n_bins, tracks).
        true: Targets with shape (N_regions, n_bins, tracks).
        eps: Small epsilon for numerical stability.

    Returns:
        Per-track correlation with shape (tracks,).
    """
    pred_bins = pred.float().reshape(-1, pred.shape[-1])
    true_bins = true.float().reshape(-1, true.shape[-1])
    return pearson_r(pred_bins, true_bins, dim=0, eps=eps)


def double_center(
    x: Tensor,
) -> Tensor:
    """Remove observation and track means from a 2D matrix.

    ``x`` is expected to have shape ``(observations, tracks)``. For 128 bp
    differential metrics, observations are flattened ``(region, bin)`` pairs.
    """
    x = x.float()
    return x - x.mean(dim=0, keepdim=True) - x.mean(dim=1, keepdim=True) + x.mean()


def differential_pearson_r(
    pred: Tensor,
    true: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Compute double-centered differential Pearson R.

    For shape ``(N_regions, n_bins, tracks)``, this treats each 128 bp bin in
    each region as an observation, removes each track's mean, removes each
    observation's mean across tracks, then computes one Pearson R over all
    residualized values.

    Args:
        pred: Predictions with shape (N_regions, n_bins, tracks) or
            (observations, tracks).
        true: Targets with the same shape as pred.
        eps: Small epsilon for numerical stability.

    Returns:
        Scalar Pearson R over double-centered prediction and target matrices.
    """
    pred_matrix = pred.float().reshape(-1, pred.shape[-1])
    true_matrix = true.float().reshape(-1, true.shape[-1])

    pred_centered = double_center(pred_matrix)
    true_centered = double_center(true_matrix)

    return pearson_r(
        pred_centered.reshape(-1),
        true_centered.reshape(-1),
        dim=0,
        eps=eps,
    )


def double_centered_r2(
    pred: Tensor,
    true: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Compute variance explained after track and cell-type centering.

    Inputs have shape ``(regions, bins, cell_types)`` or
    ``(observations, cell_types)``. Flattened genomic observations form rows
    and output cell types form columns. Prediction and target matrices are
    independently centered over observations within each track and then over
    cell types within each observation before computing ``1 - SSE / SST``.
    """
    pred_matrix = pred.float().reshape(-1, pred.shape[-1])
    true_matrix = true.float().reshape(-1, true.shape[-1])
    pred_centered = double_center(pred_matrix)
    true_centered = double_center(true_matrix)
    residual_sum_squares = (true_centered - pred_centered).pow(2).sum()
    total_sum_squares = true_centered.pow(2).sum()
    if total_sum_squares <= eps:
        return torch.tensor(float("nan"), device=true.device)
    return 1.0 - residual_sum_squares / total_sum_squares


def compute_metrics(
    pred: Tensor,
    true: Tensor,
    track_names: Optional[list] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute comprehensive metrics for genomic predictions.

    Args:
        pred: Predictions with shape (batch, seq_len, tracks).
        true: Targets with shape (batch, seq_len, tracks).
        track_names: Optional list of track names for labeling.
        eps: Small epsilon for numerical stability.

    Returns:
        Dictionary with metrics:
            - profile_pearson_r: Mean profile correlation (across batch and tracks)
            - profile_pearson_r_per_track: Per-track mean profile correlation
            - bin_pearson_r: Mean bin-level correlation (across tracks)
            - bin_pearson_r_per_track: Per-track bin-level correlation
            - differential_pearson_r: Double-centered differential correlation
            - double_centered_r2: Variance explained after double centering
    """
    results = {}

    # Profile Pearson R: correlation over positions
    profile_r = profile_pearson_r(pred, true, eps=eps)  # (batch, tracks)
    results["profile_pearson_r"] = profile_r.mean().item()

    # Per-track profile Pearson R (averaged over batch)
    profile_r_per_track = profile_r.mean(dim=0)  # (tracks,)

    # Bin Pearson R: correlation over all region/bin observations per track
    bin_r = bin_pearson_r(pred, true, eps=eps)  # (tracks,)
    results["bin_pearson_r"] = bin_r.mean().item()
    results["differential_pearson_r"] = differential_pearson_r(pred, true, eps=eps).item()
    results["double_centered_r2"] = double_centered_r2(pred, true, eps=eps).item()

    # Add per-track metrics if track names provided
    if track_names is not None:
        for i, name in enumerate(track_names):
            results[f"profile_pearson_r_{name}"] = profile_r_per_track[i].item()
            results[f"bin_pearson_r_{name}"] = bin_r[i].item()

    return results


def spearman_r(
    pred: Tensor,
    true: Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> Tensor:
    """Compute Spearman rank correlation coefficient.
    
    Args:
        pred: Predicted values.
        true: True values.
        dim: Dimension to compute correlation over.
        eps: Small epsilon for numerical stability.
        
    Returns:
        Spearman correlation coefficient.
    """
    # Convert to ranks
    def to_ranks(x: Tensor, dim: int) -> Tensor:
        return x.argsort(dim=dim).argsort(dim=dim).float()
    
    pred_ranks = to_ranks(pred, dim)
    true_ranks = to_ranks(true, dim)
    
    return pearson_r(pred_ranks, true_ranks, dim=dim, eps=eps)


class AlphaGenomeMetrics:
    """Compute validation metrics for AlphaGenome.
    
    Computes per-head Pearson correlation and optionally other metrics.
    Extensible via custom metric functions.
    
    Args:
        heads: List of head names to compute metrics for.
        additional_metrics: Dict of name -> callable(pred, true) for extra metrics.
        
    Example:
        >>> metrics = AlphaGenomeMetrics()
        >>> results = metrics(outputs, targets)
        >>> print(results)
        {'atac_pearson_r': 0.85, 'dnase_pearson_r': 0.82, ...}
    """
    
    def __init__(
        self,
        heads: Optional[list] = None,
        additional_metrics: Optional[Dict[str, callable]] = None,
    ):
        self.heads = heads
        self.additional_metrics = additional_metrics or {}
    
    def __call__(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Tensor],
    ) -> Dict[str, float]:
        """Compute metrics for all heads.
        
        Args:
            outputs: Model outputs dict.
            targets: Target values dict.
            
        Returns:
            Dict of metric names to values.
        """
        results = {}
        heads = self.heads or list(outputs.keys())
        
        for head in heads:
            if head not in outputs or head not in targets:
                continue
                
            pred = self._extract_tensor(outputs[head])
            true = self._extract_tensor(targets[head])
            
            if pred is None or true is None:
                continue
            
            # Compute Pearson R (primary metric)
            r = pearson_r(pred.flatten(), true.flatten()).item()
            results[f'{head}_pearson_r'] = r
            
            # Compute additional metrics
            for metric_name, metric_fn in self.additional_metrics.items():
                try:
                    value = metric_fn(pred, true)
                    if isinstance(value, Tensor):
                        value = value.item()
                    results[f'{head}_{metric_name}'] = value
                except Exception:
                    pass  # Skip failed metrics
        
        # Compute average Pearson R across heads
        pearson_values = [v for k, v in results.items() if k.endswith('_pearson_r')]
        if pearson_values:
            results['avg_pearson_r'] = sum(pearson_values) / len(pearson_values)
        
        return results
    
    def _extract_tensor(self, x: Any) -> Optional[Tensor]:
        """Extract tensor from possibly nested structure."""
        if isinstance(x, Tensor):
            return x
        if isinstance(x, dict):
            # Use highest resolution
            res_keys = [k for k in x.keys() if isinstance(k, int)]
            if res_keys:
                return x[min(res_keys)]
            # Try first value
            for v in x.values():
                if isinstance(v, Tensor):
                    return v
        return None


__all__ = [
    'pearson_r',
    'profile_pearson_r',
    'count_pearson_r',
    'bin_pearson_r',
    'double_center',
    'differential_pearson_r',
    'compute_metrics',
    'spearman_r',
    'AlphaGenomeMetrics',
]
