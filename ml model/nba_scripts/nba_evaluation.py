"""
NBA Model Evaluation
====================
Implements evaluation metrics per the proposal.

Metrics per spec:
- RMSE and MAE on held-out drafts, with minutes-weighted variants
- Negative log predictive density (NLPD) for probabilistic calibration
- Calibration (PIT histograms, coverage of 50/80/95% intervals)
- Spearman rank correlation on draft class
- Top-K precision/recall (K = 5, 10, 30)
- Expected utility under risk preferences
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, Optional
import warnings


def rmse(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Root Mean Square Error.
    
    Per spec: "RMSE on held-out drafts"
    """
    sq_error = (y_true - y_pred) ** 2
    if weights is not None:
        return np.sqrt(np.average(sq_error, weights=weights))
    return np.sqrt(np.mean(sq_error))


def mae(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Mean Absolute Error.
    
    Per spec: "MAE on held-out drafts"
    """
    abs_error = np.abs(y_true - y_pred)
    if weights is not None:
        return np.average(abs_error, weights=weights)
    return np.mean(abs_error)


def nlpd(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray
) -> float:
    """
    Negative Log Predictive Density (Gaussian assumption).
    
    Per spec: "Negative log predictive density (NLPD)"
    
    Lower is better. Measures both accuracy and calibration.
    """
    # Gaussian log-likelihood
    var = y_pred_std ** 2 + 1e-6
    log_likelihood = -0.5 * np.log(2 * np.pi * var) - 0.5 * (y_true - y_pred_mean) ** 2 / var
    return -np.mean(log_likelihood)


def compute_pit(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray
) -> np.ndarray:
    """
    Compute Probability Integral Transform values.
    
    Per spec: "calibration (PIT histograms)"
    
    For a calibrated model, PIT values should be uniform.
    """
    pit = stats.norm.cdf(y_true, loc=y_pred_mean, scale=y_pred_std + 1e-6)
    return pit


def pit_histogram_uniformity_test(pit_values: np.ndarray, n_bins: int = 10) -> float:
    """
    Test uniformity of PIT values using chi-squared test.
    
    Returns p-value; high p-value indicates good calibration.
    """
    hist, _ = np.histogram(pit_values, bins=n_bins, range=(0, 1))
    expected = len(pit_values) / n_bins
    chi2, p_value = stats.chisquare(hist, f_exp=[expected] * n_bins)
    return p_value


def interval_coverage(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    coverage_levels: Tuple[float, ...] = (0.50, 0.80, 0.95)
) -> Dict[float, float]:
    """
    Compute empirical coverage of prediction intervals.
    
    Per spec: "coverage of 50/80/95% intervals"
    
    Returns dict mapping target coverage to actual coverage.
    A calibrated model should have actual ≈ target.
    """
    results = {}
    for level in coverage_levels:
        z = stats.norm.ppf((1 + level) / 2)
        lower = y_pred_mean - z * y_pred_std
        upper = y_pred_mean + z * y_pred_std
        in_interval = (y_true >= lower) & (y_true <= upper)
        results[level] = np.mean(in_interval)
    return results


def spearman_rank_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman rank correlation.
    
    Per spec: "Spearman rank correlation on draft class"
    """
    corr, _ = stats.spearmanr(y_true, y_pred)
    return corr


def top_k_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int,
    threshold_percentile: float = 90
) -> float:
    """
    Precision at top-K predictions.
    
    Per spec: "Top-K precision/recall (K = 5, 10, 30)"
    
    Among the K players we predict as best, what fraction are truly elite?
    """
    threshold = np.percentile(y_true, threshold_percentile)
    
    # Get top K by prediction
    top_k_indices = np.argsort(y_pred)[-k:]
    
    # Count how many are truly above threshold
    true_positives = np.sum(y_true[top_k_indices] >= threshold)
    
    return true_positives / k


def top_k_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int,
    threshold_percentile: float = 90
) -> float:
    """
    Recall at top-K predictions.
    
    Per spec: "Top-K precision/recall (K = 5, 10, 30)"
    
    Among the true elite players, what fraction did we identify in our top K?
    """
    threshold = np.percentile(y_true, threshold_percentile)
    actual_elite = y_true >= threshold
    n_elite = np.sum(actual_elite)
    
    if n_elite == 0:
        return np.nan
    
    # Get top K by prediction
    top_k_indices = np.argsort(y_pred)[-k:]
    predicted_elite = np.zeros(len(y_pred), dtype=bool)
    predicted_elite[top_k_indices] = True
    
    # Count how many elite we found
    true_positives = np.sum(actual_elite & predicted_elite)
    
    return true_positives / n_elite


def expected_utility_starter_threshold(
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    starter_threshold: float = 1.0
) -> np.ndarray:
    """
    Compute P(peak_RAPM > starter_threshold) for each player.
    
    Per spec: "Expected utility under risk preferences 
              (e.g., probability impact exceeds starter threshold)"
    """
    prob_above = 1 - stats.norm.cdf(starter_threshold, loc=y_pred_mean, scale=y_pred_std + 1e-6)
    return prob_above


def evaluate_model(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth peak RAPM
        y_pred_mean: Model predictions (mean)
        y_pred_std: Model uncertainty (std dev); if None, skip probabilistic metrics
        weights: Sample weights (e.g., minutes-based)
    
    Returns:
        Dict with all metric values
    """
    results = {}
    
    # Point prediction metrics
    results['rmse'] = rmse(y_true, y_pred_mean)
    results['mae'] = mae(y_true, y_pred_mean)
    results['spearman'] = spearman_rank_correlation(y_true, y_pred_mean)
    
    # Weighted versions
    if weights is not None:
        results['rmse_weighted'] = rmse(y_true, y_pred_mean, weights)
        results['mae_weighted'] = mae(y_true, y_pred_mean, weights)
    
    # Top-K metrics
    for k in [5, 10, 30]:
        results[f'precision_at_{k}'] = top_k_precision(y_true, y_pred_mean, k)
        results[f'recall_at_{k}'] = top_k_recall(y_true, y_pred_mean, k)
    
    # Probabilistic metrics (require uncertainty estimates)
    if y_pred_std is not None:
        results['nlpd'] = nlpd(y_true, y_pred_mean, y_pred_std)
        
        # Coverage
        coverage = interval_coverage(y_true, y_pred_mean, y_pred_std)
        for level, cov in coverage.items():
            results[f'coverage_{int(level*100)}'] = cov
        
        # PIT uniformity
        pit_values = compute_pit(y_true, y_pred_mean, y_pred_std)
        results['pit_uniformity_pvalue'] = pit_histogram_uniformity_test(pit_values)
    
    return results


def evaluate_model_by_draft_class(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    draft_class: np.ndarray,
    y_pred_std: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics PER DRAFT CLASS and average.
    
    Per critique: "Top-K / Spearman should be computed within each draft class"
    
    Pooling across years makes ranking metrics easier/misleading because
    different eras have different talent levels. This computes metrics
    within each cohort and then averages.
    
    Args:
        y_true: Ground truth peak RAPM
        y_pred_mean: Model predictions (mean)
        draft_class: Draft class/season for each sample
        y_pred_std: Model uncertainty (optional)
        weights: Sample weights (optional)
    
    Returns:
        Dict with averaged metrics across draft classes
    """
    unique_classes = np.unique(draft_class[~np.isnan(draft_class)])
    
    # Collect per-class metrics
    class_metrics = []
    
    for dc in unique_classes:
        mask = draft_class == dc
        n_samples = mask.sum()
        
        if n_samples < 5:  # Skip classes with too few samples
            continue
        
        yt = y_true[mask]
        yp = y_pred_mean[mask]
        ys = y_pred_std[mask] if y_pred_std is not None else None
        wt = weights[mask] if weights is not None else None
        
        class_result = evaluate_model(yt, yp, ys, wt)
        class_result['draft_class'] = dc
        class_result['n_samples'] = n_samples
        class_metrics.append(class_result)
    
    if not class_metrics:
        return {'error': 'No valid draft classes found'}
    
    # Aggregate: average across classes
    aggregated = {}
    for key in class_metrics[0].keys():
        if key in ['draft_class', 'n_samples']:
            continue
        values = [m[key] for m in class_metrics if not np.isnan(m.get(key, np.nan))]
        if values:
            aggregated[f'{key}_avg'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
    
    aggregated['n_draft_classes'] = len(class_metrics)
    aggregated['total_samples'] = sum(m['n_samples'] for m in class_metrics)
    
    return aggregated


if __name__ == '__main__':
    print("NBA Model Evaluation module loaded.")
    print("Key functions: evaluate_model, evaluate_model_by_draft_class, interval_coverage")
