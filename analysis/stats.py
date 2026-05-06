"""Statistical testing utilities for multi-system RAG evaluation."""

import numpy as np


def paired_permutation_test(a, b, n_resamples=10_000, seed=42):
    """Two-sided paired permutation test on matched per-query scores.

    Returns (observed_mean_diff, p_value).
    H0: mean(a - b) == 0.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diffs = a - b
    observed = np.mean(diffs)
    count = sum(
        abs(np.mean(rng.choice([-1, 1], size=len(diffs)) * diffs)) >= abs(observed)
        for _ in range(n_resamples)
    )
    return float(observed), count / n_resamples


def bootstrap_ci(data, statistic_fn=None, n_resamples=10_000, ci=0.95, seed=42):
    """Bootstrap confidence interval.

    Returns (point_estimate, lower, upper).
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    if statistic_fn is None:
        statistic_fn = np.mean
    estimate = float(statistic_fn(data))
    resampled = [
        float(statistic_fn(rng.choice(data, size=len(data), replace=True)))
        for _ in range(n_resamples)
    ]
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(resampled, alpha * 100))
    upper = float(np.percentile(resampled, (1.0 - alpha) * 100))
    return estimate, lower, upper


def cohens_d(a, b):
    """Cohen's d effect size between two paired samples."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled_var = (np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0
    if pooled_var == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled_var))


def holm_bonferroni(p_values):
    """Holm-Bonferroni correction for multiple comparisons.

    Returns adjusted p-values in the same order as the input.
    """
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(range(n), key=lambda i: p_values[i])
    corrected = [0.0] * n
    max_so_far = 0.0
    for rank, i in enumerate(indexed):
        adjusted = min(p_values[i] * (n - rank), 1.0)
        adjusted = max(adjusted, max_so_far)
        corrected[i] = adjusted
        max_so_far = adjusted
    return corrected


def compare_systems(scores_a, scores_b, system_a="A", system_b="B",
                    n_resamples=10_000, seed=42):
    """Full pairwise comparison: permutation test + bootstrap CI + Cohen's d.

    Returns a dict with all statistics, ready to drop into a results table.
    """
    mean_diff, p_val = paired_permutation_test(scores_a, scores_b,
                                               n_resamples=n_resamples, seed=seed)
    est_a, lo_a, hi_a = bootstrap_ci(scores_a, n_resamples=n_resamples, seed=seed)
    est_b, lo_b, hi_b = bootstrap_ci(scores_b, n_resamples=n_resamples, seed=seed)
    d = cohens_d(scores_a, scores_b)
    return {
        "system_a": system_a,
        "system_b": system_b,
        "mean_a": est_a,
        "ci95_a": (lo_a, hi_a),
        "mean_b": est_b,
        "ci95_b": (lo_b, hi_b),
        "mean_diff": mean_diff,
        "p_value": p_val,
        "cohens_d": d,
    }
