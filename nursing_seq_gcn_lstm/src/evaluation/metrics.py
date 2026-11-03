"""Metrics and paired statistics.

Sequence accuracy at a given length is the proportion of records whose predicted
prefix of that length matches the reference prefix exactly. The per-record value
used in the paired tests is the proportion of the evaluated lengths at which the
match holds, which is continuous on the unit interval and therefore admissible
for a signed-rank test and for a standardised mean difference.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats


def prefix_accuracy(prediction: np.ndarray, reference: np.ndarray,
                    lengths: Sequence[int]) -> Dict[int, float]:
    out = {}
    n_items = reference.shape[1]
    for length in lengths:
        length = min(int(length), n_items)
        out[length] = float(np.all(
            prediction[:, :length] == reference[:, :length], axis=1).mean() * 100.0)
    return out


def per_record_accuracy(prediction: np.ndarray, reference: np.ndarray,
                        lengths: Sequence[int]) -> np.ndarray:
    n_items = reference.shape[1]
    hits = [np.all(prediction[:, :min(int(L), n_items)] == reference[:, :min(int(L), n_items)],
                   axis=1) for L in lengths]
    return np.mean(np.stack(hits, axis=1), axis=1) * 100.0


def mean_accuracy(prediction: np.ndarray, reference: np.ndarray,
                  lengths: Sequence[int]) -> float:
    return float(np.mean(list(prefix_accuracy(prediction, reference, lengths).values())))


def _rank_matrix(order: np.ndarray) -> np.ndarray:
    rank = np.empty_like(order, dtype=np.float64)
    rows = np.arange(order.shape[0])[:, None]
    rank[rows, order] = np.arange(order.shape[1], dtype=np.float64)
    return rank


def kendall_tau(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Kendall tau-a per record between two permutations of the same items."""
    r1, r2 = _rank_matrix(prediction), _rank_matrix(reference)
    product = (r1[:, :, None] - r1[:, None, :]) * (r2[:, :, None] - r2[:, None, :])
    n = prediction.shape[1]
    total = n * (n - 1) / 2.0
    concordant = np.sum(product > 0, axis=(1, 2)) / 2.0
    discordant = np.sum(product < 0, axis=(1, 2)) / 2.0
    return (concordant - discordant) / total


def stage_kendall_tau(prediction: np.ndarray, reference: np.ndarray,
                      exec_step: np.ndarray, low: int, high: int,
                      min_items: int = 3) -> np.ndarray:
    """Tau on the sub-order of interventions executed inside a stage."""
    values = []
    for p in range(prediction.shape[0]):
        items = np.where((exec_step[p] >= low) & (exec_step[p] < high))[0]
        if len(items) < min_items:
            continue
        local = {int(item): index for index, item in enumerate(items)}
        sub_pred = [local[int(v)] for v in prediction[p] if int(v) in local]
        sub_ref = [local[int(v)] for v in reference[p] if int(v) in local]
        if len(sub_pred) != len(sub_ref) or len(sub_pred) < min_items:
            continue
        tau, _ = stats.kendalltau(sub_pred, sub_ref)
        if np.isfinite(tau):
            values.append(tau)
    return np.asarray(values)


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - target)))


def r2_score(prediction: np.ndarray, target: np.ndarray) -> float:
    residual = np.sum((target - prediction) ** 2)
    total = np.sum((target - target.mean()) ** 2)
    return float(1.0 - residual / max(total, 1e-9))


def cohen_d_paired(difference: np.ndarray) -> float:
    sd = np.std(difference, ddof=1)
    return 0.0 if sd < 1e-9 else float(np.mean(difference) / sd)


def wilcoxon_p(difference: np.ndarray) -> float:
    if np.allclose(difference, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(difference, zero_method="wilcox")[1])
    except ValueError:
        return 1.0


def bootstrap_ci(difference: np.ndarray, n_boot: int, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(difference)
    means = np.array([difference[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    m = len(values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, values[idx] * (m - rank))
        adjusted[idx] = min(1.0, running)
    return adjusted


def effect_label(d: float) -> str:
    magnitude = abs(d)
    if magnitude < 0.2:
        return "Negligible"
    if magnitude < 0.5:
        return "Small"
    if magnitude < 0.8:
        return "Moderate"
    if magnitude < 2.0:
        return "Large"
    return "Very Large"


def paired_summary(proposed: np.ndarray, other: np.ndarray, n_boot: int,
                   seed: int) -> Dict[str, float]:
    difference = proposed - other
    low, high = bootstrap_ci(difference, n_boot, seed)
    return {
        "mean_accuracy_proposed": float(proposed.mean()),
        "mean_accuracy_other": float(other.mean()),
        "mean_difference": float(difference.mean()),
        "ci_low": low, "ci_high": high,
        "cohen_d": cohen_d_paired(difference),
        "p_value": wilcoxon_p(difference),
    }
