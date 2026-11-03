"""Graph operators.

    w_ij^II(t) = c_ij * s_i^t * (1 - s_j^t)                     (1)
    w_ik^IM(t) = F_ik^t / sum_k' F_ik'^t                        (2)

The temporal weight is the partial F statistic comparing a restricted
autoregression of the intervention state against the same model augmented with a
lagged indicator. With window L_G and lag order q the full model spends 2q + 1
parameters, so the denominator degrees of freedom are L_G - 2q - 1; at the
configured L_G = 4, q = 1 that value is one.
"""

from __future__ import annotations

import numpy as np


def dependency_weights(interval: np.ndarray, conflict: np.ndarray,
                       use_conflict: bool = True) -> np.ndarray:
    """Equation 1 for every step of one record."""
    steps, n_int = interval.shape
    strength = conflict if use_conflict else np.ones_like(conflict)
    executed = interval.reshape(steps, n_int, 1)
    pending = 1.0 - interval.reshape(steps, 1, n_int)
    weights = strength.reshape(1, n_int, n_int) * executed * pending
    return weights * (1.0 - np.eye(n_int).reshape(1, n_int, n_int))


def _ols_rss(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Batched residual sum of squares; design is (..., rows, cols)."""
    gram = np.einsum("...ri,...rj->...ij", design, design)
    rhs = np.einsum("...ri,...r->...i", design, target)
    gram = gram + np.eye(gram.shape[-1]) * 1e-8
    beta = np.linalg.solve(gram, rhs[..., None])[..., 0]
    residual = target - np.einsum("...ri,...i->...r", design, beta)
    return np.einsum("...r,...r->...", residual, residual)


def granger_weights(interval: np.ndarray, indicator: np.ndarray,
                    window: int, lag: int) -> np.ndarray:
    """Equation 2 for one record, returning (T, n, m) weights."""
    steps, n_int = interval.shape
    n_ind = indicator.shape[1]
    weights = np.full((steps, n_int, n_ind), 1.0 / n_ind, dtype=np.float64)
    if steps < window + 2:
        return weights

    rows = window
    targets, restricted, full = [], [], []
    for t in range(window + 1, steps):
        idx = np.arange(t - window, t)
        lag_idx = idx - 1
        y, y_lag = interval[idx, :], interval[lag_idx, :]
        x_lag = indicator[lag_idx, :]

        ones = np.ones((rows, n_int, n_ind))
        y_rep = np.broadcast_to(y[:, :, None], (rows, n_int, n_ind))
        ylag_rep = np.broadcast_to(y_lag[:, :, None], (rows, n_int, n_ind))
        x_rep = np.broadcast_to(x_lag[:, None, :], (rows, n_int, n_ind))

        targets.append(y_rep)
        restricted.append(np.stack([ones, ylag_rep], axis=-1))
        full.append(np.stack([ones, ylag_rep, x_rep], axis=-1))

    # Move the window axis beside the feature axis so that the batched least
    # squares contracts rows only.
    target_all = np.stack(targets, 0).transpose(0, 2, 3, 1)
    rest_all = np.stack(restricted, 0).transpose(0, 2, 3, 1, 4)
    full_all = np.stack(full, 0).transpose(0, 2, 3, 1, 4)

    df_den = window - 2 * lag - 1
    if df_den <= 0:
        return weights

    rss_r = _ols_rss(rest_all, target_all)
    rss_f = _ols_rss(full_all, target_all)
    f_stat = np.clip(((rss_r - rss_f) / lag) / (np.maximum(rss_f, 1e-10) / df_den),
                     0.0, None)

    denom = f_stat.sum(axis=2, keepdims=True)
    weights[window + 1:] = np.where(denom > 1e-12, f_stat / np.maximum(denom, 1e-12),
                                    1.0 / n_ind)
    return weights


def normalise_adjacency(weights: np.ndarray, add_self_loop: bool = False) -> np.ndarray:
    adj = weights.astype(np.float64, copy=True)
    if add_self_loop:
        n = adj.shape[-1]
        adj = adj + np.eye(n).reshape((1,) * (adj.ndim - 2) + (n, n))
    denom = adj.sum(axis=-1, keepdims=True)
    return adj / np.where(denom > 1e-12, denom, 1.0)


def build_adjacency(cohort, config, use_conflict: bool = True,
                    uniform_temporal: bool = False, static_topology: bool = False,
                    window_override: int | None = None,
                    time_averaged: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Assemble A^II (N, T, n, n) and A^IM (N, T, n, m) for the whole cohort."""
    n_pat, tmax = len(cohort), cohort.X.shape[1]
    n_int, n_ind = config.n_interventions, config.n_indicators
    AI = np.zeros((n_pat, tmax, n_int, n_int), dtype=np.float32)
    AM = np.zeros((n_pat, tmax, n_int, n_ind), dtype=np.float32)

    for p in range(n_pat):
        horizon = int(cohort.lengths[p])
        interval = cohort.interval[p, :horizon]
        indicator = cohort.X[p, :horizon, n_int:]

        dep = normalise_adjacency(
            dependency_weights(interval, cohort.conflict, use_conflict),
            add_self_loop=True,
        )

        if uniform_temporal:
            temp = np.full((horizon, n_int, n_ind), 1.0 / n_ind, dtype=np.float64)
        else:
            window = config.window_L if window_override is None else window_override
            temp = granger_weights(interval, indicator, window, config.lag_q)

        if static_topology:
            dep, temp = np.repeat(dep[:1], horizon, 0), np.repeat(temp[:1], horizon, 0)
        if time_averaged:
            dep = np.repeat(dep.mean(axis=0, keepdims=True), horizon, axis=0)
            temp = np.repeat(temp.mean(axis=0, keepdims=True), horizon, axis=0)

        AI[p, :horizon] = dep.astype(np.float32)
        AM[p, :horizon] = temp.astype(np.float32)

    return AI, AM
