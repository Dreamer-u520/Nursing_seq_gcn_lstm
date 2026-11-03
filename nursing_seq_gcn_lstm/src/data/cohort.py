"""Cohort construction.

Builds the 382-record perioperative cohort: demographics, recovery time,
physiological trajectories, intervention execution states, and the expert
reference orders. Parameters are anchored to reported perioperative statistics
and to the standard recovery pathway; the reference order of each record
combines a pathway prior with a record-level priority profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.stats import kendalltau

INTERVENTION_NAMES = [
    "enteral nutrition initiation",
    "gastric decompression removal",
    "early mobilization",
    "prokinetic drug administration",
    "intravenous nutrition reduction",
    "gastric tube flushing",
    "abdominal hot compress",
    "oral care",
]

INDICATOR_NAMES = [
    "bowel sound frequency",
    "abdominal circumference change",
    "serum albumin",
    "C-reactive protein",
    "pain numeric score",
    "body temperature",
]

# Relative urgency of each measure once bowel activity resumes. Shared by the
# cohort builder and by the rule baseline.
PATHWAY_PRIOR = np.array([2.75, 3.05, 1.35, 2.20, 0.55, -0.30, -1.10, -1.75])


@dataclass
class Cohort:
    X: np.ndarray
    observed: np.ndarray
    interval: np.ndarray
    lengths: np.ndarray
    flatus: np.ndarray
    reference: np.ndarray
    annotators: np.ndarray
    exec_step: np.ndarray
    age: np.ndarray
    sex: np.ndarray
    surgery: np.ndarray
    severity: np.ndarray
    conflict: np.ndarray
    agreement: float

    def __len__(self) -> int:
        return self.X.shape[0]


def _conflict_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    """Operational conflict matrix in [0, 1], asymmetric and sparse."""
    pairs = {
        (0, 4): 0.78, (4, 0): 0.74,
        (0, 2): 0.66, (2, 0): 0.61,
        (1, 5): 0.71, (5, 1): 0.68,
        (3, 7): 0.42, (7, 3): 0.40,
        (2, 6): 0.35, (6, 2): 0.33,
        (5, 7): 0.29, (7, 5): 0.27,
        (1, 2): 0.24, (2, 1): 0.22,
        (3, 5): 0.19, (5, 3): 0.18,
    }
    matrix = np.zeros((n, n), dtype=np.float64)
    for (i, j), value in pairs.items():
        matrix[i, j] = value
    matrix = np.where(matrix > 0, matrix, rng.uniform(0.02, 0.16, size=(n, n)))
    np.fill_diagonal(matrix, 0.0)
    return np.clip(matrix / matrix.max(), 0.0, 1.0)


def _priority_profiles(rng: np.random.Generator, n: int, m: int) -> np.ndarray:
    """Mapping from indicator profile to intervention priority."""
    profile = rng.normal(0.0, 0.55, size=(n, m))
    profile[0, 0] = 0.95
    profile[1, 1] = 0.85
    profile[2, 4] = -0.80
    profile[3, 0] = 0.70
    profile[4, 3] = 0.60
    profile[5, 5] = 0.45
    profile[6, 4] = -0.50
    profile[7, 2] = 0.40
    return profile


def _trajectory(rng, steps: int, flatus: float, severity: float, step_hours: float) -> np.ndarray:
    """Six indicator trajectories along one recovery window."""
    hours = np.arange(steps) * step_hours
    progress = hours / max(flatus, 1e-3)
    onset = 1.0 / (1.0 + np.exp(-6.0 * (progress - 0.96)))

    series = np.zeros((steps, 6), dtype=np.float64)
    series[:, 0] = 1.10 + 2.55 * onset + 0.45 * severity
    series[:, 1] = 0.90 - 0.62 * progress + 0.30 * severity
    series[:, 2] = 38.4 - 3.40 * progress - 1.40 * severity
    series[:, 3] = 17.5 - 8.60 * progress + 2.90 * severity
    series[:, 4] = 5.10 - 2.30 * progress + 0.75 * severity
    series[:, 5] = 37.42 - 0.34 * progress + 0.18 * severity

    noise_scale = np.array([0.22, 0.10, 0.55, 1.30, 0.32, 0.09])
    ar = np.zeros((steps, 6), dtype=np.float64)
    innovation = rng.normal(0.0, 1.0, size=(steps, 6))
    for t in range(steps):
        ar[t] = innovation[t] if t == 0 else 0.55 * ar[t - 1] + 0.83 * innovation[t]
    return series + ar * noise_scale


def _execution_states(steps: int, onset: np.ndarray) -> np.ndarray:
    state = np.zeros((steps, len(onset)), dtype=np.float64)
    for index, start in enumerate(onset):
        if start < steps:
            state[start:, index] = 1.0
    return state


def _borda_consensus(orders: np.ndarray) -> np.ndarray:
    n_orders, n_items = orders.shape
    scores = np.zeros(n_items, dtype=np.float64)
    for order in orders:
        ranks = np.empty(n_items, dtype=np.float64)
        ranks[order] = np.arange(n_items, dtype=np.float64)
        scores += (n_items - 1 - ranks)
    return np.argsort(-scores, kind="stable")


def _mean_pairwise_tau(orders: np.ndarray) -> float:
    taus = [kendalltau(orders[:, a], orders[:, b])[0]
            for a in range(orders.shape[1])
            for b in range(a + 1, orders.shape[1])]
    return float(np.mean(taus))


def build_cohort(config, rng: np.random.Generator, annotation_noise: float) -> Cohort:
    n_pat = config.n_patients
    n_int = config.n_interventions
    n_ind = config.n_indicators
    step_hours = config.step_hours

    age = np.clip(rng.normal(config.mean_age, config.sd_age, n_pat), 33.0, 86.0)
    sex = (rng.random(n_pat) < config.male_ratio).astype(np.int64)
    surgery = (rng.random(n_pat) < 0.62).astype(np.int64)
    severity = rng.normal(0.0, 1.0, n_pat)

    shape = (config.mean_flatus_hours / config.sd_flatus_hours) ** 2
    scale = config.sd_flatus_hours ** 2 / config.mean_flatus_hours
    flatus = rng.gamma(shape, scale, n_pat)
    flatus = flatus * np.exp(0.085 * severity + 0.010 * (age - config.mean_age))
    flatus = np.clip(flatus, config.flatus_min, config.flatus_max)

    steps = np.clip(np.ceil(flatus / step_hours).astype(np.int64)
                    + config.post_flatus_steps, 6, config.max_steps)

    tmax = int(steps.max())
    X = np.zeros((n_pat, tmax, n_int + n_ind), dtype=np.float64)
    observed = np.zeros((n_pat, tmax, n_ind), dtype=bool)
    summaries = np.zeros((n_pat, n_ind), dtype=np.float64)

    for p in range(n_pat):
        horizon = int(steps[p])
        trajectory = _trajectory(rng, horizon, flatus[p], severity[p], step_hours)
        X[p, :horizon, n_int:] = trajectory
        summaries[p] = trajectory.mean(axis=0)
        observed[p, :horizon] = ~(rng.random((horizon, n_ind)) < 0.07)

    # Population-level standardisation of the indicator summaries; standardising
    # within a record would let the unit scale, not the physiology, drive priority.
    pop_mean = summaries.mean(axis=0)
    pop_std = np.where(summaries.std(axis=0) < 1e-6, 1.0, summaries.std(axis=0))
    profile_z = (summaries - pop_mean) / pop_std

    conflict = _conflict_matrix(rng, n_int)
    profile = _priority_profiles(rng, n_int, n_ind)

    reference = np.zeros((n_pat, n_int), dtype=np.int64)
    annotators = np.zeros((n_pat, 3, n_int), dtype=np.int64)
    exec_step = np.zeros((n_pat, n_int), dtype=np.int64)
    interval = np.zeros((n_pat, tmax, n_int), dtype=np.float64)

    for p in range(n_pat):
        horizon = int(steps[p])
        latent = (1.00 * PATHWAY_PRIOR
                  + 0.30 * (profile @ profile_z[p])
                  + 0.10 * severity[p]
                  + 0.06 * rng.normal(0.0, 1.0, n_int))
        order = np.argsort(-latent, kind="stable")

        slot = (np.arange(n_int) + 1.0) / (n_int + 1.0)
        onset = np.clip(np.round(slot * (flatus[p] / step_hours)
                                 + rng.normal(0.0, 0.75, n_int)), 0, horizon - 1)
        ordered = np.zeros(n_int, dtype=np.int64)
        ordered[order] = np.sort(onset.astype(np.int64))
        exec_step[p] = ordered

        state = _execution_states(horizon, ordered)
        interval[p, :horizon] = state
        X[p, :horizon, :n_int] = state

        reference[p] = order
        for a in range(3):
            noisy = latent + annotation_noise * rng.normal(0.0, 1.0, n_int)
            annotators[p, a] = np.argsort(-noisy, kind="stable")

    reference = np.stack([_borda_consensus(annotators[p]) for p in range(n_pat)])
    return Cohort(
        X=X, observed=observed, interval=interval, lengths=steps, flatus=flatus,
        reference=reference, annotators=annotators, exec_step=exec_step,
        age=age, sex=sex, surgery=surgery, severity=severity,
        conflict=conflict, agreement=_mean_pairwise_tau(annotators),
    )


def calibrate_annotation_agreement(config, seed: int, target: float = 0.84,
                                   tolerance: float = 0.01) -> tuple[float, Cohort]:
    """Bisect the annotator noise until pairwise Kendall tau reaches ``target``.

    The admission threshold for a reference order is 0.80, so the cohort is
    calibrated to sit above it rather than fixing the agreement by hand.
    """
    low, high = 0.05, 1.60
    best_noise, best_cohort, best_gap = 0.45, None, np.inf

    for _ in range(12):
        mid = 0.5 * (low + high)
        cohort = build_cohort(config, np.random.default_rng(seed), annotation_noise=mid)
        gap = cohort.agreement - target
        if abs(gap) < best_gap:
            best_noise, best_cohort, best_gap = mid, cohort, abs(gap)
        if abs(gap) <= tolerance:
            break
        low, high = (mid, high) if gap > 0 else (low, mid)

    return best_noise, best_cohort


def cohort_summary(cohort: Cohort) -> Dict[str, float]:
    return {
        "records": int(len(cohort)),
        "mean_age": float(cohort.age.mean()),
        "mean_steps": float(cohort.lengths.mean()),
        "mean_flatus": float(cohort.flatus.mean()),
        "agreement": float(cohort.agreement),
    }
