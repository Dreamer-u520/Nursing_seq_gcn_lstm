"""Dataset assembly: imputation, standardisation, features, splits, batching.

An external cohort manifest placed at ``data/source/cohort_manifest.json`` takes
precedence; otherwise the cohort is built by :mod:`src.data.cohort` and written
to ``data/cohort``. The two locations are never mixed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau
from torch.utils.data import Dataset

from .cohort import Cohort, INDICATOR_NAMES, INTERVENTION_NAMES, build_cohort, calibrate_annotation_agreement
from .graph import build_adjacency

MANIFEST = "cohort_manifest.json"


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def resolve_cohort(config, logger) -> Tuple[Cohort, str, float]:
    manifest = Path(config.source_dir) / MANIFEST
    if manifest.exists():
        logger.info("External cohort manifest detected at %s", manifest)
        raise NotImplementedError(
            "Register the reader for the external manifest in "
            "src/data/dataset.py::resolve_cohort."
        )

    logger.info("No external manifest under %s; building the cohort.", config.source_dir)
    noise, cohort = calibrate_annotation_agreement(config, seed=config.seed)
    logger.info("Annotation noise sigma=%.3f, pairwise tau=%.3f", noise, cohort.agreement)
    return cohort, "internal", noise


def impute_indicators(X: np.ndarray, observed: np.ndarray, lengths: np.ndarray,
                      n_int: int, train_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Carry-forward imputation with a training-population fallback."""
    filled = X.copy()
    was_missing = ~observed

    for p in range(X.shape[0]):
        horizon = int(lengths[p])
        block = filled[p, :horizon, n_int:]
        last = np.zeros(block.shape[1])
        for t in range(horizon):
            last = np.where(observed[p, t], block[t], last)
            block[t] = last

    fallback = np.zeros(filled.shape[2] - n_int)
    counts = np.zeros_like(fallback)
    for p in np.where(train_mask)[0]:
        horizon = int(lengths[p])
        fallback += filled[p, :horizon, n_int:].sum(axis=0)
        counts += horizon
    fallback = fallback / np.maximum(counts, 1.0)

    for p in range(X.shape[0]):
        horizon = int(lengths[p])
        never = ~np.any(observed[p, :horizon], axis=0)
        if np.any(never):
            block = filled[p, :horizon, n_int:]
            block[:, never] = fallback[never]
            filled[p, :horizon, n_int:] = block

    return filled, was_missing


def standardise(X: np.ndarray, lengths: np.ndarray, n_int: int,
                train_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = np.concatenate(
        [X[p, :int(lengths[p]), n_int:] for p in np.where(train_mask)[0]], axis=0
    )
    mean = stacked.mean(axis=0)
    std = np.where(stacked.std(axis=0) < 1e-6, 1.0, stacked.std(axis=0))

    scaled = X.copy()
    scaled[:, :, n_int:] = (X[:, :, n_int:] - mean) / std
    return scaled, mean, std


def build_node_features(values: np.ndarray, config) -> np.ndarray:
    """Current value, first difference, then the node-category one-hot block."""
    n_pat, tmax, n_nodes = values.shape
    feats = np.zeros((n_pat, tmax, n_nodes, config.feature_dim), dtype=np.float32)
    feats[:, :, :, 0] = values
    feats[:, 1:, :, 1] = values[:, 1:, :] - values[:, :-1, :]
    for node in range(n_nodes):
        feats[:, :, node, 2 + node] = 1.0
    return feats


def make_splits(n: int, ratio: Tuple[float, float, float], seed: int) -> Split:
    index = np.random.default_rng(seed + 91).permutation(n)
    n_train = int(round(n * ratio[0]))
    n_val = int(round(n * ratio[1]))
    return Split(
        train=np.sort(index[:n_train]),
        val=np.sort(index[n_train:n_train + n_val]),
        test=np.sort(index[n_train + n_val:]),
    )


def summarise_features(cohort: Cohort, config) -> np.ndarray:
    """Static feature matrix for the non-sequential comparators."""
    n_int = config.n_interventions
    rows = []
    for p in range(len(cohort)):
        horizon = int(cohort.lengths[p])
        block = cohort.X[p, :horizon, n_int:]
        steps = np.arange(horizon)
        slope = np.array([np.polyfit(steps, block[:, j], 1)[0] if horizon > 1 else 0.0
                          for j in range(block.shape[1])])
        rows.append(np.concatenate([block.mean(axis=0), block[-1], slope,
                                    [cohort.age[p], cohort.sex[p], cohort.surgery[p], horizon]]))
    return np.asarray(rows, dtype=np.float64)


class CohortDataset(Dataset):
    def __init__(self, cohort: Cohort, features: np.ndarray, adjacency,
                 index: np.ndarray, config):
        self.index = np.asarray(index)
        self.features = torch.from_numpy(features).float()
        self.AI, self.AM = adjacency
        self.reference = torch.from_numpy(cohort.reference).long()
        self.flatus = torch.from_numpy(cohort.flatus).float()
        self.lengths = torch.from_numpy(cohort.lengths).long()
        self.age = torch.from_numpy(cohort.age).float()
        self.severity = torch.from_numpy(cohort.severity).float()
        self.exec_step = torch.from_numpy(cohort.exec_step).long()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        p = int(self.index[item])
        length = int(self.lengths[p])
        mask = torch.zeros(self.features.shape[1], dtype=torch.bool)
        mask[:length] = True
        return {
            "X": self.features[p], "AI": self.AI[p], "AM": self.AM[p], "mask": mask,
            "reference": self.reference[p], "flatus": self.flatus[p],
            "length": self.lengths[p], "age": self.age[p],
            "severity": self.severity[p], "exec_step": self.exec_step[p],
            "index": torch.tensor(p, dtype=torch.long),
        }


def dataset_statistics(cohort: Cohort, split: Split) -> pd.DataFrame:
    rows = []
    groups = [("Training Set", split.train), ("Validation Set", split.val),
              ("Test Set", split.test), ("Overall", np.arange(len(cohort)))]
    for name, idx in groups:
        taus = [kendalltau(cohort.annotators[p][a], cohort.annotators[p][b])[0]
                for p in idx for a in range(3) for b in range(a + 1, 3)]
        rows.append({
            "Statistic": name,
            "SampleSize": len(idx),
            "MeanAge": round(float(cohort.age[idx].mean()), 1),
            "SdAge": round(float(cohort.age[idx].std(ddof=1)), 1),
            "MalePercent": round(float(cohort.sex[idx].mean() * 100.0), 1),
            "MeanSteps": round(float(cohort.lengths[idx].mean()), 1),
            "SdSteps": round(float(cohort.lengths[idx].std(ddof=1)), 1),
            "MeanFlatusHours": round(float(cohort.flatus[idx].mean()), 1),
            "SdFlatusHours": round(float(cohort.flatus[idx].std(ddof=1)), 1),
            "MeanPairwiseTau": round(float(np.mean(taus)), 2),
        })
    return pd.DataFrame(rows)


def build_bundle(config, logger):
    cohort, provenance, noise = resolve_cohort(config, logger)
    split = make_splits(len(cohort), config.split_ratio, config.seed)

    train_mask = np.zeros(len(cohort), dtype=bool)
    train_mask[split.train] = True
    logger.info("Cohort n=%d | train/val/test %d/%d/%d", len(cohort),
                len(split.train), len(split.val), len(split.test))

    filled, was_missing = impute_indicators(
        cohort.X, cohort.observed, cohort.lengths, config.n_interventions, train_mask)
    scaled, mean, std = standardise(filled, cohort.lengths, config.n_interventions, train_mask)
    features = build_node_features(scaled, config)
    AI, AM = build_adjacency(cohort, config)
    logger.info("Features %s | A_II %s | A_IM %s", features.shape, AI.shape, AM.shape)

    return {
        "cohort": cohort, "split": split, "features": features, "AI": AI, "AM": AM,
        "flat": summarise_features(cohort, config), "observed": cohort.observed,
        "was_missing": was_missing, "scaled": scaled,
        "indicator_mean": mean, "indicator_std": std,
        "provenance": provenance, "annotation_noise": noise,
    }


def persist_cohort(config, bundle, logger) -> None:
    path = Path(config.cohort_dir)
    path.mkdir(parents=True, exist_ok=True)
    cohort = bundle["cohort"]

    np.savez_compressed(
        path / "cohort.npz",
        X=cohort.X, observed=cohort.observed, interval=cohort.interval,
        lengths=cohort.lengths, flatus=cohort.flatus, reference=cohort.reference,
        annotators=cohort.annotators, exec_step=cohort.exec_step, age=cohort.age,
        sex=cohort.sex, surgery=cohort.surgery, severity=cohort.severity,
        conflict=cohort.conflict,
    )

    rows = []
    for p in range(min(5, len(cohort))):
        for t in range(int(cohort.lengths[p])):
            row = {"record": p, "step": t, "hour": t * config.step_hours}
            for j, name in enumerate(INDICATOR_NAMES):
                row[name] = round(float(cohort.X[p, t, config.n_interventions + j]), 4)
            for j, name in enumerate(INTERVENTION_NAMES):
                row[name] = int(cohort.interval[p, t, j])
            rows.append(row)
    pd.DataFrame(rows).to_csv(path / "sample_records.csv", index=False)

    with open(path / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({
            "records": int(len(cohort)),
            "source": bundle["provenance"],
            "annotation_noise": bundle["annotation_noise"],
            "mean_pairwise_tau": cohort.agreement,
            "interventions": INTERVENTION_NAMES,
            "indicators": INDICATOR_NAMES,
            "files": ["cohort.npz", "sample_records.csv"],
        }, handle, indent=2)
