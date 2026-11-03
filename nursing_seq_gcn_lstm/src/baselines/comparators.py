"""Comparison methods outside the proposed architecture.

The rule ordering is parameter free: it reads the conflict matrix together with
the standard recovery pathway. The two tree models regress a priority score for
each intervention from static trajectory summaries and rank by that score, so
they carry neither temporal nor graph structure.
"""

from __future__ import annotations

import numpy as np


class RuleOrdering:
    name = "Conflict-Matrix Greedy Rule"

    def __init__(self, conflict: np.ndarray, n_interventions: int):
        from ..data.cohort import PATHWAY_PRIOR

        self.conflict = conflict
        self.n_interventions = n_interventions
        self.pathway = PATHWAY_PRIOR

    def predict(self, cohort) -> np.ndarray:
        orders = np.zeros((len(cohort), self.n_interventions), dtype=np.int64)
        for p in range(len(cohort)):
            horizon = int(cohort.lengths[p])
            done = cohort.interval[p, :horizon].mean(axis=0)
            urgency = self.conflict @ (1.0 - done)
            score = self.pathway + 1.20 * (urgency / max(float(urgency.max()), 1e-6))
            orders[p] = np.argsort(-score, kind="stable")
        return orders

    def predict_flatus(self, cohort) -> np.ndarray:
        return np.full(len(cohort), float(cohort.flatus.mean()))


class PriorityRegressor:
    def __init__(self, kind: str = "random_forest", seed: int = 0):
        self.kind = kind
        self.seed = seed
        self.name = "Random Forest" if kind == "random_forest" else "CatBoost"
        self.model = None
        self.flatus_model = None
        self.n_interventions = 8

    @staticmethod
    def _pairs(features: np.ndarray, reference: np.ndarray):
        rows, targets = [], []
        n_int = reference.shape[1]
        for p in range(features.shape[0]):
            rank = np.empty(n_int, dtype=np.float64)
            rank[reference[p]] = np.arange(n_int, dtype=np.float64)
            for item in range(n_int):
                one_hot = np.zeros(n_int)
                one_hot[item] = 1.0
                rows.append(np.concatenate([features[p], one_hot]))
                targets.append(-rank[item])
        return np.asarray(rows), np.asarray(targets)

    def fit(self, features: np.ndarray, reference: np.ndarray, flatus: np.ndarray):
        if self.kind == "random_forest":
            from sklearn.ensemble import RandomForestRegressor as Backend
            params = dict(n_estimators=120, max_depth=6, random_state=self.seed, n_jobs=-1)
        else:
            from catboost import CatBoostRegressor as Backend
            params = dict(iterations=120, depth=4, verbose=0,
                          random_seed=self.seed, allow_writing_files=False)

        rows, targets = self._pairs(features, reference)
        self.model = Backend(**params).fit(rows, targets)
        self.flatus_model = Backend(**params).fit(features, flatus)
        self.n_interventions = reference.shape[1]
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        n_pat, n_int = features.shape[0], self.n_interventions
        scores = np.zeros((n_pat, n_int))
        for item in range(n_int):
            one_hot = np.zeros((n_pat, n_int))
            one_hot[:, item] = 1.0
            scores[:, item] = self.model.predict(np.concatenate([features, one_hot], axis=1))
        return np.argsort(-scores, axis=1, kind="stable")

    def predict_flatus(self, features: np.ndarray) -> np.ndarray:
        return self.flatus_model.predict(features)
