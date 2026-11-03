"""Experiment drivers. Every table and figure is written from arrays computed
during the run; no result value is stored in the source tree.
"""

from __future__ import annotations

import copy
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy import stats as st

from . import figures
from ..baselines.comparators import PriorityRegressor, RuleOrdering
from ..core.runtime import peak_memory_mb
from ..data.dataset import CohortDataset, dataset_statistics
from ..data.graph import build_adjacency
from ..evaluation import metrics
from ..models.network import SequencingModel
from ..training.loop import evaluate, fit, make_loader


class Runner:
    def __init__(self, config, bundle, logger):
        self.config = config
        self.bundle = bundle
        self.logger = logger
        self.cohort = bundle["cohort"]
        self.split = bundle["split"]
        self.results: Dict[str, dict] = {}
        self._adj_cache: Dict[str, tuple] = {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _datasets(self, features=None, adjacency=None):
        features = self.bundle["features"] if features is None else features
        adjacency = ((self.bundle["AI"], self.bundle["AM"])
                     if adjacency is None else adjacency)
        return tuple(
            CohortDataset(self.cohort, features, adjacency, index, self.config)
            for index in (self.split.train, self.split.val, self.split.test)
        )

    def _collect(self, model, dataset) -> dict:
        out = evaluate(model, make_loader(dataset, self.config.batch_size, False))
        out["per_record_acc"] = metrics.per_record_accuracy(
            out["permutation"], out["reference"], self.config.lengths)
        out["tau"] = metrics.kendall_tau(out["permutation"], out["reference"])
        out["index"] = dataset.index
        return out

    def _register(self, name, payload):
        self.results[name] = payload
        self.logger.info("%s | mean acc %.2f | tau %.3f | MAE %.2f h", name,
                         payload["mean_accuracy"], float(np.mean(payload["tau"])),
                         metrics.mae(payload["predicted_flatus"], payload["flatus"]))

    def train_variant(self, name, encoder="typed", decoder="pointer",
                      use_ii=True, use_im=True, shared_weight=False,
                      adjacency=None, features=None, config=None, seed_index=0):
        config = self.config if config is None else config
        train, val, test = self._datasets(features, adjacency)

        torch.manual_seed(config.seed + seed_index)
        np.random.seed(config.seed + seed_index)
        model = SequencingModel(config, encoder=encoder, decoder=decoder,
                                use_ii=use_ii, use_im=use_im,
                                shared_weight=shared_weight)

        memory_before = peak_memory_mb()
        model, history, elapsed = fit(
            model,
            make_loader(train, config.batch_size, True, config.seed + seed_index),
            make_loader(val, config.batch_size, False),
            config, self.logger, tag=f"{name}_s{seed_index}")
        memory_after = peak_memory_mb()

        started = time.perf_counter()
        out = self._collect(model, test)
        latency = (time.perf_counter() - started) * 1000.0 / max(len(test), 1)

        out.update({
            "mean_accuracy": metrics.mean_accuracy(out["permutation"],
                                                   out["reference"], config.lengths),
            "history": pd.DataFrame(history), "elapsed": elapsed,
            "params": model.parameter_count(), "latency_ms": latency,
            "memory_gb": max(memory_after - memory_before, 0.0) / 1024.0,
            "epochs": len(history), "model": model,
        })
        return out

    def _merge(self, runs: List[dict]) -> dict:
        base = copy.copy(runs[0])
        permutation = np.round(np.mean([r["permutation"] for r in runs], axis=0)).astype(np.int64)
        base["permutation"] = permutation
        base["per_record_acc"] = np.mean([r["per_record_acc"] for r in runs], axis=0)
        base["predicted_flatus"] = np.mean([r["predicted_flatus"] for r in runs], axis=0)
        base["tau"] = metrics.kendall_tau(permutation, base["reference"])
        base["mean_accuracy"] = float(base["per_record_acc"].mean())
        base["elapsed"] = float(np.mean([r["elapsed"] for r in runs]))
        base["latency_ms"] = float(np.mean([r["latency_ms"] for r in runs]))
        base["memory_gb"] = float(np.max([r["memory_gb"] for r in runs]))
        base["epochs"] = float(np.mean([r["epochs"] for r in runs]))
        return base

    def _cached(self, key, builder):
        if key not in self._adj_cache:
            self._adj_cache[key] = builder()
        return self._adj_cache[key]

    def _static_adjacency(self):
        return self._cached("static", lambda: build_adjacency(
            self.cohort, self.config, static_topology=True))

    def _averaged_adjacency(self):
        return self._cached("averaged", lambda: build_adjacency(
            self.cohort, self.config, time_averaged=True))

    def _uniform_adjacency(self):
        return self._cached("uniform", lambda: build_adjacency(
            self.cohort, self.config, uniform_temporal=True))

    def _noconflict_adjacency(self):
        return self._cached("noconflict", lambda: build_adjacency(
            self.cohort, self.config, use_conflict=False))

    # ------------------------------------------------------------------
    # main comparison
    # ------------------------------------------------------------------
    def run_comparison(self):
        plan = [
            ("Proposed Model", dict(encoder="typed", decoder="pointer")),
            ("GCN-LSTM-Homogeneous", dict(encoder="homo", decoder="pointer")),
            ("GAT+LSTM", dict(encoder="gat", decoder="pointer")),
            ("Static GCN+LSTM", dict(encoder="typed", decoder="pointer",
                                     adjacency=self._static_adjacency())),
            ("Transformer", dict(encoder="transformer", decoder="pointer")),
            ("Seq2Seq+Attn", dict(encoder="typed", decoder="seq2seq")),
            ("LSTM", dict(encoder="plain", decoder="pointer")),
        ]
        for name, kwargs in plan:
            runs = [self.train_variant(name, seed_index=s, **kwargs)
                    for s in range(self.config.n_seeds)]
            self._register(name, self._merge(runs))

        self._tree_comparators()
        self._rule_comparator()

    def _tree_comparators(self):
        flat = self.bundle["flat"]
        reference, flatus = self.cohort.reference, self.cohort.flatus

        for kind, name in [("catboost", "CatBoost"), ("random_forest", "Random Forest")]:
            permutations, predictions = [], []
            for seed in range(self.config.n_seeds):
                model = PriorityRegressor(kind=kind, seed=self.config.seed + seed)
                model.fit(flat[self.split.train], reference[self.split.train],
                          flatus[self.split.train])
                permutations.append(model.predict(flat[self.split.test]))
                predictions.append(model.predict_flatus(flat[self.split.test]))

            permutation, target = permutations[0], reference[self.split.test]
            self._register(name, {
                "permutation": permutation, "reference": target,
                "flatus": flatus[self.split.test],
                "predicted_flatus": predictions[0],
                "per_record_acc": metrics.per_record_accuracy(
                    permutation, target, self.config.lengths),
                "tau": metrics.kendall_tau(permutation, target),
                "mean_accuracy": metrics.mean_accuracy(permutation, target,
                                                       self.config.lengths),
                "index": self.split.test, "elapsed": 0.0, "params": 0,
                "latency_ms": 0.0, "memory_gb": 0.0, "epochs": 0,
            })

    def _rule_comparator(self):
        model = RuleOrdering(self.cohort.conflict, self.config.n_interventions)
        permutation = model.predict(self.cohort)[self.split.test]
        target = self.cohort.reference[self.split.test]
        self._register("Conflict-Matrix Greedy Rule", {
            "permutation": permutation, "reference": target,
            "flatus": self.cohort.flatus[self.split.test],
            "predicted_flatus": model.predict_flatus(self.cohort)[self.split.test],
            "per_record_acc": metrics.per_record_accuracy(
                permutation, target, self.config.lengths),
            "tau": metrics.kendall_tau(permutation, target),
            "mean_accuracy": metrics.mean_accuracy(permutation, target,
                                                   self.config.lengths),
            "index": self.split.test, "elapsed": 0.0, "params": 0,
            "latency_ms": 0.0, "memory_gb": 0.0, "epochs": 0,
        })

    # ------------------------------------------------------------------
    # tables
    # ------------------------------------------------------------------
    def table_dataset(self) -> pd.DataFrame:
        raw = dataset_statistics(self.cohort, self.split)
        labels = {
            "SampleSize": "Sample Size (n)",
            "MeanAge": "Mean Age (years)",
            "MalePercent": "Male Proportion (%)",
            "MeanSteps": "Mean Number of Observation Time Steps",
            "MeanFlatusHours": "Mean Time to First Flatus (hours)",
            "MeanPairwiseTau": "Mean Pairwise Kendall's tau of the Reference Sequences",
        }
        formatted = {"MeanAge": ("MeanAge", "SdAge"),
                     "MeanSteps": ("MeanSteps", "SdSteps"),
                     "MeanFlatusHours": ("MeanFlatusHours", "SdFlatusHours")}

        rows = []
        for key, label in labels.items():
            row = {"Statistic": label}
            for _, record in raw.iterrows():
                if key in formatted:
                    mean_col, sd_col = formatted[key]
                    row[record["Statistic"]] = f"{record[mean_col]:.1f} \u00b1 {record[sd_col]:.1f}"
                else:
                    row[record["Statistic"]] = record[key]
            rows.append(row)

        frame = pd.DataFrame(rows)
        frame.to_csv(self.config.table_dir / "Table1_dataset_characteristics.csv", index=False)
        return frame

    def table_significance(self) -> pd.DataFrame:
        order = ["Proposed Model", "GCN-LSTM-Homogeneous", "GAT+LSTM", "Static GCN+LSTM",
                 "Transformer", "Conflict-Matrix Greedy Rule", "Seq2Seq+Attn",
                 "LSTM", "CatBoost", "Random Forest"]
        proposed = self.results["Proposed Model"]["per_record_acc"]

        rows, raw_p = [], []
        for name in order:
            payload = self.results[name]
            if name == "Proposed Model":
                rows.append({"Method": name, "MeanAccuracy": round(payload["mean_accuracy"], 2),
                             "MeanDifference": 0.0, "CI_Low": np.nan, "CI_High": np.nan,
                             "CohenD": np.nan, "Effect": "reference", "p": np.nan})
                continue

            summary = metrics.paired_summary(proposed, payload["per_record_acc"],
                                             self.config.bootstrap_n, self.config.seed)
            difference = summary["mean_difference"]
            rows.append({
                "Method": name,
                "MeanAccuracy": round(summary["mean_accuracy_other"], 2),
                "MeanDifference": round(-difference, 2),
                "CI_Low": round(-summary["ci_high"], 2),
                "CI_High": round(-summary["ci_low"], 2),
                "CohenD": round(abs(summary["cohen_d"]), 2),
                "Effect": metrics.effect_label(summary["cohen_d"]),
                "p": summary["p_value"],
            })
            raw_p.append(summary["p_value"])

        adjusted = metrics.holm_adjust(raw_p)
        for row, value in zip(rows[1:], adjusted):
            row["HolmP"] = value
        rows[0]["HolmP"] = np.nan

        frame = pd.DataFrame(rows)
        for column in ("p", "HolmP"):
            frame[column] = frame[column].apply(
                lambda v: "" if pd.isna(v) else f"{v:.4f}")
        return frame

    # ------------------------------------------------------------------
    # figures
    # ------------------------------------------------------------------
    def figure_accuracy_length(self) -> pd.DataFrame:
        rows = []
        for name, payload in self.results.items():
            table = metrics.prefix_accuracy(payload["permutation"], payload["reference"],
                                            self.config.lengths)
            std = (np.nan if name == "Conflict-Matrix Greedy Rule"
                   else float(np.std(list(table.values()), ddof=1)))
            rows += [{"Method": name, "Length": L, "Mean": round(v, 2), "Std": std}
                     for L, v in table.items()]

        frame = pd.DataFrame(rows)
        figures.plot_accuracy_by_length(
            frame, self.config.figure_dir / "Figure5_accuracy_by_length.png")
        frame.to_csv(self.config.figure_data_dir / "Figure5_Data.csv", index=False)
        return frame

    def figure_flatus(self) -> pd.DataFrame:
        payload = self.results["Proposed Model"]
        recorded = payload["flatus"]
        names = ["LSTM", "Transformer", "GCN-LSTM-Homogeneous", "Proposed Model"]
        predictions = {name: self.results[name]["predicted_flatus"] for name in names}

        figures.plot_flatus_scatter(
            recorded, predictions, self.config.figure_dir / "Figure7_flatus_scatter.png")

        frame = pd.DataFrame({"RecordedFirstFlatus": recorded})
        for name, values in predictions.items():
            frame[f"Estimated_{name}"] = values
        frame.to_csv(self.config.figure_data_dir / "Figure7_Data.csv", index=False)
        return frame

    # ------------------------------------------------------------------
    # ablation
    # ------------------------------------------------------------------
    def run_ablation(self) -> pd.DataFrame:
        variants = {
            "Full Model": dict(),
            "A1 w/o Dynamic Graph Update": dict(adjacency=self._averaged_adjacency()),
            "A2 w/o Causal Edges": dict(use_im=False),
            "A3 w/o Dependency Edges": dict(use_ii=False),
            "A4 w/o Type-Specific Weight Matrices": dict(shared_weight=True),
            "A5 w/o Granger Dynamic Weights": dict(adjacency=self._uniform_adjacency()),
            "A6 w/o Operational Conflict Matrix": dict(adjacency=self._noconflict_adjacency()),
        }

        reference = self.results["Proposed Model"]["per_record_acc"]
        rows, raw_p = [], []
        for name, kwargs in variants.items():
            if name == "Full Model":
                payload = self.results["Proposed Model"]
                p_value = np.nan
            else:
                payload = self._merge([self.train_variant(name, seed_index=s, **kwargs)
                                       for s in range(self.config.n_seeds)])
                self._register(name, payload)
                differences = reference - payload["per_record_acc"]
                p_value = (1.0 if np.allclose(differences, 0)
                           else float(st.ttest_rel(reference, payload["per_record_acc"])[1]))
                raw_p.append(p_value)

            values = payload["per_record_acc"]
            rows.append({
                "Variant": name,
                "Accuracy": round(payload["mean_accuracy"], 2),
                "AccuracySd": round(float(np.std(values, ddof=1)), 2),
                "Tau": round(float(np.mean(payload["tau"])), 3),
                "TauSd": round(float(np.std(payload["tau"], ddof=1)), 3),
                "MAE": round(metrics.mae(payload["predicted_flatus"], payload["flatus"]), 2),
                "p": p_value,
            })

        adjusted = metrics.holm_adjust(raw_p)
        cursor = 0
        for row in rows:
            if row["Variant"] == "Full Model":
                row["HolmP"], row["Mark"] = np.nan, ""
            else:
                row["HolmP"] = adjusted[cursor]
                row["Mark"] = ("**" if adjusted[cursor] < 0.01
                               else "*" if adjusted[cursor] < 0.05 else "ns")
                cursor += 1

        frame = pd.DataFrame(rows)
        frame.to_csv(self.config.table_dir / "Table3_ablation.csv", index=False)
        return frame

    # ------------------------------------------------------------------
    # decision modules and cost
    # ------------------------------------------------------------------
    def run_decision_modules(self) -> pd.DataFrame:
        rows = []
        for name, decoder in [("Softmax Classifier", "softmax"),
                              ("Seq2Seq + Attention", "seq2seq"),
                              ("Pointer Network", "pointer")]:
            if decoder == "pointer":
                payload = self.results["Proposed Model"]
            else:
                payload = self._merge([self.train_variant(name, decoder=decoder, seed_index=s)
                                       for s in range(self.config.n_seeds)])
                self._register(name, payload)

            rows.append({
                "DecisionModule": name,
                "SequenceAccuracy": round(payload["mean_accuracy"], 2),
                "AccuracySd": round(float(np.std(payload["per_record_acc"], ddof=1)), 2),
                "Tau": round(float(np.mean(payload["tau"])), 3),
                "TauSd": round(float(np.std(payload["tau"], ddof=1)), 3),
                "MAE": round(metrics.mae(payload["predicted_flatus"], payload["flatus"]), 2),
                "ConvergenceEpochs": int(round(payload["epochs"])),
                "LatencyMs": round(payload["latency_ms"], 2),
            })

        frame = pd.DataFrame(rows)
        frame.to_csv(self.config.table_dir / "Table4_decision_modules.csv", index=False)
        return frame

    def table_cost(self) -> pd.DataFrame:
        order = ["Proposed Model", "GCN-LSTM-Homogeneous", "GAT+LSTM", "Static GCN+LSTM",
                 "Transformer", "Seq2Seq+Attn", "LSTM", "CatBoost", "Random Forest"]
        rows = []
        for name in order:
            payload = self.results.get(name)
            if payload is None:
                continue
            rows.append({
                "Model": name,
                "ParametersM": round(payload["params"] / 1e6, 3),
                "TrainingSecondsPerEpoch": round(payload["elapsed"] / max(payload["epochs"], 1), 3),
                "PeakMemoryGB": round(payload["memory_gb"], 3),
                "InferenceMsPerCase": round(payload["latency_ms"], 3),
            })

        frame = pd.DataFrame(rows)
        frame.to_csv(self.config.table_dir / "Table5_computational_cost.csv", index=False)
        figures.plot_cost_bars(
            frame, self.config.figure_dir / "Figure11_computational_cost.png")
        return frame

    # ------------------------------------------------------------------
    # sensitivity, generalisation, robustness
    # ------------------------------------------------------------------
    def run_sensitivity(self) -> pd.DataFrame:
        rows = []

        def record(factor, setting, payload):
            rows.append({
                "Factor": factor, "Setting": setting,
                "Accuracy": round(payload["mean_accuracy"], 2),
                "Tau": round(float(np.mean(payload["tau"])), 3),
                "MAE": round(metrics.mae(payload["predicted_flatus"], payload["flatus"]), 2),
            })

        for window in sorted({3, self.config.window_L, self.config.window_L * 2}):
            adjacency = build_adjacency(self.cohort, self.config, window_override=window)
            record("Granger window", f"L_G={window}",
                   self.train_variant(f"window_{window}", adjacency=adjacency))

        for hidden in sorted({max(self.config.hidden_dim // 2, 16), self.config.hidden_dim}):
            variant = copy.deepcopy(self.config)
            variant.hidden_dim = hidden
            record("Hidden size", f"H={hidden}",
                   self.train_variant(f"hidden_{hidden}", config=variant))

        for dim in sorted({max(self.config.gcn_dim // 2, 8), self.config.gcn_dim}):
            variant = copy.deepcopy(self.config)
            variant.gcn_dim = dim
            record("Embedding size", f"d'={dim}",
                   self.train_variant(f"gcn_{dim}", config=variant))

        frame = pd.DataFrame(rows)
        frame.to_csv(self.config.table_dir / "Table6_sensitivity.csv", index=False)
        figures.plot_two_factor(
            pd.DataFrame({"Method": "Proposed Model", "Condition": frame["Setting"],
                          "Accuracy": frame["Accuracy"]}),
            self.config.figure_dir / "Figure8_sensitivity.png", "Configuration")
        return frame

    def run_generalization(self):
        names = ["Proposed Model", "GCN-LSTM-Homogeneous", "Transformer", "LSTM"]
        index = self.results["Proposed Model"]["index"]
        ages = self.cohort.age[index]
        exec_step = self.cohort.exec_step[index]

        young, elderly = self.config.age_bins
        groups = {"Young": ages < young,
                  "Middle aged": (ages >= young) & (ages < elderly),
                  "Elderly": ages >= elderly}
        stages = {"Initial": (0, self.config.stage_bounds[0]),
                  "Intermediate": (self.config.stage_bounds[0], self.config.stage_bounds[1]),
                  "Late": (self.config.stage_bounds[1], self.config.max_steps)}

        rows, tau_records = [], []
        for name in names:
            payload = self.results[name]
            for label, mask in groups.items():
                if mask.sum() == 0:
                    continue
                rows.append({"Dimension": "Patient group", "Category": label,
                             "Method": name, "Metric": "Accuracy",
                             "Value": round(float(np.mean(payload["per_record_acc"][mask])), 2)})
                tau_records.append(pd.DataFrame({
                    "Dimension": "Patient group", "Category": label,
                    "Method": name, "Tau": payload["tau"][mask]}))

            for label, (low, high) in stages.items():
                tau_values = metrics.stage_kendall_tau(
                    payload["permutation"], payload["reference"], exec_step, low, high)
                if len(tau_values) == 0:
                    continue
                rows.append({"Dimension": "Intervention stage", "Category": label,
                             "Method": name, "Metric": "KendallTau",
                             "Value": round(float(tau_values.mean()), 3)})
                tau_records.append(pd.DataFrame({
                    "Dimension": "Intervention stage", "Category": label,
                    "Method": name, "Tau": tau_values}))

        frame = pd.DataFrame(rows)
        frame.to_csv(self.config.table_dir / "Table7_generalization.csv", index=False)

        tau_frame = pd.concat(tau_records, ignore_index=True)
        tau_frame.to_csv(self.config.figure_data_dir / "Figure6_Data.csv", index=False)
        figures.plot_tau_distribution(
            tau_frame, self.config.figure_dir / "Figure6_tau_group_stage.png")

        bars = frame[(frame["Dimension"] == "Patient group")
                     & (frame["Metric"] == "Accuracy")]
        figures.plot_two_factor(
            bars.rename(columns={"Category": "Condition", "Value": "Accuracy"}),
            self.config.figure_dir / "Figure9_generalization.png", "Patient group")
        return frame, tau_frame

    def run_robustness(self) -> pd.DataFrame:
        base = self.bundle["features"]
        rows = []

        def record(condition, setting, payload):
            rows.append({
                "Condition": condition, "Setting": setting,
                "Accuracy": round(payload["mean_accuracy"], 2),
                "Tau": round(float(np.mean(payload["tau"])), 3),
                "MAE": round(metrics.mae(payload["predicted_flatus"], payload["flatus"]), 2),
            })

        for level in [0.0, 0.10, 0.20]:
            record("Observation noise", f"sigma={level:.2f}",
                   self.train_variant(f"noise_{level}",
                                      features=self._perturb(base, noise=level)))
        for rate in [0.0, 0.10, 0.25]:
            record("Missing indicators", f"rate={rate:.2f}",
                   self.train_variant(f"missing_{rate}",
                                      features=self._perturb(base, missing=rate)))

        frame = pd.DataFrame(rows)
        frame.to_csv(self.config.table_dir / "Table8_robustness.csv", index=False)
        figures.plot_two_factor(
            pd.DataFrame({"Method": "Proposed Model", "Condition": frame["Setting"],
                          "Accuracy": frame["Accuracy"]}),
            self.config.figure_dir / "Figure10_robustness.png", "Perturbation level")
        return frame

    def _perturb(self, features: np.ndarray, noise: float = 0.0,
                 missing: float = 0.0) -> np.ndarray:
        rng = np.random.default_rng(self.config.seed + 7)
        out = features.copy()
        block = out[:, :, self.config.n_interventions:, :]
        if noise > 0:
            block += rng.normal(0.0, noise, block.shape)
        if missing > 0:
            block[rng.random(block.shape[:-1]) < missing] = 0.0
        return out

    # ------------------------------------------------------------------
    # intermediate process and checkpoint
    # ------------------------------------------------------------------
    def make_intermediate_figures(self) -> None:
        payload = self.results.get("Proposed Model", {})
        model = payload.get("model")
        if model is None:
            self.logger.warning("No trained model available")
            return

        out_dir = self.config.intermediate_dir
        cohort = self.cohort
        record = int(self.split.test[np.argmax(cohort.lengths[self.split.test])])
        horizon = int(cohort.lengths[record])

        figures.plot_conflict_matrix(cohort.conflict, out_dir / "int_conflict_matrix.png")
        figures.plot_graph_evolution(
            cohort.interval[record, :horizon], self.bundle["AI"][record, :horizon],
            self.bundle["AM"][record, :horizon], float(cohort.flatus[record]),
            self.config.step_hours, out_dir / "int_2_1_graph_evolution.png")
        figures.plot_granger_profile(
            self.bundle["AM"][record, :horizon], out_dir / "int_2_1_granger_profile.png")

        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(self.bundle["features"][record, :horizon]).float().unsqueeze(0)
            a_ii = torch.from_numpy(self.bundle["AI"][record, :horizon]).float().unsqueeze(0)
            a_im = torch.from_numpy(self.bundle["AM"][record, :horizon]).float().unsqueeze(0)
            steps = [model.graph(x[:, t], a_ii[:, t], a_im[:, t], self.config.n_interventions)
                     for t in range(horizon)]
            embeddings = torch.stack(steps, dim=1).squeeze(0).cpu().numpy()
            trace = model.contrib_trace

        figures.plot_message_passing(
            [entry.get("dependency", 0.0) for entry in trace],
            [entry.get("temporal", 0.0) for entry in trace],
            embeddings, out_dir / "int_2_2_message_passing.png")

        if payload.get("gates") is not None:
            figures.plot_gate_dynamics(payload["gates"], out_dir / "int_2_3_gate_dynamics.png")
        if payload.get("attention") is not None:
            figures.plot_attention_mask(payload["attention"].mean(axis=0),
                                        out_dir / "int_2_4_attention_mask.png")

        figures.plot_imputation(
            cohort.X[record, :horizon], self.bundle["scaled"][record, :horizon],
            self.bundle["was_missing"][record, :horizon], self.config.n_interventions,
            self.config.step_hours, out_dir / "int_2_5_imputation.png",
            self.bundle["indicator_mean"], self.bundle["indicator_std"])

        if len(payload.get("history", [])):
            figures.plot_training_curves(payload["history"],
                                         out_dir / "int_train_convergence.png")

    def save_checkpoint(self) -> None:
        model = self.results.get("Proposed Model", {}).get("model")
        if model is None:
            return
        torch.save({"state_dict": model.state_dict(), "config": self.config.to_dict()},
                   self.config.checkpoint_dir / "best_model.pth")

    def load_checkpoint(self, encoder="typed", decoder="pointer"):
        payload = torch.load(self.config.checkpoint_dir / "best_model.pth",
                             map_location="cpu", weights_only=False)
        model = SequencingModel(self.config, encoder=encoder, decoder=decoder)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
