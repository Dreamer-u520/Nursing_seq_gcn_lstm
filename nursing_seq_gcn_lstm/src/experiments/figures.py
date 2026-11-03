"""Figure production. Each figure is drawn from arrays produced by the run."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..evaluation import metrics
from ..data.cohort import INDICATOR_NAMES, INTERVENTION_NAMES

PALETTE = {
    "Proposed Model": "#1B4F8A",
    "GCN-LSTM-Homogeneous": "#2E8B8B",
    "GAT+LSTM": "#6A9E3F",
    "Static GCN+LSTM": "#D47C0F",
    "Transformer": "#E05A5A",
    "Conflict-Matrix Greedy Rule": "#444444",
    "Seq2Seq+Attn": "#B05CB0",
    "LSTM": "#A0522D",
    "CatBoost": "#708090",
    "Random Forest": "#B0B0B0",
}

MARKERS = {
    "Proposed Model": "o", "GCN-LSTM-Homogeneous": "s", "GAT+LSTM": "^",
    "Static GCN+LSTM": "D", "Transformer": "p", "Conflict-Matrix Greedy Rule": "P",
    "Seq2Seq+Attn": "v", "LSTM": "h", "CatBoost": "X", "Random Forest": "*",
}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 14, "axes.linewidth": 1.1,
        "savefig.dpi": 600, "figure.dpi": 120, "axes.unicode_minus": False,
    })


def save(fig, path: Path, tight: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_accuracy_by_length(summary: pd.DataFrame, path: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6.4))
    for method, block in summary.groupby("Method", sort=False):
        block = block.sort_values("Length")
        color, marker = PALETTE.get(method, "#333333"), MARKERS.get(method, "o")
        if block["Std"].isna().all():
            ax.plot(block["Length"], block["Mean"], label=method, color=color,
                    marker=marker, linewidth=2.0, linestyle=(0, (5, 2)), markersize=8)
        else:
            ax.errorbar(block["Length"], block["Mean"], yerr=block["Std"], label=method,
                        color=color, marker=marker, linewidth=1.8, markersize=7,
                        capsize=3, elinewidth=1.0)
    ax.set_xlabel("Intervention Sequence Length (Nodes)")
    ax.set_ylabel("Exact-Match Sequence Accuracy (%)")
    ax.set_xticks(sorted(summary["Length"].unique()))
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, ncol=2, loc="lower left", framealpha=0.92,
              edgecolor="#888888")
    return save(fig, path)


def plot_tau_distribution(records: pd.DataFrame, path: Path) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0))

    for ax, dimension in zip(axes, ["Patient group", "Intervention stage"]):
        block = records[records["Dimension"] == dimension]
        categories = list(dict.fromkeys(block["Category"]))
        methods = list(dict.fromkeys(block["Method"]))
        positions, data, colors = [], [], []
        offset = 0.0
        for category in categories:
            for index, method in enumerate(methods):
                values = block[(block["Category"] == category)
                               & (block["Method"] == method)]["Tau"].values
                if len(values) == 0:
                    continue
                positions.append(offset + index * 0.8)
                data.append(values)
                colors.append(PALETTE.get(method, "#333333"))
            offset += len(methods) * 0.8 + 1.0

        box = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                         showfliers=False,
                         medianprops={"color": "black", "linewidth": 1.2})
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
            patch.set_edgecolor("#333333")

        centers = [positions[i] + (len(methods) - 1) * 0.4
                   for i in range(0, len(positions), len(methods))]
        ax.set_xticks(centers)
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylabel(r"Kendall's $\tau$")
        ax.set_title(dimension, fontsize=13)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#CCCCCC")
        ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE.get(m, "#333333"),
                             alpha=0.65, edgecolor="#333333") for m in methods]
    fig.legend(handles, methods, loc="lower center", ncol=len(methods), fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return save(fig, path, tight=False)


def plot_flatus_scatter(recorded: np.ndarray, predictions: Dict[str, np.ndarray],
                        path: Path) -> Path:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    low, high = np.min(recorded) - 4, np.max(recorded) + 4
    line = np.linspace(low, high, 200)

    for ax, (name, predicted) in zip(axes.flatten(), predictions.items()):
        error = np.abs(recorded - predicted)
        colors = np.where(error < 3, "#009E73",
                          np.where(error < 6, "#E69F00", "#CC79A7"))
        ax.set_axisbelow(True)
        ax.grid(linestyle="--", linewidth=0.7, color="#DDDDDD")
        ax.fill_between(line, line - 6, line + 6, color="#BBBBBB", alpha=0.35, zorder=1)
        ax.plot(line, line, color="#333333", linestyle="--", linewidth=1.4, zorder=2)
        ax.scatter(recorded, predicted, c=colors, s=38, alpha=0.85,
                   edgecolors="white", linewidths=0.4, zorder=3)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("Recorded First Flatus Time (h)")
        ax.set_ylabel("Estimated First Flatus Time (h)")
        ax.set_title(name, fontsize=13)
        ax.text(0.04, 0.96,
                f"MAE = {metrics.mae(predicted, recorded):.1f} h\n"
                f"$R^2$ = {metrics.r2_score(predicted, recorded):.2f}\n"
                f"$n$ = {len(recorded)}",
                transform=ax.transAxes, fontsize=10.5, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#AAAAAA", alpha=0.85))

    fig.tight_layout()
    return save(fig, path, tight=False)


def plot_two_factor(frame: pd.DataFrame, path: Path, x_label: str,
                    y_label: str = "Mean Sequence Accuracy (%)") -> Path:
    apply_style()
    groups = list(dict.fromkeys(frame["Condition"]))
    methods = list(dict.fromkeys(frame["Method"]))
    x = np.arange(len(groups))
    width = 0.78 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(max(9.0, 1.1 * len(groups)), 5.2))
    for index, method in enumerate(methods):
        values, errors = [], []
        for group in groups:
            block = frame[(frame["Condition"] == group) & (frame["Method"] == method)]
            values.append(float(block["Accuracy"].mean()) if len(block) else np.nan)
            errors.append(float(block["Accuracy"].std(ddof=1)) if len(block) > 1 else 0.0)
        ax.bar(x + index * width - 0.39 + width / 2, values, width * 0.9,
               yerr=errors, capsize=2.5, label=method,
               color=PALETTE.get(method, "#1B4F8A"), alpha=0.85,
               edgecolor="#333333", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#CCCCCC")
    ax.set_axisbelow(True)
    if len(methods) > 1:
        ax.legend(fontsize=9, ncol=min(4, len(methods)), frameon=False)
    fig.tight_layout()
    return save(fig, path, tight=False)


def plot_cost_bars(frame: pd.DataFrame, path: Path) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    order = frame["Model"].tolist()
    positions = np.arange(len(order))

    for ax, column, label, color in [
        (axes[0], "ParametersM", "Parameters (M)", "#1B4F8A"),
        (axes[1], "InferenceMsPerCase", "Inference time (ms/case)", "#D47C0F"),
    ]:
        ax.barh(positions, frame[column], color=color, alpha=0.85,
                edgecolor="#333333", linewidth=0.5)
        ax.set_yticks(positions)
        ax.set_yticklabels(order, fontsize=9)
        ax.set_xlabel(label)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", linewidth=0.7, color="#CCCCCC")
        ax.set_axisbelow(True)

    fig.tight_layout()
    return save(fig, path, tight=False)


# ----------------------------------------------------------------------
# intermediate process figures
# ----------------------------------------------------------------------
def plot_graph_evolution(interval, a_ii, a_im, flatus, step_hours, path: Path) -> Path:
    apply_style()
    steps = interval.shape[0]
    picks = np.linspace(0, steps - 1, min(steps, 6)).astype(int)

    fig, axes = plt.subplots(2, len(picks), figsize=(3.0 * len(picks), 6.2))
    for column, t in enumerate(picks):
        ax = axes[0, column]
        image = ax.imshow(a_ii[t], cmap="YlGnBu", vmin=0, vmax=max(a_ii.max(), 1e-6))
        ax.set_title(f"step {t} | {t * step_hours:.0f} h", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)

        ax = axes[1, column]
        image = ax.imshow(a_im[t], cmap="OrRd", vmin=0, vmax=max(a_im.max(), 1e-6),
                          aspect="auto")
        ax.set_xlabel("indicator", fontsize=9)
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)

    axes[0, 0].set_ylabel(r"$A^{II}_t$", fontsize=12)
    axes[1, 0].set_ylabel(r"$A^{IM}_t$", fontsize=12)
    return save(fig, path, tight=False)


def plot_granger_profile(a_im: np.ndarray, path: Path) -> Path:
    apply_style()
    steps = a_im.shape[0]
    pairs = [(0, 0), (1, 1), (3, 0), (5, 5)]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for (item, indicator), color in zip(pairs, ["#1B4F8A", "#2E8B8B", "#D47C0F", "#B05CB0"]):
        ax.plot(np.arange(steps), a_im[:, item, indicator], marker="o", markersize=4,
                linewidth=1.6, color=color,
                label=f"{INTERVENTION_NAMES[item]} / {INDICATOR_NAMES[indicator]}")
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"$w^{IM}_{ik}(t)$")
    ax.grid(linestyle="--", linewidth=0.7, color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    return save(fig, path, tight=False)


def plot_message_passing(dependency: Sequence[float], temporal: Sequence[float],
                         embeddings: np.ndarray, path: Path) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    axes[0].plot(dependency, marker="o", color="#1B4F8A", label="dependency path")
    axes[0].plot(temporal, marker="s", color="#D47C0F", label="temporal path")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Mean absolute activation")
    axes[0].grid(linestyle="--", linewidth=0.7, color="#CCCCCC")
    axes[0].legend(fontsize=9, frameon=False)

    flat = embeddings.reshape(-1, embeddings.shape[-1])
    centered = flat - flat.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh[:2].T
    axes[1].scatter(projected[:, 0], projected[:, 1],
                    c=np.repeat(np.arange(embeddings.shape[0]), embeddings.shape[1]),
                    cmap="viridis", s=22, alpha=0.85)
    axes[1].set_xlabel("component 1")
    axes[1].set_ylabel("component 2")
    axes[1].set_title("Structural embedding projection", fontsize=12)
    axes[1].grid(linestyle="--", linewidth=0.7, color="#CCCCCC")
    fig.tight_layout()
    return save(fig, path, tight=False)


def plot_gate_dynamics(gates: np.ndarray, path: Path) -> Path:
    apply_style()
    mean_gate, std_gate = gates.mean(axis=0), gates.std(axis=0)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    steps = np.arange(mean_gate.shape[0])
    for index, (label, color) in enumerate(zip(["forget gate", "write gate", "output gate"],
                                               ["#1B4F8A", "#D47C0F", "#2E8B8B"])):
        ax.plot(steps, mean_gate[:, index], marker="o", color=color, label=label)
        ax.fill_between(steps, mean_gate[:, index] - std_gate[:, index],
                        mean_gate[:, index] + std_gate[:, index], color=color, alpha=0.12)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Gate activation")
    ax.grid(linestyle="--", linewidth=0.7, color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    return save(fig, path, tight=False)


def plot_attention_mask(attention: np.ndarray, path: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(attention, cmap="magma", aspect="auto")
    ax.set_xlabel("Candidate intervention index")
    ax.set_ylabel("Decoding step")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    return save(fig, path, tight=False)


def plot_imputation(raw: np.ndarray, filled: np.ndarray, missing: np.ndarray,
                    n_int: int, step_hours: float, path: Path,
                    mean=None, std=None) -> Path:
    apply_style()
    horizon = raw.shape[0]
    hours = np.arange(horizon) * step_hours
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2))

    for index, ax in enumerate(axes.flatten()[:6]):
        observed_values = raw[:, n_int + index].copy()
        if mean is not None and std is not None:
            observed_values = (observed_values - mean[index]) / std[index]
        observed_values[missing[:, index]] = np.nan

        ax.plot(hours, filled[:, n_int + index], color="#1B4F8A",
                linewidth=1.6, label="imputed trajectory")
        ax.scatter(hours, observed_values, color="#D47C0F", s=22, zorder=3,
                   label="observed")
        gaps = np.where(missing[:, index])[0]
        if len(gaps):
            ax.scatter(hours[gaps], filled[gaps, n_int + index], marker="x",
                       color="#B03060", s=42, zorder=4, label="carried forward")
        ax.set_title(INDICATOR_NAMES[index], fontsize=10)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Standardised value")
        ax.grid(linestyle="--", linewidth=0.6, color="#DDDDDD")
        if index == 0:
            ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    return save(fig, path, tight=False)


def plot_training_curves(history: pd.DataFrame, path: Path) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    specs = [("train_loss", "Training loss", "o", "#1B4F8A", "Epoch"),
             ("reward", r"Sampled return (Kendall's $\tau$)", "s", "#D47C0F", "Epoch"),
             ("val_accuracy", "Validation accuracy", "^", "#2E8B8B", "%")]
    for ax, (column, title, marker, color, ylabel) in zip(axes, specs):
        ax.plot(history["epoch"], history[column], marker=marker, color=color)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(linestyle="--", linewidth=0.7, color="#CCCCCC")
    fig.tight_layout()
    return save(fig, path, tight=False)


def plot_conflict_matrix(matrix: np.ndarray, path: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    image = ax.imshow(matrix, cmap="YlOrRd")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xlabel("Pending intervention")
    ax.set_ylabel("Executed intervention")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    return save(fig, path, tight=False)
