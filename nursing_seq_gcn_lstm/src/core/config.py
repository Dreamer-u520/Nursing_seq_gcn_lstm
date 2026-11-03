"""Run configuration.

Single source of truth for every path and hyper-parameter. Two presets are
provided: ``quick`` for a fast end-to-end pass and ``full`` for the evaluation
budget reported in the paper. Select with ``--profile``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # reproducibility
    seed: int = 2024
    n_seeds: int = 1
    profile: str = "quick"

    # cohort
    n_patients: int = 382
    split_ratio: Tuple[float, float, float] = (0.7, 0.1, 0.2)
    mean_age: float = 61.4
    sd_age: float = 9.7
    male_ratio: float = 0.586
    mean_flatus_hours: float = 52.5
    sd_flatus_hours: float = 11.3
    flatus_min: float = 22.0
    flatus_max: float = 88.0
    step_hours: float = 6.0
    max_steps: int = 20
    post_flatus_steps: int = 4

    # graph
    n_interventions: int = 8
    n_indicators: int = 6
    feature_dim: int = 16
    window_L: int = 4
    lag_q: int = 1
    f_alpha: float = 0.05

    # model
    gcn_dim: int = 32
    hidden_dim: int = 48
    pointer_dim: int = 32
    dropout: float = 0.10

    # optimisation
    epochs: int = 30
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 10
    pg_coef: float = 0.1
    entropy_coef: float = 0.01
    ema_decay: float = 0.95
    regression_coef: float = 0.3

    # evaluation
    lengths: Tuple[int, ...] = (3, 4, 5, 6, 7, 8)
    bootstrap_n: int = 300
    age_bins: Tuple[float, float] = (55.0, 70.0)
    stage_bounds: Tuple[int, int] = (7, 14)

    # layout
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    output_dir: Path = field(default_factory=lambda: ROOT / "outputs")

    @property
    def source_dir(self) -> Path:
        return self.data_dir / "source"

    @property
    def cohort_dir(self) -> Path:
        return self.data_dir / "cohort"

    @property
    def table_dir(self) -> Path:
        return self.output_dir / "experiment_results" / "tables"

    @property
    def figure_dir(self) -> Path:
        return self.output_dir / "experiment_results" / "figures"

    @property
    def figure_data_dir(self) -> Path:
        return self.output_dir / "experiment_results" / "figures_data"

    @property
    def intermediate_dir(self) -> Path:
        return self.output_dir / "intermediate_process"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs"

    def all_dirs(self) -> list[Path]:
        return [
            self.source_dir, self.cohort_dir, self.table_dir, self.figure_dir,
            self.figure_data_dir, self.intermediate_dir,
            self.checkpoint_dir, self.log_dir,
        ]

    def to_dict(self) -> dict:
        return {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in asdict(self).items()
        }

    def apply_profile(self, profile: str) -> "Config":
        self.profile = profile
        if profile == "quick":
            self.n_seeds = 1
            self.epochs = 30
            self.gcn_dim, self.hidden_dim, self.pointer_dim = 32, 48, 32
            self.batch_size, self.lr = 64, 3e-3
            self.bootstrap_n = 300
        elif profile == "full":
            self.n_seeds = 5
            self.epochs = 60
            self.gcn_dim, self.hidden_dim, self.pointer_dim = 64, 128, 64
            self.batch_size, self.lr = 32, 1e-3
            self.bootstrap_n = 2000
        else:
            raise ValueError(f"unknown profile: {profile}")
        return self


def load_config(profile: str = "quick") -> Config:
    return Config().apply_profile(profile)
