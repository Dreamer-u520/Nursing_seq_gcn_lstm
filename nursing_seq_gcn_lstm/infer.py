"""Inference from a saved checkpoint.

    python infer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.core.config import load_config
from src.core.runtime import build_logger, prepare_dirs, set_seed
from src.data.dataset import CohortDataset, build_bundle
from src.evaluation import metrics
from src.models.network import SequencingModel
from src.training.loop import evaluate, make_loader


def main() -> int:
    config = load_config("quick")
    prepare_dirs(config)
    logger = build_logger(config.log_dir / "infer.log")
    set_seed(config.seed)

    checkpoint = config.checkpoint_dir / "best_model.pth"
    if not checkpoint.exists():
        logger.error("Checkpoint missing at %s; run main.py first", checkpoint)
        return 1

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    bundle = build_bundle(config, logger)

    model = SequencingModel(config, encoder="typed", decoder="pointer")
    model.load_state_dict(payload["state_dict"])
    model.eval()

    dataset = CohortDataset(bundle["cohort"], bundle["features"],
                            (bundle["AI"], bundle["AM"]), bundle["split"].test, config)
    out = evaluate(model, make_loader(dataset, config.batch_size, False))

    accuracy = metrics.mean_accuracy(out["permutation"], out["reference"], config.lengths)
    tau = metrics.kendall_tau(out["permutation"], out["reference"])
    error = metrics.mae(out["predicted_flatus"], out["flatus"])

    frame = pd.DataFrame({
        "RecordIndex": dataset.index,
        "ReferenceOrder": ["-".join(map(str, row)) for row in out["reference"]],
        "PredictedOrder": ["-".join(map(str, row)) for row in out["permutation"]],
        "RecordedFlatus": np.round(out["flatus"], 2),
        "EstimatedFlatus": np.round(out["predicted_flatus"], 2),
        "KendallTau": np.round(tau, 4),
    })
    target = config.table_dir / "Inference_predictions.csv"
    frame.to_csv(target, index=False)

    logger.info("Accuracy %.2f%% | tau %.3f | MAE %.2f h", accuracy, float(tau.mean()), error)
    logger.info("Predictions written to %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
