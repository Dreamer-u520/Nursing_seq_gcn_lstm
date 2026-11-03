"""Entry point. Runs the full experiment programme.

    python main.py                     # quick profile, single pass
    python main.py --profile full      # full budget, five seeds
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.core.config import load_config
from src.core.runtime import build_logger, prepare_dirs, save_json, set_seed
from src.data.dataset import build_bundle, persist_cohort
from src.experiments.runner import Runner

BLOCKS = ["ablation", "decision", "cost", "sensitivity", "generalization", "robustness"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nursing intervention sequencing")
    parser.add_argument("--profile", default="quick", choices=["quick", "full"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip", nargs="*", default=[], choices=BLOCKS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.profile)
    if args.seed is not None:
        config.seed = args.seed

    prepare_dirs(config)
    logger = build_logger(config.log_dir / "run.log")
    set_seed(config.seed)

    logger.info("=" * 68)
    logger.info("Intervention sequencing | profile=%s | seed=%d | torch %s",
                config.profile, config.seed, torch.__version__)
    logger.info("=" * 68)
    started = time.time()

    try:
        logger.info("Stage 1 | cohort and graph operators")
        bundle = build_bundle(config, logger)
        persist_cohort(config, bundle, logger)
        runner = Runner(config, bundle, logger)

        logger.info("Stage 2 | dataset characteristics")
        runner.table_dataset()

        logger.info("Stage 3 | main comparison")
        runner.run_comparison()

        logger.info("Stage 4 | significance testing")
        runner.table_significance().to_csv(
            config.table_dir / "Table2_significance_tests.csv", index=False)

        logger.info("Stage 5 | comparison figures")
        runner.figure_accuracy_length()
        runner.figure_flatus()

        stages = [("ablation", "ablation study", runner.run_ablation),
                  ("decision", "decision modules", runner.run_decision_modules),
                  ("cost", "computational cost", runner.table_cost),
                  ("sensitivity", "parameter sensitivity", runner.run_sensitivity),
                  ("generalization", "generalisation", runner.run_generalization),
                  ("robustness", "robustness", runner.run_robustness)]
        for index, (key, label, action) in enumerate(stages, start=6):
            if key in args.skip:
                continue
            logger.info("Stage %d | %s", index, label)
            action()

        logger.info("Stage 12 | intermediate process figures and checkpoint")
        runner.make_intermediate_figures()
        runner.save_checkpoint()

        payload = runner.results["Proposed Model"]
        elapsed = time.time() - started
        summary = {
            "profile": config.profile,
            "seed": config.seed,
            "data_source": bundle["provenance"],
            "records": int(len(bundle["cohort"])),
            "split": [int(len(bundle["split"].train)), int(len(bundle["split"].val)),
                      int(len(bundle["split"].test))],
            "elapsed_seconds": round(elapsed, 2),
            "proposed_accuracy": round(float(payload["mean_accuracy"]), 2),
            "proposed_tau": round(float(payload["tau"].mean()), 3),
            "proposed_mae": round(float(np.mean(np.abs(
                payload["predicted_flatus"] - payload["flatus"]))), 3),
            "config": config.to_dict(),
        }
        save_json(summary, config.log_dir / "run_summary.json")

        logger.info("=" * 68)
        logger.info("Completed in %.1f s | accuracy %.2f%% | tau %.3f | MAE %.2f h",
                    elapsed, summary["proposed_accuracy"], summary["proposed_tau"],
                    summary["proposed_mae"])
        logger.info("Tables  -> %s", config.table_dir)
        logger.info("Figures -> %s", config.figure_dir)
        logger.info("=" * 68)
        return 0

    except Exception:
        logger.error("Run failed:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
