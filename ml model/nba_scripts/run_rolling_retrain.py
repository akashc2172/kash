#!/usr/bin/env python3
"""
Rolling Retrain Orchestrator
============================
Runs iterative season-anchored latent-model retraining windows.

Example:
python3 nba_scripts/run_rolling_retrain.py --start-anchor 2019 --end-anchor 2024
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = BASE_DIR / "nba_scripts" / "train_latent_model.py"


def run_cmd(cmd: list[str]) -> int:
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=BASE_DIR)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling retrain schedule for latent model")
    parser.add_argument("--start-anchor", type=int, default=2020, help="First test anchor season")
    parser.add_argument("--end-anchor", type=int, default=2024, help="Last test anchor season")
    parser.add_argument("--train-window", type=int, default=8, help="Train window size in seasons")
    parser.add_argument("--val-window", type=int, default=1, help="Validation window size in seasons")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temporal-decay-half-life", type=float, default=4.0)
    parser.add_argument("--temporal-decay-min", type=float, default=0.2)
    args = parser.parse_args()

    if args.end_anchor < args.start_anchor:
        raise ValueError("end-anchor must be >= start-anchor")

    failures = []
    for anchor in range(args.start_anchor, args.end_anchor + 1):
        test_start = anchor
        test_end = anchor
        val_end = anchor - 1
        val_start = max(val_end - args.val_window + 1, 2000)
        train_end = val_start - 1
        train_start = max(train_end - args.train_window + 1, 2000)

        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--train-start", str(train_start),
            "--train-end", str(train_end),
            "--val-start", str(val_start),
            "--val-end", str(val_end),
            "--test-start", str(test_start),
            "--test-end", str(test_end),
            "--temporal-decay-half-life", str(args.temporal_decay_half_life),
            "--temporal-decay-min", str(args.temporal_decay_min),
        ]
        rc = run_cmd(cmd)
        if rc != 0:
            failures.append(anchor)

    if failures:
        logger.error("Rolling retrain failed for anchors: %s", failures)
        raise SystemExit(1)
    logger.info("Rolling retrain completed successfully for all anchors.")


if __name__ == "__main__":
    main()

