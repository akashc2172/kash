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
from datetime import datetime
import json
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = BASE_DIR / "nba_scripts" / "train_latent_model.py"
MODELS_DIR = BASE_DIR / "models"
AUDIT_DIR = BASE_DIR / "data" / "audit"


def run_cmd(cmd: list[str]) -> int:
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=BASE_DIR)
    return proc.returncode


def _latest_latent_dir(before: set[str]) -> Path | None:
    dirs = sorted([p for p in MODELS_DIR.glob("latent_model_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    for d in reversed(dirs):
        if d.name not in before:
            return d
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling retrain schedule for latent model")
    parser.add_argument("--start-anchor", type=int, default=2020, help="First test anchor season")
    parser.add_argument("--end-anchor", type=int, default=2024, help="Last test anchor season")
    parser.add_argument("--train-window", type=int, default=8, help="Train window size in seasons")
    parser.add_argument("--expanding-window", action=argparse.BooleanOptionalAction, default=True, help="Use expanding train window (active loop)")
    parser.add_argument("--base-train-start", type=int, default=2010, help="Earliest train season for expanding mode")
    parser.add_argument("--val-window", type=int, default=1, help="Validation window size in seasons")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--objective-profile", choices=["epm_first", "rapm_first", "balanced"], default="epm_first")
    parser.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=True, help="Warm-start each year from prior year model")
    parser.add_argument("--allow-warm-start-expanding", action=argparse.BooleanOptionalAction, default=False, help="Override safeguard and allow warm-start with expanding windows")
    parser.add_argument("--temporal-decay-half-life", type=float, default=4.0)
    parser.add_argument("--temporal-decay-min", type=float, default=0.2)
    parser.add_argument("--rapm-min-nba-seasons", type=int, default=3, help="RAPM maturity gate (passed to trainer)")
    args = parser.parse_args()

    if args.end_anchor < args.start_anchor:
        raise ValueError("end-anchor must be >= start-anchor")

    failures = []
    run_rows = []
    prev_checkpoint = None
    effective_warm_start = bool(args.warm_start)
    if args.expanding_window and args.warm_start and not args.allow_warm_start_expanding:
        effective_warm_start = False
        logger.info("Warm-start disabled for expanding-window mode (set --allow-warm-start-expanding to override).")
    for anchor in range(args.start_anchor, args.end_anchor + 1):
        test_start = anchor
        test_end = anchor
        val_end = anchor - 1
        val_start = max(val_end - args.val_window + 1, 2000)
        train_end = val_start - 1
        if args.expanding_window:
            train_start = max(args.base_train_start, 2000)
        else:
            train_start = max(train_end - args.train_window + 1, 2000)

        before = {p.name for p in MODELS_DIR.glob("latent_model_*") if p.is_dir()}

        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--objective-profile", str(args.objective_profile),
            "--rapm-min-nba-seasons", str(int(args.rapm_min_nba_seasons)),
            "--asof-year", str(test_start),
            "--train-start", str(train_start),
            "--train-end", str(train_end),
            "--val-start", str(val_start),
            "--val-end", str(val_end),
            "--test-start", str(test_start),
            "--test-end", str(test_end),
            "--temporal-decay-half-life", str(args.temporal_decay_half_life),
            "--temporal-decay-min", str(args.temporal_decay_min),
        ]
        if effective_warm_start and prev_checkpoint:
            cmd.extend(["--init-model-path", str(prev_checkpoint)])
        rc = run_cmd(cmd)
        model_dir = _latest_latent_dir(before)
        model_cfg = {}
        eval_metrics = {}
        if model_dir is not None:
            cfg_path = model_dir / "model_config.json"
            eval_path = model_dir / "eval_metrics.json"
            if cfg_path.exists():
                try:
                    model_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                except Exception:
                    model_cfg = {}
            if eval_path.exists():
                try:
                    eval_metrics = json.loads(eval_path.read_text(encoding="utf-8"))
                except Exception:
                    eval_metrics = {}
            ckpt_best = model_dir / "model_best.pt"
            ckpt_final = model_dir / "model.pt"
            prev_checkpoint = ckpt_best if ckpt_best.exists() else (ckpt_final if ckpt_final.exists() else None)

        run_rows.append(
            {
                "anchor_season": anchor,
                "returncode": rc,
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
                "expanding_window": int(bool(args.expanding_window)),
                "warm_start_effective": int(bool(effective_warm_start and prev_checkpoint is not None)),
                "objective_profile": args.objective_profile,
                "model_dir": str(model_dir) if model_dir else "",
                "monitor_metric": model_cfg.get("monitor_metric"),
                "best_epoch": eval_metrics.get("best_epoch"),
                "best_monitor": eval_metrics.get("best_monitor"),
                "test_epm_rmse": (eval_metrics.get("test") or {}).get("epm_rmse"),
                "test_epm_corr": (eval_metrics.get("test") or {}).get("epm_corr"),
                "test_rapm_rmse": (eval_metrics.get("test") or {}).get("rapm_rmse"),
                "test_rapm_corr": (eval_metrics.get("test") or {}).get("rapm_corr"),
            }
        )
        if rc != 0:
            failures.append(anchor)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = AUDIT_DIR / f"rolling_retrain_report_{stamp}.csv"
    out_json = AUDIT_DIR / f"rolling_retrain_report_{stamp}.json"
    pd.DataFrame(run_rows).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(run_rows, indent=2), encoding="utf-8")
    logger.info("Saved rolling report: %s", out_csv)

    if failures:
        logger.error("Rolling retrain failed for anchors: %s", failures)
        raise SystemExit(1)
    logger.info("Rolling retrain completed successfully for all anchors.")


if __name__ == "__main__":
    main()
