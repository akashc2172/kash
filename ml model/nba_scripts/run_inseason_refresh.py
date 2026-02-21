#!/usr/bin/env python3
"""
In-season refresh orchestrator (no retraining).

Behavior:
1) Select a locked latent model checkpoint (latest by default).
2) Run inference table build + prediction + ranking export.
3) Write a compact audit artifact with paths and basic quality checks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
AUDIT_DIR = BASE_DIR / "data" / "audit"
INFERENCE_DIR = BASE_DIR / "data" / "inference"
INFER_SCRIPT = BASE_DIR / "nba_scripts" / "nba_prospect_inference.py"


def _latest_latent_model() -> Path | None:
    model_files = sorted(
        [p / "model.pt" for p in MODELS_DIR.glob("latent_model_*") if (p / "model.pt").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return model_files[0] if model_files else None


def _latest_pred_file() -> Path | None:
    files = sorted(
        INFERENCE_DIR.glob("prospect_predictions_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _run_cmd(cmd: list[str]) -> Dict[str, Any]:
    started = datetime.utcnow()
    proc = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
    ended = datetime.utcnow()
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "started_utc": started.isoformat() + "Z",
        "ended_utc": ended.isoformat() + "Z",
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-40:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run in-season refresh using latest locked model")
    parser.add_argument("--model-path", type=str, default="", help="Optional explicit model.pt path")
    parser.add_argument("--split-id", type=str, default="ALL__ALL")
    parser.add_argument("--skip-rank-export", action="store_true", help="Skip CSV/XLSX ranking export")
    args = parser.parse_args()

    model_path = Path(args.model_path) if args.model_path else _latest_latent_model()
    if model_path is None or not model_path.exists():
        print("No model checkpoint found. Provide --model-path or train a latent model first.", file=sys.stderr)
        return 2

    recal_path = model_path.parent / "season_recalibration.json"
    cmd = [
        sys.executable,
        str(INFER_SCRIPT),
        "--split-id",
        str(args.split_id),
        "--model-path",
        str(model_path),
    ]
    if recal_path.exists():
        cmd.extend(["--recalibration-path", str(recal_path)])
    if args.skip_rank_export:
        cmd.append("--skip-rank-export")

    run = _run_cmd(cmd)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_json = AUDIT_DIR / f"inseason_refresh_{stamp}.json"

    out: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "model_path": str(model_path),
        "recalibration_path": str(recal_path) if recal_path.exists() else None,
        "run": run,
    }

    # Post-run quick checks
    pred = _latest_pred_file()
    out["latest_prediction_file"] = str(pred) if pred else None
    if pred and pred.exists():
        df = pd.read_parquet(pred)
        out["prediction_rows"] = int(len(df))
        out["prediction_columns"] = int(df.shape[1])
        for c in ["pred_rank_score", "pred_rank_target", "pred_year1_epm", "pred_peak_rapm", "pred_dev_rate"]:
            if c in df.columns:
                out[f"{c}_non_null_rate"] = float(pd.to_numeric(df[c], errors="coerce").notna().mean())
    else:
        out["prediction_rows"] = 0
        out["prediction_columns"] = 0

    # Export paths
    expected = [
        INFERENCE_DIR / "season_rankings_latest_best_current.csv",
        INFERENCE_DIR / "season_rankings_latest_best_current_qualified.csv",
        INFERENCE_DIR / "season_rankings_latest_best_current_matched.csv",
        INFERENCE_DIR / "season_rankings_latest_best_current_matched_qualified.csv",
        INFERENCE_DIR / "season_rankings_top25_best_current_tabs.xlsx",
    ]
    out["exports"] = {str(p): p.exists() for p in expected}

    out["passed"] = bool(run["returncode"] == 0 and out["prediction_rows"] > 0)
    audit_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"audit": str(audit_json), "passed": out["passed"], "model_path": str(model_path)}, indent=2))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

