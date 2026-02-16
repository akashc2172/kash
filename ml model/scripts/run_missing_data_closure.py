#!/usr/bin/env python3
"""
Execute missing-data closure workflow for full 2010-2025 model scope.

Default mode is dry-run. Pass --execute to run commands.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_cmd(cmd: list[str], cwd: Path, execute: bool) -> None:
    pretty = " ".join(cmd)
    print(f"[CMD] {pretty}")
    if not execute:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def pick_cli_python(explicit: str | None) -> str:
    if explicit:
        return explicit

    candidates = [sys.executable, "python3.13", "python3"]
    for candidate in candidates:
        try:
            subprocess.run(
                [candidate, "-c", "import typer"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return candidate
        except Exception:
            continue
    raise RuntimeError(
        "No Python interpreter with 'typer' available. "
        "Pass --cli-python with an environment that has project dependencies installed."
    )


def unique_season_type_pairs(df: pd.DataFrame, season_col: str = "season", type_col: str = "season_type") -> list[tuple[int, str]]:
    if df.empty:
        return []
    pairs = (
        df[[season_col, type_col]]
        .dropna()
        .drop_duplicates()
        .sort_values([season_col, type_col])
        .itertuples(index=False, name=None)
    )
    return [(int(s), str(t)) for s, t in pairs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run missing-data closure pipeline.")
    parser.add_argument("--db", default="data/warehouse.duckdb")
    parser.add_argument("--audit-dir", default="data/audit")
    parser.add_argument("--start-season", type=int, default=2010)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--execute", action="store_true", help="Actually execute commands (default is dry-run).")
    parser.add_argument("--cli-python", default=None, help="Python binary for cbd_pbp CLI commands.")
    parser.add_argument("--skip-postseason-fetch", action="store_true")
    parser.add_argument("--skip-feature-repair", action="store_true")
    parser.add_argument("--skip-rebuild", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    py = sys.executable
    cli_py = pick_cli_python(args.cli_python)

    audit_cmd = [
        py,
        "scripts/run_missing_data_audit.py",
        "--db",
        args.db,
        "--audit-dir",
        args.audit_dir,
        "--start-season",
        str(args.start_season),
        "--end-season",
        str(args.end_season),
    ]
    run_cmd(audit_cmd, cwd=repo_root, execute=args.execute)

    audit_dir = (repo_root / args.audit_dir).resolve()
    plays_manifest = pd.read_csv(audit_dir / "reingest_manifest_plays.csv")
    subs_manifest = pd.read_csv(audit_dir / "reingest_manifest_subs.csv")
    lineups_manifest = pd.read_csv(audit_dir / "reingest_manifest_lineups.csv")
    gate = json.loads((audit_dir / "model_readiness_gate.json").read_text(encoding="utf-8"))

    missing_postseason = (
        gate.get("checks", {})
        .get("postseason_manifest_complete", {})
        .get("missing_postseason_seasons", [])
    )

    if not args.skip_postseason_fetch:
        for season in sorted(int(s) for s in missing_postseason):
            cmd = [
                cli_py,
                "-m",
                "cbd_pbp.cli",
                "fetch-ingest",
                "--season",
                str(season),
                "--season-type",
                "postseason",
                "--include-lineups",
                "False",
                "--out",
                args.db,
            ]
            run_cmd(cmd, cwd=repo_root, execute=args.execute)

    # Pass 1: plays + participants
    for season, season_type in unique_season_type_pairs(plays_manifest):
        cmd = [
            cli_py,
            "-m",
            "cbd_pbp.cli",
            "resume-ingest-endpoints",
            "--season",
            str(season),
            "--season-type",
            season_type,
            "--endpoints",
            "plays",
            "--out",
            args.db,
        ]
        run_cmd(cmd, cwd=repo_root, execute=args.execute)

    # Pass 2: substitutions
    for season, season_type in unique_season_type_pairs(subs_manifest):
        cmd = [
            cli_py,
            "-m",
            "cbd_pbp.cli",
            "resume-ingest-endpoints",
            "--season",
            str(season),
            "--season-type",
            season_type,
            "--endpoints",
            "subs",
            "--out",
            args.db,
        ]
        run_cmd(cmd, cwd=repo_root, execute=args.execute)

    # Pass 3: lineups
    for season, season_type in unique_season_type_pairs(lineups_manifest):
        cmd = [
            cli_py,
            "-m",
            "cbd_pbp.cli",
            "resume-ingest-endpoints",
            "--season",
            str(season),
            "--season-type",
            season_type,
            "--endpoints",
            "lineups",
            "--out",
            args.db,
        ]
        run_cmd(cmd, cwd=repo_root, execute=args.execute)

    if not args.skip_rebuild:
        run_cmd(
            [cli_py, "-m", "cbd_pbp.cli", "build-derived", "--season", "2025", "--season-type", "regular", "--out", args.db],
            cwd=repo_root,
            execute=args.execute,
        )

    if not args.skip_feature_repair:
        run_cmd(
            [
                py,
                "scripts/repair_college_feature_store.py",
                "--input",
                "data/college_feature_store/college_features_v1.parquet",
                "--in-place",
                "--report",
                "data/audit/feature_store_repair_report.json",
            ],
            cwd=repo_root,
            execute=args.execute,
        )

    if not args.skip_rebuild:
        run_cmd([py, "college_scripts/build_prospect_career_store_v2.py"], cwd=repo_root, execute=args.execute)
        run_cmd([py, "nba_scripts/build_unified_training_table.py"], cwd=repo_root, execute=args.execute)

    # Final audit/readiness gate refresh.
    run_cmd(audit_cmd, cwd=repo_root, execute=args.execute)
    print("Missing-data closure workflow complete.")


if __name__ == "__main__":
    main()
