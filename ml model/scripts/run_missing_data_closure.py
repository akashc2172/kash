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
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def run_cmd(cmd: list[str], cwd: Path, execute: bool) -> None:
    pretty = " ".join(cmd)
    print(f"[CMD] {pretty}")
    if not execute:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def load_retry_cache(audit_dir: Path) -> dict:
    p = audit_dir / "retry_policy_cache.json"
    if not p.exists():
        return {"entries": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"entries": {}}


def is_retry_eligible(entry: dict, force_recheck: bool) -> bool:
    if not entry:
        return True
    state = entry.get("state")
    if state == "terminal":
        return bool(force_recheck)
    cooldown_until = entry.get("cooldown_until")
    if not cooldown_until:
        return True
    try:
        cooldown_dt = datetime.fromisoformat(str(cooldown_until).replace("Z", "+00:00"))
        return datetime.now(timezone.utc) >= cooldown_dt
    except Exception:
        return True


def filter_manifest_for_endpoint(manifest_df: pd.DataFrame, endpoint: str, retry_cache: dict, force_recheck: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    skipped = []
    entries = retry_cache.get("entries", {})
    for _, r in manifest_df.iterrows():
        gid = str(r["game_id"])
        key = f"{gid}|{endpoint}"
        entry = entries.get(key, {})
        if is_retry_eligible(entry, force_recheck):
            rows.append(r)
        else:
            skipped.append(
                {
                    "game_id": gid,
                    "season": int(r["season"]),
                    "season_type": str(r["season_type"]),
                    "endpoint": endpoint,
                    "attempted_at": utc_now_iso(),
                    "result_class": "skipped_cached",
                }
            )
    out = pd.DataFrame(rows, columns=manifest_df.columns)
    skipped_df = pd.DataFrame(skipped)
    return out, skipped_df


def write_temp_game_ids(df: pd.DataFrame, temp_dir: Path, label: str) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    p = temp_dir / f"{label}.txt"
    with p.open("w", encoding="utf-8") as f:
        for gid in df["game_id"].astype(str).tolist():
            f.write(f"{gid}\n")
    return p


def append_ingest_attempts(audit_dir: Path, rows_df: pd.DataFrame) -> None:
    if rows_df.empty:
        return
    p = audit_dir / "ingest_attempts.csv"
    cols = ["game_id", "season", "season_type", "endpoint", "attempted_at", "result_class"]
    rows_df = rows_df[cols].copy()
    if p.exists():
        prev = pd.read_csv(p, dtype={"game_id": str})
        rows_df = pd.concat([prev, rows_df], ignore_index=True)
    rows_df.to_csv(p, index=False)


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
    parser.add_argument("--skip-bridges", action="store_true", help="Skip rebuilding manual scrape bridges.")
    parser.add_argument("--force-recheck", action="store_true", help="Ignore terminal/cooldown cache and retry all.")
    parser.add_argument("--audit-only", action="store_true", help="Run audit only (no ingest commands).")
    parser.add_argument("--probe-missing-api", action="store_true", help="Enable API probing in audit stage.")
    parser.add_argument("--probe-max-games", type=int, default=200, help="Max missing game IDs to probe per audit run.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    py = sys.executable
    cli_py = pick_cli_python(args.cli_python)

    # Always refresh manual scrape bridges unless explicitly skipped.
    # This makes in-progress NCAA.org scraping immediately visible in dual-source coverage.
    if not args.skip_bridges:
        run_cmd(
            [cli_py, "-m", "cbd_pbp.cli", "build-bridges", "--scrape-root", "data/manual_scrapes", "--out", args.db],
            cwd=repo_root,
            execute=args.execute,
        )

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
    if args.force_recheck:
        audit_cmd.append("--force-recheck")
    if args.probe_missing_api:
        audit_cmd.extend(["--probe-missing-api", "--probe-max-games", str(args.probe_max_games)])
    run_cmd(audit_cmd, cwd=repo_root, execute=args.execute)

    audit_dir = (repo_root / args.audit_dir).resolve()
    plays_manifest = pd.read_csv(audit_dir / "reingest_manifest_plays.csv")
    subs_manifest = pd.read_csv(audit_dir / "reingest_manifest_subs.csv")
    lineups_manifest = pd.read_csv(audit_dir / "reingest_manifest_lineups.csv")
    retry_cache = load_retry_cache(audit_dir)
    attempt_rows = []
    gate = json.loads((audit_dir / "model_readiness_gate.json").read_text(encoding="utf-8"))

    missing_postseason = (
        gate.get("checks", {})
        .get("postseason_manifest_complete", {})
        .get("missing_postseason_seasons", [])
    )

    if not args.audit_only and not args.skip_postseason_fetch:
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
                "--no-include-lineups",
                "--out",
                args.db,
            ]
            run_cmd(cmd, cwd=repo_root, execute=args.execute)

    # Endpoint-specific retry suppression.
    plays_retry, plays_skipped = filter_manifest_for_endpoint(plays_manifest, "plays", retry_cache, args.force_recheck)
    subs_retry, subs_skipped = filter_manifest_for_endpoint(subs_manifest, "subs", retry_cache, args.force_recheck)
    lineups_retry, lineups_skipped = filter_manifest_for_endpoint(lineups_manifest, "lineups", retry_cache, args.force_recheck)
    for df in [plays_skipped, subs_skipped, lineups_skipped]:
        if not df.empty and args.execute:
            attempt_rows.append(df)

    temp_dir = audit_dir / ".tmp_ingest_lists"

    # Pass 1: plays + participants
    for season, season_type in unique_season_type_pairs(plays_retry):
        season_df = plays_retry[(plays_retry["season"] == season) & (plays_retry["season_type"] == season_type)]
        list_file = write_temp_game_ids(season_df, temp_dir, f"plays_{season}_{season_type}")
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
            "--only-game-ids-file",
            str(list_file),
            "--out",
            args.db,
        ]
        if not args.audit_only:
            run_cmd(cmd, cwd=repo_root, execute=args.execute)
        if args.execute:
            attempt_rows.append(
                pd.DataFrame(
                    {
                        "game_id": season_df["game_id"].astype(str),
                        "season": season,
                        "season_type": season_type,
                        "endpoint": "plays",
                        "attempted_at": utc_now_iso(),
                        "result_class": "ingested",
                    }
                )
            )

    # Pass 2: substitutions
    for season, season_type in unique_season_type_pairs(subs_retry):
        season_df = subs_retry[(subs_retry["season"] == season) & (subs_retry["season_type"] == season_type)]
        list_file = write_temp_game_ids(season_df, temp_dir, f"subs_{season}_{season_type}")
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
            "--only-game-ids-file",
            str(list_file),
            "--out",
            args.db,
        ]
        if not args.audit_only:
            run_cmd(cmd, cwd=repo_root, execute=args.execute)
        if args.execute:
            attempt_rows.append(
                pd.DataFrame(
                    {
                        "game_id": season_df["game_id"].astype(str),
                        "season": season,
                        "season_type": season_type,
                        "endpoint": "subs",
                        "attempted_at": utc_now_iso(),
                        "result_class": "ingested",
                    }
                )
            )

    # Pass 3: lineups
    for season, season_type in unique_season_type_pairs(lineups_retry):
        season_df = lineups_retry[(lineups_retry["season"] == season) & (lineups_retry["season_type"] == season_type)]
        list_file = write_temp_game_ids(season_df, temp_dir, f"lineups_{season}_{season_type}")
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
            "--only-game-ids-file",
            str(list_file),
            "--out",
            args.db,
        ]
        if not args.audit_only:
            run_cmd(cmd, cwd=repo_root, execute=args.execute)
        if args.execute:
            attempt_rows.append(
                pd.DataFrame(
                    {
                        "game_id": season_df["game_id"].astype(str),
                        "season": season,
                        "season_type": season_type,
                        "endpoint": "lineups",
                        "attempted_at": utc_now_iso(),
                        "result_class": "ingested",
                    }
                )
            )

    if not args.audit_only and not args.skip_rebuild:
        run_cmd(
            [cli_py, "-m", "cbd_pbp.cli", "build-derived", "--season", "2025", "--season-type", "regular", "--out", args.db],
            cwd=repo_root,
            execute=args.execute,
        )

    if not args.audit_only and not args.skip_feature_repair:
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

    if not args.audit_only and not args.skip_rebuild:
        run_cmd([py, "college_scripts/build_prospect_career_store_v2.py"], cwd=repo_root, execute=args.execute)
        run_cmd([py, "nba_scripts/build_unified_training_table.py"], cwd=repo_root, execute=args.execute)

    if attempt_rows:
        append_ingest_attempts(audit_dir, pd.concat(attempt_rows, ignore_index=True))

    # Final audit/readiness gate refresh.
    if not args.audit_only:
        run_cmd(audit_cmd, cwd=repo_root, execute=args.execute)
    print("Missing-data closure workflow complete.")


if __name__ == "__main__":
    main()
