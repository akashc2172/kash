#!/usr/bin/env python3
"""
Repair college feature store integrity issues:
1) deterministically collapse duplicate (season, athlete_id, split_id) rows
2) backfill team_pace and conference from team-season references
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


KEY_COLS = ["season", "athlete_id", "split_id"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_team_season_reference(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "teamId"]
    base = df[df["teamId"].notna()].copy()

    pace_ref = (
        base[base["team_pace"].notna()]
        .groupby(keys, as_index=False)["team_pace"]
        .median()
        .rename(columns={"team_pace": "team_pace_ref"})
    )

    conf_mode = (
        base[base["conference"].notna()]
        .groupby(keys + ["conference"], as_index=False)
        .size()
        .sort_values(keys + ["size", "conference"], ascending=[True, True, False, True])
        .drop_duplicates(keys)
        .rename(columns={"conference": "conference_ref"})[keys + ["conference_ref"]]
    )

    ref = pace_ref.merge(conf_mode, on=keys, how="outer")
    return ref


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair duplicate keys and context nulls in college_features_v1.parquet.")
    parser.add_argument("--input", default="data/college_feature_store/college_features_v1.parquet")
    parser.add_argument("--output", default="data/college_feature_store/college_features_v1.repaired.parquet")
    parser.add_argument("--in-place", action="store_true", help="Overwrite --input with repaired output (creates .bak).")
    parser.add_argument("--report", default="data/audit/feature_store_repair_report.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_path = (repo_root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    output_path = (repo_root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    report_path = (repo_root / args.report).resolve() if not Path(args.report).is_absolute() else Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    before_rows = len(df)
    before_distinct = df[KEY_COLS].drop_duplicates().shape[0]
    before_dup_rows = before_rows - before_distinct
    before_null_team_pace = int(df["team_pace"].isna().sum())
    before_null_conference = int(df["conference"].isna().sum())

    # Fill context from team-season references.
    ref = build_team_season_reference(df)
    df = df.merge(ref, how="left", on=["season", "teamId"])
    df["team_pace"] = df["team_pace"].fillna(df["team_pace_ref"])
    df["conference"] = df["conference"].fillna(df["conference_ref"])
    df = df.drop(columns=["team_pace_ref", "conference_ref"])

    # Deterministic duplicate collapse.
    sort_cols = [c for c in ["season", "athlete_id", "split_id", "minutes_total", "shots_total", "fga_total"] if c in df.columns]
    ascending = [True, True, True] + [False] * (len(sort_cols) - 3)
    df = df.sort_values(sort_cols, ascending=ascending)
    df = df.drop_duplicates(subset=KEY_COLS, keep="first")

    after_rows = len(df)
    after_distinct = df[KEY_COLS].drop_duplicates().shape[0]
    after_dup_rows = after_rows - after_distinct
    after_null_team_pace = int(df["team_pace"].isna().sum())
    after_null_conference = int(df["conference"].isna().sum())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.in_place:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        input_path.replace(backup_path)
        df.to_parquet(input_path, index=False)
        written_to = str(input_path)
        backup_written = str(backup_path)
    else:
        df.to_parquet(output_path, index=False)
        written_to = str(output_path)
        backup_written = None

    report = {
        "generated_at": utc_now_iso(),
        "input_file": str(input_path),
        "output_file": written_to,
        "backup_file": backup_written,
        "before": {
            "rows": before_rows,
            "distinct_keys": before_distinct,
            "duplicate_rows": before_dup_rows,
            "null_team_pace": before_null_team_pace,
            "null_conference": before_null_conference,
        },
        "after": {
            "rows": after_rows,
            "distinct_keys": after_distinct,
            "duplicate_rows": after_dup_rows,
            "null_team_pace": after_null_team_pace,
            "null_conference": after_null_conference,
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote repaired file: {written_to}")
    if backup_written:
        print(f"Wrote backup file: {backup_written}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
