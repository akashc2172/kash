#!/usr/bin/env python3
"""
Canonical missing-data audit for the full 2010-2025 model scope.

Outputs under data/audit/:
  - missing_games_by_endpoint.csv
  - missing_games_by_season.csv
  - feature_store_integrity_report.json
  - reingest_manifest_plays.csv
  - reingest_manifest_subs.csv
  - reingest_manifest_lineups.csv
  - model_readiness_gate.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


ENDPOINT_TABLES = {
    "plays": "stg_plays",
    "participants": "stg_participants",
    "subs": "stg_subs",
    "lineups": "stg_lineups",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_dim_games(con: duckdb.DuckDBPyConnection, start_season: int, end_season: int) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            CAST(id AS VARCHAR) AS game_id,
            id AS game_id_int,
            season,
            seasonType AS season_type
        FROM dim_games
        WHERE season BETWEEN ? AND ?
        """,
        [start_season, end_season],
    ).fetchdf()


def load_distinct_game_ids(con: duckdb.DuckDBPyConnection, table_name: str, game_col: str = "gameId") -> pd.DataFrame:
    return con.execute(f"SELECT DISTINCT {game_col} AS game_id FROM {table_name}").fetchdf()


def build_coverage(dim_games: pd.DataFrame, have_game_ids: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = dim_games.merge(have_game_ids, how="left", on="game_id", indicator=True)
    merged["covered"] = (merged["_merge"] == "both").astype(int)
    missing = merged[merged["covered"] == 0][["game_id", "game_id_int", "season", "season_type"]].copy()

    summary = (
        merged.groupby(["season_type", "season"], as_index=False)
        .agg(expected_games=("game_id", "count"), covered_games=("covered", "sum"))
    )
    summary["missing_games"] = summary["expected_games"] - summary["covered_games"]
    summary["coverage_rate"] = summary["covered_games"] / summary["expected_games"].clip(lower=1)
    return summary, missing


def compute_feature_store_integrity(feature_path: Path) -> dict:
    con = duckdb.connect()
    try:
        q = con.execute(
            f"""
            SELECT
                COUNT(*) AS total_rows,
                COUNT(DISTINCT season || '|' || athlete_id || '|' || split_id) AS distinct_keys,
                COUNT(*) - COUNT(DISTINCT season || '|' || athlete_id || '|' || split_id) AS duplicate_rows,
                SUM(CASE WHEN team_pace IS NULL THEN 1 ELSE 0 END) AS null_team_pace,
                SUM(CASE WHEN conference IS NULL THEN 1 ELSE 0 END) AS null_conference
            FROM read_parquet('{feature_path.as_posix()}')
            """
        ).fetchone()

        dup_groups = con.execute(
            f"""
            SELECT COUNT(*) AS duplicate_groups
            FROM (
                SELECT season, athlete_id, split_id, COUNT(*) AS n
                FROM read_parquet('{feature_path.as_posix()}')
                GROUP BY 1,2,3
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    finally:
        con.close()

    total_rows = int(q[0])
    return {
        "total_rows": total_rows,
        "distinct_keys": int(q[1]),
        "duplicate_rows": int(q[2]),
        "duplicate_groups": int(dup_groups),
        "null_team_pace": int(q[3]),
        "null_team_pace_rate": float(q[3] / total_rows if total_rows else 0.0),
        "null_conference": int(q[4]),
        "null_conference_rate": float(q[4] / total_rows if total_rows else 0.0),
    }


def compute_target_coverage(rapm_csv: Path, year1_epm_parquet: Path) -> dict:
    con = duckdb.connect()
    try:
        rapm_seasons = [
            int(r[0])
            for r in con.execute(
                f"SELECT DISTINCT season FROM read_csv_auto('{rapm_csv.as_posix()}') ORDER BY season"
            ).fetchall()
        ]

        year1_cols = [
            "year1_epm_tot",
            "year1_epm_off",
            "year1_epm_def",
            "year1_mp",
            "year1_tspct",
            "year1_usg",
        ]
        row = con.execute(
            f"""
            SELECT
                COUNT(*) AS n,
                {", ".join([f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS null_{c}" for c in year1_cols])}
            FROM read_parquet('{year1_epm_parquet.as_posix()}')
            """
        ).fetchone()
    finally:
        con.close()

    n = int(row[0])
    null_rates = {}
    for i, c in enumerate(year1_cols, start=1):
        nulls = int(row[i])
        null_rates[c] = {"nulls": nulls, "null_rate": float(nulls / n if n else 0.0)}

    return {
        "historical_rapm_seasons": rapm_seasons,
        "historical_rapm_min_season": min(rapm_seasons) if rapm_seasons else None,
        "historical_rapm_max_season": max(rapm_seasons) if rapm_seasons else None,
        "year1_rows": n,
        "year1_null_rates": null_rates,
    }


def build_readiness_gate(
    endpoint_summary_df: pd.DataFrame,
    fact_parity: dict,
    feature_report: dict,
    target_report: dict,
    postseason_missing_seasons: list[int],
    start_season: int,
    end_season: int,
    subs_floor: float,
    lineups_floor: float,
    fact_parity_floor: float,
    feature_null_rate_max: float,
    year1_null_rate_max: float,
) -> dict:
    checks = {}

    plays_missing = int(endpoint_summary_df[endpoint_summary_df["endpoint"] == "plays"]["missing_games"].sum())
    participants_missing = int(
        endpoint_summary_df[endpoint_summary_df["endpoint"] == "participants"]["missing_games"].sum()
    )
    checks["plays_participants_complete"] = {
        "pass": plays_missing == 0 and participants_missing == 0,
        "plays_missing_games": plays_missing,
        "participants_missing_games": participants_missing,
    }

    subs = endpoint_summary_df[endpoint_summary_df["endpoint"] == "subs"].copy()
    lineups = endpoint_summary_df[endpoint_summary_df["endpoint"] == "lineups"].copy()
    subs_coverage = float(subs["covered_games"].sum() / max(subs["expected_games"].sum(), 1))
    lineups_coverage = float(lineups["covered_games"].sum() / max(lineups["expected_games"].sum(), 1))
    checks["subs_lineups_floor"] = {
        "pass": (subs_coverage >= subs_floor) and (lineups_coverage >= lineups_floor),
        "subs_coverage_rate": subs_coverage,
        "lineups_coverage_rate": lineups_coverage,
        "subs_floor": subs_floor,
        "lineups_floor": lineups_floor,
    }

    checks["postseason_manifest_complete"] = {
        "pass": len(postseason_missing_seasons) == 0,
        "missing_postseason_seasons": postseason_missing_seasons,
    }

    fact_game_coverage = float(fact_parity["fact_player_game"]["coverage_vs_stg_plays"])
    fact_impact_coverage = float(fact_parity["fact_player_game_impact"]["coverage_vs_stg_plays"])
    checks["fact_parity"] = {
        "pass": (fact_game_coverage >= fact_parity_floor) and (fact_impact_coverage >= fact_parity_floor),
        "coverage_vs_stg_plays": {
            "fact_player_game": fact_game_coverage,
            "fact_player_game_impact": fact_impact_coverage,
        },
        "minimum_required": fact_parity_floor,
    }

    checks["feature_store_integrity"] = {
        "pass": (
            feature_report["duplicate_rows"] == 0
            and feature_report["null_team_pace_rate"] <= feature_null_rate_max
            and feature_report["null_conference_rate"] <= feature_null_rate_max
        ),
        "duplicate_rows": feature_report["duplicate_rows"],
        "null_team_pace_rate": feature_report["null_team_pace_rate"],
        "null_conference_rate": feature_report["null_conference_rate"],
        "null_rate_max": feature_null_rate_max,
    }

    expected_rapm = set(range(start_season, end_season + 1))
    actual_rapm = set(target_report["historical_rapm_seasons"])
    missing_rapm = sorted(expected_rapm - actual_rapm)
    year1_core = target_report["year1_null_rates"]["year1_epm_tot"]["null_rate"]
    checks["target_coverage"] = {
        "pass": len(missing_rapm) == 0 and year1_core <= year1_null_rate_max,
        "missing_historical_rapm_seasons": missing_rapm,
        "year1_epm_tot_null_rate": year1_core,
        "year1_epm_tot_null_rate_max": year1_null_rate_max,
    }

    passed = all(v["pass"] for v in checks.values())
    return {
        "generated_at": utc_now_iso(),
        "scope": {"start_season": start_season, "end_season": end_season, "season_types": ["regular", "postseason"]},
        "passed": passed,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full missing-data audit and generate manifests/reports.")
    parser.add_argument("--db", default="data/warehouse.duckdb")
    parser.add_argument("--audit-dir", default="data/audit")
    parser.add_argument("--start-season", type=int, default=2010)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--subs-floor", type=float, default=0.85)
    parser.add_argument("--lineups-floor", type=float, default=0.60)
    parser.add_argument("--fact-parity-floor", type=float, default=0.95)
    parser.add_argument("--feature-null-rate-max", type=float, default=0.20)
    parser.add_argument("--year1-null-rate-max", type=float, default=0.20)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    db_path = (repo_root / args.db).resolve() if not Path(args.db).is_absolute() else Path(args.db)
    audit_dir = (repo_root / args.audit_dir).resolve() if not Path(args.audit_dir).is_absolute() else Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)

    feature_path = repo_root / "data/college_feature_store/college_features_v1.parquet"
    rapm_path = repo_root / "data/historical_rapm_results_lambda1000.csv"
    year1_path = repo_root / "data/warehouse_v2/fact_player_year1_epm.parquet"

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        dim_games = load_dim_games(con, args.start_season, args.end_season)

        endpoint_summary_rows = []
        missing_map: dict[str, pd.DataFrame] = {}
        season_rollup = (
            dim_games.groupby(["season", "season_type"], as_index=False)
            .agg(expected_games=("game_id", "count"))
            .sort_values(["season", "season_type"])
        )

        for endpoint, table in ENDPOINT_TABLES.items():
            have = load_distinct_game_ids(con, table, game_col="gameId")
            summary_df, missing_df = build_coverage(dim_games, have)
            summary_df["endpoint"] = endpoint
            endpoint_summary_rows.append(summary_df)
            missing_map[endpoint] = missing_df

            rename = summary_df.rename(
                columns={
                    "covered_games": f"{endpoint}_covered",
                    "missing_games": f"{endpoint}_missing",
                    "coverage_rate": f"{endpoint}_coverage_rate",
                }
            )
            season_rollup = season_rollup.merge(
                rename[["season", "season_type", f"{endpoint}_covered", f"{endpoint}_missing", f"{endpoint}_coverage_rate"]],
                how="left",
                on=["season", "season_type"],
            )

        endpoint_summary = pd.concat(endpoint_summary_rows, ignore_index=True)
        endpoint_summary = endpoint_summary[
            ["endpoint", "season_type", "season", "expected_games", "covered_games", "missing_games", "coverage_rate"]
        ].sort_values(["endpoint", "season", "season_type"])

        expected_postseason = list(range(args.start_season, args.end_season + 1))
        found_postseason = sorted(
            dim_games.loc[dim_games["season_type"] == "postseason", "season"].dropna().astype(int).unique().tolist()
        )
        postseason_missing = sorted(set(expected_postseason) - set(found_postseason))

        stg_plays_games = load_distinct_game_ids(con, "stg_plays", game_col="gameId")
        fact_player_game_games = load_distinct_game_ids(con, "fact_player_game", game_col="gameId")
        fact_player_impact_games = load_distinct_game_ids(con, "fact_player_game_impact", game_col="gameId")

        stg_plays_covered = dim_games.merge(stg_plays_games, how="inner", on="game_id")
        fact_pg_covered = dim_games.merge(fact_player_game_games, how="inner", on="game_id")
        fact_pi_covered = dim_games.merge(fact_player_impact_games, how="inner", on="game_id")

        stg_games = int(stg_plays_covered["game_id"].nunique())
        fact_pg_games = int(fact_pg_covered["game_id"].nunique())
        fact_pi_games = int(fact_pi_covered["game_id"].nunique())
        fact_parity = {
            "fact_player_game": {
                "games": fact_pg_games,
                "coverage_vs_stg_plays": float(fact_pg_games / stg_games if stg_games else 0.0),
            },
            "fact_player_game_impact": {
                "games": fact_pi_games,
                "coverage_vs_stg_plays": float(fact_pi_games / stg_games if stg_games else 0.0),
            },
            "stg_plays_games": stg_games,
        }
    finally:
        con.close()

    # Plays manifest uses union of plays+participants gaps so pass 1 can close both.
    plays_missing = missing_map["plays"][["game_id", "game_id_int", "season", "season_type"]].copy()
    plays_missing["missing_plays"] = True
    participants_missing = missing_map["participants"][["game_id", "game_id_int", "season", "season_type"]].copy()
    participants_missing["missing_participants"] = True

    plays_manifest = plays_missing.merge(
        participants_missing,
        how="outer",
        on=["game_id", "game_id_int", "season", "season_type"],
    )
    plays_manifest["missing_plays"] = plays_manifest["missing_plays"].eq(True)
    plays_manifest["missing_participants"] = plays_manifest["missing_participants"].eq(True)
    plays_manifest = plays_manifest.sort_values(["season", "season_type", "game_id_int"])

    subs_manifest = missing_map["subs"][["game_id", "game_id_int", "season", "season_type"]].copy()
    subs_manifest = subs_manifest.sort_values(["season", "season_type", "game_id_int"])

    lineups_manifest = missing_map["lineups"][["game_id", "game_id_int", "season", "season_type"]].copy()
    lineups_manifest = lineups_manifest.sort_values(["season", "season_type", "game_id_int"])

    feature_report = compute_feature_store_integrity(feature_path)
    target_report = compute_target_coverage(rapm_path, year1_path)
    readiness = build_readiness_gate(
        endpoint_summary_df=endpoint_summary,
        fact_parity=fact_parity,
        feature_report=feature_report,
        target_report=target_report,
        postseason_missing_seasons=postseason_missing,
        start_season=args.start_season,
        end_season=args.end_season,
        subs_floor=args.subs_floor,
        lineups_floor=args.lineups_floor,
        fact_parity_floor=args.fact_parity_floor,
        feature_null_rate_max=args.feature_null_rate_max,
        year1_null_rate_max=args.year1_null_rate_max,
    )

    endpoint_csv = audit_dir / "missing_games_by_endpoint.csv"
    season_csv = audit_dir / "missing_games_by_season.csv"
    feature_json = audit_dir / "feature_store_integrity_report.json"
    plays_csv = audit_dir / "reingest_manifest_plays.csv"
    subs_csv = audit_dir / "reingest_manifest_subs.csv"
    lineups_csv = audit_dir / "reingest_manifest_lineups.csv"
    gate_json = audit_dir / "model_readiness_gate.json"

    endpoint_summary.to_csv(endpoint_csv, index=False)
    season_rollup.to_csv(season_csv, index=False)
    plays_manifest.to_csv(plays_csv, index=False)
    subs_manifest.to_csv(subs_csv, index=False)
    lineups_manifest.to_csv(lineups_csv, index=False)

    feature_report_full = {
        "generated_at": utc_now_iso(),
        "feature_file": str(feature_path.resolve()),
        "integrity": feature_report,
        "target_coverage": target_report,
        "fact_parity": fact_parity,
        "postseason_missing_seasons": postseason_missing,
    }
    feature_json.write_text(json.dumps(feature_report_full, indent=2), encoding="utf-8")
    gate_json.write_text(json.dumps(readiness, indent=2), encoding="utf-8")

    print(f"Wrote: {endpoint_csv}")
    print(f"Wrote: {season_csv}")
    print(f"Wrote: {feature_json}")
    print(f"Wrote: {plays_csv}")
    print(f"Wrote: {subs_csv}")
    print(f"Wrote: {lineups_csv}")
    print(f"Wrote: {gate_json}")
    print(f"Readiness passed: {readiness['passed']}")


if __name__ == "__main__":
    main()
