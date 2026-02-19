#!/usr/bin/env python3
"""
Export manual scrape target lists for unresolved games by season.

By default, exports 2023-2024 and 2024-2025 (warehouse seasons 2024 and 2025):
  - data/audit/manual_scrape_targets_2023-2024.csv
  - data/audit/manual_scrape_targets_2024-2025.csv
  - data/audit/manual_scrape_targets_combined.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def season_folder_label(season: int) -> str:
    return f"{season-1}-{season}"


def build_targets(con: duckdb.DuckDBPyConnection, seasons: list[int]) -> pd.DataFrame:
    q = """
    WITH expected AS (
      SELECT
        CAST(g.id AS VARCHAR) AS game_id,
        g.id AS game_id_int,
        g.season,
        g.seasonType AS season_type,
        CAST(g.startDate AS DATE) AS game_date,
        g.homeTeam AS home_team,
        g.awayTeam AS away_team
      FROM dim_games g
      WHERE g.season IN ({seasons_sql})
    ),
    p AS (SELECT DISTINCT gameId AS game_id FROM stg_plays),
    part AS (SELECT DISTINCT gameId AS game_id FROM stg_participants),
    subs AS (SELECT DISTINCT gameId AS game_id FROM stg_subs),
    lu AS (SELECT DISTINCT gameId AS game_id FROM stg_lineups),
    mg AS (SELECT DISTINCT CAST(cbd_game_id AS VARCHAR) AS game_id FROM bridge_game_cbd_scrape)
    SELECT
      e.game_id,
      e.game_id_int,
      e.season,
      e.season_type,
      e.game_date,
      e.home_team,
      e.away_team,
      (p.game_id IS NOT NULL) AS has_plays,
      (part.game_id IS NOT NULL) AS has_participants,
      (subs.game_id IS NOT NULL) AS has_subs,
      (lu.game_id IS NOT NULL) AS has_lineups,
      (mg.game_id IS NOT NULL) AS has_manual_bridge
    FROM expected e
    LEFT JOIN p USING(game_id)
    LEFT JOIN part USING(game_id)
    LEFT JOIN subs USING(game_id)
    LEFT JOIN lu USING(game_id)
    LEFT JOIN mg USING(game_id)
    """

    seasons_sql = ",".join(str(s) for s in seasons)
    df = con.execute(q.format(seasons_sql=seasons_sql)).fetchdf()
    df["needs_manual_priority"] = (~df["has_plays"]) | (~df["has_participants"])
    df["needs_manual_any"] = (~df["has_plays"]) | (~df["has_participants"]) | (~df["has_subs"]) | (~df["has_lineups"])
    df["season_folder"] = df["season"].map(season_folder_label)
    df["note"] = ""
    df.loc[df["needs_manual_priority"], "note"] = "missing plays/participants"
    df.loc[(~df["needs_manual_priority"]) & (df["needs_manual_any"]), "note"] = "plays present; missing subs/lineups"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Export manual scrape target lists.")
    parser.add_argument("--db", default="data/warehouse.duckdb")
    parser.add_argument("--audit-dir", default="data/audit")
    parser.add_argument("--seasons", nargs="*", type=int, default=[2024, 2025])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    db_path = (repo_root / args.db).resolve() if not Path(args.db).is_absolute() else Path(args.db)
    out_dir = (repo_root / args.audit_dir).resolve() if not Path(args.audit_dir).is_absolute() else Path(args.audit_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = build_targets(con, args.seasons)
    finally:
        con.close()

    combined = df[df["needs_manual_any"]].copy().sort_values(["season", "season_type", "game_date", "game_id_int"])
    combined_path = out_dir / "manual_scrape_targets_combined.csv"
    combined.to_csv(combined_path, index=False)

    for season in sorted(set(args.seasons)):
        season_df = combined[combined["season"] == season].copy()
        p = out_dir / f"manual_scrape_targets_{season_folder_label(season)}.csv"
        season_df.to_csv(p, index=False)
        print(f"Wrote: {p} ({len(season_df)} rows)")

    priority = combined[combined["needs_manual_priority"]]
    print(f"Wrote: {combined_path} ({len(combined)} rows)")
    print(f"Priority (missing plays/participants): {len(priority)}")


if __name__ == "__main__":
    main()
