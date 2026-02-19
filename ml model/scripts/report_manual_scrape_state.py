#!/usr/bin/env python3
"""
Quick report of NCAA manual scrape ingestion state and bridge coverage.
"""

from __future__ import annotations

from pathlib import Path
import duckdb


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scrape_root = root / "data/manual_scrapes"
    db_path = root / "data/warehouse.duckdb"

    seasons = []
    if scrape_root.exists():
        for p in sorted(scrape_root.iterdir()):
            if p.is_dir():
                n_csv = len(list(p.glob("*.csv")))
                seasons.append((p.name, n_csv))

    print("Manual scrape folders:")
    if not seasons:
        print("  (none)")
    else:
        for name, n_csv in seasons:
            print(f"  - {name}: {n_csv} csv")

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        bg = con.execute("SELECT COUNT(*) FROM bridge_game_cbd_scrape").fetchone()[0]
        bp = con.execute("SELECT COUNT(*) FROM bridge_player_cbd_scrape").fetchone()[0]
        print(f"\nBridge rows: game={bg}, player={bp}")

        q = """
        WITH expected AS (
          SELECT CAST(id AS VARCHAR) AS game_id, season
          FROM dim_games
          WHERE season BETWEEN 2023 AND 2025
        ),
        api AS (
          SELECT DISTINCT gameId AS game_id FROM stg_plays
        ),
        manual AS (
          SELECT DISTINCT CAST(cbd_game_id AS VARCHAR) AS game_id FROM bridge_game_cbd_scrape
        )
        SELECT
          e.season,
          COUNT(*) AS expected_games,
          SUM(CASE WHEN a.game_id IS NOT NULL THEN 1 ELSE 0 END) AS api_games,
          SUM(CASE WHEN m.game_id IS NOT NULL THEN 1 ELSE 0 END) AS manual_games,
          SUM(CASE WHEN a.game_id IS NOT NULL OR m.game_id IS NOT NULL THEN 1 ELSE 0 END) AS covered_either
        FROM expected e
        LEFT JOIN api a USING(game_id)
        LEFT JOIN manual m USING(game_id)
        GROUP BY 1
        ORDER BY 1
        """
        print("\nCoverage snapshot (2023-2025):")
        print(con.execute(q).fetchdf().to_string(index=False))
    finally:
        con.close()


if __name__ == "__main__":
    main()
