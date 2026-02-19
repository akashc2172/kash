import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DB_PATH = 'data/warehouse.duckdb'
OUT_DIR = Path('data/college_feature_store')
OUT_FILE = OUT_DIR / "derived_box_stats_v1.parquet"

def extract_stats_from_pbp():
    logger.info("Connecting to DuckDB...")
    con = duckdb.connect(DB_PATH, read_only=True)
    logger.info("Computing derived box stats from structured PBP...")
    out_df = con.execute(
        """
        WITH base AS (
          SELECT
            CAST(season AS BIGINT) AS season,
            gameId,
            playType,
            participants,
            shotInfo
          FROM fact_play_raw
          WHERE season IS NOT NULL
        ),
        games AS (
          SELECT
            b.season,
            CAST(p.id AS BIGINT) AS athlete_id,
            COUNT(DISTINCT b.gameId) AS college_games_played
          FROM base b, UNNEST(b.participants) AS t(p)
          WHERE p.id IS NOT NULL
          GROUP BY 1, 2
        ),
        assists AS (
          SELECT
            season,
            CAST(shotInfo.assistedBy.id AS BIGINT) AS athlete_id,
            COUNT(*) AS college_ast_total
          FROM base
          WHERE shotInfo.assistedBy.id IS NOT NULL
          GROUP BY 1, 2
        ),
        steals AS (
          SELECT
            season,
            CAST(participants[1].id AS BIGINT) AS athlete_id,
            COUNT(*) AS college_stl_total
          FROM base
          WHERE playType = 'Steal'
            AND participants[1].id IS NOT NULL
          GROUP BY 1, 2
        ),
        blocks AS (
          SELECT
            season,
            CAST(participants[1].id AS BIGINT) AS athlete_id,
            COUNT(*) AS college_blk_total
          FROM base
          WHERE playType = 'Block Shot'
            AND participants[1].id IS NOT NULL
          GROUP BY 1, 2
        ),
        turnovers AS (
          SELECT
            season,
            CAST(participants[1].id AS BIGINT) AS athlete_id,
            COUNT(*) AS college_tov_total
          FROM base
          WHERE playType IN ('Lost Ball Turnover', 'Foul Turnover', 'Traveling', 'Bad Pass Turnover', 'Out Of Bounds Turnover')
            AND participants[1].id IS NOT NULL
          GROUP BY 1, 2
        )
        SELECT
          g.season,
          g.athlete_id,
          COALESCE(a.college_ast_total, 0) AS college_ast_total,
          COALESCE(s.college_stl_total, 0) AS college_stl_total,
          COALESCE(bk.college_blk_total, 0) AS college_blk_total,
          COALESCE(t.college_tov_total, 0) AS college_tov_total,
          g.college_games_played
        FROM games g
        LEFT JOIN assists a USING (season, athlete_id)
        LEFT JOIN steals s USING (season, athlete_id)
        LEFT JOIN blocks bk USING (season, athlete_id)
        LEFT JOIN turnovers t USING (season, athlete_id)
        """
    ).df()
    con.close()
    
    # Save
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    out_df.to_parquet(OUT_FILE, index=False)
    logger.info(f"Saved {len(out_df):,} rows to {OUT_FILE}")
    
    # Validation (2018 check)
    logger.info("Validation: Top Assist Leaders 2018")
    v18 = out_df[out_df['season'] == 2018].sort_values('college_ast_total', ascending=False).head(5)
    print(v18[['athlete_id', 'college_ast_total', 'college_games_played']].to_string())

if __name__ == "__main__":
    extract_stats_from_pbp()
