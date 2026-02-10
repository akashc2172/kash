"""
Create Unified View
===================
Creates fact_play_unified by joining CBD plays with Manual Lineups.

Logic:
- Start with CBD plays (fact_play_raw).
- Join game_crosswalk to link to manual GameID.
- Join manual plays (fact_play_historical_combined) to get onFloor.
- PRIORITY:
  - Lineups: Manual > CBD
  - PlayText: CBD (Standardized)
  - Scores: CBD
"""

import duckdb
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

DB_PATH = "data/warehouse.duckdb"

def main():
    con = duckdb.connect(DB_PATH)
    
    # Verify tables exist
    tables = con.execute("SHOW TABLES").fetchdf()['name'].tolist()
    if 'game_crosswalk' not in tables:
        logger.error("game_crosswalk table missing. Run build_game_crosswalk.py first.")
        return
        
    logger.info("Creating fact_play_unified view...")
    
    # Note: We need to register the parquet file as a view to join it
    if os.path.exists("data/fact_play_historical_combined.parquet"):
        con.execute("CREATE OR REPLACE VIEW fact_play_historical AS SELECT * FROM 'data/fact_play_historical_combined.parquet'")
    else:
         logger.warning("data/fact_play_historical_combined.parquet not found. Unified view will be partial.")
         con.execute("CREATE OR REPLACE VIEW fact_play_historical AS SELECT CAST(NULL AS VARCHAR) as gameSourceId, CAST(NULL AS JSON) as onFloor")

    # Unified Query
    # Logic:
    # 1. Base is CBD fact_play_raw (all modern games + ingested history).
    # 2. Left Join Crosswalk.
    # 3. Left Join Historical via Crosswalk match.
    #    PROBLEM: Historical is Play-by-Play, CBD is Play-by-Play.
    #    We can't join row-for-row accurately without time synchronization!
    #    The User asked to "Merge Lineups".
    #    
    #    If we just want the LINEUPS, we can assume the manual data has better lineup tracking.
    #    But joining PBP lines is hard (timestamps drift).
    #    
    #    ALTERNATIVE STRATEGY (User's Prompt Hint):
    #    "SELECT COALESCE(cbd.gameId, hist.gameSourceId)..."
    #    User implies a UNION of games, or enriching.
    #
    #    Actually, if we have CBD PBP for a game, we prefer CBD PBP text/clock/score.
    #    But we want Manual Lineups.
    #    Manual Lineups are computed per-play in `onFloor`.
    #    We can't easily map `onFloor` from Manual Play X to CBD Play Y unless we align them.
    #
    #    Fallback: If match exists, maybe we just use the MANUAL data for that game entirely?
    #    User's prompt: "fact_play_unified ... COALESCE ... left join"
    #    The user query implies joining on gameId.
    #    But `fact_play_historical` has many rows per game. `fact_play_raw` has many rows per game.
    #    A join on `gameId` would create a Cartesian Product (NxM rows)!
    #    
    #    CORRECT INTERPRETATION:
    #    The user likely wants to use Manual Data for years where CBD lacks lineups (2011-2023),
    #    and CBD data for years where it's good (2024-2025 or modern).
    #    OR, they want to use CBD text but "fill" lineups? That's hard.
    #    
    #    Let's look at the "Coverage Matrix":
    #    2011-2023: CBD=Yes, Manual=Yes. User says "Match + merge lineups".
    #    This implies taking the BEST of both.
    #    Given synchronization diffs, the safest "Merge" is:
    #    - If Manual exists, USE MANUAL PBP (it has lineups).
    #    - If Manual missing, USE CBD PBP.
    #    - This avoids the N*M join explosion.
    #    - AND Manual PBP has cleaned text/score too.
    #
    #    So we prefer Manual content when available because it contains the `onFloor` truth.
    
    query = """
    CREATE OR REPLACE TABLE fact_play_unified AS
    SELECT
        -- IDs
        COALESCE(xw.cbd_game_id, cbd.gameId) as gameId,
        COALESCE(hist.season, cbd.season) as season,
        
        -- Source of Truth Selection
        CASE 
            WHEN hist.gameSourceId IS NOT NULL THEN 'MANUAL'
            ELSE 'CBD'
        END as source_type,
        
        -- Content
        CASE 
            WHEN hist.gameSourceId IS NOT NULL THEN hist.clock
            ELSE cbd.clock
        END as clock,
        
        CASE 
            WHEN hist.gameSourceId IS NOT NULL THEN hist.playText
            ELSE cbd.playText
        END as playText,
        
        CASE 
            WHEN hist.gameSourceId IS NOT NULL THEN hist.homeScore
            ELSE cbd.homeScore
        END as homeScore,

        CASE 
            WHEN hist.gameSourceId IS NOT NULL THEN hist.awayScore
            ELSE cbd.awayScore
        END as awayScore,

        -- LINEUPS (The Goal)
        CASE 
            WHEN hist.gameSourceId IS NOT NULL THEN hist.onFloor
            ELSE cbd.onFloor
        END as onFloor
        
    FROM fact_play_raw cbd
    FULL OUTER JOIN game_crosswalk xw ON cbd.gameId = xw.cbd_game_id
    FULL OUTER JOIN fact_play_historical hist ON (xw.manual_game_id = hist.gameSourceId)
    WHERE (cbd.gameId IS NOT NULL OR hist.gameSourceId IS NOT NULL)
    
    -- Filter out duplicates? 
    -- If we FULL OUTER JOIN, we get:
    -- 1. CBD rows (no match) -> CBD Source
    -- 2. CBD + Hist (Match) -> Cartesian Product? YES. BAD.
    
    -- We cannot join PBP tables directly on GameID.
    -- We must UNION them, filtering out the "loser" of the preference.
    """
    
    # REVISED QUERY: UNION ALL with Filtering
    # 1. Select all Manual Games (with CBD ID mapped if exists)
    # 2. Select all CBD Games that are NOT in the Crosswalk (to avoid dupe)
    
    sql = """
    CREATE OR REPLACE TABLE fact_play_unified AS
    
    -- 1. High Fidelity Manual Data (Preferred for Lineups)
    SELECT
        COALESCE(xw.cbd_game_id, 'MANUAL_' || hist.gameSourceId) as gameId,
        hist.season,
        'MANUAL' as source_type,
        hist.clock,
        hist.playText,
        hist.homeScore,
        hist.awayScore,
        hist.onFloor
    FROM fact_play_historical hist
    LEFT JOIN game_crosswalk xw ON hist.gameSourceId = xw.manual_game_id
    
    UNION ALL
    
    -- 2. CBD Data (Only for games NOT matched in Manual)
    SELECT
        cbd.gameId,
        cbd.season,
        'CBD' as source_type,
        cbd.clock,
        cbd.playText,
        cbd.homeScore,
        cbd.awayScore,
        cbd.onFloor
        -- Add other CBD cols if needed
    FROM fact_play_raw cbd
    WHERE cbd.gameId NOT IN (SELECT cbd_game_id FROM game_crosswalk)
    """
    
    con.execute(sql)
    logger.info("Created fact_play_unified table.")
    
    # Validation
    counts = con.execute("SELECT source_type, COUNT(*), COUNT(DISTINCT gameId) FROM fact_play_unified GROUP BY 1").fetchdf()
    print("\n---------- UNIFIED SUMMARY ----------")
    print(counts)
    print("-------------------------------------")

if __name__ == "__main__":
    main()
