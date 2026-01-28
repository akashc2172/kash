"""
College Data Explorer
=====================
Explore the richness of the raw CollegeBasketballData (CBD) in DuckDB.

This script queries `data/warehouse.duckdb` to show:
1. Sample PBP sequence (what an actual possession looks like)
2. Shot chart details (location, type, assistance)
3. Lineup data (who is on the floor)
4. Advanced context (leverage, score state)

Run this to see "what cool shit we can do" with the raw data.
"""

import duckdb
import pandas as pd
from datetime import datetime

# Connect to warehouse
con = duckdb.connect('data/warehouse.duckdb')

def explore_pbp_richness(limit: int = 5):
    """Show details of a single possession sequence."""
    print("\n=== 1. Rich Play-by-Play Sequence ===")
    
    # Get a random game with plays
    game_id = con.execute("SELECT gameId FROM stg_plays LIMIT 1").fetchone()[0]
    
    query = f"""
    SELECT 
        period, secondsRemaining, 
        homeScore, awayScore, homeWinProbability as win_prob,
        playType, shootingPlay, scoringPlay,
        playText,
        participants
    FROM stg_plays
    WHERE gameId = '{game_id}'
    AND period = 1 AND secondsRemaining BETWEEN 1100 AND 1200
    ORDER BY secondsRemaining DESC
    LIMIT {limit}
    """
    df = con.query(query).to_df()
    print(df.to_string(index=False))

def explore_shot_richness(limit: int = 5):
    """Show shot chart details (distance, type, context)."""
    print("\n=== 2. Detailed Shot Tracking ===")
    
    query = f"""
    SELECT 
        shooter_name, made, shot_range,
        assisted, assistAthleteId as assister_id,
        is_garbage, is_high_leverage
    FROM stg_shots
    WHERE shot_range IS NOT NULL
    LIMIT {limit}
    """
    df = con.query(query).to_df()
    print(df.to_string(index=False))

def explore_lineup_context(limit: int = 5):
    """Show lineup/participant details."""
    print("\n=== 3. Lineup / On-Court Context ===")
    
    query = f"""
    SELECT 
        gameId, playId, 
        participants
    FROM stg_plays
    WHERE len(participants) > 0
    LIMIT {limit}
    """
    df = con.query(query).to_df()
    # Pretty print participants struct
    for _, row in df.iterrows():
        print(f"Game {row['gameId']} | Play {row['playId']}")
        print(f"  Participants: {row['participants']}")

def main():
    print(f"--- College Data Exploration ---")
    print(f"Source: data/warehouse.duckdb")
    
    try:
        explore_pbp_richness()
        explore_shot_richness()
        explore_lineup_context()
        
        print("\n=== Potential 'Cool Shit' ===")
        print("- Win Probability models (win_prob column)")
        print("- Clutch/Leverage stats (is_high_leverage)")
        print("- Lineup impact (Plus/Minus from stints)")
        print("- Assist networks (assistedBy in shots)")
        print("- Shot charts (range/zone data)")
        
    except Exception as e:
        print(f"\nError exploring data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
