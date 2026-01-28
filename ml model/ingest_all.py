#!/usr/bin/env python3
"""
Full Multi-Season Ingest: 2021-2025
Handles all seasons in one run. Can resume from where it left off.
"""
import os
import sys
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cbbd
import pandas as pd
from cbd_pbp.warehouse import Warehouse
from cbd_pbp.ingest import ingest_games_only

# All seasons to ingest (newest first - resuming from 2023)
SEASONS = [2023, 2022, 2021]
DB_PATH = "data/warehouse.duckdb"

def get_api_key():
    with open(".env") as f:
        for line in f:
            if line.startswith("CBD_API_KEY"):
                return line.split("=")[1].strip().strip('"').strip("'")
    raise ValueError("CBD_API_KEY not found in .env")

def ensure_dim_games_for_season(wh, season, api_key):
    """Check if season exists in dim_games, if not fetch and insert."""
    existing = wh.query_df(f"SELECT COUNT(*) as cnt FROM dim_games WHERE season = {season}")
    count = existing['cnt'].iloc[0] if not existing.empty else 0
    
    if count > 0:
        print(f"  dim_games for {season}: {count} games already exist, skipping fetch.")
        return
    
    print(f"  Fetching dim_games for {season}...")
    config = cbbd.Configuration(
        host="https://api.collegebasketballdata.com",
        access_token=api_key
    )
    
    with cbbd.ApiClient(config) as client:
        games_api = cbbd.GamesApi(client)
        games = games_api.get_games(season=season, season_type="regular")
        df = pd.DataFrame([g.to_dict() for g in games])
        print(f"  Fetched {len(df)} games for {season}")
    
    # Use ensure_table to handle column alignment safely
    wh.ensure_table("dim_games", df, pk=["id"])

def main():
    print("=" * 70)
    print(f"Full Multi-Season Ingest Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Seasons: " + ", ".join(map(str, SEASONS)))
    print("=" * 70)
    
    api_key = get_api_key()
    wh = Warehouse(DB_PATH)
    
    for season in SEASONS:
        print(f"\n{'='*70}")
        print(f"SEASON {season}")
        print("=" * 70)
        
        # Step 1: Ensure dim_games has this season
        ensure_dim_games_for_season(wh, season, api_key)
        
        # Step 2: Ingest plays/lineups/subs
        print(f"  Ingesting per-game data for {season}...")
        ingest_games_only(wh, season=season, season_type="regular")
        
        print(f"✓ Season {season} complete!")
    
    wh.close()
    print("\n" + "=" * 70)
    print(f"All seasons complete! {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

if __name__ == "__main__":
    main()
