#!/usr/bin/env python3
"""
Multi-Season Ingest Script
Ingests historical seasons (2021-2024) after current season completes.
Run this after the 2025 ingest finishes.
"""
import os
import sys

# Add parent to path so we can import cbd_pbp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbd_pbp.warehouse import Warehouse
from cbd_pbp.ingest import ingest_games_only

SEASONS = list(range(2020, 2004, -1))  # 2020 down to 2005

DB_PATH = "data/warehouse.duckdb"

def ingest_dim_games_for_season(wh, season):
    """Fetch and insert dim_games for a season."""
    import cbbd
    import pandas as pd
    
    # Read API key
    with open(".env") as f:
        for line in f:
            if line.startswith("CBD_API_KEY"):
                api_key = line.split("=")[1].strip().strip('"').strip("'")
                break
    
    config = cbbd.Configuration(
        host="https://api.collegebasketballdata.com",
        access_token=api_key
    )
    
    with cbbd.ApiClient(config) as client:
        games_api = cbbd.GamesApi(client)
        games = games_api.get_games(season=season, season_type="regular")
        df = pd.DataFrame([g.to_dict() for g in games])
        print(f"Fetched {len(df)} games for {season}")
    
    # Append to dim_games safely
    wh.ensure_table("dim_games", df, pk=["id"])
    print(f"Added {season} to dim_games")

def main():
    print("=" * 60)
    print("Multi-Season Ingest: 2021-2024")
    print("=" * 60)
    
    wh = Warehouse(DB_PATH)
    
    for season in SEASONS:
        print(f"\n{'='*60}")
        print(f"Processing Season {season}")
        print("=" * 60)
        
        # First, add games for this season to dim_games
        print(f"Step 1: Fetching dim_games for {season}...")
        ingest_dim_games_for_season(wh, season)
        
        # Then run the per-game ingest
        print(f"Step 2: Ingesting plays/lineups/subs for {season}...")
        ingest_games_only(wh, season=season, season_type="regular")
        
        print(f"✓ Season {season} complete!")
    
    wh.close()
    print("\n" + "=" * 60)
    print("All historical seasons ingested!")
    print("=" * 60)

if __name__ == "__main__":
    main()
