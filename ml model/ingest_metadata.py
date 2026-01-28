#!/usr/bin/env python3
"""
Metadata Ingest Script
Fetches player bio info (Height, Weight, Experience, etc.) for all seasons.
Run this AFTER the main game ingest finishes (or concurrently if DB allows, but DuckDB locks).
"""
import os
import sys
import pandas as pd
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cbbd
from cbd_pbp.warehouse import Warehouse

# Seasons to fetch metadata for
SEASONS = list(range(2025, 2004, -1))
DB_PATH = "data/warehouse.duckdb"

def get_api_key():
    with open(".env") as f:
        for line in f:
            if line.startswith("CBD_API_KEY"):
                return line.split("=")[1].strip().strip('"').strip("'")
    raise ValueError("CBD_API_KEY not found in .env")

def _models_to_df(models: list) -> pd.DataFrame:
    if not models:
        return pd.DataFrame()
    return pd.DataFrame([m.to_dict() for m in models])

def main():
    print("=" * 60)
    print("Player Metadata Ingest (Height/Weight/Bio)")
    print("=" * 60)
    
    api_key = get_api_key()
    wh = Warehouse(DB_PATH)
    config = cbbd.Configuration(
        host="https://api.collegebasketballdata.com",
        access_token=api_key
    )
    
    with cbbd.ApiClient(config) as client:
        stats_api = cbbd.StatsApi(client)
        recruiting_api = cbbd.RecruitingApi(client)
        
        for season in SEASONS:
            print(f"\nProcessing Season {season} metadata...")
            
            # 1. Player Season Stats (often contains height/weight/class)
            try:
                print(f"  Fetching player stats/bio...")
                stats = stats_api.get_player_season_stats(season=season)
                df = _models_to_df(stats)
                if not df.empty:
                    wh.ensure_table("fact_player_season_stats", df, pk=None)
                    print(f"    ✓ Stored {len(df)} player records")
            except Exception as e:
                print(f"    x Error fetching player stats: {e}")

            # 2. Recruiting Info (explicit height/weight/stars)
            try:
                print(f"  Fetching recruiting info...")
                recruits = recruiting_api.get_recruits(year=season)
                df = _models_to_df(recruits)
                if not df.empty:
                    wh.ensure_table("fact_recruiting_players", df, pk=None)
                    print(f"    ✓ Stored {len(df)} recruiting records")
            except Exception as e:
                print(f"    x Error fetching recruiting: {e}")
                
    wh.close()
    print("\n" + "=" * 60)
    print("Metadata Ingest Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
