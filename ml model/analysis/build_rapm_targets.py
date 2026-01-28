#!/usr/bin/env python3
"""
Build Training Targets and Warehouse Tables
===========================================
1. fact_player_peak_rapm: 3-year peak RAPM targets
2. dim_player_nba: Player metadata (rookie year, draft year)
3. fact_player_year1_epm: Year-1 EPM stats (for two-hop modeling)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# === Configuration ===
MP_MIN_ROOKIE = 200     # Min minutes to qualify as a "played" season for rookie logic
MP_MIN_YEAR1 = 200      # Min minutes to qualify for Year 1 stats reliability

# Paths
RAPM_PATH = Path("data/nba_six_factor_rapm_clean.csv")
MERGED_SEASON_PATH = Path("data/nba_merged/nba_player_season_merged_2004_2025.parquet")
OUT_DIR = Path("data/warehouse_v1")

def safe_int(val):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building warehouse in {OUT_DIR}...")
    
    # ---------------------------------------------------------
    # Step A: Build Canonical RAPM Peak Label Table
    # ---------------------------------------------------------
    print("\n--- Step A: Building fact_player_peak_rapm ---")
    rapm_df = pd.read_csv(RAPM_PATH)
    
    # Enforce types
    rapm_df['nba_id'] = rapm_df['nba_id'].apply(safe_int)
    rapm_df['Latest_Year'] = rapm_df['Latest_Year'].apply(safe_int)
    
    # Derive windows
    # Year_Interval is assumed to be "3Y" for this file based on clean step
    rapm_df['peak_end_season_year'] = rapm_df['Latest_Year']
    rapm_df['peak_start_season_year'] = rapm_df['peak_end_season_year'] - 2
    rapm_df['peak_window'] = (
        rapm_df['peak_start_season_year'].astype(str) + "-" + 
        rapm_df['peak_end_season_year'].astype(str)
    )
    
    # Create Fact Table
    fact_rapm = pd.DataFrame()
    fact_rapm['nba_id'] = rapm_df['nba_id']
    fact_rapm['player_name'] = rapm_df['player_name'] # For QA/Sanity checks
    
    # Targets
    fact_rapm['y_peak_ovr'] = rapm_df['OVR_RAPM']
    fact_rapm['y_peak_off'] = rapm_df['Off_RAPM']
    fact_rapm['y_peak_def'] = rapm_df['Def_RAPM']
    
    # Reliability
    fact_rapm['peak_poss'] = rapm_df['Off_Poss']
    
    # Window info
    fact_rapm['peak_start_year'] = rapm_df['peak_start_season_year']
    fact_rapm['peak_end_year'] = rapm_df['peak_end_season_year']
    fact_rapm['peak_window'] = rapm_df['peak_window']
    
    # Drop rows with invalid ID
    fact_rapm = fact_rapm.dropna(subset=['nba_id'])
    fact_rapm['nba_id'] = fact_rapm['nba_id'].astype(int)
    
    print(f"RAPM Fact Rows: {len(fact_rapm)}")
    print(fact_rapm[['nba_id', 'player_name', 'y_peak_ovr', 'peak_poss']].head(3))
    
    # ---------------------------------------------------------
    # Step B: Build Metadata (Dim Player) & Year 1 EPM
    # ---------------------------------------------------------
    print("\n--- Step B: Building dim_player_nba & fact_player_year1_epm ---")
    seasons_df = pd.read_parquet(MERGED_SEASON_PATH)
    
    # Ensure types
    seasons_df['nba_player_id'] = seasons_df['nba_player_id'].astype(int)
    seasons_df['season_year'] = seasons_df['season_year'].astype(int)
    seasons_df['minutes'] = seasons_df['minutes'].fillna(0)
    
    # Handle d_y (draft year) cleaning
    # d_y is in basketball-excel often as float
    seasons_df['d_y'] = pd.to_numeric(seasons_df['d_y'], errors='coerce')
    
    dim_rows = []
    year1_rows = []
    
    # Group by player
    for pid, group in seasons_df.groupby('nba_player_id'):
        group = group.sort_values('season_year')
        
        # 1. Basic Metadata
        last_row = group.iloc[-1]
        name = last_row['player_name']
        
        # Draft Year: take max valid d_y (usually stable)
        draft_year = None
        valid_dys = group['d_y'].dropna()
        if not valid_dys.empty:
            draft_year = int(valid_dys.max())
            
        # 2. Determine Rookie Season
        # Rule: Use draft_year if they played in that season
        rookie_year = None
        if draft_year:
            # Check if they have a row for that season
            if (group['season_year'] == draft_year).any():
                rookie_year = draft_year
        
        # Rule: Else first season with mp >= MP_MIN
        if rookie_year is None:
            qualifying_seasons = group[group['minutes'] >= MP_MIN_ROOKIE]
            if not qualifying_seasons.empty:
                rookie_year = int(qualifying_seasons.iloc[0]['season_year'])
            else:
                # Fallback: Just the first appeared season
                rookie_year = int(group.iloc[0]['season_year'])
        
        dim_rows.append({
            'nba_id': pid,
            'player_name': name,
            'draft_year': draft_year,
            'rookie_season_year': rookie_year
        })
        
        # 3. Year 1 EPM stats
        # Defined as stats from the rookie_season_year (if eligible)
        y1_data = {
            'nba_id': pid,
            'year1_season_year': rookie_year,
            'year1_epm_tot': None,
            'year1_epm_off': None,
            'year1_epm_def': None,
            'missing_year1': 1,
            'delayed_debut': 0
        }
        
        # Get rookie season Stats
        rookie_stats = group[group['season_year'] == rookie_year]
        
        if not rookie_stats.empty:
            r_stat = rookie_stats.iloc[0]
            if r_stat['minutes'] >= MP_MIN_YEAR1 and r_stat['has_epm'] == 1:
                y1_data['year1_epm_tot'] = r_stat.get('epm__tot')
                y1_data['year1_epm_off'] = r_stat.get('epm__off')
                y1_data['year1_epm_def'] = r_stat.get('epm__def')
                y1_data['missing_year1'] = 0
                
        # Delayed debut check
        # "delayed_debut=1 if first eligible season > rookie_season_year"
        elig_seasons = group[group['minutes'] >= MP_MIN_YEAR1]
        if not elig_seasons.empty:
            first_elig_year = int(elig_seasons.iloc[0]['season_year'])
            if first_elig_year > rookie_year:
                y1_data['delayed_debut'] = 1
                
        year1_rows.append(y1_data)
        
    dim_player_df = pd.DataFrame(dim_rows)
    fact_year1_df = pd.DataFrame(year1_rows)
    
    print(f"Dim Players: {len(dim_player_df)}")
    print(f"Fact Year 1: {len(fact_year1_df)}")
    
    # ---------------------------------------------------------
    # Step C: Save
    # ---------------------------------------------------------
    print("\n--- Saving Tables ---")
    
    p1 = OUT_DIR / "fact_player_peak_rapm.parquet"
    p2 = OUT_DIR / "dim_player_nba.parquet"
    p3 = OUT_DIR / "fact_player_year1_epm.parquet"
    
    # Parquet requires string cols for objects usually
    # RAPM
    fact_rapm.to_parquet(p1, index=False)
    
    # DIM
    # Convert object to string if any
    for c in dim_player_df.select_dtypes(['object']).columns:
        dim_player_df[c] = dim_player_df[c].astype(str)
    dim_player_df.to_parquet(p2, index=False)
    
    # YEAR 1
    fact_year1_df.to_parquet(p3, index=False)
    
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    print(f"Saved: {p3}")
    print("Done.")

if __name__ == "__main__":
    main()
