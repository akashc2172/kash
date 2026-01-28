#!/usr/bin/env python3
"""
Merge basketball-excel players + EPM (Dunks & Threes) data
Creates a canonical NBA player-season dataset (2004-2025)
"""
import os
import re
import pandas as pd
from pathlib import Path

# === Step 0: Set base dirs ===
BE_DIR = Path("/Users/akashc/my-trankcopy/ml model/data/basketball_excel/players")
EPM_DIR = Path("/Users/akashc/my-trankcopy/ml model/data/epm ")  # Note: trailing space
OUT_DIR = Path("/Users/akashc/my-trankcopy/ml model/data/nba_merged")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_years(directory: Path, pattern: str) -> set:
    """Extract years from filenames matching pattern."""
    years = set()
    if not directory.exists():
        return years
    for f in directory.iterdir():
        match = re.search(pattern, f.name)
        if match:
            years.add(int(match.group(1)))
    return years


def safe_int(val):
    """Safely convert to int, handling various edge cases."""
    if pd.isna(val):
        return None
    try:
        # Handle strings with whitespace
        if isinstance(val, str):
            val = val.strip()
            if val == '':
                return None
        return int(float(val))
    except (ValueError, TypeError):
        return None


def load_basketball_excel(season_year: int) -> pd.DataFrame:
    """Load and standardize basketball-excel season file."""
    path = BE_DIR / f"players_{season_year}_regular.csv"
    if not path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    
    # Standardize columns
    df['season_year'] = season_year
    df['nba_player_id'] = df['nid'].apply(safe_int)
    df['player_name'] = df['nm'].astype(str)
    df['team'] = df['tid'].astype(str)
    df['minutes'] = pd.to_numeric(df['mp'], errors='coerce')
    
    # Drop rows where nba_player_id couldn't be parsed
    invalid = df['nba_player_id'].isna()
    if invalid.any():
        print(f"  [BE] Warning: Dropping {invalid.sum()} rows with invalid nba_player_id")
        df = df[~invalid]
    
    df['nba_player_id'] = df['nba_player_id'].astype(int)
    
    # Check uniqueness
    dups = df.duplicated(subset=['nba_player_id'], keep=False)
    if dups.any():
        print(f"  [BE] Warning: {dups.sum()} duplicate player IDs found, keeping first")
        df = df.drop_duplicates(subset=['nba_player_id'], keep='first')
    
    return df


def load_epm(season_year: int) -> pd.DataFrame:
    """Load and standardize EPM season file."""
    path = EPM_DIR / f"Dunks & Threes Stats {season_year}.csv"
    if not path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    
    # Standardize columns
    df['season_year'] = season_year
    df['nba_player_id'] = df['player_id'].apply(safe_int)
    
    # Drop rows where nba_player_id couldn't be parsed
    invalid = df['nba_player_id'].isna()
    if invalid.any():
        print(f"  [EPM] Warning: Dropping {invalid.sum()} rows with invalid nba_player_id")
        df = df[~invalid]
    
    df['nba_player_id'] = df['nba_player_id'].astype(int)
    
    # Rename all other columns with epm__ prefix (except season_year and nba_player_id)
    keep_cols = {'season_year', 'nba_player_id'}
    rename_map = {col: f"epm__{col}" for col in df.columns if col not in keep_cols}
    df = df.rename(columns=rename_map)
    
    # Check uniqueness
    dups = df.duplicated(subset=['nba_player_id'], keep=False)
    if dups.any():
        print(f"  [EPM] Warning: {dups.sum()} duplicate player IDs found, keeping first")
        df = df.drop_duplicates(subset=['nba_player_id'], keep='first')
    
    return df


def merge_season(be_df: pd.DataFrame, epm_df: pd.DataFrame, season_year: int) -> tuple:
    """Merge basketball-excel with EPM for a single season."""
    if be_df.empty:
        return pd.DataFrame(), {'season_year': season_year, 'n_be': 0, 'n_epm': len(epm_df), 'n_matched': 0, 'match_rate': 0.0}
    
    n_be = len(be_df)
    n_epm = len(epm_df)
    
    if epm_df.empty:
        merged = be_df.copy()
        merged['has_epm'] = 0
        n_matched = 0
    else:
        merged = be_df.merge(
            epm_df,
            on=['nba_player_id', 'season_year'],
            how='left'
        )
        # has_epm = 1 if epm__off is present (one of the EPM columns)
        epm_cols = [c for c in merged.columns if c.startswith('epm__')]
        if epm_cols:
            merged['has_epm'] = merged[epm_cols[0]].notna().astype(int)
        else:
            merged['has_epm'] = 0
        n_matched = merged['has_epm'].sum()
    
    match_rate = n_matched / n_be if n_be > 0 else 0.0
    
    qa = {
        'season_year': season_year,
        'n_be': n_be,
        'n_epm': n_epm,
        'n_matched': n_matched,
        'match_rate': round(match_rate, 4)
    }
    
    return merged, qa


def main():
    print("=" * 70)
    print("NBA Player-Season Merge: basketball-excel + EPM")
    print("=" * 70)
    
    # === Step 1: Enumerate seasons ===
    be_years = extract_years(BE_DIR, r'players_(\d{4})_regular\.csv')
    epm_years = extract_years(EPM_DIR, r'Dunks & Threes Stats (\d{4})\.csv')
    
    print(f"\nBasketball-Excel seasons: {sorted(be_years)}")
    print(f"EPM seasons: {sorted(epm_years)}")
    
    # Use all BE years (left join means we keep all BE data)
    all_seasons = sorted(be_years)
    matched_seasons = sorted(be_years & epm_years)
    
    print(f"Processing seasons: {all_seasons}")
    print(f"Seasons with EPM available: {matched_seasons}")
    
    # === Steps 2-5: Load, merge, concatenate ===
    all_merged = []
    qa_reports = []
    
    for season_year in all_seasons:
        print(f"\n--- Season {season_year} ---")
        
        # Step 2: Load basketball-excel
        be_df = load_basketball_excel(season_year)
        print(f"  Basketball-Excel: {len(be_df)} players")
        
        # Step 3: Load EPM
        epm_df = load_epm(season_year)
        print(f"  EPM: {len(epm_df)} players")
        
        # Step 4: Merge
        merged, qa = merge_season(be_df, epm_df, season_year)
        print(f"  Merged: {len(merged)} rows, {qa['n_matched']} matched ({qa['match_rate']*100:.1f}%)")
        
        if not merged.empty:
            all_merged.append(merged)
        qa_reports.append(qa)
    
    # Step 5: Concatenate
    print("\n" + "=" * 70)
    print("Concatenating all seasons...")
    
    if not all_merged:
        print("ERROR: No data to merge!")
        return
    
    final_df = pd.concat(all_merged, ignore_index=True)
    
    # Enforce final uniqueness
    dups = final_df.duplicated(subset=['nba_player_id', 'season_year'], keep=False)
    if dups.any():
        print(f"WARNING: {dups.sum()} duplicate (player_id, season) pairs found, keeping first")
        final_df = final_df.drop_duplicates(subset=['nba_player_id', 'season_year'], keep='first')
    
    print(f"Final dataset: {len(final_df)} rows")
    
    # === Step 6: QA Report ===
    qa_df = pd.DataFrame(qa_reports)
    total_rows = len(final_df)
    total_matched = final_df['has_epm'].sum()
    overall_rate = total_matched / total_rows if total_rows > 0 else 0
    
    print(f"\nQA Summary:")
    print(f"  Total rows: {total_rows}")
    print(f"  Rows with EPM: {total_matched}")
    print(f"  Overall match rate: {overall_rate*100:.1f}%")
    
    # === Step 7: Save ===
    # Convert object columns to string for parquet compatibility
    for col in final_df.select_dtypes(include=['object']).columns:
        final_df[col] = final_df[col].astype(str)
    
    parquet_path = OUT_DIR / "nba_player_season_merged_2004_2025.parquet"
    qa_path = OUT_DIR / "qa_merge_report.csv"
    
    final_df.to_parquet(parquet_path, index=False)
    qa_df.to_csv(qa_path, index=False)
    
    print(f"\nSaved:")
    print(f"  {parquet_path}")
    print(f"  {qa_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
