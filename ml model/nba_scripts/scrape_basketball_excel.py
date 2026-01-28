#!/usr/bin/env python3
"""
Basketball-Excel.com Data Scraper
Scrapes NBA statistics from 2005-2025 via the discovered API endpoints.

API Endpoints discovered:
- Players: /api/get/players?sz={year}&st={type}&leagueDataType=1
- Teams: /api/get/teams?sz={year}&st={type}&leagueDataType=2  
- Standings: /api/get/standing?sz={year}
- Games: /api/get/games?sz={year}&st={type}

Parameters:
- sz: Season year (e.g., 2024 for 2024-25 season)
- st: Season type (0=Regular Season, 1=Playoffs)
- leagueDataType: 1=Player stats, 2=Team stats
"""
import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_URL = "https://basketball-excel.com/api/get"
OUTPUT_DIR = Path("data/basketball_excel")

# Seasons available (2005-2025)
START_YEAR = 2005
END_YEAR = 2025

# Season types
SEASON_TYPES = {
    0: "regular",
    1: "playoffs"
}

def fetch_json(endpoint: str, params: dict, retries: int = 3) -> dict | None:
    """Fetch JSON from API with retry logic."""
    url = f"{BASE_URL}/{endpoint}"
    
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"  Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    return None


def convert_to_dataframe(data: dict) -> pd.DataFrame:
    """Convert API response (headers + rowSet) to DataFrame."""
    if not data or 'headers' not in data or 'rowSet' not in data:
        return pd.DataFrame()
    
    headers = data['headers']
    rows = data['rowSet']
    return pd.DataFrame(rows, columns=headers)


def scrape_players(year: int, season_type: int) -> pd.DataFrame:
    """Scrape player statistics for a given year and season type."""
    params = {
        'sz': year,
        'st': season_type,
        'leagueDataType': 1
    }
    data = fetch_json('players', params)
    df = convert_to_dataframe(data)
    if not df.empty:
        df['season_year'] = year
        df['season_type'] = SEASON_TYPES[season_type]
    return df


def scrape_teams(year: int, season_type: int) -> pd.DataFrame:
    """Scrape team statistics for a given year and season type."""
    params = {
        'sz': year,
        'st': season_type,
        'leagueDataType': 2
    }
    data = fetch_json('teams', params)
    df = convert_to_dataframe(data)
    if not df.empty:
        df['season_year'] = year
        df['season_type'] = SEASON_TYPES[season_type]
    return df


def scrape_standings(year: int) -> pd.DataFrame:
    """Scrape standings for a given year."""
    params = {'sz': year}
    data = fetch_json('standing', params)
    df = convert_to_dataframe(data)
    if not df.empty:
        df['season_year'] = year
    return df


def scrape_games(year: int, season_type: int) -> pd.DataFrame:
    """Scrape games for a given year and season type."""
    params = {
        'sz': year,
        'st': season_type
    }
    data = fetch_json('games', params)
    df = convert_to_dataframe(data)
    if not df.empty:
        df['season_year'] = year
        df['season_type'] = SEASON_TYPES[season_type]
    return df


def save_dataframe(df: pd.DataFrame, category: str, year: int, season_type: str | None = None):
    """Save DataFrame to CSV and Parquet files."""
    if df.empty:
        return
    
    subdir = OUTPUT_DIR / category
    subdir.mkdir(parents=True, exist_ok=True)
    
    if season_type:
        filename = f"{category}_{year}_{season_type}"
    else:
        filename = f"{category}_{year}"
    
    # Convert object columns to string to avoid mixed-type parquet errors
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=['object']).columns:
        df_copy[col] = df_copy[col].astype(str)
    
    # Save as CSV and Parquet
    df.to_csv(subdir / f"{filename}.csv", index=False)
    df_copy.to_parquet(subdir / f"{filename}.parquet", index=False)
    print(f"    Saved {len(df)} rows to {filename}")


def scrape_all(start_year: int = START_YEAR, end_year: int = END_YEAR):
    """Scrape all data from start_year to end_year."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 70)
    print(f"Basketball-Excel.com Scraper")
    print(f"Scraping seasons {start_year} to {end_year}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 70)
    
    all_players = []
    all_teams = []
    all_standings = []
    all_games = []
    
    for year in range(start_year, end_year + 1):
        print(f"\n{'='*50}")
        print(f"Season {year}-{str(year+1)[-2:]}")
        print(f"{'='*50}")
        
        # Scrape standings (no season type needed)
        print(f"  Standings...")
        standings_df = scrape_standings(year)
        if not standings_df.empty:
            all_standings.append(standings_df)
            save_dataframe(standings_df, 'standings', year)
        
        # Scrape by season type (regular and playoffs)
        for st_code, st_name in SEASON_TYPES.items():
            print(f"  {st_name.title()} Season:")
            
            # Players
            print(f"    Players...")
            players_df = scrape_players(year, st_code)
            if not players_df.empty:
                all_players.append(players_df)
                save_dataframe(players_df, 'players', year, st_name)
            
            # Teams
            print(f"    Teams...")
            teams_df = scrape_teams(year, st_code)
            if not teams_df.empty:
                all_teams.append(teams_df)
                save_dataframe(teams_df, 'teams', year, st_name)
            
            # Games
            print(f"    Games...")
            games_df = scrape_games(year, st_code)
            if not games_df.empty:
                all_games.append(games_df)
                save_dataframe(games_df, 'games', year, st_name)
            
            # Small delay to be respectful
            time.sleep(0.5)
        
        print(f"✓ Season {year} complete")
    
    # Save combined files
    print(f"\n{'='*70}")
    print("Saving combined files...")
    
    def to_parquet_safe(df: pd.DataFrame, path: Path):
        """Convert object columns to string and save to parquet."""
        df_copy = df.copy()
        for col in df_copy.select_dtypes(include=['object']).columns:
            df_copy[col] = df_copy[col].astype(str)
        df_copy.to_parquet(path, index=False)
    
    if all_players:
        combined_players = pd.concat(all_players, ignore_index=True)
        to_parquet_safe(combined_players, OUTPUT_DIR / 'all_players.parquet')
        print(f"  all_players.parquet: {len(combined_players)} total rows")
    
    if all_teams:
        combined_teams = pd.concat(all_teams, ignore_index=True)
        to_parquet_safe(combined_teams, OUTPUT_DIR / 'all_teams.parquet')
        print(f"  all_teams.parquet: {len(combined_teams)} total rows")
    
    if all_standings:
        combined_standings = pd.concat(all_standings, ignore_index=True)
        to_parquet_safe(combined_standings, OUTPUT_DIR / 'all_standings.parquet')
        print(f"  all_standings.parquet: {len(combined_standings)} total rows")
    
    if all_games:
        combined_games = pd.concat(all_games, ignore_index=True)
        to_parquet_safe(combined_games, OUTPUT_DIR / 'all_games.parquet')
        print(f"  all_games.parquet: {len(combined_games)} total rows")
    
    print(f"\n{'='*70}")
    print(f"Scraping complete! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data saved to: {OUTPUT_DIR.absolute()}")
    print(f"{'='*70}")


def scrape_single_season(year: int):
    """Scrape a single season (useful for testing)."""
    scrape_all(start_year=year, end_year=year)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape basketball-excel.com data")
    parser.add_argument('--start', type=int, default=START_YEAR, help=f'Start year (default: {START_YEAR})')
    parser.add_argument('--end', type=int, default=END_YEAR, help=f'End year (default: {END_YEAR})')
    parser.add_argument('--year', type=int, help='Scrape single year only')
    
    args = parser.parse_args()
    
    if args.year:
        scrape_single_season(args.year)
    else:
        scrape_all(args.start, args.end)
