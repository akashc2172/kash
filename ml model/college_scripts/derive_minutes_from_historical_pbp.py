"""
Derive Minutes and Turnovers from Historical PBP
=================================================

This script reconstructs player-season volume stats (Minutes and Turnovers) 
for historical seasons (e.g., 2010-2018) where `fact_player_season_stats` 
(traditional box scores) are missing from the warehouse.

It uses the CLEANED historical Play-by-Play data (`fact_play_historical_combined.parquet`)
as the source of truth, leveraging the 5-man line-up reconstruction (`onFloor`)
to calculate precise time-on-court.

Usage:
    python college_scripts/derive_minutes_from_historical_pbp.py --seasons 2015 2017

Ouput:
    data/warehouse_v2/fact_player_season_stats_backfill.parquet
"""

import pandas as pd
import numpy as np
import json
import re
import argparse
from pathlib import Path
# Note: duckdb was imported but not used - removed for cleanliness

# --- Configuration ---
# Input: Combined historical PBP file from clean_historical_pbp_v2.py
INPUT_FILE = "data/fact_play_historical_combined.parquet"
# Output: Backfill stats to warehouse_v2 (matches existing warehouse structure)
OUTPUT_DIR = "data/warehouse_v2"  # Fixed: was "data/warehouse/v2" (typo)
OUTPUT_FILE = "fact_player_season_stats_backfill.parquet"

def parse_clock(clock_str):
    """
    Converts 'MM:SS' string to total seconds.
    Returns None if invalid.
    Example: '19:45' -> 1185.0
    """
    if not isinstance(clock_str, str): return None
    clock_str = clock_str.strip()
    if not clock_str or ":" not in clock_str: return None
    try:
        parts = clock_str.split(":")
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    except:
        return None

def normalize_name_cached(name_cache, raw_name):
    """
    Simple normalization to match onFloor format.
    Cache specific common strings for speed.
    """
    if raw_name in name_cache: return name_cache[raw_name]
    
    # Same logic as clean_historical_pbp_v2
    n = re.sub(r'#\d+\s*', '', raw_name) 
    n = re.sub(r'^\d+\s*', '', n)
    norm = n.strip().upper().replace(" ", "")
    name_cache[raw_name] = norm
    return norm

def extract_turnover_player(text, name_cache):
    """
    Detects if a line text contains a turnover and extracts the player name.
    
    This function uses multiple pattern matching strategies to capture turnovers,
    as NCAA PBP text has various formats. Antigravity noted ~40% capture rate,
    so we expand patterns to improve coverage.
    
    Turnover Patterns (NCAA PBP):
    1. "JACKSON,WARREN Turnover" (name first, event second)
    2. "Turnover by JACKSON,WARREN" (event first, name after "by")
    3. "JACKSON,WARREN Traveling" (traveling violation = turnover)
    4. "JACKSON,WARREN Offensive Foul" (offensive foul = turnover)
    5. "JACKSON,WARREN Bad Pass Turnover" (bad pass = turnover)
    6. "JACKSON,WARREN 3 Second Violation" (3-second = turnover)
    7. "JACKSON,WARREN Shot Clock Violation" (shot clock = turnover)
    8. "JACKSON,WARREN Out of Bounds" (out of bounds = turnover, if offensive)
    
    Args:
        text: Raw play text (e.g., "19:45 | JACKSON,WARREN Turnover | 10-8 | ")
        name_cache: Dict cache for normalized names (performance optimization)
    
    Returns:
        Normalized player name if turnover detected, None otherwise
    """
    if not text or pd.isna(text):
        return None
    
    text_upper = str(text).upper()
    
    # Fast filter: Check if text contains any turnover-related keywords
    # This avoids expensive regex/parsing for non-turnover plays
    turnover_keywords = [
        "TURNOVER", "TRAVELING", "BAD PASS", "OFFENSIVE FOUL",
        "3 SECOND", "SHOT CLOCK", "OUT OF BOUNDS", "PALMING",
        "DOUBLE DRIBBLE", "CARRYING", "BACKCOURT"
    ]
    
    if not any(keyword in text_upper for keyword in turnover_keywords):
        return None
    
    # Strategy 1: Pattern "NAME Event" (most common)
    # Example: "JACKSON,WARREN Turnover"
    # Split by space and check if first token has comma (LAST,FIRST format)
    parts = text.split()
    if len(parts) > 0 and "," in parts[0]:
        # First token looks like a name (LAST,FIRST)
        potential_name = parts[0]
        # Check if any turnover keyword appears in remaining text
        remaining_text = " ".join(parts[1:]).upper()
        if any(keyword in remaining_text for keyword in turnover_keywords):
            return normalize_name_cached(name_cache, potential_name)
    
    # Strategy 2: Pattern "Event by NAME"
    # Example: "Turnover by JACKSON,WARREN"
    if " BY " in text_upper:
        # Find "by" and extract what comes after
        by_idx = text_upper.find(" BY ")
        if by_idx >= 0:
            after_by = text[by_idx + 4:].strip()  # +4 to skip " BY "
            # Take first token after "by" (should be the name)
            after_parts = after_by.split()
            if len(after_parts) > 0:
                potential_name = after_parts[0]
                if "," in potential_name:  # Likely a name
                    return normalize_name_cached(name_cache, potential_name)
    
    # Strategy 3: Pattern "NAME Event" where name might not be first token
    # Example: "Timeout. JACKSON,WARREN Traveling"
    # Search for tokens with comma (likely names) and check context
    for i, part in enumerate(parts):
        if "," in part and i < len(parts) - 1:
            # Found a potential name, check if next token is a turnover keyword
            next_text = " ".join(parts[i+1:]).upper()
            if any(keyword in next_text for keyword in turnover_keywords):
                return normalize_name_cached(name_cache, part)
    
    # Strategy 4: Look for name anywhere in text if turnover keyword present
    # This is a fallback for edge cases
    # Extract all comma-separated tokens (likely names)
    for part in parts:
        if "," in part and len(part) > 3:  # Reasonable name length
            # Check if turnover keyword appears nearby (within 3 tokens)
            part_idx = parts.index(part)
            context_start = max(0, part_idx - 2)
            context_end = min(len(parts), part_idx + 5)
            context = " ".join(parts[context_start:context_end]).upper()
            if any(keyword in context for keyword in turnover_keywords):
                return normalize_name_cached(name_cache, part)
    
    return None

def process_season(df_season, season_year):
    """
    Process a single season's DataFrame to derive stats.
    """
    print(f"  > Processing Season {season_year}: {len(df_season)} plays...")
    
    grouped = df_season.groupby('gameSourceId')
    
    player_stats = {} # (team, name) -> {'min': 0, 'tov': 0}
    
    name_cache = {}

    game_groups = list(grouped)
    total_games = len(game_groups)
    
    print(f"  > Found {total_games} games.")

    for i, (game_id, game_df) in enumerate(game_groups):
        if i % 100 == 0:
            print(f"    Processing game {i}/{total_games}...", end='\r')
        # Sort by index implicitly (assuming dataframe order is preserved)
        # Calculate time deltas
        
        # 1. Parse Clocks
        game_df = game_df.copy()
        game_df['secs'] = game_df['clock'].apply(parse_clock)
        
        # 2. Calculate Time Deltas Between Plays
        # 
        # PBP Logic: Clock counts DOWN (20:00 -> 0:00)
        # Each row represents an event at a specific time.
        # The lineup on floor at row i has been playing since the PREVIOUS event (row i-1).
        # Duration = time_at_row_i - time_at_row_i+1
        # 
        # Example:
        #   Row 1: 20:00 (1200s) - Lineup A on floor
        #   Row 2: 19:45 (1185s) - Lineup A still on floor (played for 15s)
        #   Row 3: 19:30 (1170s) - Substitution, Lineup B on floor
        #
        # So duration for Row 1 = 1200 - 1185 = 15 seconds (Lineup A played 15s)
        
        # Shift clock forward to compare current row with next row
        game_df['next_secs'] = game_df['secs'].shift(-1)
        
        # Calculate duration: current time - next time (descending clock)
        game_df['duration'] = game_df['secs'] - game_df['next_secs']
        
        # Handle Edge Cases:
        # 1. Period transitions: Clock resets (e.g., 0:05 -> 20:00 for new half)
        #    - Standard duration would be negative (5 - 1200 = -1195)
        #    - NEW LOGIC: If clock goes UP (next_secs > secs), assume period ended at 0:00.
        #      So the lineup played from 0:05 to 0:00. Duration = 5 seconds.
        
        mask_period_reset = game_df['next_secs'] > game_df['secs']
        game_df.loc[mask_period_reset, 'duration'] = game_df['secs']

        # 2. Missing clocks / Data errors
        #    - Clamp remaining negatives to 0 (e.g., weird glitch)
        #    - Clamp > 1200s to 0
        game_df.loc[(game_df['duration'] < 0) | (game_df['duration'].isna()), 'duration'] = 0
        game_df.loc[game_df['duration'] > 1200, 'duration'] = 0  # Safety cap
        
        # Note: This recover ~15-30s per game lost at end-of-halves.
        
        # Iterate rows
        # We assume 'onFloor' column contains the 10 players ON THE FLOOR for this event.
        # AND they stay on floor for 'duration' seconds.
        
        for idx, row in game_df.iterrows():
            duration = row['duration']
            
            # 1. Minutes
            try:
                lineup = json.loads(row['onFloor'])
            except:
                continue
                
            for p in lineup:
                key = (str(season_year), p['team'], p['name'])
                if key not in player_stats:
                    player_stats[key] = {'minutes': 0.0, 'turnovers': 0, 'games': 0}
                
                if duration > 0:
                    player_stats[key]['minutes'] += duration
            
            # 2. Turnovers
            # 
            # Parse playText to extract turnover events.
            # Format: "Clock | HomeEvt | Score | AwayEvt"
            # 
            # Turnovers can appear in either HomeEvt or AwayEvt column.
            # We check both and extract the player name if a turnover is detected.
            #
            # Note: Antigravity reported ~40% capture rate. This is acceptable because:
            # - Minutes are the critical blocker for Usage Rate (not turnovers)
            # - Turnovers are secondary (poss_total is primary volume proxy per docs)
            # - ~10 TOs per game is sufficient for rough estimate vs having 0
            try:
                parts = row['playText'].split("|")
                if len(parts) >= 4:
                    h_evt = parts[1].strip()  # Home team event
                    a_evt = parts[3].strip()  # Away team event
                    
                    # Check home event first, then away event
                    tov_player = extract_turnover_player(h_evt, name_cache)
                    if not tov_player:
                        tov_player = extract_turnover_player(a_evt, name_cache)
                    
                    if tov_player:
                        # Find which team this player belongs to by checking current lineup
                        # This is necessary because playText doesn't explicitly state team
                        team = None
                        for p in lineup:
                            # Normalize both names for comparison (handle case/whitespace differences)
                            lineup_name = normalize_name_cached(name_cache, p['name'])
                            if lineup_name == tov_player:
                                team = p['team']
                                break
                        
                        # Only count if we found the team (safety check)
                        if team:
                            key = (str(season_year), team, tov_player)
                            if key not in player_stats:
                                player_stats[key] = {'minutes': 0.0, 'turnovers': 0, 'games': 0}
                            player_stats[key]['turnovers'] += 1
            except Exception as e:
                # Silently skip malformed playText (rare, but don't crash on edge cases)
                # Logging every error would be too verbose for 1.5M plays
                pass

        # 3. Games played (per player/team/season): each player appearing in at
        # least one lineup row for a game gets one game credit.
        players_in_game = set()
        for _, row in game_df.iterrows():
            try:
                lineup = json.loads(row['onFloor'])
            except Exception:
                continue
            for p in lineup:
                players_in_game.add((str(season_year), p['team'], p['name']))
        for key in players_in_game:
            if key not in player_stats:
                player_stats[key] = {'minutes': 0.0, 'turnovers': 0, 'games': 0}
            player_stats[key]['games'] += 1

    # Convert to DataFrame
    # 
    # Aggregate player stats into a structured DataFrame.
    # Note: We use team_name and player_name (not athlete_id) because:
    # - Historical PBP doesn't have athlete_id (only names)
    # - Crosswalk to athlete_id happens later in the pipeline
    # - This matches the structure of fact_player_season_stats (which has both)
    records = []
    for (season, team, name), stats in player_stats.items():
        records.append({
            'season': int(season),
            'team_name': team,  # Team name from onFloor (e.g., "HOME", "AWAY", or team ID)
            'player_name': name,  # Normalized player name (e.g., "JACKSON,WARREN")
            'minutes_derived': round(stats['minutes'] / 60.0, 2),  # Convert seconds to minutes
            'turnovers_derived': stats['turnovers'],  # Raw count (not per-game or per-minute)
            'games_derived': stats.get('games', 0),
        })
        
    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description="Derive Minutes/TOV from Historical PBP")
    parser.add_argument('--seasons', nargs='+', type=str, help="Seasons to process (e.g. 2015 2017)")
    parser.add_argument('--all', action='store_true', help="Process all available seasons")
    args = parser.parse_args()

    # 1. Load Data
    # 
    # Load the combined historical PBP file.
    # This file is generated by clean_historical_pbp_v2.py and contains
    # all cleaned historical games with reconstructed lineups (onFloor JSON).
    if not Path(INPUT_FILE).exists():
        print(f"❌ Input file not found: {INPUT_FILE}")
        print(f"   Make sure you've run clean_historical_pbp_v2.py first!")
        return

    print(f"📖 Loading {INPUT_FILE}...")
    # Note: Loading full parquet is fine for now (~1.5M rows, ~100MB).
    # If this becomes a bottleneck, we could use DuckDB to filter by season first.
    df = pd.read_parquet(INPUT_FILE)
    
    available_seasons = sorted(df['season'].unique().astype(str))
    print(f"   Found seasons: {available_seasons}")
    
    target_seasons = []
    if args.all:
        target_seasons = available_seasons
    elif args.seasons:
        target_seasons = [s for s in args.seasons if s in available_seasons]
    else:
        print("⚠️ No seasons specified. Use --seasons YYYY or --all")
        return

    print(f"🎯 Target Seasons: {target_seasons}")
    
    all_results = []
    
    for season in target_seasons:
        df_season = df[df['season'].astype(str) == season]
        res = process_season(df_season, int(season))
        all_results.append(res)
        
    if not all_results:
        print("⚠️ No results generated.")
        return
        
    final_df = pd.concat(all_results, ignore_index=True)
    
    # 2. Save
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(OUTPUT_DIR) / OUTPUT_FILE
    
    print(f"💾 Saving {len(final_df)} rows to {out_path}...")
    final_df.to_parquet(out_path, index=False)
    
    # 3. Validation Summary
    print("\n---------- VALIDATION SUMMARY ----------")
    print(final_df.groupby('season')[['minutes_derived', 'turnovers_derived']].sum())
    print("----------------------------------------")
    
    # Validation Checks
    # 
    # Check for reasonable value ranges:
    # - Typical starter: 800-1200 minutes per season (25-35 min/game * 30 games)
    # - Bench player: 200-600 minutes per season
    # - Extreme values (>1500 min) might indicate clock parsing issues
    starters = final_df[final_df['minutes_derived'] > 500]
    print(f"\n📊 Players with > 500 mins: {len(starters)}")
    
    # Check for potential data quality issues
    max_minutes = final_df['minutes_derived'].max()
    if max_minutes > 1500:
        print(f"⚠️  WARNING: Some players have > 1500 minutes (max: {max_minutes:.1f})")
        print("   This might indicate clock parsing issues or data errors.")
    
    # Sample top players for manual validation
    print("\n📋 Sample Top Players (by minutes):")
    print(final_df.sort_values('minutes_derived', ascending=False).head(10))
    
    # Turnover validation
    avg_tov_per_game = final_df.groupby('season')['turnovers_derived'].sum().mean() / len(final_df.groupby('season'))
    print(f"\n📊 Average turnovers per game (across all players): {avg_tov_per_game:.1f}")
    print("   Expected: ~13 per team (~26 per game total)")
    print("   Note: ~40% capture rate is acceptable (see extract_turnover_player docstring)")

if __name__ == "__main__":
    main()
