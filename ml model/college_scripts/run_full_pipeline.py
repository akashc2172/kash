"""
Full Pipeline Runner for Historical Data
========================================

This script runs the complete pipeline for specified seasons:
1. Clean historical PBP (if needed)
2. Derive minutes/turnovers from PBP
3. Calculate RAPM

Usage:
    python college_scripts/run_full_pipeline.py --seasons 2012 2015

Note: The cleaning step processes ALL seasons in manual_scrapes/ and outputs
a combined file. The backfill and RAPM steps can be run for specific seasons.
"""

import subprocess
import sys
import argparse
from pathlib import Path
import pandas as pd

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"📋 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    else:
        print(result.stdout)
        return True

def check_combined_file_has_seasons(seasons):
    """Check if combined file contains the required seasons."""
    combined_file = Path("data/fact_play_historical_combined.parquet")
    if not combined_file.exists():
        return False
    
    df = pd.read_parquet(combined_file)
    available_seasons = set(df['season'].unique())
    required_seasons = set(seasons)
    
    missing = required_seasons - available_seasons
    if missing:
        print(f"⚠️  Warning: Combined file missing seasons: {missing}")
        print(f"   Available: {sorted(available_seasons)}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run full pipeline for historical data")
    parser.add_argument('--seasons', nargs='+', type=int, 
                       help="Seasons to process (e.g., 2012 2015)")
    parser.add_argument('--skip-clean', action='store_true',
                       help="Skip cleaning step (use existing combined file)")
    parser.add_argument('--skip-backfill', action='store_true',
                       help="Skip backfill step")
    parser.add_argument('--skip-rapm', action='store_true',
                       help="Skip RAPM calculation")
    args = parser.parse_args()
    
    if not args.seasons:
        print("❌ Please specify seasons with --seasons (e.g., --seasons 2012 2015)")
        return
    
    seasons_str = " ".join(map(str, args.seasons))
    print(f"\n🚀 Running Full Pipeline for Seasons: {seasons_str}")
    
    # Step 1: Clean Historical PBP
    if not args.skip_clean:
        print("\n📝 Step 1: Cleaning Historical PBP...")
        print("   Note: This processes ALL seasons in manual_scrapes/")
        print("   Output: data/fact_play_historical_combined.parquet")
        
        if not run_command(
            [sys.executable, "college_scripts/utils/clean_historical_pbp_v2.py"],
            "Cleaning Historical PBP"
        ):
            print("❌ Cleaning failed. Exiting.")
            return
        
        # Verify seasons are in combined file
        if not check_combined_file_has_seasons(args.seasons):
            print("❌ Combined file doesn't contain required seasons. Exiting.")
            return
    else:
        print("\n⏭️  Skipping cleaning step (using existing combined file)")
        if not check_combined_file_has_seasons(args.seasons):
            print("❌ Combined file doesn't contain required seasons.")
            print("   Run without --skip-clean to regenerate.")
            return
    
    # Step 2: Derive Minutes/Turnovers
    if not args.skip_backfill:
        print("\n📊 Step 2: Deriving Minutes/Turnovers from PBP...")
        print(f"   Processing seasons: {seasons_str}")
        
        if not run_command(
            [sys.executable, "college_scripts/derive_minutes_from_historical_pbp.py", 
             "--seasons"] + list(map(str, args.seasons)),
            "Deriving Minutes/Turnovers"
        ):
            print("❌ Backfill failed. Exiting.")
            return
    else:
        print("\n⏭️  Skipping backfill step")
    
    # Step 3: Calculate RAPM
    if not args.skip_rapm:
        print("\n📈 Step 3: Calculating RAPM...")
        print("   Note: This processes ALL seasons in combined file")
        print("   Output: data/historical_rapm_results_lambda1000.csv")
        
        if not run_command(
            [sys.executable, "college_scripts/calculate_historical_rapm.py"],
            "Calculating RAPM"
        ):
            print("❌ RAPM calculation failed. Exiting.")
            return
    else:
        print("\n⏭️  Skipping RAPM calculation")
    
    print("\n" + "="*60)
    print("✅ Full Pipeline Complete!")
    print("="*60)
    print(f"\nProcessed seasons: {seasons_str}")
    print("\nOutput files:")
    print("  - data/fact_play_historical_combined.parquet (cleaned PBP)")
    print("  - data/warehouse_v2/fact_player_season_stats_backfill.parquet (minutes/TOV)")
    print("  - data/historical_rapm_results_lambda1000.csv (RAPM results)")

if __name__ == "__main__":
    main()
