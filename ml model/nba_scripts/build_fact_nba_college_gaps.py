"""
Build Fact NBA-College Gaps (Phase 4)
=====================================
Calculates the "Adaptation Gap" for all bridged players.

Logic:
    gap_ts_legacy = NBA_Year1_TS - College_Final_TS
    gap_usg_legacy = NBA_Year1_USG - College_Final_USG
    gap_3p_rate = NBA_Year1_3PAr - College_Final_3PAr (Optional)

Inputs:
    - data/warehouse_v2/fact_player_year1_epm.parquet (NBA)
    - data/college_feature_store/prospect_career_v1.parquet (College - must be V2 with TS/Usg)
    - data/warehouse_v2/dim_player_nba_college_crosswalk.parquet (Mappings)

Output:
    - data/warehouse_v2/fact_player_nba_college_gaps.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
WAREHOUSE_DIR = Path("data/warehouse_v2")
COLLEGE_DIR = Path("data/college_feature_store")
OUT_FILE = WAREHOUSE_DIR / "fact_player_nba_college_gaps.parquet"

def build_gaps():
    logger.info("Loading datasets...")
    
    # 1. Load NBA Year 1
    nba_y1 = pd.read_parquet(WAREHOUSE_DIR / "fact_player_year1_epm.parquet")
    # Need: nba_id, year1_tspct, year1_usg
    
    # 2. Load College Career (V2)
    # Note: Ensure the user ran V2 build before this!
    college_df = pd.read_parquet(COLLEGE_DIR / "prospect_career_v1.parquet")
    # Need: athlete_id, final_trueShootingPct, final_usage
    
    # Check if V2 columns exist
    if 'final_trueShootingPct' not in college_df.columns:
        logger.error("CRITICAL: College store is missing 'final_trueShootingPct'. Please run build_prospect_career_store_v2.py first.")
        return
        
    # 3. Load Crosswalk
    crosswalk = pd.read_parquet(WAREHOUSE_DIR / "dim_player_nba_college_crosswalk.parquet")
    
    # -------------------------------------------------------------------------
    # Processing
    # -------------------------------------------------------------------------
    
    # Join NBA -> Crosswalk -> College
    merged = nba_y1.merge(crosswalk, on='nba_id', how='inner')
    merged = merged.merge(college_df, on='athlete_id', how='inner')
    
    logger.info(f"Bridged {len(merged)} players for Gap Analysis.")
    
    # CURSOR NOTE: Check coverage breakdown
    has_nba_ts = merged['year1_tspct'].notna().sum()
    has_college_ts = merged['final_trueShootingPct'].notna().sum()
    has_nba_usg = merged['year1_usg'].notna().sum()
    has_college_usg = merged['final_usage'].notna().sum()
    
    logger.info(f"Coverage breakdown:")
    logger.info(f"  NBA TS% available: {has_nba_ts} / {len(merged)}")
    logger.info(f"  College TS% available: {has_college_ts} / {len(merged)}")
    logger.info(f"  NBA Usage available: {has_nba_usg} / {len(merged)}")
    logger.info(f"  College Usage available: {has_college_usg} / {len(merged)}")
    
    # Calculate Gaps
    # Gap = NBA - College (Expect negative values for efficiency/usage)
    
    # TS% (Scale: 0.0-1.0)
    # Check scaling. EPM TS% is usually 0-1 or 0-100?
    # nba_y1['year1_tspct'] usually on 0.55 scale.
    # college['final_trueShootingPct'] on 0.55 scale.
    
    merged['gap_ts_legacy'] = merged['year1_tspct'] - merged['final_trueShootingPct']
    
    # Usage (Scale: 0.0-1.0 or 0-100?)
    # EPM Usg is usually 0-100? Or 0.25?
    # CURSOR NOTE: Check scaling - EPM typically reports usage as decimal (0.0-1.0)
    # but some sources use percentage (0-100). Check max value to auto-detect.
    if merged['year1_usg'].max() > 1.0:
        logger.warning(f"Detected usage > 1.0 (max={merged['year1_usg'].max():.2f}), converting from percentage to decimal")
        merged['year1_usg'] /= 100.0
         
    # College usages derived in V2 are clamped 0-0.6.
    # CURSOR NOTE: Both should be on same scale (0.0-1.0) now
    merged['gap_usg_legacy'] = merged['year1_usg'] - merged['final_usage']
    
    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    
    final_output = merged[['nba_id', 'athlete_id', 'gap_ts_legacy', 'gap_usg_legacy']].copy()
    
    # Add existence flags
    final_output['has_gap_data'] = (final_output['gap_ts_legacy'].notna()) & (final_output['gap_usg_legacy'].notna())
    final_output['has_gap_data'] = final_output['has_gap_data'].astype(int)
    
    final_output.to_parquet(OUT_FILE, index=False)
    logger.info(f"Saved gaps to {OUT_FILE}")
    
    # Validation stats
    valid_gaps = final_output[final_output['has_gap_data'] == 1]
    if len(valid_gaps) > 0:
        avg_ts_gap = valid_gaps['gap_ts_legacy'].mean()
        avg_usg_gap = valid_gaps['gap_usg_legacy'].mean()
        median_ts_gap = valid_gaps['gap_ts_legacy'].median()
        median_usg_gap = valid_gaps['gap_usg_legacy'].median()
        
        logger.info(f"Validation Stats (n={len(valid_gaps):,}):")
        logger.info(f"  TS Gap: Mean={avg_ts_gap:.3f}, Median={median_ts_gap:.3f} (Expect ~ -0.05 to -0.10)")
        logger.info(f"  Usg Gap: Mean={avg_usg_gap:.3f}, Median={median_usg_gap:.3f} (Expect ~ -0.05)")
        
        # CURSOR NOTE: Check for outliers (gaps that are too positive might indicate scaling issues)
        ts_outliers = valid_gaps[valid_gaps['gap_ts_legacy'] > 0.05]
        usg_outliers = valid_gaps[valid_gaps['gap_usg_legacy'] > 0.10]
        if len(ts_outliers) > len(valid_gaps) * 0.1:
            logger.warning(f"  ⚠️  {len(ts_outliers)} players have positive TS gaps (>0.05) - may indicate scaling issue")
        if len(usg_outliers) > len(valid_gaps) * 0.1:
            logger.warning(f"  ⚠️  {len(usg_outliers)} players have large positive Usg gaps (>0.10) - may indicate scaling issue")
    else:
        logger.warning("No valid gap data to validate!")

if __name__ == "__main__":
    build_gaps()
