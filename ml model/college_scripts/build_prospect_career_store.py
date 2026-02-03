"""
Prospect Career Store Build Script (Optimized)
==============================================
Input: data/college_feature_store/college_features_v1.parquet
Output: data/college_feature_store/prospect_career_v1.parquet
"""

import pandas as pd
import numpy as np
import os
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_FILE = 'data/college_feature_store/college_features_v1.parquet'
OUTPUT_FILE = 'data/college_feature_store/prospect_career_v1.parquet'

def build_career_store():
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        return

    logger.info(f"Loading base features from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    df_all = df[df['split_id'] == 'ALL__ALL'].copy()
    df_all = df_all.sort_values(['athlete_id', 'season'])
    
    available_metrics = [
        'minutes_total', 'fga_total', 'ast_total', 'tov_total', 
        'stl_total', 'blk_total', 'rim_fg_pct', 'three_fg_pct', 'ft_pct'
    ]
    available_metrics = [m for m in available_metrics if m in df_all.columns]
    
    logger.info(f"Generating vectorized metrics for {len(df_all):,} rows...")

    # 1. Career Metadata
    df_all['season_rank'] = df_all.groupby('athlete_id').cumcount() + 1
    career_counts = df_all.groupby('athlete_id').size().rename('career_years')
    
    # 2. Final Season Snapshots
    df_final = df_all.groupby('athlete_id').last()
    
    # 3. Vectorized Deltas (YoY)
    for m in available_metrics:
        df_all[f'delta_{m}'] = df_all.groupby('athlete_id')[m].diff().fillna(0)
    
    # 4. Trajectory (Simple Slope: Final - First / Years)
    # Linear Regression is better but 2-point slope is a good proxy for most recruits
    first_vals = df_all.groupby('athlete_id')[available_metrics].first()
    last_vals = df_all.groupby('athlete_id')[available_metrics].last()
    
    # Join career_years for normalization
    stats = pd.concat([first_vals.add_prefix('first_'), last_vals.add_prefix('last_')], axis=1)
    stats = stats.join(career_counts)
    
    for m in available_metrics:
        # Simple slope: (Last - First) / (Years - 1)
        stats[f'slope_{m}'] = (stats[f'last_{m}'] - stats[f'first_{m}']) / np.maximum(1, stats['career_years'] - 1)

    # 5. Weighted Career (Recency weight: 1.0, 0.8, 0.6, 0.4)
    # We assign weights based on season_rank relative to career_years
    # For speed, we do a simple weighted sum
    df_all = df_all.merge(career_counts.to_frame('total_years'), left_on='athlete_id', right_index=True)
    df_all['weight'] = 1.0 - (df_all['total_years'] - df_all['season_rank']) * 0.2
    df_all['weight'] = df_all['weight'].clip(lower=0.2)
    
    weighted_stats = pd.DataFrame(index=career_counts.index)
    for m in available_metrics:
        df_all[f'wt_{m}'] = df_all[m].fillna(0) * df_all['weight']
        weighted_stats[f'career_wt_{m}'] = df_all.groupby('athlete_id')[f'wt_{m}'].sum() / df_all.groupby('athlete_id')['weight'].sum()

    # 6. Physical Growth (Placeholder)
    # We'll just take first/last if available
    if 'heightInches' in df_all.columns:
        stats['height_at_entry'] = df_all.groupby('athlete_id')['heightInches'].first()
        stats['height_final'] = df_all.groupby('athlete_id')['heightInches'].last()
        stats['delta_height'] = stats['height_final'] - stats['height_at_entry']
    else:
        stats['delta_height'] = 0.0

    # Combine All
    # df_final has the latest season and core info
    final_output = df_final[['season', 'teamId']].join(career_counts)
    final_output = final_output.join(stats[[f'slope_{m}' for m in available_metrics] + ['delta_height']])
    final_output = final_output.join(weighted_stats)
    
    # Add YoY Deltas from the last row of df_all
    deltas_final = df_all.groupby('athlete_id')[[f'delta_{m}' for m in available_metrics]].last()
    final_output = final_output.join(deltas_final)
    
    # Add final snapshots
    final_output = final_output.join(last_vals.add_prefix('final_'))

    # Final cleanup
    final_output = final_output.reset_index()
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_output.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"Saved optimized career store to {OUTPUT_FILE} ({len(final_output):,} athletes)")

if __name__ == "__main__":
    build_career_store()
