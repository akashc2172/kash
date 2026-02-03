"""
Unified Training Table Builder
==============================
Merges college features + NBA targets + gaps into a single training matrix.

Inputs:
- data/college_feature_store/college_features_v1.parquet (per-season features)
- data/college_feature_store/prospect_career_v1.parquet (career aggregates)
- data/warehouse_v2/dim_player_nba_college_crosswalk.parquet
- data/warehouse_v2/fact_player_year1_epm.parquet
- data/warehouse_v2/fact_player_peak_rapm.parquet
- data/warehouse_v2/fact_player_nba_college_gaps.parquet

Output:
- data/training/unified_training_table.parquet

Key Design Decisions:
1. Only include players with BOTH college features AND at least one NBA target
2. Tier 2 (spatial) features are included with explicit NaN (masked in model)
3. Era normalization applied to drift-prone features
4. Coverage masks added for Tier 2 availability
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
COLLEGE_FEATURE_STORE = BASE_DIR / "data/college_feature_store"
WAREHOUSE_V2 = BASE_DIR / "data/warehouse_v2"
OUTPUT_DIR = BASE_DIR / "data/training"

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Tier 1: Universal features (2010-2025, 100% coverage)
TIER1_SHOT_PROFILE = [
    'rim_att', 'rim_made', 'rim_fg_pct', 'rim_share',
    'mid_att', 'mid_made', 'mid_fg_pct', 'mid_share',
    'three_att', 'three_made', 'three_fg_pct', 'three_share',
    'ft_att', 'ft_made', 'ft_pct',
    'fga_total', 'shots_total',
]

TIER1_CREATION = [
    'assisted_rim_made', 'assisted_mid_made', 'assisted_three_made',
    'unassisted_rim_made', 'unassisted_mid_made', 'unassisted_three_made',
]

TIER1_IMPACT = [
    'on_net_rating', 'on_ortg', 'on_drtg',
    'seconds_on', 'games_played', 'poss_on',
]

TIER1_CONTEXT = [
    'teamId',
    'team_pace', 'is_power_conf',
    # Additional draft-time-safe context + “impact-adjacent” box signals
    'minutes_total',
    'ast_total', 'tov_total', 'stl_total', 'blk_total',
    'orb_total', 'drb_total', 'trb_total',
]

# Tier 2: Spatial features (2019+ only, ~25% coverage)
TIER2_SPATIAL = [
    'avg_shot_dist', 'shot_dist_var',
    'corner_3_rate', 'corner_3_pct',
    'deep_3_rate', 'rim_purity',
    'xy_shots', 'xy_3_shots', 'xy_rim_shots',  # Coverage counts
]

# Career aggregate features (from prospect_career_v1)
CAREER_FEATURES = [
    'career_years',
    'final_trueShootingPct', 'final_usage', 'final_poss_total',
    'final_rim_fg_pct', 'final_three_fg_pct', 'final_ft_pct',
    'final_avg_shot_dist', 'final_corner_3_rate', 'final_rim_purity',
    'slope_trueShootingPct', 'slope_usage', 'slope_rim_fg_pct', 'slope_three_fg_pct',
    'career_wt_trueShootingPct', 'career_wt_usage',
    'delta_trueShootingPct', 'delta_usage',
]

# Target columns
PRIMARY_TARGET = 'y_peak_ovr'
AUX_TARGETS = ['gap_ts_legacy', 'year1_epm_tot', 'year1_epm_off', 'year1_epm_def']
BINARY_TARGET = 'made_nba'  # Derived: year1_mp >= 100

# Exposure/weight columns
EXPOSURE_COLS = ['year1_mp', 'peak_poss']


def load_college_features(split_id: str = 'ALL__ALL') -> pd.DataFrame:
    """Load college season features, filtered to specified split."""
    path = COLLEGE_FEATURE_STORE / "college_features_v1.parquet"
    if not path.exists():
        logger.warning(f"College features not found: {path}")
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    df = df[df['split_id'] == split_id].copy()
    # The feature store may contain duplicate rows for the same athlete/season/split.
    # These are typically exact duplicates from upstream joins; drop to prevent noise.
    df = df.drop_duplicates(subset=['athlete_id', 'season', 'split_id'])
    logger.info(f"Loaded {len(df):,} college feature rows (split={split_id})")
    return df


def load_career_features() -> pd.DataFrame:
    """Load career aggregate features."""
    path = COLLEGE_FEATURE_STORE / "prospect_career_v1.parquet"
    if not path.exists():
        logger.warning(f"Career features not found: {path}")
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} career feature rows")
    return df


def load_trajectory_features() -> pd.DataFrame:
    """Load trajectory stub features (list-valued sequences)."""
    path = OUTPUT_DIR / "trajectory_stub_v1.parquet"
    if not path.exists():
        logger.warning(f"Trajectory features not found: {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} trajectory feature rows")
    return df


def load_crosswalk() -> pd.DataFrame:
    """Load college-to-NBA crosswalk."""
    path = WAREHOUSE_V2 / "dim_player_nba_college_crosswalk.parquet"
    if not path.exists():
        logger.warning(f"Crosswalk not found: {path}")
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} crosswalk entries")
    return df


def load_nba_targets() -> pd.DataFrame:
    """Load and merge NBA target tables."""
    targets = pd.DataFrame()
    
    # Peak RAPM (primary target)
    peak_path = WAREHOUSE_V2 / "fact_player_peak_rapm.parquet"
    if peak_path.exists():
        peak = pd.read_parquet(peak_path)
        targets = peak[['nba_id', 'y_peak_ovr', 'y_peak_off', 'y_peak_def', 'peak_poss']].copy()
        logger.info(f"Loaded {len(peak):,} peak RAPM targets")
    
    # Year-1 EPM (auxiliary targets)
    y1_path = WAREHOUSE_V2 / "fact_player_year1_epm.parquet"
    if y1_path.exists():
        y1 = pd.read_parquet(y1_path)
        y1_cols = ['nba_id', 'year1_epm_tot', 'year1_epm_off', 'year1_epm_def', 
                   'year1_mp', 'year1_tspct', 'year1_usg']
        y1_cols = [c for c in y1_cols if c in y1.columns]
        
        if targets.empty:
            targets = y1[y1_cols].copy()
        else:
            targets = targets.merge(y1[y1_cols], on='nba_id', how='outer')
        logger.info(f"Loaded {len(y1):,} year-1 EPM targets")
    
    # Gaps (auxiliary targets)
    gaps_path = WAREHOUSE_V2 / "fact_player_nba_college_gaps.parquet"
    if gaps_path.exists():
        gaps = pd.read_parquet(gaps_path)
        gap_cols = ['nba_id', 'gap_ts_legacy', 'gap_usg_legacy', 'gap_dist_leap', 'gap_corner_rate']
        gap_cols = [c for c in gap_cols if c in gaps.columns]
        
        if targets.empty:
            targets = gaps[gap_cols].copy()
        else:
            targets = targets.merge(gaps[gap_cols], on='nba_id', how='outer')
        logger.info(f"Loaded {len(gaps):,} gap targets")
    
    return targets


def get_final_college_season(college_features: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the final college season for each athlete.
    This is what we use to predict NBA outcomes.
    """
    if college_features.empty:
        return pd.DataFrame()
    
    # Sort and get last season.
    # Important: GroupBy.last() can skip NaNs per-column, which can accidentally pull
    # values from earlier seasons. We want the literal final season row.
    df = college_features.sort_values(['athlete_id', 'season'])
    # Remove exact duplicates only. Do not collapse athlete-season rows, since transfers
    # can legitimately create multiple rows for the same season.
    df = df.drop_duplicates()

    # Handle transfers / multi-team seasons:
    # - Sum count columns across rows within (athlete_id, season)
    # - Choose teamId as the team with max minutes_total
    # - Average team_pace weighted by minutes_total (if available)
    sum_cols = [
        'shots_total', 'fga_total', 'ft_att',
        'rim_att', 'rim_made',
        'three_att', 'three_made',
        'mid_att', 'mid_made',
        'ft_made',
        'assisted_made_rim', 'assisted_made_three', 'assisted_made_mid',
        'xy_shots', 'sum_dist_ft', 'corner_3_att', 'corner_3_made',
        'xy_3_shots', 'xy_rim_shots', 'deep_3_att', 'rim_rest_att',
        'sum_dist_sq_ft',
        'ast_total', 'tov_total', 'stl_total', 'blk_total',
        'orb_total', 'drb_total', 'trb_total',
        'minutes_total',
        'team_pace',
    ]
    sum_cols = [c for c in sum_cols if c in df.columns]

    grp = df.groupby(['athlete_id', 'season'], as_index=False)
    agg = grp[sum_cols].sum(min_count=1)

    # Meta from max minutes row
    meta_cols = ['teamId', 'is_power_conf']
    meta_cols = [c for c in meta_cols if c in df.columns]
    if 'minutes_total' in df.columns and meta_cols:
        meta = (
            df.sort_values(['athlete_id', 'season', 'minutes_total'], ascending=[True, True, False])
            .drop_duplicates(['athlete_id', 'season'])[['athlete_id', 'season'] + meta_cols]
        )
        agg = agg.merge(meta, how='left', on=['athlete_id', 'season'])

    # Recompute core rates + spatial fields (match college_features conventions).
    if 'rim_att' in agg.columns and 'rim_made' in agg.columns:
        agg['rim_fg_pct'] = np.where(agg['rim_att'] > 0, agg['rim_made'] / agg['rim_att'], np.nan)
    if 'three_att' in agg.columns and 'three_made' in agg.columns:
        agg['three_fg_pct'] = np.where(agg['three_att'] > 0, agg['three_made'] / agg['three_att'], np.nan)
    if 'mid_att' in agg.columns and 'mid_made' in agg.columns:
        agg['mid_fg_pct'] = np.where(agg['mid_att'] > 0, agg['mid_made'] / agg['mid_att'], np.nan)
    if 'ft_att' in agg.columns and 'ft_made' in agg.columns:
        agg['ft_pct'] = np.where(agg['ft_att'] > 0, agg['ft_made'] / agg['ft_att'], np.nan)

    # Spatial recompute + gating
    if {'shots_total', 'xy_shots'}.issubset(agg.columns):
        agg['xy_coverage'] = np.where(agg['shots_total'] > 0, agg['xy_shots'] / agg['shots_total'], np.nan)
    if {'xy_shots', 'sum_dist_ft'}.issubset(agg.columns):
        gate_xy = agg['xy_shots'] >= 25
        avg_dist = np.where(gate_xy & (agg['xy_shots'] > 0), agg['sum_dist_ft'] / agg['xy_shots'], np.nan)
        agg['avg_shot_dist'] = avg_dist
    if {'xy_shots', 'sum_dist_sq_ft', 'avg_shot_dist'}.issubset(agg.columns):
        gate_xy = agg['xy_shots'] >= 25
        var_dist = np.where(
            gate_xy & (agg['xy_shots'] > 0),
            (agg['sum_dist_sq_ft'] / agg['xy_shots']) - np.square(agg['avg_shot_dist']),
            np.nan,
        )
        agg['shot_dist_var'] = np.where(np.isnan(var_dist), np.nan, np.maximum(0.0, var_dist))

    if {'xy_3_shots', 'three_att', 'corner_3_att'}.issubset(agg.columns):
        gate_xy_3 = agg['xy_3_shots'] >= 15
        agg['corner_3_rate'] = np.where(gate_xy_3 & (agg['three_att'] > 0), agg['corner_3_att'] / agg['three_att'], np.nan)
    if {'xy_3_shots', 'corner_3_att', 'corner_3_made'}.issubset(agg.columns):
        gate_xy_3 = agg['xy_3_shots'] >= 15
        agg['corner_3_pct'] = np.where(gate_xy_3 & (agg['corner_3_att'] > 0), agg['corner_3_made'] / agg['corner_3_att'], np.nan)
    if {'xy_3_shots', 'three_att', 'deep_3_att'}.issubset(agg.columns):
        gate_xy_3 = agg['xy_3_shots'] >= 15
        agg['deep_3_rate'] = np.where(gate_xy_3 & (agg['three_att'] > 0), agg['deep_3_att'] / agg['three_att'], np.nan)
    if {'xy_rim_shots', 'rim_att', 'rim_rest_att'}.issubset(agg.columns):
        gate_xy_rim = agg['xy_rim_shots'] >= 20
        agg['rim_purity'] = np.where(gate_xy_rim & (agg['rim_att'] > 0), agg['rim_rest_att'] / agg['rim_att'], np.nan)

    # Now take the literal final season row per athlete.
    agg = agg.sort_values(['athlete_id', 'season'])
    final = agg.groupby('athlete_id').tail(1).reset_index(drop=True)
    
    # Rename to indicate these are "final season" features
    cols_to_rename = {}
    for col in TIER1_SHOT_PROFILE + TIER1_CREATION + TIER1_IMPACT + TIER1_CONTEXT + TIER2_SPATIAL:
        if col in final.columns:
            cols_to_rename[col] = f'college_{col}'
    
    final = final.rename(columns=cols_to_rename)
    final = final.rename(columns={'season': 'college_final_season'})
    
    logger.info(f"Extracted {len(final):,} final college seasons")
    return final


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features and coverage masks."""
    df = df.copy()
    
    # Binary target: made_nba (played >= 100 minutes in Year 1)
    if 'year1_mp' in df.columns:
        df['made_nba'] = (df['year1_mp'] >= 100).astype(int)
    else:
        df['made_nba'] = np.nan
    
    # Coverage masks for Tier 2
    if 'college_xy_shots' in df.columns:
        df['has_spatial_data'] = (df['college_xy_shots'] >= 25).astype(int)
    else:
        df['has_spatial_data'] = 0
    
    # Assisted share rates (from counts)
    for zone in ['rim', 'mid', 'three']:
        made_col = f'college_{zone}_made'
        assisted_col = f'college_assisted_{zone}_made'
        if made_col in df.columns and assisted_col in df.columns:
            df[f'college_assisted_share_{zone}'] = np.where(
                df[made_col] > 0,
                df[assisted_col] / df[made_col],
                np.nan
            )

    # Shot-mix shares (derived from counts; missingness-safe).
    # These features are important for era adjustment (3P rates drift a lot).
    if 'college_shots_total' in df.columns:
        shots = df['college_shots_total'].astype(float)
        for zone in ['rim', 'mid', 'three']:
            att = f'college_{zone}_att'
            if att in df.columns:
                df[f'college_{zone}_share'] = np.where(shots > 0, df[att].astype(float) / shots, np.nan)

    # Simple per-minute defensive activity rates (impact-adjacent).
    if 'college_minutes_total' in df.columns:
        mins = df['college_minutes_total'].astype(float)
        for stat in ['stl_total', 'blk_total', 'tov_total', 'ast_total']:
            col = f'college_{stat}'
            if col in df.columns:
                df[f'{col}_per40'] = np.where(mins > 0, df[col].astype(float) / mins * 40.0, np.nan)
    
    # True Shooting % (if not already present)
    # Missingness-safe: do not fabricate 0s when components are missing.
    if 'final_trueShootingPct' not in df.columns:
        rim_made = df['college_rim_made'] if 'college_rim_made' in df.columns else np.nan
        mid_made = df['college_mid_made'] if 'college_mid_made' in df.columns else np.nan
        three_made = df['college_three_made'] if 'college_three_made' in df.columns else np.nan
        ft_made = df['college_ft_made'] if 'college_ft_made' in df.columns else np.nan
        fga = df['college_fga_total'] if 'college_fga_total' in df.columns else np.nan
        fta = df['college_ft_att'] if 'college_ft_att' in df.columns else np.nan

        any_pts = pd.Series(rim_made).notna() | pd.Series(mid_made).notna() | pd.Series(three_made).notna() | pd.Series(ft_made).notna()
        pts = np.where(
            any_pts,
            2 * pd.Series(rim_made).fillna(0) + 2 * pd.Series(mid_made).fillna(0) + 3 * pd.Series(three_made).fillna(0) + pd.Series(ft_made).fillna(0),
            np.nan,
        )
        denom = 2 * (pd.Series(fga).astype(float) + 0.44 * pd.Series(fta).astype(float))
        df['final_trueShootingPct'] = np.where((denom > 0) & pd.Series(pts).notna(), pts / denom, np.nan)
    
    return df


def apply_era_normalization(df: pd.DataFrame, era_col: str = 'college_final_season') -> pd.DataFrame:
    """
    Z-score features by era (season) to handle drift.
    Only applies to features known to drift across eras.
    """
    df = df.copy()
    
    if era_col not in df.columns:
        logger.warning(f"Era column '{era_col}' not found, skipping normalization")
        return df
    
    # Features to normalize (rates that drift)
    drift_features = [
        # shooting + mix
        'college_rim_fg_pct', 'college_mid_fg_pct', 'college_three_fg_pct', 'college_ft_pct',
        'college_rim_share', 'college_mid_share', 'college_three_share',
        # creation-ish
        'college_assisted_share_rim', 'college_assisted_share_three',
        # career-level summary rates
        'final_trueShootingPct', 'final_usage',
        # activity / role
        'college_tov_total_per40', 'college_ast_total_per40',
        'college_stl_total_per40', 'college_blk_total_per40',
    ]
    
    for col in drift_features:
        if col in df.columns:
            grouped = df.groupby(era_col)[col]
            mean = grouped.transform('mean')
            std = grouped.transform('std').replace(0, 1)
            df[f'{col}_z'] = (df[col] - mean) / std
            logger.debug(f"Normalized {col} by era")
    
    return df


def apply_team_residualization(
    df: pd.DataFrame,
    season_col: str = 'college_final_season',
    team_col: str = 'college_teamId',
) -> pd.DataFrame:
    """
    Remove team-season means from select features (team context adjustment).
    This is a lightweight alternative to full adjusted-plus-minus for non-point outcomes.
    """
    df = df.copy()
    if season_col not in df.columns or team_col not in df.columns:
        return df

    features = [
        'final_trueShootingPct',
        'final_usage',
        'college_three_fg_pct',
        'college_three_share',
        'college_ast_total_per40',
        'college_tov_total_per40',
        'college_stl_total_per40',
        'college_blk_total_per40',
    ]
    for col in features:
        if col not in df.columns:
            continue
        grp = df.groupby([season_col, team_col])[col]
        mu = grp.transform('mean')
        df[f'{col}_team_resid'] = df[col] - mu

    return df


def build_unified_training_table(
    use_career_features: bool = True,
    use_trajectory_features: bool = True,
    apply_normalization: bool = True,
    min_targets: int = 1,
) -> pd.DataFrame:
    """
    Build the unified training table.
    
    Args:
        use_career_features: Include career aggregate features
        apply_normalization: Apply era-based z-score normalization
        min_targets: Minimum number of non-null targets required
    
    Returns:
        DataFrame ready for model training
    """
    logger.info("=" * 60)
    logger.info("Building Unified Training Table")
    logger.info("=" * 60)
    
    # 1. Load all data sources
    college_features = load_college_features()
    career_features = load_career_features()
    trajectory_features = load_trajectory_features()
    crosswalk = load_crosswalk()
    nba_targets = load_nba_targets()
    
    # Check for required data
    if crosswalk.empty:
        logger.error("Crosswalk is required but not found!")
        return pd.DataFrame()
    
    if nba_targets.empty:
        logger.error("NBA targets are required but not found!")
        return pd.DataFrame()
    
    # 2. Extract final college season features
    if not college_features.empty:
        final_college = get_final_college_season(college_features)
    else:
        final_college = pd.DataFrame()
    
    # 3. Join college features via crosswalk
    # crosswalk has: athlete_id (college) -> nba_id
    logger.info("Joining college features to NBA targets via crosswalk...")
    
    df = crosswalk[['athlete_id', 'nba_id']].copy()
    
    # Join final college season
    if not final_college.empty:
        df = df.merge(final_college, on='athlete_id', how='left')
    
    # Join career features
    if use_career_features and not career_features.empty:
        df = df.merge(career_features, on='athlete_id', how='left', suffixes=('', '_career'))
    
    # Join trajectory features (new multi-season encoding)
    if use_trajectory_features and not trajectory_features.empty:
        traj_cols = ['athlete_id'] + [c for c in trajectory_features.columns if c != 'athlete_id']
        df = df.merge(trajectory_features[traj_cols], on='athlete_id', how='left', suffixes=('', '_traj'))
        logger.info(f"  Added {len(trajectory_features.columns)-1} trajectory features")
    
    # Join NBA targets
    df = df.merge(nba_targets, on='nba_id', how='inner')  # Inner: must have at least one target
    
    logger.info(f"Joined dataset: {len(df):,} rows")
    
    # 4. Compute derived features
    df = compute_derived_features(df)
    
    # 5. Apply era normalization
    if apply_normalization:
        df = apply_era_normalization(df)
        df = apply_team_residualization(df)
    
    # 6. Filter: require minimum number of targets
    target_cols = [PRIMARY_TARGET] + AUX_TARGETS
    target_cols = [c for c in target_cols if c in df.columns]
    
    if target_cols:
        target_count = df[target_cols].notna().sum(axis=1)
        df = df[target_count >= min_targets]
        logger.info(f"After target filter (min={min_targets}): {len(df):,} rows")
    
    # 7. Log coverage statistics
    log_coverage_stats(df)
    
    return df


def log_coverage_stats(df: pd.DataFrame) -> None:
    """Log coverage statistics for the training table."""
    logger.info("\n" + "=" * 40)
    logger.info("COVERAGE STATISTICS")
    logger.info("=" * 40)
    
    logger.info(f"Total rows: {len(df):,}")
    
    # Target coverage
    target_cols = [PRIMARY_TARGET] + AUX_TARGETS + [BINARY_TARGET]
    for col in target_cols:
        if col in df.columns:
            n = df[col].notna().sum()
            pct = n / len(df) * 100
            logger.info(f"  {col}: {n:,} ({pct:.1f}%)")
    
    # Tier 2 coverage
    if 'has_spatial_data' in df.columns:
        n = df['has_spatial_data'].sum()
        pct = n / len(df) * 100
        logger.info(f"  has_spatial_data: {n:,} ({pct:.1f}%)")
    
    # Era distribution
    if 'college_final_season' in df.columns:
        logger.info("\nSeason distribution:")
        dist = df['college_final_season'].value_counts().sort_index()
        for season, count in dist.items():
            logger.info(f"  {season}: {count:,}")


def save_training_table(df: pd.DataFrame, filename: str = "unified_training_table.parquet") -> Path:
    """Save the training table to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved training table to {output_path}")
    return output_path


def main():
    """Main entry point."""
    # Build the table
    df = build_unified_training_table(
        use_career_features=True,
        apply_normalization=True,
        min_targets=1,
    )
    
    if df.empty:
        logger.error("Failed to build training table - no data!")
        return
    
    # Save
    save_training_table(df)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("UNIFIED TRAINING TABLE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Columns: {len(df.columns)}")
    logger.info(f"Feature columns: {[c for c in df.columns if c.startswith('college_') or c.startswith('final_') or c.startswith('career_')]}")


if __name__ == "__main__":
    main()
