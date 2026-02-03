"""
Enhanced Feature Computation Module
====================================
Computes athleticism, pressure, and decision discipline features
that extend the base college feature store.

New Feature Blocks (Jan 2025):
- Athleticism: dunk_rate, putback_rate, transition_freq, rim_pressure_index
- Defense Activity: deflection_proxy, contest_proxy
- Decision Discipline: pressure_handle_proxy, clutch_shooting_delta
- Shot Creation: self_creation_rate, self_creation_eff
- Context: leverage_poss_share

Usage:
    from compute_enhanced_features import compute_all_enhanced_features
    df_enhanced = compute_all_enhanced_features(df_base, df_leverage_splits)
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Constants for shrinkage
BETA_PRIOR_ALPHA = 2.0  # Prior successes
BETA_PRIOR_BETA = 2.0   # Prior failures
MIN_ATTEMPTS_FOR_RATE = 10  # Minimum attempts before computing rate


def beta_shrink(made: pd.Series, att: pd.Series, 
                alpha: float = BETA_PRIOR_ALPHA, 
                beta: float = BETA_PRIOR_BETA) -> pd.Series:
    """
    Apply beta-binomial shrinkage to rate estimates.
    
    Formula: (made + alpha) / (att + alpha + beta)
    
    This pulls small-sample rates toward the prior mean (alpha / (alpha + beta)).
    """
    return (made + alpha) / (att + alpha + beta)


def compute_athleticism_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute athleticism-related features.
    
    Requires columns:
    - rim_att, rim_made: Rim shot attempts and makes
    - ft_att: Free throw attempts
    - fga_total: Total field goal attempts
    - shots_total: Total shots including FTs
    - orb_total: Offensive rebounds (from box stats)
    - poss_est: Estimated possessions
    
    Optional (for dunk detection):
    - dunk_att: Dunk attempts (if available from playType)
    - transition_att, transition_made: Transition attempts/makes
    - putback_att: Putback attempts
    
    Returns DataFrame with new columns added.
    """
    df = df.copy()
    
    # Dunk Rate: dunks / rim attempts
    # If dunk_att not available, set to NaN (requires playType parsing)
    if 'dunk_att' in df.columns:
        df['dunk_rate'] = np.where(
            df['rim_att'] >= MIN_ATTEMPTS_FOR_RATE,
            beta_shrink(df['dunk_att'], df['rim_att']),
            np.nan
        )
        df['dunk_freq'] = np.where(
            df['fga_total'] >= MIN_ATTEMPTS_FOR_RATE,
            df['dunk_att'] / df['fga_total'],
            np.nan
        )
    else:
        df['dunk_rate'] = np.nan
        df['dunk_freq'] = np.nan
    df['dunk_rate_missing'] = df['dunk_rate'].isna().astype(int)
    df['dunk_freq_missing'] = df['dunk_freq'].isna().astype(int)
    
    # Putback Rate: putback attempts / (OREBs + 1)
    # Requires linking OREB events to subsequent shots within 2s
    if 'putback_att' in df.columns and 'orb_total' in df.columns:
        df['putback_rate'] = np.where(
            df['orb_total'] >= 5,
            df['putback_att'] / (df['orb_total'] + 1),
            np.nan
        )
    else:
        df['putback_rate'] = np.nan
    df['putback_rate_missing'] = df['putback_rate'].isna().astype(int)
    
    # Transition Frequency & Efficiency
    if 'transition_att' in df.columns:
        df['transition_freq'] = np.where(
            df['fga_total'] >= MIN_ATTEMPTS_FOR_RATE,
            df['transition_att'] / df['fga_total'],
            np.nan
        )
        if 'transition_made' in df.columns:
            df['transition_eff'] = np.where(
                df['transition_att'] >= MIN_ATTEMPTS_FOR_RATE,
                beta_shrink(df['transition_made'], df['transition_att']),
                np.nan
            )
        else:
            df['transition_eff'] = np.nan
    else:
        df['transition_freq'] = np.nan
        df['transition_eff'] = np.nan
    df['transition_freq_missing'] = df['transition_freq'].isna().astype(int)
    df['transition_eff_missing'] = df['transition_eff'].isna().astype(int)
    
    # Rim Pressure Index: (rim_fga + 0.44*FTA) / poss_est
    # Measures ability to get to rim and draw fouls
    if 'poss_est' in df.columns:
        rim_plus_ft = df['rim_att'] + 0.44 * df['ft_att']
        df['rim_pressure_index'] = np.where(
            df['poss_est'] >= 50,
            rim_plus_ft / df['poss_est'],
            np.nan
        )
    else:
        df['rim_pressure_index'] = np.nan
    df['rim_pressure_index_missing'] = df['rim_pressure_index'].isna().astype(int)
    
    return df


def compute_defense_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute defense activity features.
    
    Requires columns:
    - stl_total, blk_total: Steals and blocks
    - poss_est: Estimated possessions
    - foul_total or pf_total: Personal fouls (if available)
    
    Returns DataFrame with new columns added.
    """
    df = df.copy()
    
    # First compute base rates if not present
    if 'stl_rate' not in df.columns and 'stl_total' in df.columns:
        df['stl_rate'] = np.where(
            df['poss_est'] >= 50,
            df['stl_total'] / df['poss_est'],
            np.nan
        )
    
    if 'blk_rate' not in df.columns and 'blk_total' in df.columns:
        df['blk_rate'] = np.where(
            df['poss_est'] >= 50,
            df['blk_total'] / df['poss_est'],
            np.nan
        )
    
    # Foul rate
    foul_col = 'pf_total' if 'pf_total' in df.columns else 'foul_total'
    if foul_col in df.columns:
        df['foul_rate'] = np.where(
            df['poss_est'] >= 50,
            df[foul_col] / df['poss_est'],
            np.nan
        )
    else:
        df['foul_rate'] = np.nan
    
    # Deflection Proxy: 1.5*stl_rate + 0.5*blk_rate
    # Weighted combination emphasizing steals (more indicative of active hands)
    if 'stl_rate' in df.columns and 'blk_rate' in df.columns:
        df['deflection_proxy'] = 1.5 * df['stl_rate'].fillna(0) + 0.5 * df['blk_rate'].fillna(0)
        df['deflection_proxy'] = np.where(
            df['stl_rate'].notna() | df['blk_rate'].notna(),
            df['deflection_proxy'],
            np.nan
        )
    else:
        df['deflection_proxy'] = np.nan
    df['deflection_proxy_missing'] = df['deflection_proxy'].isna().astype(int)
    
    # Contest Proxy: blk_rate / (blk_rate + foul_rate + 0.01)
    # Measures clean contests (blocks without fouling)
    if 'blk_rate' in df.columns and 'foul_rate' in df.columns:
        denom = df['blk_rate'].fillna(0) + df['foul_rate'].fillna(0) + 0.01
        df['contest_proxy'] = np.where(
            df['blk_rate'].notna() & df['foul_rate'].notna(),
            df['blk_rate'].fillna(0) / denom,
            np.nan
        )
    else:
        df['contest_proxy'] = np.nan
    df['contest_proxy_missing'] = df['contest_proxy'].isna().astype(int)
    
    return df


def compute_pressure_features(df_all: pd.DataFrame, 
                               df_high_lev: Optional[pd.DataFrame] = None,
                               df_low_lev: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute decision discipline / pressure features.
    
    Requires:
    - df_all: Base features for ALL split
    - df_high_lev: Features for HIGH_LEVERAGE split (optional)
    - df_low_lev: Features for LOW_LEVERAGE split (optional)
    
    If leverage splits not provided, attempts to extract from df_all using split_id.
    
    Returns DataFrame with pressure features added.
    """
    df = df_all.copy()
    
    # Try to extract leverage splits from split_id if not provided
    if df_high_lev is None and 'split_id' in df.columns:
        df_high_lev = df[df['split_id'].str.contains('HIGH_LEVERAGE')].copy()
        df_low_lev = df[df['split_id'].str.contains('LOW_LEVERAGE')].copy()
        df = df[df['split_id'] == 'ALL__ALL'].copy()
    
    if df_high_lev is None or df_low_lev is None or len(df_high_lev) == 0:
        logger.warning("Leverage splits not available, pressure features will be NaN")
        df['pressure_handle_proxy'] = np.nan
        df['clutch_shooting_delta'] = np.nan
        df['pressure_handle_proxy_missing'] = 1
        df['clutch_shooting_delta_missing'] = 1
        return df
    
    # Compute TO rate for each split
    # TO rate = tov / (fga + 0.44*fta + tov)
    def compute_to_rate(d):
        if 'tov_total' not in d.columns:
            return pd.Series(np.nan, index=d.index)
        denom = d['fga_total'] + 0.44 * d['ft_att'] + d['tov_total']
        return np.where(denom > 10, d['tov_total'] / denom, np.nan)
    
    # Compute TS% proxy for each split
    def compute_ts_proxy(d):
        pts = d['rim_made'] * 2 + d['three_made'] * 3 + d['mid_made'] * 2 + d['ft_made']
        tsa = d['fga_total'] + 0.44 * d['ft_att']
        return np.where(tsa > 10, pts / (2 * tsa), np.nan)
    
    # Merge high/low leverage stats
    key_cols = ['season', 'athlete_id']
    
    # Ensure key columns exist
    if not all(c in df_high_lev.columns for c in key_cols):
        logger.warning("Key columns missing from leverage splits")
        df['pressure_handle_proxy'] = np.nan
        df['clutch_shooting_delta'] = np.nan
        df['pressure_handle_proxy_missing'] = 1
        df['clutch_shooting_delta_missing'] = 1
        return df
    
    df_high_lev['to_rate_high'] = compute_to_rate(df_high_lev)
    df_high_lev['ts_proxy_high'] = compute_ts_proxy(df_high_lev)
    
    df_low_lev['to_rate_low'] = compute_to_rate(df_low_lev)
    df_low_lev['ts_proxy_low'] = compute_ts_proxy(df_low_lev)
    
    # Merge
    df = df.merge(
        df_high_lev[key_cols + ['to_rate_high', 'ts_proxy_high']],
        on=key_cols, how='left'
    )
    df = df.merge(
        df_low_lev[key_cols + ['to_rate_low', 'ts_proxy_low']],
        on=key_cols, how='left'
    )
    
    # Pressure Handle Proxy: TO_rate(high) - TO_rate(low)
    # Negative = handles pressure well (fewer TOs under pressure)
    df['pressure_handle_proxy'] = df['to_rate_high'] - df['to_rate_low']
    df['pressure_handle_proxy_missing'] = (
        df['to_rate_high'].isna() | df['to_rate_low'].isna()
    ).astype(int)
    
    # Clutch Shooting Delta: TS%(high) - TS%(low)
    # Positive = rises to occasion (better efficiency under pressure)
    df['clutch_shooting_delta'] = df['ts_proxy_high'] - df['ts_proxy_low']
    df['clutch_shooting_delta_missing'] = (
        df['ts_proxy_high'].isna() | df['ts_proxy_low'].isna()
    ).astype(int)
    
    # Clean up temp columns
    for col in ['to_rate_high', 'to_rate_low', 'ts_proxy_high', 'ts_proxy_low']:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    return df


def compute_creation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute shot creation features.
    
    Requires columns:
    - assisted_made_rim, assisted_made_three, assisted_made_mid: Assisted makes by zone
    - rim_made, three_made, mid_made: Total makes by zone
    - fga_total: Total FGA
    
    Returns DataFrame with creation features added.
    """
    df = df.copy()
    
    # Total assisted makes
    assisted_cols = ['assisted_made_rim', 'assisted_made_three', 'assisted_made_mid']
    if all(c in df.columns for c in assisted_cols):
        df['assisted_made_total'] = (
            df['assisted_made_rim'].fillna(0) + 
            df['assisted_made_three'].fillna(0) + 
            df['assisted_made_mid'].fillna(0)
        )
    else:
        df['assisted_made_total'] = 0
    
    # Total makes
    made_cols = ['rim_made', 'three_made', 'mid_made']
    if all(c in df.columns for c in made_cols):
        df['fg_made_total'] = (
            df['rim_made'].fillna(0) + 
            df['three_made'].fillna(0) + 
            df['mid_made'].fillna(0)
        )
    else:
        df['fg_made_total'] = 0
    
    # Unassisted makes
    df['unassisted_made'] = df['fg_made_total'] - df['assisted_made_total']
    
    # Self Creation Rate: unassisted FGA / total FGA
    # Approximation: unassisted_made / fg_made_total (assumes similar make rates)
    df['self_creation_rate'] = np.where(
        df['fg_made_total'] >= MIN_ATTEMPTS_FOR_RATE,
        df['unassisted_made'] / df['fg_made_total'],
        np.nan
    )
    df['self_creation_rate_missing'] = df['self_creation_rate'].isna().astype(int)
    
    # Self Creation Efficiency: unassisted points / (2 * unassisted FGA)
    # Approximation using makes (assumes 2pt value for simplicity)
    # More accurate would require tracking unassisted attempts separately
    df['self_creation_eff'] = np.where(
        df['unassisted_made'] >= 5,
        beta_shrink(df['unassisted_made'], df['unassisted_made'] * 1.5),  # Rough FGA estimate
        np.nan
    )
    df['self_creation_eff_missing'] = df['self_creation_eff'].isna().astype(int)
    
    return df


def compute_leverage_context(df: pd.DataFrame, 
                              df_high_lev: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute leverage context features.
    
    Requires:
    - df: Base features
    - df_high_lev: High leverage split features
    
    Returns DataFrame with context features added.
    """
    df = df.copy()
    
    if df_high_lev is None and 'split_id' in df.columns:
        df_high_lev = df[df['split_id'].str.contains('HIGH_LEVERAGE')].copy()
        df = df[df['split_id'] == 'ALL__ALL'].copy()
    
    if df_high_lev is None or len(df_high_lev) == 0:
        df['leverage_poss_share'] = np.nan
        df['leverage_poss_share_missing'] = 1
        return df
    
    key_cols = ['season', 'athlete_id']
    
    if not all(c in df_high_lev.columns for c in key_cols):
        df['leverage_poss_share'] = np.nan
        df['leverage_poss_share_missing'] = 1
        return df
    
    # Get high leverage shots as proxy for high leverage possessions
    if 'shots_total' in df_high_lev.columns:
        high_lev_shots = df_high_lev.groupby(key_cols)['shots_total'].sum().reset_index()
        high_lev_shots = high_lev_shots.rename(columns={'shots_total': 'high_lev_shots'})
        
        df = df.merge(high_lev_shots, on=key_cols, how='left')
        
        df['leverage_poss_share'] = np.where(
            df['shots_total'] >= MIN_ATTEMPTS_FOR_RATE,
            df['high_lev_shots'].fillna(0) / df['shots_total'],
            np.nan
        )
        
        df = df.drop(columns=['high_lev_shots'], errors='ignore')
    else:
        df['leverage_poss_share'] = np.nan
    
    df['leverage_poss_share_missing'] = df['leverage_poss_share'].isna().astype(int)
    
    return df


def compute_all_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all enhanced features in one pass.
    
    Args:
        df: Base feature DataFrame with split_id column
    
    Returns:
        DataFrame with all enhanced features added (for ALL__ALL split only)
    """
    logger.info("Computing enhanced features...")
    
    # Filter to ALL__ALL split for base
    if 'split_id' in df.columns:
        df_all = df[df['split_id'] == 'ALL__ALL'].copy()
        df_high = df[df['split_id'].str.startswith('HIGH_LEVERAGE')].copy()
        df_low = df[df['split_id'].str.startswith('LOW_LEVERAGE')].copy()
    else:
        df_all = df.copy()
        df_high = None
        df_low = None
    
    # Apply all feature computations
    df_all = compute_athleticism_features(df_all)
    logger.info("  Computed athleticism features")
    
    df_all = compute_defense_features(df_all)
    logger.info("  Computed defense features")
    
    df_all = compute_pressure_features(df_all, df_high, df_low)
    logger.info("  Computed pressure features")
    
    df_all = compute_creation_features(df_all)
    logger.info("  Computed creation features")
    
    df_all = compute_leverage_context(df_all, df_high)
    logger.info("  Computed leverage context features")
    
    # Summary
    new_features = [
        'dunk_rate', 'dunk_freq', 'putback_rate', 'transition_freq', 'transition_eff',
        'rim_pressure_index', 'deflection_proxy', 'contest_proxy',
        'pressure_handle_proxy', 'clutch_shooting_delta',
        'self_creation_rate', 'self_creation_eff', 'leverage_poss_share'
    ]
    
    available = [f for f in new_features if f in df_all.columns and df_all[f].notna().any()]
    logger.info(f"  Available enhanced features: {available}")
    
    return df_all


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Enhanced Feature Computation Module")
    print("=" * 40)
    print("\nFeature Blocks:")
    print("  - Athleticism: dunk_rate, dunk_freq, putback_rate, transition_freq, transition_eff, rim_pressure_index")
    print("  - Defense: deflection_proxy, contest_proxy")
    print("  - Pressure: pressure_handle_proxy, clutch_shooting_delta")
    print("  - Creation: self_creation_rate, self_creation_eff")
    print("  - Context: leverage_poss_share")
