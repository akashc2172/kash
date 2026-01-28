"""
NBA Feature Transforms
======================
Implements transformations specified in NBA_Generative_Prospect_Model_Proposal_v12.

Transformations:
- z-score by era (season)
- logit for percentages (bounded [0,1])
- weights via 1/mp for noise modeling
- indicator columns for missingness (MAR handling)

Usage:
    from nba_feature_transforms import apply_all_transforms
    df = apply_all_transforms(df)
"""

import numpy as np
import pandas as pd
from typing import List, Optional

# Constants
MIN_MP_THRESHOLD = 200  # Minimum minutes for reliable Year-1 stats
MIN_POSS_THRESHOLD = 1000  # Minimum possessions for peak RAPM reliability
EPSILON = 1e-6  # Avoid divide-by-zero


def logit_transform(x: pd.Series, clip_eps: float = 0.01) -> pd.Series:
    """
    Apply logit transform to percentage columns (bounded [0,1]).
    Clips to [clip_eps, 1-clip_eps] to avoid inf.
    
    Auto-detects if input is in [0,100] scale and converts to [0,1].
    
    Per spec: "logit for percentages"
    """
    # Auto-detect 0-100 scale: if max > 1.5, assume 0-100
    if x.max() > 1.5:
        import warnings
        warnings.warn(
            f"Detected 0-100 scale in percentage column (max={x.max():.1f}), converting to 0-1",
            UserWarning
        )
        x = x / 100.0
    
    clipped = x.clip(lower=clip_eps, upper=1-clip_eps)
    return np.log(clipped / (1 - clipped))


def zscore_by_era(df: pd.DataFrame, col: str, era_col: str = 'season') -> pd.Series:
    """
    Z-score a column within each era (season).
    
    Per spec: "z-score within era"
    """
    grouped = df.groupby(era_col)[col]
    mean = grouped.transform('mean')
    std = grouped.transform('std').replace(0, 1)  # Avoid divide by zero
    return (df[col] - mean) / std


def compute_mp_weights(mp: pd.Series, mp_ref: float = 2000.0) -> pd.Series:
    """
    Compute reliability weights based on minutes played.
    
    Per spec: "w = min(1, mp / mp_ref); σ² ∝ 1/(mp+ε)"
    """
    return (mp / mp_ref).clip(upper=1.0)


def compute_mp_variance(mp: pd.Series) -> pd.Series:
    """
    Compute observation noise variance inversely proportional to minutes.
    
    Per spec: "σ² ∝ 1/(mp+ε)"
    """
    return 1.0 / (mp + EPSILON)


def compute_poss_weights(poss: pd.Series, poss_ref: float = 10000.0) -> pd.Series:
    """
    Compute reliability weights based on possessions (for peak RAPM).
    
    Per spec: "σ² ∝ 1/(peak_poss+ε)"
    """
    return (poss / poss_ref).clip(upper=1.0)


def add_missing_indicator(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Add a binary indicator column for missingness (MAR handling).
    
    Per spec: "mark as missing; treat as MAR with indicator; do not impute with zeros"
    """
    indicator_col = f'{col}_missing'
    df[indicator_col] = df[col].isna().astype(int)
    return df


def identify_percentage_columns() -> List[str]:
    """
    Returns list of columns that should receive logit transform.
    These are efficiency/percentage metrics bounded [0,1].
    """
    return [
        'tspct', 'efg', 'fgpct_rim', 'fgpct_mid', 'fg2pct', 'fg3pct', 'ftpct',
        'year1_tspct',
        # Add more as needed from the registry
    ]


def identify_rate_columns() -> List[str]:
    """
    Returns list of columns that should be z-scored by era.
    These are rate/advanced stats.
    """
    return [
        'orbpct', 'drbpct', 'astpct', 'topct', 'stlpct', 'blkpct',
        'year1_usg', 'year1_epm_tot', 'year1_epm_off', 'year1_epm_def',
        'year1_epm_ewins',
        # Add more as needed from the registry
    ]


def identify_aux_observation_columns() -> List[str]:
    """
    Columns used as auxiliary observations p(a | z).
    Per spec: "Year-1 per-season NBA stat used in the 'aux head' likelihood"
    """
    return [
        'mpg', 'off', 'def', 'tot', 'ewins', 'usg', 'tspct', 'efg',
        'fgpct_rim', 'fgpct_mid', 'fg2pct', 'fg3pct', 'ftpct',
        'orbpct', 'drbpct', 'astpct', 'topct', 'stlpct', 'blkpct',
        'fga_rim_75', 'fga_mid_75', 'fg3a_75', 'fta_75', 'fga_75',
    ]


def apply_all_transforms(df: pd.DataFrame, era_col: str = 'rookie_season_year') -> pd.DataFrame:
    """
    Apply all transformations specified in the variable registry.
    
    Args:
        df: Input DataFrame
        era_col: Column to use for era-based z-scoring. Default is 'rookie_season_year'.
                Common options: 'rookie_season_year', 'draft_year', 'season'
    
    Returns a new DataFrame with transformed columns.
    """
    df = df.copy()
    
    # Validate era_col exists
    if era_col not in df.columns:
        import warnings
        warnings.warn(
            f"Era column '{era_col}' not found in DataFrame. Z-score transforms will be skipped. "
            f"Available columns: {[c for c in df.columns if 'year' in c.lower() or 'season' in c.lower()]}",
            UserWarning
        )
    
    # 1. Logit transform for percentages
    for col in identify_percentage_columns():
        if col in df.columns:
            new_col = f'{col}_logit'
            df[new_col] = logit_transform(df[col])
    
    # 2. Z-score by era for rate columns (only if era_col exists)
    if era_col in df.columns:
        for col in identify_rate_columns():
            if col in df.columns:
                new_col = f'{col}_z'
                df[new_col] = zscore_by_era(df, col, era_col)
    
    # 3. Compute weights
    if 'year1_mp' in df.columns:
        df['year1_mp_weight'] = compute_mp_weights(df['year1_mp'])
        df['year1_mp_variance'] = compute_mp_variance(df['year1_mp'])
    
    if 'peak_poss' in df.columns:
        df['peak_poss_weight'] = compute_poss_weights(df['peak_poss'])
        df['peak_poss_variance'] = 1.0 / (df['peak_poss'] + EPSILON)
    
    return df


def filter_reliable_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to samples with sufficient exposure.
    
    Per spec: 
    - "Drop from loss if year1_mp below threshold"
    - "Downweight by peak_poss; exclude if peak_poss below threshold"
    """
    mask = pd.Series(True, index=df.index)
    
    if 'year1_mp' in df.columns:
        mask &= (df['year1_mp'] >= MIN_MP_THRESHOLD) | df['year1_mp'].isna()
    
    if 'peak_poss' in df.columns:
        mask &= (df['peak_poss'] >= MIN_POSS_THRESHOLD) | df['peak_poss'].isna()
    
    return df[mask]


if __name__ == '__main__':
    print("NBA Feature Transforms module loaded.")
    print(f"Percentage columns: {identify_percentage_columns()}")
    print(f"Rate columns: {identify_rate_columns()}")
    print(f"Aux observation columns: {identify_aux_observation_columns()}")
