"""
NBA Data Loader
===============
Loads and joins NBA Warehouse v2 tables to create the modeling dataset.

Tables joined:
- dim_player_crosswalk: Identity mapping
- dim_player_nba: Bio/Anthro features
- fact_player_year1_epm: Rookie season stats (inputs + aux targets)
- fact_player_peak_rapm: Primary target (peak 3Y RAPM)

Per spec: "nba_id is used only to link sources; never fed as a feature"
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List

# Default paths
WAREHOUSE_DIR = Path("data/warehouse_v2")

# =============================================================================
# LEAKAGE PREVENTION: Explicit allowlist of draft-time-safe feature columns
# =============================================================================
# These are the ONLY columns that should be used as model inputs (X).
# All other columns are either:
# - Identity columns (never features)
# - Post-draft information (would leak)
# - Targets/labels (not inputs)

DRAFT_TIME_SAFE_COLUMNS = {
    # Bio/Anthro (known at draft time from combine or college records)
    'ht_first', 'ht_max', 'ht_peak_delta',
    'wt_first', 'wt_max', 'wt_peak_delta',
    
    # Draft metadata (known at draft time by definition)
    'draft_year',
    
    # NOTE: rookie_season_year is borderline - it's derived, not draft-time.
    # We include it ONLY for era normalization, NOT as a predictive feature.
    # If using it as a feature, the model could learn "era shortcuts."
}

# Prefixes for college features (all draft-time safe by definition)
COLLEGE_FEATURE_PREFIXES = {
    'cbb__',      # College basketball box stats
    'pbp__',      # Play-by-play derived features
    'cbd__',      # CollegeBasketballData features
    'shot__',     # Shot profile features
    'lineup__',   # Lineup-based impact features
    'college_',   # Generic college prefix
}

# Columns that are FORBIDDEN in X (would cause leakage)
FORBIDDEN_FEATURE_COLUMNS = {
    # Identity columns
    'nba_id', 'player_name', 'pid', 'bbr_id',
    
    # NBA performance (post-draft leakage)
    'year1_epm_tot', 'year1_epm_off', 'year1_epm_def', 'year1_epm_ewins',
    'year1_mp', 'year1_usg', 'year1_tspct',
    
    # Target columns
    'y_peak_ovr', 'y_peak_off', 'y_peak_def', 'peak_poss',
    'peak_start_year', 'peak_end_year',
    
    # All year1 box stats are post-draft
    'mpg', 'usg', 'tspct', 'efg',
    'fgpct_rim', 'fgpct_mid', 'fg2pct', 'fg3pct', 'ftpct',
    'orbpct', 'drbpct', 'astpct', 'topct', 'stlpct', 'blkpct',
    'fga_rim_75', 'fga_mid_75', 'fg3a_75', 'fta_75', 'fga_75',
    # Phase 1 Additions: All Year-1 Basketball-Excel stats are post-draft (auxiliary only)
    'year1_corner_3_att', 'year1_dunk_att', 'year1_dist_3p',
    'year1_ast_rim_pct', 'year1_pullup_2p_freq',
    'year1_deflections', 'year1_on_ortg', 'year1_off_ortg',
    'has_year1_be',  # Flag column (not a feature)
}


def assert_no_leakage(feature_columns: List[str], context: str = "features") -> None:
    """
    Assert that feature columns contain ONLY draft-time-safe columns.
    
    Per critique: "if any feature accidentally leaks post-draft info,
    your results will look amazing and be useless."
    
    Recognizes:
    - Explicit allowlist (DRAFT_TIME_SAFE_COLUMNS)
    - College feature prefixes (COLLEGE_FEATURE_PREFIXES)
    
    Raises AssertionError if any forbidden column is detected.
    """
    forbidden_found = set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS
    if forbidden_found:
        raise AssertionError(
            f"LEAKAGE DETECTED in {context}! The following columns are post-draft "
            f"or identity columns and cannot be used as features: {forbidden_found}"
        )
    
    # Check which columns are allowed via prefix
    def is_college_feature(col: str) -> bool:
        return any(col.startswith(prefix) for prefix in COLLEGE_FEATURE_PREFIXES)
    
    # Find columns not in allowlist and not college features
    unknown = set()
    for col in feature_columns:
        if col not in DRAFT_TIME_SAFE_COLUMNS and col not in FORBIDDEN_FEATURE_COLUMNS:
            if not is_college_feature(col):
                unknown.add(col)
    
    if unknown:
        import warnings
        warnings.warn(
            f"Columns not in explicit allowlist (review for leakage): {unknown}",
            UserWarning
        )


def load_crosswalk(warehouse_dir: Path = WAREHOUSE_DIR) -> pd.DataFrame:
    """Load identity/crosswalk table."""
    return pd.read_parquet(warehouse_dir / "dim_player_crosswalk.parquet")


def load_bio(warehouse_dir: Path = WAREHOUSE_DIR) -> pd.DataFrame:
    """
    Load bio/anthropometrics table.
    
    Key columns per spec:
    - ht_first, ht_max, ht_peak_delta: Height metrics
    - wt_first, wt_max, wt_peak_delta: Weight metrics
    - draft_year, rookie_season_year: Temporal anchors
    """
    return pd.read_parquet(warehouse_dir / "dim_player_nba.parquet")


def load_year1_stats(warehouse_dir: Path = WAREHOUSE_DIR) -> pd.DataFrame:
    """
    Load Year-1 (rookie season) statistics.
    
    Per spec: "Year-1 per-season NBA stat used in the 'aux head' likelihood p(a | z)"
    
    Key columns:
    - year1_epm_tot/off/def: Auxiliary supervision targets
    - year1_mp: Exposure/reliability weight
    - Shooting splits, advanced rates: Aux observations
    """
    return pd.read_parquet(warehouse_dir / "fact_player_year1_epm.parquet")


def load_peak_rapm(warehouse_dir: Path = WAREHOUSE_DIR) -> pd.DataFrame:
    """
    Load peak 3-year RAPM (primary target).
    
    Per spec: "Primary target: y_i,peak = peak 3-year RAPM"
    
    Key columns:
    - y_peak_ovr: TARGET LABEL
    - y_peak_off, y_peak_def: Components
    - peak_poss: Exposure/reliability weight
    """
    return pd.read_parquet(warehouse_dir / "fact_player_peak_rapm.parquet")


def build_modeling_dataset(
    warehouse_dir: Path = WAREHOUSE_DIR,
    include_missing_year1: bool = True
) -> pd.DataFrame:
    """
    Build the full modeling dataset by joining all tables.
    
    Join strategy (per spec):
    - Left join from bio (all players with bio)
    - Left join year1 stats (some players missing year1)
    - Left join peak rapm (some players missing peak)
    
    Returns DataFrame with columns for:
    - Identity: nba_id, player_name
    - Bio/Anthro: ht_*, wt_*, draft_year
    - Year1 Stats: year1_epm_*, year1_mp, shooting splits
    - Target: y_peak_ovr, y_peak_off, y_peak_def, peak_poss
    """
    # Load all tables
    bio = load_bio(warehouse_dir)
    year1 = load_year1_stats(warehouse_dir)
    peak = load_peak_rapm(warehouse_dir)
    
    # Remove duplicate columns from year1 before join
    year1_cols_to_keep = [c for c in year1.columns if c not in bio.columns or c == 'nba_id']
    year1_clean = year1[year1_cols_to_keep]
    
    # Join: bio <- year1 <- peak
    df = bio.merge(year1_clean, on='nba_id', how='left')
    df = df.merge(peak, on='nba_id', how='left')
    
    # Add flags per spec
    df['has_year1'] = df['year1_epm_tot'].notna().astype(int)
    df['has_peak'] = df['y_peak_ovr'].notna().astype(int)
    
    # Filter if requested (per spec: "must be non-null for joins")
    if not include_missing_year1:
        df = df[df['has_year1'] == 1]
    
    return df


def get_feature_columns() -> dict:
    """
    Return column groupings for modeling.
    
    Per spec variable roles:
    - Identity: Join keys only, never features
    - Input features: Bio/Anthro, NOT year1 stats
    - Aux observations: Year1 stats for p(a|z)
    - Aux targets: Year1 EPM for multi-task supervision
    - Primary target: Peak RAPM
    """
    return {
        'identity': ['nba_id', 'player_name', 'pid', 'bbr_id'],
        
        'bio_features': [
            'ht_first', 'ht_max', 'ht_peak_delta',
            'wt_first', 'wt_max', 'wt_peak_delta',
            'draft_year', 'rookie_season_year',
        ],
        
        'aux_observations': [
            'mpg', 'usg', 'tspct', 'efg',
            'fgpct_rim', 'fgpct_mid', 'fg2pct', 'fg3pct', 'ftpct',
            'orbpct', 'drbpct', 'astpct', 'topct', 'stlpct', 'blkpct',
            'fga_rim_75', 'fga_mid_75', 'fg3a_75', 'fta_75', 'fga_75',
            # Phase 1 Additions (Basketball-Excel)
            'year1_corner_3_att', 'year1_dunk_att', 'year1_dist_3p',
            'year1_ast_rim_pct', 'year1_pullup_2p_freq', 
            'year1_deflections', 'year1_on_ortg', 'year1_off_ortg',
        ],
        
        'aux_targets': [
            'year1_epm_tot', 'year1_epm_off', 'year1_epm_def', 'year1_epm_ewins',
        ],
        
        'primary_target': ['y_peak_ovr', 'y_peak_off', 'y_peak_def'],
        
        'exposure_weights': ['year1_mp', 'peak_poss'],
        
        'metadata': [
            'season', 'peak_start_year', 'peak_end_year',
            'has_year1', 'has_peak', 'missing_year1',
        ],
    }


def prepare_train_test_split(
    df: pd.DataFrame,
    test_seasons: Optional[list] = None,
    test_frac: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Split data for training/testing.
    
    Per spec: "out-of-sample predictive accuracy for y_i,peak under calibrated uncertainty"
    
    Strategy options:
    1. Temporal split by draft class (preferred for realism)
    2. Random split (for comparison)
    """
    if test_seasons is not None:
        # Temporal split
        train = df[~df['rookie_season_year'].isin(test_seasons)]
        test = df[df['rookie_season_year'].isin(test_seasons)]
    else:
        # Random split
        test = df.sample(frac=test_frac, random_state=random_state)
        train = df.drop(test.index)
    
    return train, test


if __name__ == '__main__':
    print("NBA Data Loader module loaded.")
    print(f"Warehouse directory: {WAREHOUSE_DIR}")
    print(f"Feature columns: {list(get_feature_columns().keys())}")
