"""
Test Suite for Trajectory Features
==================================
Tests for player-season panel and trajectory feature computation.

Tests:
1. As-of leakage (no post-draft seasons)
2. Coverage bias (Tier2 NaN when no spatial)
3. Transfer segmentation (stint_id reset)
4. Ordering (season_idx monotonic)
5. Late bloom (std≈0 handling)
6. Normalization leakage (future-proof)
7. End-to-end synthetic
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from college_scripts.build_player_season_panel import (
    build_panel, compute_derived_metrics, SPATIAL_THRESHOLD
)
from college_scripts.compute_trajectory_features import (
    compute_trajectory_features, weighted_ols_slope, safe_late_bloom,
    SKILL_METRICS, SPATIAL_METRICS
)


# =============================================================================
# Test 1: As-Of Leakage Test
# =============================================================================

def test_no_post_draft_leakage():
    """Ensure no season >= draft_year in panel data."""
    # Create synthetic data with known draft years
    data = []
    for i in range(10):
        draft_year = 2020 + (i % 3)  # Draft years: 2020, 2021, 2022
        for s in range(4):
            season = 2018 + s  # Seasons: 2018, 2019, 2020, 2021
            data.append({
                'athlete_id': i,
                'season': season,
                'teamId': i,
                'poss_total': 500,
                'xy_shots': 30,
            })
    df = pd.DataFrame(data)
    
    crosswalk = pd.DataFrame({
        'athlete_id': list(range(10)),
        'draft_year': [2020 + (i % 3) for i in range(10)]
    })
    
    # Build panel
    panel = build_panel(df, crosswalk)
    
    # Merge draft_year back for verification
    panel_check = panel.merge(crosswalk, on='athlete_id')
    
    # Check: no season >= draft_year
    violations = panel_check[panel_check['season'] >= panel_check['draft_year_y']]
    assert len(violations) == 0, f"Found {len(violations)} post-draft seasons"


def test_unknown_draft_year_handling():
    """Players with unknown draft_year should keep all seasons but be marked."""
    data = []
    for s in range(3):
        data.append({
            'athlete_id': 1,
            'season': 2018 + s,
            'teamId': 1,
            'poss_total': 500,
            'xy_shots': 30,
        })
    df = pd.DataFrame(data)
    
    # Empty crosswalk = unknown draft year
    crosswalk = pd.DataFrame()
    
    panel = build_panel(df, crosswalk)
    
    # Should have all 3 seasons
    assert len(panel) == 3
    # Should be marked as unknown
    assert (panel['draft_year_known'] == 0).all()


# =============================================================================
# Test 2: Coverage Bias Test
# =============================================================================

def test_tier2_coverage_correct():
    """Tier2 trajectory features are NaN when coverage missing."""
    # Create data: player 1 has spatial, player 2 doesn't
    data = []
    for i, has_spatial in enumerate([True, False]):
        for s in range(3):
            data.append({
                'athlete_id': i,
                'season': 2018 + s,
                'season_idx': s + 1,
                'teamId': i,
                'stint_id': 1,
                'seasons_in_stint': s + 1,
                'poss_total': 500 + s * 100,
                'poss_in_stint': sum([500 + j * 100 for j in range(s + 1)]),
                'trueShootingPct': 0.55 + s * 0.02,
                'usage': 0.20,
                'rim_fg_pct': 0.60,
                'three_fg_pct': 0.35,
                'ft_pct': 0.75,
                'xy_shots': 50 if has_spatial else 10,
                'has_spatial': 1 if has_spatial else 0,
                'avg_shot_dist': 12 + s if has_spatial else np.nan,
                'corner_3_rate': 0.15 if has_spatial else np.nan,
                'rim_purity': 0.70 if has_spatial else np.nan,
                'n_seasons_total': 3,
                'transfer_flag': 0,
            })
    panel = pd.DataFrame(data)
    
    features = compute_trajectory_features(panel, save=False)
    
    # Player 0 (has spatial) should have spatial trajectory features
    p0 = features[features['athlete_id'] == 0].iloc[0]
    assert not np.isnan(p0['avg_shot_dist_slope_spatial'])
    assert p0['avg_shot_dist_spatial_traj_missing'] == 0
    
    # Player 1 (no spatial) should have NaN + missing flag
    p1 = features[features['athlete_id'] == 1].iloc[0]
    assert np.isnan(p1['avg_shot_dist_slope_spatial'])
    assert p1['avg_shot_dist_spatial_traj_missing'] == 1


def test_spatial_threshold():
    """has_spatial should respect SPATIAL_THRESHOLD constant."""
    data = [
        {'athlete_id': 1, 'season': 2020, 'teamId': 1, 'xy_shots': SPATIAL_THRESHOLD - 1, 'poss_total': 500},
        {'athlete_id': 2, 'season': 2020, 'teamId': 2, 'xy_shots': SPATIAL_THRESHOLD, 'poss_total': 500},
        {'athlete_id': 3, 'season': 2020, 'teamId': 3, 'xy_shots': SPATIAL_THRESHOLD + 1, 'poss_total': 500},
    ]
    df = pd.DataFrame(data)
    panel = build_panel(df, pd.DataFrame())
    
    assert panel[panel['athlete_id'] == 1]['has_spatial'].iloc[0] == 0
    assert panel[panel['athlete_id'] == 2]['has_spatial'].iloc[0] == 1
    assert panel[panel['athlete_id'] == 3]['has_spatial'].iloc[0] == 1


# =============================================================================
# Test 3: Transfer Segmentation Test
# =============================================================================

def test_transfer_stint_tracking():
    """Verify stint_id increments correctly at transfers."""
    data = [
        {'athlete_id': 1, 'season': 2018, 'teamId': 100, 'poss_total': 500, 'xy_shots': 30},
        {'athlete_id': 1, 'season': 2019, 'teamId': 100, 'poss_total': 600, 'xy_shots': 30},
        {'athlete_id': 1, 'season': 2020, 'teamId': 200, 'poss_total': 700, 'xy_shots': 30},  # Transfer
        {'athlete_id': 1, 'season': 2021, 'teamId': 200, 'poss_total': 800, 'xy_shots': 30},
    ]
    df = pd.DataFrame(data)
    panel = build_panel(df, pd.DataFrame())
    
    # Check stint_id
    assert list(panel['stint_id']) == [1, 1, 2, 2]
    
    # Check transfer_flag
    assert list(panel['transfer_flag']) == [0, 0, 1, 0]
    
    # Check seasons_in_stint
    assert list(panel['seasons_in_stint']) == [1, 2, 1, 2]


def test_final_stint_features():
    """Final stint features should only use post-transfer seasons."""
    data = []
    # Player who transfers: low TS at first school, high TS at second
    for s, (team, ts) in enumerate([
        (100, 0.50), (100, 0.52),  # First school
        (200, 0.58), (200, 0.62),  # Second school (better)
    ]):
        data.append({
            'athlete_id': 1,
            'season': 2018 + s,
            'season_idx': s + 1,
            'teamId': team,
            'stint_id': 1 if team == 100 else 2,
            'seasons_in_stint': (s + 1) if team == 100 else (s - 1),
            'poss_total': 500,
            'poss_in_stint': 500 * ((s + 1) if team == 100 else (s - 1)),
            'trueShootingPct': ts,
            'usage': 0.20,
            'rim_fg_pct': 0.60,
            'three_fg_pct': 0.35,
            'ft_pct': 0.75,
            'xy_shots': 30,
            'has_spatial': 1,
            'avg_shot_dist': 12,
            'corner_3_rate': 0.15,
            'rim_purity': 0.70,
            'n_seasons_total': 4,
            'transfer_flag': 1 if s == 2 else 0,
        })
    panel = pd.DataFrame(data)
    
    # Fix seasons_in_stint for second stint
    panel.loc[panel['stint_id'] == 2, 'seasons_in_stint'] = [1, 2]
    panel.loc[panel['stint_id'] == 2, 'poss_in_stint'] = [500, 1000]
    
    features = compute_trajectory_features(panel, save=False)
    
    # Final stint mean should be average of 0.58 and 0.62 = 0.60
    assert abs(features.iloc[0]['trueShootingPct_mean_final_stint'] - 0.60) < 0.01
    
    # Overall mean should be average of all four
    assert abs(features.iloc[0]['trueShootingPct_mean'] - 0.555) < 0.01


# =============================================================================
# Test 4: Ordering Test
# =============================================================================

def test_season_idx_monotonic():
    """season_idx should be monotonic increasing within each athlete."""
    data = []
    for i in range(5):
        n_seasons = np.random.randint(1, 5)
        for s in range(n_seasons):
            data.append({
                'athlete_id': i,
                'season': 2018 + s,
                'teamId': i,
                'poss_total': 500,
                'xy_shots': 30,
            })
    df = pd.DataFrame(data)
    panel = build_panel(df, pd.DataFrame())
    
    for athlete_id, group in panel.groupby('athlete_id'):
        assert group['season_idx'].is_monotonic_increasing, f"Non-monotonic for {athlete_id}"
        assert group['season_idx'].iloc[0] == 1, f"Doesn't start at 1 for {athlete_id}"


def test_season_ordering_preserved():
    """Seasons should be in chronological order."""
    data = [
        {'athlete_id': 1, 'season': 2021, 'teamId': 1, 'poss_total': 500, 'xy_shots': 30},
        {'athlete_id': 1, 'season': 2019, 'teamId': 1, 'poss_total': 500, 'xy_shots': 30},
        {'athlete_id': 1, 'season': 2020, 'teamId': 1, 'poss_total': 500, 'xy_shots': 30},
    ]
    df = pd.DataFrame(data)
    panel = build_panel(df, pd.DataFrame())
    
    # Should be sorted by season
    assert list(panel['season']) == [2019, 2020, 2021]
    assert list(panel['season_idx']) == [1, 2, 3]


# =============================================================================
# Test 5: End-to-End Synthetic
# =============================================================================

def test_e2e_synthetic():
    """Synthetic data through full pipeline without NaN explosions."""
    np.random.seed(42)
    n_athletes = 50
    data = []
    
    for i in range(n_athletes):
        n_seasons = np.random.randint(1, 5)
        team_id = i
        for s in range(n_seasons):
            if s == 2:  # Transfer after 2 seasons
                team_id = i + 1000
            data.append({
                'athlete_id': i,
                'season': 2018 + s,
                'teamId': team_id,
                'poss_total': np.random.uniform(200, 2000),
                'trueShootingPct': np.random.uniform(0.45, 0.65),
                'usage': np.random.uniform(0.15, 0.35),
                'rim_fg_pct': np.random.uniform(0.50, 0.70),
                'three_fg_pct': np.random.uniform(0.30, 0.42),
                'ft_pct': np.random.uniform(0.65, 0.85),
                'xy_shots': np.random.randint(0, 100) if 2018 + s >= 2019 else 0,
                'avg_shot_dist': np.random.uniform(8, 18) if 2018 + s >= 2019 else np.nan,
                'corner_3_rate': np.random.uniform(0.10, 0.25) if 2018 + s >= 2019 else np.nan,
                'rim_purity': np.random.uniform(0.60, 0.80) if 2018 + s >= 2019 else np.nan,
            })
    
    df = pd.DataFrame(data)
    
    # Build panel
    panel = build_panel(df, pd.DataFrame())
    
    # Compute trajectory features
    features = compute_trajectory_features(panel, save=False)
    
    # Check no unexpected NaNs in level features
    assert not features['trueShootingPct_final'].isna().any()
    assert not features['n_stints'].isna().any()
    
    # Check missingness flags are correct for 1-season players
    one_season = features[features['n_seasons'] == 1]
    if len(one_season) > 0:
        assert (one_season['trueShootingPct_slope_missing'] == 1).all()
        assert (one_season['trueShootingPct_accel_missing'] == 1).all()


def test_missingness_flags_consistent():
    """Missingness flags should match actual NaN values."""
    data = []
    # Player with 1 season (no slope/accel)
    data.append({
        'athlete_id': 1, 'season': 2020, 'season_idx': 1, 'teamId': 1,
        'stint_id': 1, 'seasons_in_stint': 1, 'poss_total': 500, 'poss_in_stint': 500,
        'trueShootingPct': 0.55, 'usage': 0.20, 'rim_fg_pct': 0.60,
        'three_fg_pct': 0.35, 'ft_pct': 0.75, 'xy_shots': 30, 'has_spatial': 1,
        'avg_shot_dist': 12, 'corner_3_rate': 0.15, 'rim_purity': 0.70,
        'n_seasons_total': 1, 'transfer_flag': 0,
    })
    # Player with 2 seasons (has slope, no accel)
    for s in range(2):
        data.append({
            'athlete_id': 2, 'season': 2020 + s, 'season_idx': s + 1, 'teamId': 2,
            'stint_id': 1, 'seasons_in_stint': s + 1, 'poss_total': 500, 'poss_in_stint': 500 * (s + 1),
            'trueShootingPct': 0.55 + s * 0.02, 'usage': 0.20, 'rim_fg_pct': 0.60,
            'three_fg_pct': 0.35, 'ft_pct': 0.75, 'xy_shots': 30, 'has_spatial': 1,
            'avg_shot_dist': 12, 'corner_3_rate': 0.15, 'rim_purity': 0.70,
            'n_seasons_total': 2, 'transfer_flag': 0,
        })
    # Player with 3 seasons (has slope and accel)
    for s in range(3):
        data.append({
            'athlete_id': 3, 'season': 2020 + s, 'season_idx': s + 1, 'teamId': 3,
            'stint_id': 1, 'seasons_in_stint': s + 1, 'poss_total': 500, 'poss_in_stint': 500 * (s + 1),
            'trueShootingPct': 0.55 + s * 0.02, 'usage': 0.20, 'rim_fg_pct': 0.60,
            'three_fg_pct': 0.35, 'ft_pct': 0.75, 'xy_shots': 30, 'has_spatial': 1,
            'avg_shot_dist': 12, 'corner_3_rate': 0.15, 'rim_purity': 0.70,
            'n_seasons_total': 3, 'transfer_flag': 0,
        })
    
    panel = pd.DataFrame(data)
    features = compute_trajectory_features(panel, save=False)
    
    # Player 1: slope missing, accel missing
    p1 = features[features['athlete_id'] == 1].iloc[0]
    assert np.isnan(p1['trueShootingPct_slope']) and p1['trueShootingPct_slope_missing'] == 1
    assert np.isnan(p1['trueShootingPct_accel']) and p1['trueShootingPct_accel_missing'] == 1
    
    # Player 2: slope present, accel missing
    p2 = features[features['athlete_id'] == 2].iloc[0]
    assert not np.isnan(p2['trueShootingPct_slope']) and p2['trueShootingPct_slope_missing'] == 0
    assert np.isnan(p2['trueShootingPct_accel']) and p2['trueShootingPct_accel_missing'] == 1
    
    # Player 3: slope present, accel present
    p3 = features[features['athlete_id'] == 3].iloc[0]
    assert not np.isnan(p3['trueShootingPct_slope']) and p3['trueShootingPct_slope_missing'] == 0
    assert not np.isnan(p3['trueShootingPct_accel']) and p3['trueShootingPct_accel_missing'] == 0


# =============================================================================
# Test 6: Normalization Leakage Test
# =============================================================================

def test_normalization_no_leakage():
    """
    If within-season normalization is added, ensure train-only stats.
    This test demonstrates the pattern to follow.
    """
    from sklearn.model_selection import train_test_split
    
    # Create synthetic features
    np.random.seed(42)
    n = 100
    features = pd.DataFrame({
        'athlete_id': range(n),
        'trueShootingPct_final': np.random.normal(0.55, 0.05, n),
    })
    
    train_ids, test_ids = train_test_split(
        features['athlete_id'].unique(), test_size=0.2, random_state=42
    )
    
    train_features = features[features['athlete_id'].isin(train_ids)]
    test_features = features[features['athlete_id'].isin(test_ids)]
    
    # Compute normalization stats on TRAIN ONLY
    train_mean = train_features['trueShootingPct_final'].mean()
    train_std = train_features['trueShootingPct_final'].std()
    
    # Apply to both (using train stats)
    train_normed = (train_features['trueShootingPct_final'] - train_mean) / train_std
    test_normed = (test_features['trueShootingPct_final'] - train_mean) / train_std
    
    # Verify: train mean ≈ 0
    assert abs(train_normed.mean()) < 0.05, "Train not centered"
    
    # Test mean can be non-zero - that's expected
    # This test just ensures we're using train stats, not test stats


# =============================================================================
# Unit Tests for Helper Functions
# =============================================================================

def test_weighted_ols_slope():
    """Test WLS slope computation."""
    # Simple case: y = 1 + 2*x with equal weights
    values = np.array([1, 3, 5, 7])
    weights = np.array([1, 1, 1, 1])
    x = np.array([0, 1, 2, 3])
    
    slope = weighted_ols_slope(values, weights, x)
    assert abs(slope - 2.0) < 0.01
    
    # With unequal weights (upweight later points)
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    slope = weighted_ols_slope(values, weights, x)
    assert abs(slope - 2.0) < 0.01  # Still 2.0 for linear data


def test_weighted_ols_slope_with_noise():
    """WLS should downweight noisy points."""
    # y = x with noise, but low-weight point is outlier
    values = np.array([1, 2, 3, 10])  # Last point is outlier
    weights = np.array([1, 1, 1, 0.01])  # But it has very low weight
    x = np.array([1, 2, 3, 4])
    
    slope = weighted_ols_slope(values, weights, x)
    # Should be close to 1.0 (ignoring outlier)
    assert abs(slope - 1.0) < 0.5


def test_safe_late_bloom():
    """Test late bloom with edge cases."""
    # Normal case
    assert abs(safe_late_bloom(0.60, 0.55, 0.03) - 1.67) < 0.1
    
    # std ≈ 0 should return 0
    assert safe_late_bloom(0.60, 0.55, 0.001) == 0.0
    
    # NaN inputs
    assert np.isnan(safe_late_bloom(np.nan, 0.55, 0.03))


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Trajectory Features Test Suite")
    print("="*60)
    
    tests = [
        test_no_post_draft_leakage,
        test_unknown_draft_year_handling,
        test_tier2_coverage_correct,
        test_spatial_threshold,
        test_transfer_stint_tracking,
        test_final_stint_features,
        test_season_idx_monotonic,
        test_season_ordering_preserved,
        test_e2e_synthetic,
        test_missingness_flags_consistent,
        test_normalization_no_leakage,
        test_weighted_ols_slope,
        test_weighted_ols_slope_with_noise,
        test_safe_late_bloom,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {test.__name__}: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
