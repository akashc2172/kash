#!/usr/bin/env python3
"""
Edge-Case Wiring Tests (No External Data Needed)
================================================
Lightweight asserts to keep the "college -> NBA" wiring robust.

Focus:
- Final-season selection does not use "last non-null" leakage
- Duplicate athlete-season rows are handled deterministically
- Inference table works without nba_id (UDFAs / current prospects)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_final_season_nan_preserved() -> None:
    # Two seasons. Final season has NaN in rim_made. We must not backfill from earlier season.
    df = pd.DataFrame(
        [
            {"athlete_id": 1, "season": 2024, "split_id": "ALL__ALL", "rim_made": 10.0, "minutes_total": 500.0},
            {"athlete_id": 1, "season": 2025, "split_id": "ALL__ALL", "rim_made": np.nan, "minutes_total": 600.0},
        ]
    )
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from nba_scripts.build_unified_training_table import get_final_college_season

    out = get_final_college_season(df)
    assert len(out) == 1
    assert np.isnan(out.loc[0, "college_rim_made"]), "final season NaN was incorrectly pulled from earlier season"


def test_duplicate_athlete_season_prefers_max_minutes() -> None:
    # Same season duplicate; should aggregate counts, and choose teamId from max minutes_total.
    df = pd.DataFrame(
        [
            {"athlete_id": 2, "season": 2025, "split_id": "ALL__ALL", "teamId": 20, "rim_att": 10.0, "rim_made": 5.0, "shots_total": 10.0, "fga_total": 10.0, "ft_att": 0.0, "ft_made": 0.0, "three_att": 0.0, "three_made": 0.0, "mid_att": 0.0, "mid_made": 0.0, "minutes_total": 10.0},
            {"athlete_id": 2, "season": 2025, "split_id": "ALL__ALL", "teamId": 21, "rim_att": 20.0, "rim_made": 10.0, "shots_total": 20.0, "fga_total": 20.0, "ft_att": 0.0, "ft_made": 0.0, "three_att": 0.0, "three_made": 0.0, "mid_att": 0.0, "mid_made": 0.0, "minutes_total": 100.0},
        ]
    )
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from nba_scripts.build_unified_training_table import get_final_college_season

    out = get_final_college_season(df)
    assert len(out) == 1
    assert out.loc[0, "college_rim_made"] == 15.0
    assert out.loc[0, "college_rim_att"] == 30.0
    assert out.loc[0, "college_teamId"] == 21


def test_inference_table_does_not_require_nba_id() -> None:
    # Minimal college features for one athlete-season.
    college = pd.DataFrame(
        [
            {
                "athlete_id": 3,
                "season": 2025,
                "split_id": "ALL__ALL",
                "teamId": 1,
                "team_pace": 68.0,
                "minutes_total": 200.0,
                "shots_total": 10.0,
                "fga_total": 4.0,
                "ft_att": 0.0,
                "rim_att": 4.0,
                "rim_made": 2.0,
                "three_att": 0.0,
                "three_made": 0.0,
                "mid_att": 0.0,
                "mid_made": 0.0,
                "ft_made": 0.0,
                "assisted_made_rim": 0.0,
                "assisted_made_three": 0.0,
                "assisted_made_mid": 0.0,
                "xy_shots": 0.0,
                "sum_dist_ft": 0.0,
                "corner_3_att": 0.0,
                "corner_3_made": 0.0,
                "xy_3_shots": 0.0,
                "xy_rim_shots": 0.0,
                "deep_3_att": 0.0,
                "rim_rest_att": 0.0,
                "sum_dist_sq_ft": 0.0,
                "ast_total": 0.0,
                "tov_total": 0.0,
                "stl_total": 0.0,
                "blk_total": 0.0,
                "orb_total": 0.0,
                "drb_total": 0.0,
                "trb_total": 0.0,
            }
        ]
    )

    career = pd.DataFrame(
        [
            {
                "athlete_id": 3,
                "career_years": 1,
                "final_trueShootingPct": 0.5,
                "final_usage": np.nan,
                "final_poss_total": 10.0,
            }
        ]
    )

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from nba_scripts.nba_prospect_inference import build_prospect_inference_table

    out = build_prospect_inference_table(college, career)
    assert "nba_id" not in out.columns
    assert len(out) == 1


def test_transfer_same_season_aggregates_counts() -> None:
    """
    Transfers can produce multiple rows for the same athlete-season (different teamId).
    Final-season extraction should aggregate count columns across those rows.
    """
    df = pd.DataFrame(
        [
            {"athlete_id": 4, "season": 2025, "split_id": "ALL__ALL", "teamId": 10, "minutes_total": 100.0, "rim_att": 2.0, "rim_made": 1.0, "shots_total": 2.0, "fga_total": 2.0, "ft_att": 0.0, "ft_made": 0.0, "three_att": 0.0, "three_made": 0.0, "mid_att": 0.0, "mid_made": 0.0, "xy_shots": 0.0, "xy_3_shots": 0.0, "xy_rim_shots": 0.0, "sum_dist_ft": 0.0, "sum_dist_sq_ft": 0.0, "corner_3_att": 0.0, "corner_3_made": 0.0, "deep_3_att": 0.0, "rim_rest_att": 0.0},
            {"athlete_id": 4, "season": 2025, "split_id": "ALL__ALL", "teamId": 11, "minutes_total": 200.0, "rim_att": 4.0, "rim_made": 3.0, "shots_total": 4.0, "fga_total": 4.0, "ft_att": 0.0, "ft_made": 0.0, "three_att": 0.0, "three_made": 0.0, "mid_att": 0.0, "mid_made": 0.0, "xy_shots": 0.0, "xy_3_shots": 0.0, "xy_rim_shots": 0.0, "sum_dist_ft": 0.0, "sum_dist_sq_ft": 0.0, "corner_3_att": 0.0, "corner_3_made": 0.0, "deep_3_att": 0.0, "rim_rest_att": 0.0},
        ]
    )
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from nba_scripts.build_unified_training_table import get_final_college_season

    out = get_final_college_season(df)
    assert len(out) == 1
    # Counts should aggregate
    assert out.loc[0, "college_rim_att"] == 6.0
    assert out.loc[0, "college_rim_made"] == 4.0
    # TeamId should be the max-minutes team (11)
    assert out.loc[0, "college_teamId"] == 11


def main() -> None:
    test_final_season_nan_preserved()
    test_duplicate_athlete_season_prefers_max_minutes()
    test_inference_table_does_not_require_nba_id()
    test_transfer_same_season_aggregates_counts()
    print("OK")


if __name__ == "__main__":
    main()
