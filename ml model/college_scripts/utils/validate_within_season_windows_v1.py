#!/usr/bin/env python3
"""Validate Within-Season Windows (v1)

Checks mask/value consistency and NaN safety for:
- data/college_feature_store/within_season_windows_v1.parquet
- data/college_feature_store/prospect_career_long_v1.parquet
- data/college_feature_store/prospect_career_v1.parquet

This validator is designed to catch accidental "missing -> 0" regressions.

Usage:
  python college_scripts/utils/validate_within_season_windows_v1.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]

WS_PATH = BASE_DIR / 'data/college_feature_store/within_season_windows_v1.parquet'
CAREER_LONG_PATH = BASE_DIR / 'data/college_feature_store/prospect_career_long_v1.parquet'
CAREER_WIDE_PATH = BASE_DIR / 'data/college_feature_store/prospect_career_v1.parquet'


def _assert_masked_nan(df: pd.DataFrame, mask_col: str, value_cols: list[str]) -> list[str]:
    errs = []
    if mask_col not in df.columns:
        return errs

    mask = df[mask_col]
    # Only evaluate where mask is explicitly 0/1
    mask0 = mask == 0
    for c in value_cols:
        if c not in df.columns:
            continue
        bad = df.loc[mask0, c].notna().sum()
        if bad:
            errs.append(f"{mask_col}=0 but {c} has {bad} non-null values")
    return errs


def validate_ws_table(df: pd.DataFrame) -> list[str]:
    errs: list[str] = []

    # Basic required columns
    for col in ['athlete_id', 'season', 'games_played_pg']:
        if col not in df.columns:
            errs.append(f"missing required column: {col}")

    # Mask/value consistency
    errs += _assert_masked_nan(df, 'has_ws_last5', ['ws_minutes_last5', 'ws_fga_last5', 'ws_pts_last5', 'ws_pps_last5', 'ws_on_net_rating_last5_mean'])
    errs += _assert_masked_nan(df, 'has_ws_last10', ['ws_minutes_last10', 'ws_fga_last10', 'ws_pts_last10', 'ws_pps_last10', 'ws_on_net_rating_last10_mean'])
    errs += _assert_masked_nan(df, 'has_ws_prev5', ['ws_delta_pps_last5_minus_prev5', 'ws_delta_minutes_last5_minus_prev5'])
    errs += _assert_masked_nan(df, 'has_ws_prev10', ['ws_delta_pps_last10_minus_prev10', 'ws_delta_minutes_last10_minus_prev10'])
    errs += _assert_masked_nan(df, 'has_ws_breakout_timing_minutes', ['ws_breakout_timing_minutes'])
    errs += _assert_masked_nan(df, 'has_ws_breakout_timing_volume', ['ws_breakout_timing_volume'])
    errs += _assert_masked_nan(df, 'has_ws_breakout_timing_eff', ['ws_breakout_timing_eff'])

    # Timing bounds if present
    for c, m in [
        ('ws_breakout_timing_minutes', 'has_ws_breakout_timing_minutes'),
        ('ws_breakout_timing_volume', 'has_ws_breakout_timing_volume'),
        ('ws_breakout_timing_eff', 'has_ws_breakout_timing_eff'),
    ]:
        if c in df.columns and m in df.columns:
            ok = df[m] == 1
            if ok.any():
                v = df.loc[ok, c]
                bad = ((v < 0) | (v > 1)).sum()
                if bad:
                    errs.append(f"{c} out of [0,1] for {bad} rows")

    return errs


def validate_career_long(df: pd.DataFrame) -> list[str]:
    errs: list[str] = []
    # Same checks but on per-season career table (joined).
    if 'games_played_pg' not in df.columns:
        return errs

    errs += _assert_masked_nan(df, 'has_ws_last5', ['ws_minutes_last5', 'ws_fga_last5', 'ws_pts_last5', 'ws_pps_last5', 'ws_on_net_rating_last5_mean'])
    errs += _assert_masked_nan(df, 'has_ws_last10', ['ws_minutes_last10', 'ws_fga_last10', 'ws_pts_last10', 'ws_pps_last10', 'ws_on_net_rating_last10_mean'])
    errs += _assert_masked_nan(df, 'has_ws_prev5', ['ws_delta_pps_last5_minus_prev5', 'ws_delta_minutes_last5_minus_prev5'])
    errs += _assert_masked_nan(df, 'has_ws_prev10', ['ws_delta_pps_last10_minus_prev10', 'ws_delta_minutes_last10_minus_prev10'])
    errs += _assert_masked_nan(df, 'has_ws_breakout_timing_minutes', ['ws_breakout_timing_minutes'])
    errs += _assert_masked_nan(df, 'has_ws_breakout_timing_volume', ['ws_breakout_timing_volume'])
    errs += _assert_masked_nan(df, 'has_ws_breakout_timing_eff', ['ws_breakout_timing_eff'])

    return errs


def validate_career_wide(df: pd.DataFrame) -> list[str]:
    errs: list[str] = []

    # Wide table uses final_* prefix.
    errs += _assert_masked_nan(df, 'final_has_ws_last5', ['final_ws_minutes_last5', 'final_ws_fga_last5', 'final_ws_pts_last5', 'final_ws_pps_last5', 'final_ws_on_net_rating_last5_mean'])
    errs += _assert_masked_nan(df, 'final_has_ws_last10', ['final_ws_minutes_last10', 'final_ws_fga_last10', 'final_ws_pts_last10', 'final_ws_pps_last10', 'final_ws_on_net_rating_last10_mean'])
    errs += _assert_masked_nan(df, 'final_has_ws_prev5', ['final_ws_delta_pps_last5_minus_prev5', 'final_ws_delta_minutes_last5_minus_prev5'])
    errs += _assert_masked_nan(df, 'final_has_ws_prev10', ['final_ws_delta_pps_last10_minus_prev10', 'final_ws_delta_minutes_last10_minus_prev10'])
    errs += _assert_masked_nan(df, 'final_has_ws_breakout_timing_minutes', ['final_ws_breakout_timing_minutes'])
    errs += _assert_masked_nan(df, 'final_has_ws_breakout_timing_volume', ['final_ws_breakout_timing_volume'])
    errs += _assert_masked_nan(df, 'final_has_ws_breakout_timing_eff', ['final_ws_breakout_timing_eff'])

    return errs


def main() -> int:
    any_errors: list[str] = []

    if WS_PATH.exists():
        ws = pd.read_parquet(WS_PATH)
        any_errors += [f"within_season_windows_v1: {e}" for e in validate_ws_table(ws)]
    else:
        print(f"SKIP: missing {WS_PATH}")

    if CAREER_LONG_PATH.exists():
        cl = pd.read_parquet(CAREER_LONG_PATH)
        any_errors += [f"prospect_career_long_v1: {e}" for e in validate_career_long(cl)]
    else:
        print(f"SKIP: missing {CAREER_LONG_PATH}")

    if CAREER_WIDE_PATH.exists():
        cw = pd.read_parquet(CAREER_WIDE_PATH)
        any_errors += [f"prospect_career_v1: {e}" for e in validate_career_wide(cw)]
    else:
        print(f"SKIP: missing {CAREER_WIDE_PATH}")

    if any_errors:
        print("FAIL")
        for e in any_errors:
            print("-", e)
        return 1

    print("OK")
    return 0


if __name__ == '__main__':
    sys.exit(main())
