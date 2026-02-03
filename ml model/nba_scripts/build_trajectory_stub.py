#!/usr/bin/env python3
"""
Build Trajectory Dataset Stub
============================
Creates a per-player sequence dataset from prospect_career_long_v1.parquet.
This is a lightweight scaffold for future sequence/temporal models.
"""

from pathlib import Path
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "data/college_feature_store/prospect_career_long_v1.parquet"
OUTPUT_FILE = BASE_DIR / "data/training/trajectory_stub_v1.parquet"

TRAJECTORY_FEATURES = [
    # Within-season windows (optional)
    "has_ws_last5",
    "ws_minutes_last5",
    "ws_fga_last5",
    "ws_pts_last5",
    "ws_pps_last5",
    "ws_on_net_rating_last5_mean",
    "has_ws_last10",
    "ws_minutes_last10",
    "ws_fga_last10",
    "ws_pts_last10",
    "ws_pps_last10",
    "ws_on_net_rating_last10_mean",
    "has_ws_prev5",
    "ws_delta_pps_last5_minus_prev5",
    "ws_delta_minutes_last5_minus_prev5",
    "has_ws_prev10",
    "ws_delta_pps_last10_minus_prev10",
    "ws_delta_minutes_last10_minus_prev10",
    "has_ws_breakout_timing_minutes",
    "ws_breakout_timing_minutes",
    "has_ws_breakout_timing_volume",
    "ws_breakout_timing_volume",
    "has_ws_breakout_timing_eff",
    "ws_breakout_timing_eff",

    "games_played",
    "minutes_per_game",
    "poss_per_game",
    "trueShootingPct",
    "usage",
    "rim_fg_pct",
    "three_fg_pct",
    "ft_pct",
    "ast_total",
    "tov_total",
    "stl_total",
    "blk_total",
    "avg_shot_dist",
    "corner_3_rate",
    "deep_3_rate",
    "rim_purity",
    "shot_dist_var",
]


def build_trajectory_stub() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_FILE}")

    logger.info(f"Loading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)

    required_cols = {"athlete_id", "season"}
    missing_required = required_cols - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {sorted(missing_required)}")

    available_features = [c for c in TRAJECTORY_FEATURES if c in df.columns]
    logger.info(f"Using {len(available_features)} trajectory features")

    df = df.sort_values(["athlete_id", "season"])

    agg_dict = {"season": list}
    for col in available_features:
        agg_dict[col] = list

    grouped = df.groupby("athlete_id", as_index=False).agg(agg_dict)

    grouped.rename(columns={"season": "seasons"}, inplace=True)
    grouped["career_years"] = grouped["seasons"].apply(len)
    grouped["final_season"] = grouped["seasons"].apply(lambda s: s[-1] if s else np.nan)

    return grouped


def save_trajectory_stub(df: pd.DataFrame) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"Saved trajectory stub: {OUTPUT_FILE} ({len(df):,} players)")


def main() -> None:
    df = build_trajectory_stub()
    save_trajectory_stub(df)


if __name__ == "__main__":
    main()
