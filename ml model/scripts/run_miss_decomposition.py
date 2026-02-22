#!/usr/bin/env python3
"""
Miss decomposition for rolling yearly rankings.

Reads:
- data/inference/rolling_yearly/{year}/rankings_{year}_nba_mapped.csv
- data/training/unified_training_table_supervised.parquet

Writes:
- data/audit/rolling_yearly/miss_report_{year}.csv
- data/audit/rolling_yearly/miss_slices_{year}.csv
- data/audit/rolling_yearly/miss_summary_all_years.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent.parent
ROLLING_DIR = BASE / "data" / "inference" / "rolling_yearly"
SUPERVISED_PATH = BASE / "data" / "training" / "unified_training_table_supervised.parquet"
OUT_DIR = BASE / "data" / "audit" / "rolling_yearly"
WATCHLIST_NAMES = [
    "rob dillingham",
    "reed sheppard",
    "ajay mitchell",
    "cole anthony",
    "onyeka okongwu",
    "patrick williams",
    "tyrese haliburton",
    "naji marshall",
    "cj elleby",
    "filip petrusev",
    "chris duarte",
    "dalano banton",
    "jalen johnson",
    "austin reaves",
    "cade cunningham",
    "evan mobley",
    "cam reddish",
    "kz okpala",
    "cameron johnson",
    "daniel gafford",
]


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _year_files(years: Iterable[int]) -> List[Path]:
    out = []
    for y in years:
        p = ROLLING_DIR / str(y) / f"rankings_{y}_nba_mapped.csv"
        if p.exists():
            out.append(p)
    return out


def _slice_stats(df: pd.DataFrame, year: int) -> pd.DataFrame:
    rows = []
    slice_cols = [
        "age_at_season",
        "class_year",
        "is_power_conf",
        "recruiting_stars",
        "career_years",
    ]
    bucket_defs = {}
    if "age_at_season" in df.columns:
        bucket_defs["age_bucket"] = pd.cut(
            _safe_numeric(df["age_at_season"]),
            bins=[0, 19, 20, 21, 22, 30],
            labels=["<=19", "20", "21", "22", "23+"],
            include_lowest=True,
        )
    if "college_height_in" in df.columns:
        bucket_defs["height_bucket"] = pd.cut(
            _safe_numeric(df["college_height_in"]),
            bins=[0, 74, 77, 80, 84, 100],
            labels=["<=6-2", "6-3..6-5", "6-6..6-8", "6-9..7-0", "7-1+"],
            include_lowest=True,
        )
    if "college_weight_lbs" in df.columns:
        bucket_defs["weight_bucket"] = pd.cut(
            _safe_numeric(df["college_weight_lbs"]),
            bins=[0, 180, 200, 220, 240, 400],
            labels=["<=180", "181-200", "201-220", "221-240", "241+"],
            include_lowest=True,
        )
    if "college_minutes_total" in df.columns:
        bucket_defs["minutes_bucket"] = pd.cut(
            _safe_numeric(df["college_minutes_total"]),
            bins=[-1, 300, 600, 900, 1200, 5000],
            labels=["<=300", "301-600", "601-900", "901-1200", "1201+"],
            include_lowest=True,
        )
    if "ctx_adj_onoff_net" in df.columns:
        x = _safe_numeric(df["ctx_adj_onoff_net"])
        if x.notna().sum() >= 8:
            bucket_defs["ctx_adj_onoff_net_q"] = pd.qcut(
                x.rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"]
            )
        else:
            rows.append(pd.DataFrame([{
                "year": int(year),
                "slice": "ctx_adj_onoff_net_q",
                "slice_value": "INSUFFICIENT_DATA",
                "n": int(len(df)),
                "mean_rank_error": float(np.nanmean(df["rank_error"])),
                "median_rank_error": float(np.nanmedian(df["rank_error"])),
                "mean_abs_rank_error": float(np.nanmean(np.abs(df["rank_error"]))),
            }]))
    if "ctx_vel_net_yoy" in df.columns:
        x = _safe_numeric(df["ctx_vel_net_yoy"])
        if x.notna().sum() >= 8:
            bucket_defs["ctx_vel_net_yoy_q"] = pd.qcut(
                x.rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"]
            )
        else:
            rows.append(pd.DataFrame([{
                "year": int(year),
                "slice": "ctx_vel_net_yoy_q",
                "slice_value": "INSUFFICIENT_DATA",
                "n": int(len(df)),
                "mean_rank_error": float(np.nanmean(df["rank_error"])),
                "median_rank_error": float(np.nanmedian(df["rank_error"])),
                "mean_abs_rank_error": float(np.nanmean(np.abs(df["rank_error"]))),
            }]))

    w = df.copy()
    for bcol, bser in bucket_defs.items():
        w[bcol] = bser

    group_cols = slice_cols + list(bucket_defs.keys())
    for col in group_cols:
        if col not in w.columns:
            continue
        g = (
            w.groupby(col, dropna=False)
            .agg(
                n=("rank_error", "size"),
                mean_rank_error=("rank_error", "mean"),
                median_rank_error=("rank_error", "median"),
                mean_abs_rank_error=("rank_error", lambda x: float(np.nanmean(np.abs(x)))),
            )
            .reset_index()
            .rename(columns={col: "slice_value"})
        )
        g.insert(0, "slice", col)
        g.insert(0, "year", int(year))
        rows.append(g)
    if not rows:
        return pd.DataFrame(columns=["year", "slice", "slice_value", "n", "mean_rank_error", "median_rank_error", "mean_abs_rank_error"])
    return pd.concat(rows, ignore_index=True)


def _norm_name(x: str) -> str:
    return "".join(ch for ch in str(x).lower().strip() if ch.isalnum() or ch == " ")


def run(start_year: int, end_year: int, top_n: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sup = pd.read_parquet(SUPERVISED_PATH)
    join_cols = [c for c in [
        "nba_id",
        "athlete_id",
        "age_at_season",
        "class_year",
        "career_years",
        "is_power_conf",
        "recruiting_stars",
        "recruiting_rank",
        "recruiting_rating",
        "college_height_in",
        "college_weight_lbs",
        "college_minutes_total",
        "college_poss_proxy",
        "transfer_event_count",
        "ctx_adj_onoff_net",
        "ctx_vel_net_yoy",
        "ctx_adj_onoff_off",
        "ctx_adj_onoff_def",
        "ctx_quality_flag",
        "path_onoff_ast_diff_per100",
        "path_onoff_reb_diff_per100",
        "path_onoff_stl_diff_per100",
        "path_onoff_blk_diff_per100",
        "path_onoff_tov_diff_per100",
        "path_onoff_transition_diff_per100",
        "path_onoff_dunk_pressure_diff",
    ] if c in sup.columns]
    sup = sup[join_cols].drop_duplicates(subset=[c for c in ["nba_id", "athlete_id"] if c in join_cols])

    summaries = []
    all_slices = []
    all_frames = []

    for p in _year_files(range(start_year, end_year + 1)):
        year = int(p.parent.name)
        df = pd.read_csv(p)
        if "rank_error" not in df.columns:
            continue

        df["rank_error"] = _safe_numeric(df["rank_error"])
        df["actual_rank"] = _safe_numeric(df.get("actual_rank", pd.Series(np.nan, index=df.index)))
        df["pred_rank"] = _safe_numeric(df.get("pred_rank", pd.Series(np.nan, index=df.index)))
        df["actual_target"] = _safe_numeric(df.get("actual_target", pd.Series(np.nan, index=df.index)))
        df = df[df["actual_rank"].notna()].copy()
        if df.empty:
            continue

        key = "nba_id" if "nba_id" in df.columns and "nba_id" in sup.columns else "athlete_id"
        if key in df.columns and key in sup.columns:
            df = df.merge(sup, on=key, how="left", suffixes=("", "_meta"))
        df["season"] = year
        all_frames.append(df.copy())

        # rank_error = pred_rank - actual_rank
        # positive -> model ranked too low (sleeper miss)
        # negative -> model ranked too high (bust miss)
        over = df.sort_values("rank_error", ascending=False).head(top_n).copy()
        over["miss_type"] = "sleeper_miss"
        under = df.sort_values("rank_error", ascending=True).head(top_n).copy()
        under["miss_type"] = "bust_miss"
        report = pd.concat([over, under], ignore_index=True)
        report_cols = [c for c in [
            "miss_type", "player_name", "nba_id", "athlete_id",
            "pred_rank", "actual_rank", "rank_error", "actual_target",
            "age_at_season", "class_year", "is_power_conf",
            "recruiting_stars", "college_height_in", "college_weight_lbs",
            "college_minutes_total", "college_poss_proxy", "transfer_event_count",
            "ctx_adj_onoff_net", "ctx_vel_net_yoy", "ctx_adj_onoff_off", "ctx_adj_onoff_def",
            "path_onoff_ast_diff_per100", "path_onoff_reb_diff_per100",
            "path_onoff_stl_diff_per100", "path_onoff_blk_diff_per100",
            "path_onoff_tov_diff_per100", "path_onoff_transition_diff_per100",
            "path_onoff_dunk_pressure_diff",
        ] if c in report.columns]
        report = report[report_cols]
        report.to_csv(OUT_DIR / f"miss_report_{year}.csv", index=False)

        slices = _slice_stats(df, year)
        slices.to_csv(OUT_DIR / f"miss_slices_{year}.csv", index=False)
        all_slices.append(slices)

        top_pred15 = df[df["pred_rank"] <= 15].copy()
        top_actual15 = df[df["actual_rank"] <= 15].copy()
        bust_regret_top15 = float(
            np.nanmean(np.clip(_safe_numeric(top_pred15["actual_rank"]) - 15.0, 0.0, None))
        ) if len(top_pred15) else np.nan
        sleeper_regret_top15 = float(
            np.nanmean(np.clip(_safe_numeric(top_actual15["pred_rank"]) - 15.0, 0.0, None))
        ) if len(top_actual15) else np.nan

        summaries.append({
            "year": year,
            "n_with_actual_rank": int(len(df)),
            "mean_abs_rank_error": float(np.nanmean(np.abs(df["rank_error"]))),
            "median_abs_rank_error": float(np.nanmedian(np.abs(df["rank_error"]))),
            "sleeper_miss_threshold_topn": float(over["rank_error"].min()) if len(over) else np.nan,
            "bust_miss_threshold_topn": float(under["rank_error"].max()) if len(under) else np.nan,
            "spearman_pred_vs_actual_rank": float(df[["pred_rank", "actual_rank"]].corr(method="spearman").iloc[0, 1]),
            "bust_regret_top15": bust_regret_top15,
            "sleeper_regret_top15": sleeper_regret_top15,
        })

    if summaries:
        pd.DataFrame(summaries).sort_values("year").to_csv(OUT_DIR / "miss_summary_all_years.csv", index=False)
    if all_slices:
        pd.concat(all_slices, ignore_index=True).to_csv(OUT_DIR / "miss_slices_all_years.csv", index=False)
    if all_frames:
        full = pd.concat(all_frames, ignore_index=True)
        top15 = full[
            full["actual_rank"].notna()
            & (full["actual_rank"] <= 15)
            & (full["season"] >= 2021)
            & (full["season"] <= 2024)
        ].copy()
        top15["abs_rank_error"] = np.abs(_safe_numeric(top15["rank_error"]))
        top15 = top15.sort_values(["season", "abs_rank_error"], ascending=[True, False])
        top15_cols = [c for c in [
            "season", "player_name", "nba_id", "athlete_id", "pred_rank", "actual_rank",
            "rank_error", "actual_target", "ctx_adj_onoff_net", "ctx_vel_net_yoy",
            "ctx_adj_onoff_off", "ctx_adj_onoff_def", "ctx_quality_flag",
            "path_onoff_ast_diff_per100", "path_onoff_reb_diff_per100",
            "path_onoff_stl_diff_per100", "path_onoff_blk_diff_per100",
            "path_onoff_tov_diff_per100", "path_onoff_transition_diff_per100",
            "path_onoff_dunk_pressure_diff",
        ] if c in top15.columns]
        top15[top15_cols].to_csv(OUT_DIR / "top15_regret_2021_2024.csv", index=False)

        watch = full.copy()
        watch["player_name_norm"] = watch.get("player_name", "").map(_norm_name)
        keep = {_norm_name(n) for n in WATCHLIST_NAMES}
        watch = watch[watch["player_name_norm"].isin(keep)].copy()
        watch_cols = [c for c in [
            "season", "player_name", "nba_id", "athlete_id", "pred_rank", "actual_rank",
            "rank_error", "actual_target", "ctx_adj_onoff_net", "ctx_vel_net_yoy",
            "college_minutes_total", "college_poss_proxy", "age_at_season", "class_year",
        ] if c in watch.columns]
        watch[watch_cols].sort_values(["season", "player_name"]).to_csv(
            OUT_DIR / "watchlist_players_miss_context.csv", index=False
        )

        y23 = full[full["season"] == 2023].copy()
        y23 = y23[y23["actual_rank"].notna() & y23["pred_rank"].notna()].copy()
        sleepers = y23.sort_values("rank_error", ascending=False).head(5)
        busts = y23.sort_values("rank_error", ascending=True).head(5)
        print("\nTop 5 Sleeper Misses (2023):")
        for _, r in sleepers.iterrows():
            print(
                f"  {r.get('player_name','?')}: pred={int(r['pred_rank'])}, actual={int(r['actual_rank'])}, "
                f"err={float(r['rank_error']):.1f}, ctx_onoff={r.get('ctx_adj_onoff_net', np.nan)}, "
                f"ctx_vel={r.get('ctx_vel_net_yoy', np.nan)}"
            )
        print("\nTop 5 Bust Misses (2023):")
        for _, r in busts.iterrows():
            print(
                f"  {r.get('player_name','?')}: pred={int(r['pred_rank'])}, actual={int(r['actual_rank'])}, "
                f"err={float(r['rank_error']):.1f}, ctx_onoff={r.get('ctx_adj_onoff_net', np.nan)}, "
                f"ctx_vel={r.get('ctx_vel_net_yoy', np.nan)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling miss decomposition")
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()
    run(args.start_year, args.end_year, args.top_n)


if __name__ == "__main__":
    main()
