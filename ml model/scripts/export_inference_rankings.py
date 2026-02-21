#!/usr/bin/env python3
"""
Export inference rankings with player names, season ranks, and per-season tabs.
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
INFERENCE_DIR = BASE_DIR / "data" / "inference"
WAREHOUSE_DB = BASE_DIR / "data" / "warehouse.duckdb"
CROSSWALK_PATH = BASE_DIR / "data" / "warehouse_v2" / "dim_player_nba_college_crosswalk.parquet"


def latest_predictions() -> Path:
    preds = sorted(
        INFERENCE_DIR.glob("prospect_predictions_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not preds:
        raise FileNotFoundError("No prospect_predictions_*.parquet found")
    return preds[0]


def load_name_map() -> pd.DataFrame:
    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    q = """
    WITH src AS (
        SELECT CAST(athleteId AS VARCHAR) AS athlete_id, name AS player_name
        FROM fact_player_season_stats
        WHERE name IS NOT NULL AND TRIM(name) <> ''
        UNION ALL
        SELECT CAST(athleteId AS VARCHAR) AS athlete_id, name AS player_name
        FROM fact_recruiting_players
        WHERE name IS NOT NULL AND TRIM(name) <> ''
        UNION ALL
        SELECT CAST(athleteId AS VARCHAR) AS athlete_id, athlete_name AS player_name
        FROM stg_participants
        WHERE athlete_name IS NOT NULL AND TRIM(athlete_name) <> ''
    ),
    ranked AS (
        SELECT athlete_id, player_name, COUNT(*) AS n,
               ROW_NUMBER() OVER (
                 PARTITION BY athlete_id
                 ORDER BY COUNT(*) DESC, LENGTH(player_name) DESC, player_name
               ) AS rn
        FROM src
        GROUP BY 1,2
    )
    SELECT athlete_id, player_name
    FROM ranked
    WHERE rn = 1
    """
    out = con.execute(q).df()
    con.close()
    out["athlete_id"] = out["athlete_id"].astype(str)
    return out


def main() -> None:
    pred_path = latest_predictions()
    pred = pd.read_parquet(pred_path)
    pred["athlete_id"] = pred["athlete_id"].astype(str)
    pred["college_final_season"] = pd.to_numeric(pred["college_final_season"], errors="coerce")
    pred["pred_peak_rapm"] = pd.to_numeric(pred["pred_peak_rapm"], errors="coerce")
    pred = pred[pred["college_final_season"].notna() & pred["pred_peak_rapm"].notna()].copy()
    pred["college_final_season"] = pred["college_final_season"].astype(int)

    names = load_name_map()
    ranked = pred.merge(names, on="athlete_id", how="left")
    ranked["player_name"] = ranked["player_name"].fillna(ranked["athlete_id"])
    if CROSSWALK_PATH.exists():
        cw = pd.read_parquet(CROSSWALK_PATH, columns=["athlete_id"]).dropna().copy()
        cw["athlete_id"] = cw["athlete_id"].astype(str)
        matched_ids = set(cw["athlete_id"].unique().tolist())
        ranked["is_nba_matched"] = ranked["athlete_id"].isin(matched_ids).astype(int)
    else:
        ranked["is_nba_matched"] = 0
    score_col = (
        "pred_rank_score"
        if "pred_rank_score" in ranked.columns
        else ("pred_peak_rapm_rank_score" if "pred_peak_rapm_rank_score" in ranked.columns else "pred_peak_rapm")
    )
    ranked = ranked.sort_values(["college_final_season", score_col], ascending=[True, False])
    ranked["season_rank_all"] = ranked.groupby("college_final_season").cumcount() + 1

    # Display-friendly minutes for downstream sheets:
    # if raw minutes are missing/zero but games are present, provide a transparent estimate.
    mins = pd.to_numeric(ranked.get("college_minutes_total", pd.Series(index=ranked.index, dtype=float)), errors="coerce")
    games = pd.to_numeric(ranked.get("college_games_played", pd.Series(index=ranked.index, dtype=float)), errors="coerce")
    ranked["college_minutes_total_display"] = np.where(
        mins > 0,
        mins,
        np.where(games > 0, games * 25.0, np.nan),
    )
    ranked["minutes_is_estimated"] = np.where((mins <= 0) & (games > 0), 1, 0)

    # Qualified prospect pool:
    # keep raw rank for completeness, and add a practical rank that excludes tiny-sample rows.
    poss = pd.to_numeric(ranked.get("college_poss_proxy", pd.Series(index=ranked.index, dtype=float)), errors="coerce")
    # Strict exposure gate to avoid tiny-sample rows dominating seasonal leaderboards.
    # Allow either possessions or minutes-based evidence to qualify.
    ranked["is_qualified_pool"] = ((games >= 14) & ((poss >= 200) | (ranked["college_minutes_total_display"] >= 400))).astype(int)
    ranked["season_rank_qualified"] = pd.NA
    for season, g in ranked.groupby("college_final_season", sort=False):
        q_idx = g.index[g["is_qualified_pool"] == 1]
        if len(q_idx):
            ranked.loc[q_idx, "season_rank_qualified"] = range(1, len(q_idx) + 1)
    ranked["season_rank"] = ranked["season_rank_qualified"]
    ranked["season_rank_matched"] = pd.NA
    for season, g in ranked.groupby("college_final_season", sort=False):
        q_idx = g.index[(g["is_qualified_pool"] == 1) & (g["is_nba_matched"] == 1)]
        if len(q_idx):
            ranked.loc[q_idx, "season_rank_matched"] = range(1, len(q_idx) + 1)

    keep = ["college_final_season", "season_rank", "athlete_id", "player_name", score_col, "pred_peak_rapm"]
    if "pred_rank_target" in ranked.columns:
        keep.insert(5, "pred_rank_target")
    if score_col != "pred_peak_rapm" and "pred_peak_rapm_reliability" in ranked.columns:
        keep.insert(5, "pred_peak_rapm_reliability")
    # Export minutes as populated display value; keep raw for audits.
    ranked["college_minutes_total_raw"] = mins
    ranked["college_minutes_total"] = ranked["college_minutes_total_display"]

    for c in ["season_rank_all", "season_rank_qualified", "season_rank_matched", "is_qualified_pool", "is_nba_matched", "college_games_played", "college_poss_proxy", "college_minutes_total", "college_minutes_total_raw", "college_minutes_total_display", "minutes_is_estimated"]:
        if c in ranked.columns:
            keep.append(c)
    for c in ["pred_dev_rate", "pred_dev_rate_std", "pred_year1_epm", "pred_gap_ts", "pred_made_nba_logit"]:
        if c in ranked.columns:
            keep.append(c)
    out_all = ranked[keep].copy()
    out_qualified = ranked[ranked["is_qualified_pool"] == 1][keep].copy()

    csv_path = INFERENCE_DIR / "season_rankings_latest_best_current.csv"
    out_all.to_csv(csv_path, index=False)
    csv_q_path = INFERENCE_DIR / "season_rankings_latest_best_current_qualified.csv"
    out_qualified.to_csv(csv_q_path, index=False)
    out_matched = out_all[out_all["is_nba_matched"] == 1].copy()
    out_matched_q = out_all[(out_all["is_nba_matched"] == 1) & (out_all["is_qualified_pool"] == 1)].copy()
    # For matched exports, primary displayed rank should be matched-cohort rank, not global-qualified rank.
    if "season_rank_matched" in out_matched.columns:
        out_matched["season_rank"] = out_matched["season_rank_matched"]
    if "season_rank_matched" in out_matched_q.columns:
        out_matched_q["season_rank"] = out_matched_q["season_rank_matched"]
    csv_m_path = INFERENCE_DIR / "season_rankings_latest_best_current_matched.csv"
    out_matched.to_csv(csv_m_path, index=False)
    csv_mq_path = INFERENCE_DIR / "season_rankings_latest_best_current_matched_qualified.csv"
    out_matched_q.to_csv(csv_mq_path, index=False)

    season_dir = INFERENCE_DIR / "season_rankings_top25_best_current_by_season_csv"
    season_dir.mkdir(parents=True, exist_ok=True)
    for season, g in out_all.groupby("college_final_season"):
        g.head(25).to_csv(season_dir / f"season_{season}_top25_all.csv", index=False)
    for season, g in out_qualified.groupby("college_final_season"):
        g.head(25).to_csv(season_dir / f"season_{season}_top25_qualified.csv", index=False)

    xlsx_path = INFERENCE_DIR / "season_rankings_top25_best_current_tabs.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for season, g in out_all.groupby("college_final_season"):
            g.head(25).to_excel(writer, sheet_name=f"{season}_all"[:31], index=False)
        for season, g in out_qualified.groupby("college_final_season"):
            g.head(25).to_excel(writer, sheet_name=f"{season}_q"[:31], index=False)
        for season, g in out_matched_q.groupby("college_final_season"):
            g.head(25).to_excel(writer, sheet_name=f"{season}_mq"[:31], index=False)

    # Dedicated qualified-only workbook (mainstay output).
    q_xlsx_path = INFERENCE_DIR / "season_rankings_top25_qualified_only_tabs.xlsx"
    with pd.ExcelWriter(q_xlsx_path, engine="openpyxl") as writer:
        for season, g in out_qualified.groupby("college_final_season"):
            g.head(25).to_excel(writer, sheet_name=f"{season}_q"[:31], index=False)
        out_qualified.head(25).to_excel(writer, sheet_name="overall_top25_q", index=False)

    # Dedicated matched+qualified workbook (NBA-anchored view).
    mq_xlsx_path = INFERENCE_DIR / "season_rankings_top25_matched_qualified_tabs.xlsx"
    with pd.ExcelWriter(mq_xlsx_path, engine="openpyxl") as writer:
        for season, g in out_matched_q.groupby("college_final_season"):
            g.head(25).to_excel(writer, sheet_name=f"{season}_mq"[:31], index=False)
        out_matched_q.head(25).to_excel(writer, sheet_name="overall_top25_mq", index=False)

    print(f"predictions={pred_path}")
    print(f"csv={csv_path}")
    print(f"csv_qualified={csv_q_path}")
    print(f"csv_matched={csv_m_path}")
    print(f"csv_matched_qualified={csv_mq_path}")
    print(f"xlsx={xlsx_path}")
    print(f"xlsx_qualified={q_xlsx_path}")
    print(f"xlsx_matched_qualified={mq_xlsx_path}")
    print(f"per_season_dir={season_dir}")
    print(
        f"rows_all={len(out_all)} rows_qualified={len(out_qualified)} "
        f"rows_matched={len(out_matched)} rows_matched_qualified={len(out_matched_q)} "
        f"seasons={out_all['college_final_season'].nunique()}"
    )


if __name__ == "__main__":
    main()
