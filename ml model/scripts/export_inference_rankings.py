#!/usr/bin/env python3
"""
Export inference rankings with player names, season ranks, and per-season tabs.
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
INFERENCE_DIR = BASE_DIR / "data" / "inference"
WAREHOUSE_DB = BASE_DIR / "data" / "warehouse.duckdb"


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
    score_col = "pred_peak_rapm_rank_score" if "pred_peak_rapm_rank_score" in ranked.columns else "pred_peak_rapm"
    ranked = ranked.sort_values(["college_final_season", score_col], ascending=[True, False])
    ranked["season_rank_all"] = ranked.groupby("college_final_season").cumcount() + 1

    # Qualified prospect pool:
    # keep raw rank for completeness, and add a practical rank that excludes tiny-sample rows.
    games = pd.to_numeric(ranked.get("college_games_played", pd.Series(index=ranked.index, dtype=float)), errors="coerce")
    poss = pd.to_numeric(ranked.get("college_poss_proxy", pd.Series(index=ranked.index, dtype=float)), errors="coerce")
    # Strict exposure gate to avoid tiny-sample rows dominating seasonal leaderboards.
    ranked["is_qualified_pool"] = ((games >= 14) & (poss >= 200)).astype(int)
    ranked["season_rank_qualified"] = pd.NA
    for season, g in ranked.groupby("college_final_season", sort=False):
        q_idx = g.index[g["is_qualified_pool"] == 1]
        if len(q_idx):
            ranked.loc[q_idx, "season_rank_qualified"] = range(1, len(q_idx) + 1)
    ranked["season_rank"] = ranked["season_rank_qualified"]

    keep = ["college_final_season", "season_rank", "athlete_id", "player_name", score_col, "pred_peak_rapm"]
    if score_col != "pred_peak_rapm" and "pred_peak_rapm_reliability" in ranked.columns:
        keep.insert(5, "pred_peak_rapm_reliability")
    for c in ["season_rank_all", "season_rank_qualified", "is_qualified_pool", "college_games_played", "college_poss_proxy", "college_minutes_total"]:
        if c in ranked.columns:
            keep.append(c)
    for c in ["pred_dev_rate", "pred_dev_rate_std", "pred_year1_epm", "pred_gap_ts", "pred_made_nba_logit"]:
        if c in ranked.columns:
            keep.append(c)
    out = ranked[ranked["is_qualified_pool"] == 1][keep].copy()

    csv_path = INFERENCE_DIR / "season_rankings_latest_best_current.csv"
    out.to_csv(csv_path, index=False)

    season_dir = INFERENCE_DIR / "season_rankings_top25_best_current_by_season_csv"
    season_dir.mkdir(parents=True, exist_ok=True)
    for season, g in out.groupby("college_final_season"):
        q = g[g["is_qualified_pool"] == 1] if "is_qualified_pool" in g.columns else g
        q.head(25).to_csv(season_dir / f"season_{season}_top25.csv", index=False)

    xlsx_path = INFERENCE_DIR / "season_rankings_top25_best_current_tabs.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for season, g in out.groupby("college_final_season"):
            g.head(25).to_excel(writer, sheet_name=str(season)[:31], index=False)

    print(f"predictions={pred_path}")
    print(f"csv={csv_path}")
    print(f"xlsx={xlsx_path}")
    print(f"per_season_dir={season_dir}")
    print(f"rows={len(out)} seasons={out['college_final_season'].nunique()}")


if __name__ == "__main__":
    main()
