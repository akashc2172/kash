#!/usr/bin/env python3
"""
Peak-EPM alignment diagnostics.

Builds a workbook that explains:
- how predicted rank score aligns with actual peak EPM targets,
- what peak window year each player is using,
- whether error drifts by class/age/development proxies.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
INFERENCE_DIR = BASE_DIR / "data" / "inference"
UNIFIED_PATH = BASE_DIR / "data" / "training" / "unified_training_table.parquet"
OUT_XLSX = INFERENCE_DIR / "season_rankings_peak_epm_diagnostics.xlsx"


def _safe_corr(g: pd.DataFrame, x: str, y: str, method: str = "spearman") -> float:
    if g[x].notna().sum() < 5 or g[y].notna().sum() < 5:
        return np.nan
    return float(g[x].corr(g[y], method=method))


def _safe_median(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return np.nan
    return float(s.median())


def main() -> None:
    ranked_path = INFERENCE_DIR / "season_rankings_latest_best_current_qualified.csv"
    if not ranked_path.exists():
        raise FileNotFoundError(f"Missing rankings CSV: {ranked_path}")
    if not UNIFIED_PATH.exists():
        raise FileNotFoundError(f"Missing unified table: {UNIFIED_PATH}")

    ranked = pd.read_csv(ranked_path)
    unified = pd.read_parquet(UNIFIED_PATH)

    ranked["athlete_id"] = ranked["athlete_id"].astype(str)
    ranked["college_final_season"] = pd.to_numeric(ranked["college_final_season"], errors="coerce")
    ranked["pred_rank_score"] = pd.to_numeric(ranked["pred_rank_score"], errors="coerce")

    keep_u = [
        c
        for c in [
            "athlete_id",
            "college_final_season",
            "draft_year_proxy",
            "age_at_season",
            "career_years",
            "y_peak_epm_3y",
            "y_peak_epm_window",
            "epm_obs_seasons",
            "epm_obs_minutes",
            "epm_peak_window_end_year",
            "year1_epm_tot",
        ]
        if c in unified.columns
    ]
    u = unified[keep_u].copy()
    u["athlete_id"] = u["athlete_id"].astype(str)
    u["college_final_season"] = pd.to_numeric(u["college_final_season"], errors="coerce")

    d = ranked.merge(u, on=["athlete_id", "college_final_season"], how="left")
    d["epm_years_to_peak"] = pd.to_numeric(d.get("epm_peak_window_end_year"), errors="coerce") - pd.to_numeric(
        d.get("draft_year_proxy"), errors="coerce"
    )
    d["pred_rank_score_z"] = (d["pred_rank_score"] - d["pred_rank_score"].mean()) / max(d["pred_rank_score"].std(), 1e-8)
    d["actual_peak_epm_3y_z"] = (
        (pd.to_numeric(d.get("y_peak_epm_3y"), errors="coerce") - pd.to_numeric(d.get("y_peak_epm_3y"), errors="coerce").mean())
        / max(pd.to_numeric(d.get("y_peak_epm_3y"), errors="coerce").std(), 1e-8)
    )
    d["actual_peak_epm_window_z"] = (
        (pd.to_numeric(d.get("y_peak_epm_window"), errors="coerce") - pd.to_numeric(d.get("y_peak_epm_window"), errors="coerce").mean())
        / max(pd.to_numeric(d.get("y_peak_epm_window"), errors="coerce").std(), 1e-8)
    )

    # Per-season summary.
    season_rows = []
    for season, g in d.groupby("college_final_season", sort=True):
        g3 = g[g["y_peak_epm_3y"].notna()]
        gw = g[g["y_peak_epm_window"].notna()]
        season_rows.append(
            {
                "college_final_season": int(season) if pd.notna(season) else np.nan,
                "rows": int(len(g)),
                "rows_with_peak_epm_3y": int(len(g3)),
                "rows_with_peak_epm_window": int(len(gw)),
                "spearman_pred_vs_peak_epm_3y": _safe_corr(g3, "pred_rank_score", "y_peak_epm_3y", "spearman"),
                "pearson_pred_vs_peak_epm_3y": _safe_corr(g3, "pred_rank_score", "y_peak_epm_3y", "pearson"),
                "spearman_pred_vs_peak_epm_window": _safe_corr(gw, "pred_rank_score", "y_peak_epm_window", "spearman"),
                "median_epm_obs_seasons": _safe_median(gw.get("epm_obs_seasons", pd.Series(dtype=float))) if len(gw) else np.nan,
                "median_epm_years_to_peak": _safe_median(gw.get("epm_years_to_peak", pd.Series(dtype=float))) if len(gw) else np.nan,
                "median_age_at_season": _safe_median(g.get("age_at_season", pd.Series(dtype=float))) if len(g) else np.nan,
            }
        )
    season_summary = pd.DataFrame(season_rows).sort_values("college_final_season")

    # Age bins.
    age_df = d[pd.to_numeric(d.get("age_at_season"), errors="coerce").notna()].copy()
    if len(age_df):
        age_df["age_bin"] = pd.cut(pd.to_numeric(age_df["age_at_season"], errors="coerce"), bins=[17, 19, 20, 21, 22, 30])
        age_summary = (
            age_df.groupby("age_bin", observed=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "rows": len(g),
                        "peak_epm3y_non_null_rate": float(g["y_peak_epm_3y"].notna().mean()),
                        "pred_vs_peak_epm3y_spearman": _safe_corr(g[g["y_peak_epm_3y"].notna()], "pred_rank_score", "y_peak_epm_3y", "spearman"),
                        "pred_vs_peak_epm_window_spearman": _safe_corr(g[g["y_peak_epm_window"].notna()], "pred_rank_score", "y_peak_epm_window", "spearman"),
                        "median_epm_years_to_peak": _safe_median(g.get("epm_years_to_peak", pd.Series(dtype=float))),
                    }
                )
            )
            .reset_index()
        )
    else:
        age_summary = pd.DataFrame(columns=["age_bin", "rows"])

    # Top misses.
    miss_cols = [
        "college_final_season",
        "season_rank",
        "athlete_id",
        "player_name",
        "pred_rank_target",
        "pred_rank_score",
        "y_peak_epm_3y",
        "y_peak_epm_window",
        "year1_epm_tot",
        "epm_obs_seasons",
        "epm_peak_window_end_year",
        "epm_years_to_peak",
        "age_at_season",
        "career_years",
        "actual_epm3y_rank_class_qualified",
        "actual_epm_window_rank_class_qualified",
    ]
    available_miss_cols = [c for c in miss_cols if c in d.columns]
    dm = d[available_miss_cols].copy()
    if "y_peak_epm_3y" in dm.columns:
        dm["err_peak_epm_3y"] = pd.to_numeric(dm["pred_rank_score"], errors="coerce") - pd.to_numeric(dm["y_peak_epm_3y"], errors="coerce")
        top_over = dm[dm["err_peak_epm_3y"].notna()].sort_values("err_peak_epm_3y", ascending=False).head(200)
        top_under = dm[dm["err_peak_epm_3y"].notna()].sort_values("err_peak_epm_3y", ascending=True).head(200)
    else:
        top_over = pd.DataFrame()
        top_under = pd.DataFrame()

    # Global summary.
    g3 = d[d["y_peak_epm_3y"].notna()].copy()
    gw = d[d["y_peak_epm_window"].notna()].copy()
    summary = pd.DataFrame(
        [
            {"metric": "rows_total", "value": float(len(d))},
            {"metric": "rows_peak_epm_3y", "value": float(len(g3))},
            {"metric": "rows_peak_epm_window", "value": float(len(gw))},
            {"metric": "pred_vs_peak_epm_3y_spearman", "value": _safe_corr(g3, "pred_rank_score", "y_peak_epm_3y", "spearman")},
            {"metric": "pred_vs_peak_epm_3y_pearson", "value": _safe_corr(g3, "pred_rank_score", "y_peak_epm_3y", "pearson")},
            {"metric": "pred_vs_peak_epm_window_spearman", "value": _safe_corr(gw, "pred_rank_score", "y_peak_epm_window", "spearman")},
            {"metric": "median_epm_obs_seasons", "value": _safe_median(gw.get("epm_obs_seasons", pd.Series(dtype=float))) if len(gw) else np.nan},
            {"metric": "median_epm_years_to_peak", "value": _safe_median(gw.get("epm_years_to_peak", pd.Series(dtype=float))) if len(gw) else np.nan},
            {"metric": "pred_rank_score_std", "value": float(pd.to_numeric(d["pred_rank_score"], errors="coerce").std())},
        ]
    )

    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        season_summary.to_excel(writer, sheet_name="season_summary", index=False)
        age_summary.to_excel(writer, sheet_name="age_bias", index=False)
        if len(top_over):
            top_over.to_excel(writer, sheet_name="top_overpred", index=False)
        if len(top_under):
            top_under.to_excel(writer, sheet_name="top_underpred", index=False)
        dcols = [c for c in [
            "college_final_season",
            "season_rank",
            "athlete_id",
            "player_name",
            "pred_rank_target",
            "pred_rank_score",
            "y_peak_epm_3y",
            "y_peak_epm_window",
            "actual_epm3y_rank_class_qualified",
            "actual_epm_window_rank_class_qualified",
            "epm_obs_seasons",
            "epm_obs_minutes",
            "epm_peak_window_end_year",
            "epm_years_to_peak",
            "draft_year_proxy",
            "age_at_season",
            "career_years",
            "college_games_played",
            "college_minutes_total_raw",
            "college_poss_proxy",
        ] if c in d.columns]
        d[dcols].sort_values(["college_final_season", "season_rank"], ascending=[True, True]).to_excel(
            writer, sheet_name="qualified_full", index=False
        )

    print(f"wrote={OUT_XLSX}")


if __name__ == "__main__":
    main()
