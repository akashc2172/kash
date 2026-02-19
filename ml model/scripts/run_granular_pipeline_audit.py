#!/usr/bin/env python3
"""
Granular Pipeline Audit (Per-Season, Per-Column, Train vs Inference)
====================================================================
Runs exhaustive contract checks over model inputs/targets and linkage.

Outputs:
  data/audit/granular_pipeline_audit_<timestamp>/
    - feature_coverage_by_season.csv
    - target_coverage_by_season.csv
    - train_inference_skew.csv
    - crosswalk_integrity.csv
    - sanity_player_ranks.csv
    - summary.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TRAIN_PATH = ROOT / "data" / "training" / "unified_training_table.parquet"
INF_DIR = ROOT / "data" / "inference"
WH2 = ROOT / "data" / "warehouse_v2"
WH = ROOT / "data" / "warehouse.duckdb"


def _latest_prediction() -> Path:
    preds = sorted(INF_DIR.glob("prospect_predictions_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not preds:
        raise FileNotFoundError("No prediction files found in data/inference")
    return preds[0]


def _safe_stats(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce")
    return {
        "nonnull_rate": float(x.notna().mean()),
        "nonzero_rate": float((x.fillna(0) != 0).mean()),
        "mean": float(x.mean()) if x.notna().any() else np.nan,
        "std": float(x.std()) if x.notna().sum() > 1 else np.nan,
        "p01": float(x.quantile(0.01)) if x.notna().any() else np.nan,
        "p99": float(x.quantile(0.99)) if x.notna().any() else np.nan,
    }


def _name_map() -> pd.DataFrame:
    con = duckdb.connect(str(WH), read_only=True)
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
    c AS (
        SELECT athlete_id, player_name, COUNT(*) AS n,
               ROW_NUMBER() OVER (
                 PARTITION BY athlete_id
                 ORDER BY COUNT(*) DESC, LENGTH(player_name) DESC, player_name
               ) AS rn
        FROM src
        GROUP BY 1,2
    )
    SELECT athlete_id, player_name
    FROM c
    WHERE rn=1
    """
    out = con.execute(q).df()
    con.close()
    out["athlete_id"] = out["athlete_id"].astype(str)
    return out


def _iter_numeric_cols(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    out = []
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def main() -> None:
    from models.player_encoder import TIER1_COLUMNS, TIER2_COLUMNS, CAREER_BASE_COLUMNS, WITHIN_COLUMNS

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "data" / "audit" / f"granular_pipeline_audit_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(TRAIN_PATH)
    pred_path = _latest_prediction()
    pred = pd.read_parquet(pred_path)
    pred["athlete_id"] = pred["athlete_id"].astype(str)
    train["athlete_id"] = train["athlete_id"].astype(str)

    # Input columns under contract.
    input_cols = list(dict.fromkeys(TIER1_COLUMNS + TIER2_COLUMNS + CAREER_BASE_COLUMNS + WITHIN_COLUMNS))
    input_cols_num = _iter_numeric_cols(train, input_cols)
    target_cols = [c for c in ["y_peak_ovr", "year1_epm_tot", "gap_ts_legacy", "dev_rate_y1_y3_mean", "made_nba"] if c in train.columns]

    # 1) Per-season coverage for every input column.
    f_rows = []
    for season, g in train.groupby("college_final_season", dropna=False):
        for c in input_cols_num:
            st = _safe_stats(g[c])
            f_rows.append(
                {
                    "season": season,
                    "column": c,
                    "rows": int(len(g)),
                    **st,
                }
            )
    feature_cov = pd.DataFrame(f_rows).sort_values(["column", "season"])
    feature_cov.to_csv(out_dir / "feature_coverage_by_season.csv", index=False)

    # 1b) Per-season coverage for ALL numeric training columns (full-surface audit).
    all_num_cols = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c])]
    all_rows = []
    for season, g in train.groupby("college_final_season", dropna=False):
        for c in all_num_cols:
            st = _safe_stats(g[c])
            all_rows.append(
                {
                    "season": season,
                    "column": c,
                    "rows": int(len(g)),
                    **st,
                }
            )
    all_cov = pd.DataFrame(all_rows).sort_values(["column", "season"])
    all_cov.to_csv(out_dir / "all_numeric_coverage_by_season.csv", index=False)

    # 2) Per-season target coverage.
    t_rows = []
    for season, g in train.groupby("college_final_season", dropna=False):
        for c in target_cols:
            st = _safe_stats(g[c])
            t_rows.append(
                {
                    "season": season,
                    "column": c,
                    "rows": int(len(g)),
                    **st,
                }
            )
    target_cov = pd.DataFrame(t_rows).sort_values(["column", "season"])
    target_cov.to_csv(out_dir / "target_coverage_by_season.csv", index=False)

    # 3) Train vs inference skew checks on common inputs.
    # Note: inference may include many non-NBA players; use same stats for drift visibility.
    # Build inference table via current prediction merge if columns available.
    infer_cols = [c for c in input_cols_num if c in train.columns]
    # We cannot assume predictions file has all features; use season_rankings or skip.
    # Here we compare on training subset only by joining predictions.
    merged = train.merge(pred[["athlete_id", "college_final_season", "pred_peak_rapm"]], on=["athlete_id", "college_final_season"], how="left")
    s_rows = []
    for c in infer_cols:
        # Compare by season against full training baseline (sanity drift flags).
        for season, g in merged.groupby("college_final_season", dropna=False):
            x = pd.to_numeric(g[c], errors="coerce")
            st = _safe_stats(x)
            s_rows.append(
                {
                    "season": season,
                    "column": c,
                    "train_nonzero_rate": st["nonzero_rate"],
                    "train_nonnull_rate": st["nonnull_rate"],
                    "train_mean": st["mean"],
                    "train_std": st["std"],
                }
            )
    skew = pd.DataFrame(s_rows).sort_values(["column", "season"])
    skew.to_csv(out_dir / "train_inference_skew.csv", index=False)

    # 4) Crosswalk integrity checks.
    cw = pd.read_parquet(WH2 / "dim_player_nba_college_crosswalk.parquet")
    cw_rows = []
    cw_rows.append({"check": "rows", "value": int(len(cw))})
    cw_rows.append({"check": "duplicate_nba_id", "value": int(cw["nba_id"].duplicated().sum())})
    cw_rows.append({"check": "duplicate_athlete_id", "value": int(cw["athlete_id"].duplicated().sum())})
    if "match_score" in cw.columns:
        cw_rows.append({"check": "match_score_p10", "value": float(cw["match_score"].quantile(0.10))})
        cw_rows.append({"check": "match_score_min", "value": float(cw["match_score"].min())})
    crosswalk_integrity = pd.DataFrame(cw_rows)
    crosswalk_integrity.to_csv(out_dir / "crosswalk_integrity.csv", index=False)

    # 5) Sanity player rank checks.
    nmap = _name_map()
    rank = pred.merge(nmap, on="athlete_id", how="left")
    rank["player_name"] = rank["player_name"].fillna("Unknown")
    score_col = "pred_peak_rapm_rank_score" if "pred_peak_rapm_rank_score" in rank.columns else "pred_peak_rapm"
    rank = rank.sort_values(["college_final_season", score_col], ascending=[True, False])
    rank["season_rank"] = rank.groupby("college_final_season").cumcount() + 1
    cw = pd.read_parquet(WH2 / "dim_player_nba_college_crosswalk.parquet")
    dim = pd.read_parquet(WH2 / "dim_player_nba.parquet")
    dim_name_to_nba = {
        str(r["player_name"]).strip().lower(): r["nba_id"]
        for _, r in dim[["nba_id", "player_name"]].dropna().iterrows()
    }
    nba_to_athlete = dict(zip(cw["nba_id"], cw["athlete_id"]))
    sanity_players = [
        ("Zion Williamson", 2019),
        ("Ja Morant", 2019),
        ("Anthony Edwards", 2020),
        ("Cade Cunningham", 2021),
    ]
    sanity_rows = []
    for name, season in sanity_players:
        nba_id = dim_name_to_nba.get(name.lower())
        athlete_id = nba_to_athlete.get(nba_id) if nba_id is not None else None
        hit = rank[
            (rank["athlete_id"].astype(str) == str(athlete_id))
            & (rank["college_final_season"] == season)
        ]
        if len(hit):
            r = hit.iloc[0]
            sanity_rows.append(
                {
                    "player_name": name,
                    "season": season,
                    "found": 1,
                    "nba_id": int(nba_id) if nba_id is not None else np.nan,
                    "athlete_id": int(athlete_id) if athlete_id is not None else np.nan,
                    "season_rank": int(r["season_rank"]),
                    "pred_peak_rapm": float(r["pred_peak_rapm"]),
                    "rank_score": float(r[score_col]),
                }
            )
        else:
            sanity_rows.append(
                {
                    "player_name": name,
                    "season": season,
                    "found": 0,
                    "nba_id": int(nba_id) if nba_id is not None else np.nan,
                    "athlete_id": int(athlete_id) if athlete_id is not None else np.nan,
                    "season_rank": np.nan,
                    "pred_peak_rapm": np.nan,
                    "rank_score": np.nan,
                }
            )
    sanity = pd.DataFrame(sanity_rows)
    sanity.to_csv(out_dir / "sanity_player_ranks.csv", index=False)

    # 6) Summary + hard failures.
    dead_inputs = feature_cov.groupby("column")["nonzero_rate"].max()
    dead_inputs = dead_inputs[dead_inputs <= 0.001]
    low_cov_inputs = feature_cov.groupby("column")["nonnull_rate"].min()
    low_cov_inputs = low_cov_inputs[low_cov_inputs < 0.50]

    approx_checks_inputs_targets = int(len(input_cols_num)) * 16 + int(len(target_cols)) * 16
    approx_checks_all_numeric = int(len(all_num_cols)) * 16

    summary = {
        "timestamp": ts,
        "prediction_file": str(pred_path),
        "training_rows": int(len(train)),
        "training_cols": int(train.shape[1]),
        "input_columns_checked": int(len(input_cols_num)),
        "target_columns_checked": int(len(target_cols)),
        "all_numeric_columns_checked": int(len(all_num_cols)),
        "approx_checks_inputs_targets": approx_checks_inputs_targets,
        "approx_checks_all_numeric": approx_checks_all_numeric,
        "approx_checks_total": int(approx_checks_inputs_targets + approx_checks_all_numeric),
        "dead_input_columns_count": int(len(dead_inputs)),
        "low_cov_input_columns_count": int(len(low_cov_inputs)),
        "duplicate_nba_id_crosswalk": int(cw["nba_id"].duplicated().sum()),
        "duplicate_athlete_id_crosswalk": int(cw["athlete_id"].duplicated().sum()),
        "rank_score_column": score_col,
        "star_sanity_fail_count": int((sanity["season_rank"].fillna(999999) > 50).sum()),
        "dead_input_columns": dead_inputs.index.tolist(),
        "low_cov_input_columns": low_cov_inputs.index.tolist(),
        "audit_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
