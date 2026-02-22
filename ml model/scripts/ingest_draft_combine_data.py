#!/usr/bin/env python3
"""Stage 2: Ingest Combine Data with deterministic linkage + contract outputs.

Outputs:
- data/warehouse_v2/raw_nba_draft_combine.parquet
- data/warehouse_v2/fact_player_combine_measurements.parquet
- data/audit/combine_linkage_quality.csv
- data/audit/combine_unmatched.csv
"""

from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
RAW_CSV_PATHS = [
    BASE / "data" / "raw" / "raw_nba_draft_combine.csv",
    BASE / "data" / "external" / "nba-draft-combine-command-center" / "raw_nba_draft_combine.csv",
]
WAREHOUSE_V2 = BASE / "data" / "warehouse_v2"
RAW_OUT = WAREHOUSE_V2 / "raw_nba_draft_combine.parquet"
FACT_OUT = WAREHOUSE_V2 / "fact_player_combine_measurements.parquet"

CROSSWALK_PATH = WAREHOUSE_V2 / "dim_player_nba_college_crosswalk.parquet"
DIM_NBA_PATH = WAREHOUSE_V2 / "dim_player_nba.parquet"
AUDIT_DIR = BASE / "data" / "audit"
AUDIT_OUT = AUDIT_DIR / "combine_linkage_quality.csv"
AUDIT_UNMATCHED = AUDIT_DIR / "combine_unmatched.csv"


def normalize_name(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
        .str.strip()
    )


def _read_raw() -> pd.DataFrame:
    for p in RAW_CSV_PATHS:
        if p.exists():
            logger.info("Loading raw combine file: %s", p)
            return pd.read_csv(p)
    raise FileNotFoundError(f"No raw combine csv found in: {RAW_CSV_PATHS}")


def _draft_year_proxy(df: pd.DataFrame) -> pd.Series:
    draft = pd.to_numeric(df.get("draft_year"), errors="coerce")
    rookie = pd.to_numeric(df.get("rookie_season_year"), errors="coerce") - 1
    return draft.where(draft.notna(), rookie)


def main() -> None:
    df_comb = _read_raw()
    WAREHOUSE_V2.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # Standardize combine schema.
    col_map = {
        "SEASON": "combine_year",
        "PLAYER_ID": "combine_player_id",
        "PLAYER_NAME": "combine_player_name",
        "HEIGHT_WO_SHOES": "height_wo_shoes_in",
        "HEIGHT_W_SHOES": "height_w_shoes_in",
        "WEIGHT": "weight_lbs",
        "WINGSPAN": "wingspan_in",
        "STANDING_REACH": "standing_reach_in",
        "STANDING_VERTICAL_LEAP": "no_step_vertical_in",
        "MAX_VERTICAL_LEAP": "max_vertical_in",
        "LANE_AGILITY_TIME": "lane_agility_s",
        "SHUTTLE_RUN": "shuttle_run_s",
        "THREE_QUARTER_SPRINT": "three_quarter_sprint_s",
    }
    df_comb = df_comb.rename(columns={k: v for k, v in col_map.items() if k in df_comb.columns})
    df_comb["norm_name"] = normalize_name(df_comb.get("combine_player_name", pd.Series("", index=df_comb.index)))
    df_comb["combine_year"] = pd.to_numeric(df_comb.get("combine_year"), errors="coerce")
    df_comb.to_parquet(RAW_OUT, index=False)

    if not DIM_NBA_PATH.exists() or not CROSSWALK_PATH.exists():
        raise FileNotFoundError("Missing required dim/crosswalk parquet for linkage.")

    dim_nba = pd.read_parquet(DIM_NBA_PATH)[["nba_id", "player_name", "draft_year", "rookie_season_year"]].copy()
    dim_nba["norm_name_nba"] = normalize_name(dim_nba["player_name"])
    dim_nba["draft_year_proxy"] = _draft_year_proxy(dim_nba)

    cw = pd.read_parquet(CROSSWALK_PATH)[["athlete_id", "nba_id"]].dropna().drop_duplicates(subset=["nba_id"]).copy()
    athlete_map = cw[["nba_id", "athlete_id"]].copy()

    # 1) exact id match
    dim_nba["nba_id_float"] = pd.to_numeric(dim_nba["nba_id"], errors="coerce")
    df_comb["combine_player_id_float"] = pd.to_numeric(df_comb.get("combine_player_id"), errors="coerce")
    exact = df_comb.merge(
        dim_nba[["nba_id_float", "nba_id", "draft_year_proxy"]],
        left_on="combine_player_id_float",
        right_on="nba_id_float",
        how="left",
    )
    exact_hit = exact[exact["nba_id"].notna()].copy()
    exact_hit["link_method"] = "exact_nba_id"

    # 2) fuzzy constrained by draft-year proximity
    left = exact[exact["nba_id"].isna()].copy()
    left = left.drop(columns=[c for c in ["nba_id", "nba_id_float", "draft_year_proxy"] if c in left.columns])
    fuzzy = left.merge(
        dim_nba[["nba_id", "norm_name_nba", "draft_year_proxy"]],
        left_on="norm_name",
        right_on="norm_name_nba",
        how="left",
    )
    fuzzy["year_gap"] = (pd.to_numeric(fuzzy["combine_year"], errors="coerce") - pd.to_numeric(fuzzy["draft_year_proxy"], errors="coerce")).abs()
    # deterministic gate: keep only plausible year proximity
    fuzzy = fuzzy[fuzzy["year_gap"].notna() & (fuzzy["year_gap"] <= 2)].copy()
    fuzzy = fuzzy.sort_values(["combine_player_id", "year_gap", "nba_id"]).drop_duplicates(subset=["combine_player_id"], keep="first")
    fuzzy["link_method"] = "norm_name_draft_year"

    final = pd.concat([exact_hit, fuzzy], ignore_index=True, sort=False)
    final = final.merge(athlete_map, on="nba_id", how="left")
    final["is_measured"] = 1
    final["combine_source"] = "nba_draft_combine_command_center"
    final["confidence"] = np.where(final["link_method"] == "exact_nba_id", 1.0, 0.75)
    final["has_combine_measured"] = 1

    out_cols = [
        "nba_id",
        "athlete_id",
        "combine_year",
        "combine_player_name",
        "height_wo_shoes_in",
        "height_w_shoes_in",
        "weight_lbs",
        "wingspan_in",
        "standing_reach_in",
        "no_step_vertical_in",
        "max_vertical_in",
        "lane_agility_s",
        "shuttle_run_s",
        "three_quarter_sprint_s",
        "link_method",
        "combine_source",
        "is_measured",
        "confidence",
        "has_combine_measured",
    ]
    final_out = final[[c for c in out_cols if c in final.columns]].copy()
    final_out.to_parquet(FACT_OUT, index=False)

    unresolved = df_comb[~df_comb["combine_player_id"].isin(final["combine_player_id"])].copy()
    unresolved.to_csv(AUDIT_UNMATCHED, index=False)

    audit_df = pd.DataFrame([{
        "total_combine_rows": int(len(df_comb)),
        "matched_total": int(len(final_out)),
        "matched_exact_id": int((final_out.get("link_method") == "exact_nba_id").sum()) if "link_method" in final_out.columns else 0,
        "matched_name_year": int((final_out.get("link_method") == "norm_name_draft_year").sum()) if "link_method" in final_out.columns else 0,
        "unmatched": int(len(unresolved)),
        "match_rate": float(len(final_out) / len(df_comb)) if len(df_comb) else 0.0,
    }])
    audit_df.to_csv(AUDIT_OUT, index=False)

    logger.info("Saved raw combine: %s", RAW_OUT)
    logger.info("Saved combine measurements: %s", FACT_OUT)
    logger.info("Saved combine audit: %s", AUDIT_OUT)


if __name__ == "__main__":
    main()
