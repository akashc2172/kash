#!/usr/bin/env python3
"""Build NBA-mapped physicals gap queue and Jalen Williams validation artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
WAREHOUSE = BASE / "data" / "warehouse.duckdb"
WAREHOUSE_V2 = BASE / "data" / "warehouse_v2"
COLLEGE_FS = BASE / "data" / "college_feature_store" / "college_features_v1.parquet"
AUDIT = BASE / "data" / "audit"


def _load_crosswalk() -> pd.DataFrame:
    p = WAREHOUSE_V2 / "dim_player_nba_college_crosswalk.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["athlete_id", "nba_id"])
    df = pd.read_parquet(p)
    keep = [c for c in ["athlete_id", "nba_id"] if c in df.columns]
    if not keep:
        return pd.DataFrame(columns=["athlete_id", "nba_id"])
    return df[keep].dropna(subset=["athlete_id", "nba_id"]).drop_duplicates()


def _load_canonical() -> pd.DataFrame:
    p = WAREHOUSE_V2 / "fact_college_player_physicals_by_season.parquet"
    if p.exists():
        df = pd.read_parquet(p)
    else:
        con = duckdb.connect(str(WAREHOUSE), read_only=True)
        df = con.execute("SELECT * FROM fact_college_player_physicals_by_season").df()
        con.close()
    for c in ["athlete_id", "season", "team_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load_college_seasons() -> pd.DataFrame:
    df = pd.read_parquet(COLLEGE_FS)
    if "split_id" in df.columns:
        df = df[df["split_id"] == "ALL__ALL"].copy()
    keep = [c for c in ["athlete_id", "season", "teamId"] if c in df.columns]
    out = df[keep].dropna(subset=["athlete_id", "season"]).drop_duplicates()
    out = out.rename(columns={"teamId": "team_id"})
    out["athlete_id"] = pd.to_numeric(out["athlete_id"], errors="coerce")
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce")
    return out


def _load_name_dir() -> pd.DataFrame:
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    stats = con.execute(
        """
        SELECT
          CAST(athleteId AS BIGINT) AS athlete_id,
          CAST(season AS BIGINT) AS season,
          CAST(teamId AS BIGINT) AS team_id,
          mode(name) AS player_name,
          mode(team) AS team_name
        FROM fact_player_season_stats
        WHERE athleteId IS NOT NULL
        GROUP BY 1,2,3
        """
    ).df()
    shots = con.execute(
        """
        SELECT
          CAST(s.shooterAthleteId AS BIGINT) AS athlete_id,
          CAST(g.season AS BIGINT) AS season,
          CAST(s.teamId AS BIGINT) AS team_id,
          mode(s.shooter_name) AS player_name,
          mode(COALESCE(t.school,t.displayName,t.shortDisplayName,t.abbreviation)) AS team_name
        FROM stg_shots s
        JOIN dim_games g ON g.id=CAST(s.gameId AS BIGINT)
        LEFT JOIN dim_teams t ON t.id=CAST(s.teamId AS BIGINT)
        WHERE s.shooterAthleteId IS NOT NULL
        GROUP BY 1,2,3
        """
    ).df()
    con.close()
    out = pd.concat([stats, shots], ignore_index=True)
    out = out.dropna(subset=["athlete_id", "season"])
    out = out.sort_values(["athlete_id", "season"]).drop_duplicates(["athlete_id", "season", "team_id"], keep="last")
    return out


def _load_provider_attempts() -> pd.DataFrame:
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    try:
        df = con.execute(
            """
            SELECT
              CAST(season AS BIGINT) AS season,
              CAST(team_id AS BIGINT) AS team_id,
              string_agg(DISTINCT provider, '|') AS provider_attempts,
              count(*) AS raw_rows
            FROM raw_team_roster_physical
            GROUP BY 1,2
            """
        ).df()
    except Exception:
        df = pd.DataFrame(columns=["season", "team_id", "provider_attempts", "raw_rows"])
    con.close()
    return df


def build_gap_queue(jalen_athlete_id: int = 27556) -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)

    cross = _load_crosswalk()
    seasons = _load_college_seasons()
    canon = _load_canonical()
    names = _load_name_dir()
    attempts = _load_provider_attempts()

    base = seasons.merge(cross[["athlete_id"]].drop_duplicates(), on="athlete_id", how="inner")
    ckeep = ["athlete_id", "season", "team_id", "height_in", "weight_lbs"]
    canon = canon[[c for c in ckeep if c in canon.columns]].copy()
    for c in ["athlete_id", "season", "team_id"]:
        if c in canon.columns:
            canon[c] = pd.to_numeric(canon[c], errors="coerce")

    q = base.merge(canon, on=["athlete_id", "season", "team_id"], how="left")
    q["missing_height"] = q["height_in"].isna().astype(int)
    q["missing_weight"] = q["weight_lbs"].isna().astype(int)
    q = q[(q["missing_height"] == 1) | (q["missing_weight"] == 1)].copy()
    q = q.merge(names, on=["athlete_id", "season", "team_id"], how="left")
    q = q.merge(attempts, on=["season", "team_id"], how="left")
    q["provider_attempts"] = q["provider_attempts"].fillna("")
    q["candidate_sources"] = q["provider_attempts"].replace({"": "none"})

    out_cols = [
        "athlete_id", "player_name", "season", "team_name",
        "missing_height", "missing_weight", "provider_attempts", "candidate_sources",
    ]
    q = q[[c for c in out_cols if c in q.columns]].sort_values(["season", "team_name", "player_name"])
    q.to_csv(AUDIT / "physicals_gap_queue_nba_mapped.csv", index=False)

    # Jalen Williams validation.
    expected = seasons[seasons["athlete_id"] == int(jalen_athlete_id)][["athlete_id", "season", "team_id"]].drop_duplicates()
    v = expected.merge(canon, on=["athlete_id", "season", "team_id"], how="left")
    v = v.merge(names, on=["athlete_id", "season", "team_id"], how="left")
    v["has_height"] = v["height_in"].notna().astype(int)
    v["has_weight"] = v["weight_lbs"].notna().astype(int)
    v.sort_values("season").to_csv(AUDIT / "physicals_jalen_williams_validation.csv", index=False)

    # Lightweight cohort metric for gateing unresolved-by-coverage.
    denom = len(base)
    miss = len(q)
    cohort_missing_pct = (100.0 * miss / denom) if denom else 100.0
    pd.DataFrame([{
        "nba_mapped_season_rows": denom,
        "missing_physical_rows": miss,
        "missing_rate_pct": cohort_missing_pct,
    }]).to_csv(AUDIT / "physicals_nba_mapped_missing_rate.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build NBA-mapped physical gap queue and Jalen validation")
    ap.add_argument("--jalen-athlete-id", type=int, default=27556)
    args = ap.parse_args()
    build_gap_queue(jalen_athlete_id=args.jalen_athlete_id)
    print(str(AUDIT / "physicals_gap_queue_nba_mapped.csv"))
    print(str(AUDIT / "physicals_jalen_williams_validation.csv"))


if __name__ == "__main__":
    main()

