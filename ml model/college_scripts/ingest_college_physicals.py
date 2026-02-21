#!/usr/bin/env python3
"""Ingest season-by-season college physicals with provenance and canonical resolution.

Stages:
1) Ingest raw provider rows into `raw_team_roster_physical` (append-only)
2) Resolve identities to athlete_id and emit audits
3) Build canonical `fact_college_player_physicals_by_season`
4) Build derived `fact_college_player_physical_trajectory`
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import requests
from difflib import SequenceMatcher

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DB = Path("data/warehouse.duckdb")
DEFAULT_AUDIT = Path("data/audit")
DEFAULT_MANUAL = Path("data/manual_physicals")
DEFAULT_WAREHOUSE_V2 = Path("data/warehouse_v2")
ADAPTER_VERSION = "v1.1.0"

PUNCT_RE = re.compile(r"[^a-z0-9\s]")
WS_RE = re.compile(r"\s+")
SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b", flags=re.IGNORECASE)
FT_IN_RE = re.compile(r"^\s*(\d+)\s*[-']\s*(\d{1,2})\s*(?:\"|in)?\s*$")
CM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*cm\s*$", flags=re.IGNORECASE)
IN_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:in|inch|inches)?\s*$", flags=re.IGNORECASE)
KG_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*kg\s*$", flags=re.IGNORECASE)
LB_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:lb|lbs|pound|pounds)?\s*$", flags=re.IGNORECASE)

SOURCE_PRIORITY = {
    "official_roster": 4,
    "media_guide": 3,
    "conference_roster": 2,
    "recruiting_fallback": 1,
}

PROVIDER_DEFAULT_TYPE = {
    "cbd": "official_roster",
    "cbbpy": "official_roster",
    "sportsipy": "conference_roster",
    "manual": "media_guide",
    "recruiting_fallback": "recruiting_fallback",
    "nba_fallback": "recruiting_fallback",
}


@dataclass
class PhysicalRecord:
    provider: str
    season: int
    team_id: Optional[int]
    team_name: Optional[str]
    player_name_raw: str
    player_id_raw: Optional[str]
    height_raw: Optional[str]
    weight_raw: Optional[str]
    wingspan_raw: Optional[str]
    standing_reach_raw: Optional[str]
    source_url: Optional[str]
    source_type: str
    confidence: float
    is_measured_vs_listed: str
    payload_json: str
    run_id: str
    adapter_version: str


def now_utc_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    n = str(x).lower().strip()
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode("ascii")
    n = n.replace("-", " ").replace("'", " ")
    n = SUFFIX_RE.sub(" ", n)
    n = PUNCT_RE.sub(" ", n)
    n = WS_RE.sub(" ", n).strip()
    return n


def parse_height_in(raw: Any) -> Optional[float]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    m = FT_IN_RE.match(s)
    if m:
        ft = int(m.group(1))
        inch = int(m.group(2))
        return float(ft * 12 + inch)
    m = CM_RE.match(s)
    if m:
        cm = float(m.group(1))
        return float(cm / 2.54)
    m = IN_RE.match(s)
    if m:
        v = float(m.group(1))
        # If explicit inch markers are present, treat numeric as inches.
        if any(tok in s for tok in ("in", "inch", '"')):
            if v <= 9.5:
                return float(v * 12.0)
            return float(v)
        # Plain numeric can still be cm in many source tables (e.g. 194).
        if 120 <= v <= 260:
            return float(v / 2.54)
        if v <= 9.5:
            return float(v * 12.0)
        return float(v)
    try:
        v = float(s)
        if v <= 9.5:
            return float(v * 12.0)
        if 120 <= v <= 260:
            return float(v / 2.54)
        return float(v)
    except Exception:
        return None


def parse_weight_lbs(raw: Any) -> Optional[float]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    m = KG_RE.match(s)
    if m:
        kg = float(m.group(1))
        return float(kg * 2.2046226218)
    m = LB_RE.match(s)
    if m:
        return float(m.group(1))
    try:
        v = float(s)
        if v < 95:
            return float(v * 2.2046226218)
        return float(v)
    except Exception:
        return None


def parse_optional_inches(raw: Any) -> Optional[float]:
    return parse_height_in(raw)


def _validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "height_in" in df.columns:
        df.loc[(df["height_in"] < 60) | (df["height_in"] > 92), "height_in"] = np.nan
    if "weight_lbs" in df.columns:
        df.loc[(df["weight_lbs"] < 120) | (df["weight_lbs"] > 380), "weight_lbs"] = np.nan
    if "wingspan_in" in df.columns:
        df.loc[(df["wingspan_in"] < 60) | (df["wingspan_in"] > 100), "wingspan_in"] = np.nan
    if "standing_reach_in" in df.columns:
        df.loc[(df["standing_reach_in"] < 70) | (df["standing_reach_in"] > 130), "standing_reach_in"] = np.nan
    return df


def _name_score(a: str, b: str) -> float:
    a_n, b_n = normalize_name(a), normalize_name(b)
    if not a_n or not b_n:
        return 0.0
    r = SequenceMatcher(None, a_n, b_n).ratio()
    ta, tb = set(a_n.split()), set(b_n.split())
    if ta and tb:
        j = len(ta & tb) / len(ta | tb)
        r = max(r, 0.7 * r + 0.3 * j)
    return float(r)


def _team_season_universe(con: duckdb.DuckDBPyConnection, start: int, end: int) -> pd.DataFrame:
    q = """
    WITH g AS (
      SELECT season, CAST(homeTeamId AS BIGINT) AS team_id FROM dim_games
      UNION ALL
      SELECT season, CAST(awayTeamId AS BIGINT) AS team_id FROM dim_games
    )
    SELECT DISTINCT
      CAST(g.season AS BIGINT) AS season,
      CAST(g.team_id AS BIGINT) AS team_id,
      COALESCE(t.school, t.displayName, t.shortDisplayName, t.abbreviation) AS team_name
    FROM g
    LEFT JOIN dim_teams t ON t.id = g.team_id
    WHERE g.season BETWEEN ? AND ?
      AND g.team_id IS NOT NULL
    """
    return con.execute(q, [int(start), int(end)]).df()


def _player_season_directory(con: duckdb.DuckDBPyConnection, start: int, end: int) -> pd.DataFrame:
    # Strong identity base from player season stats plus shot-derived seasons for sparse years.
    stats_q = """
    SELECT DISTINCT
      CAST(athleteId AS BIGINT) AS athlete_id,
      mode(name) AS athlete_name,
      CAST(teamId AS BIGINT) AS team_id,
      mode(team) AS team_name,
      CAST(season AS BIGINT) AS season
    FROM fact_player_season_stats
    WHERE athleteId IS NOT NULL
      AND season BETWEEN ? AND ?
    GROUP BY 1,3,5
    """
    shots_q = """
    SELECT
      CAST(s.shooterAthleteId AS BIGINT) AS athlete_id,
      mode(s.shooter_name) AS athlete_name,
      CAST(s.teamId AS BIGINT) AS team_id,
      mode(COALESCE(t.school, t.displayName, t.shortDisplayName, t.abbreviation)) AS team_name,
      CAST(g.season AS BIGINT) AS season
    FROM stg_shots s
    JOIN dim_games g ON g.id = CAST(s.gameId AS BIGINT)
    LEFT JOIN dim_teams t ON t.id = CAST(s.teamId AS BIGINT)
    WHERE s.shooterAthleteId IS NOT NULL
      AND g.season BETWEEN ? AND ?
    GROUP BY 1,3,5
    """
    a = con.execute(stats_q, [int(start), int(end)]).df()
    b = con.execute(shots_q, [int(start), int(end)]).df()
    out = pd.concat([a, b], ignore_index=True)
    out = out.dropna(subset=["athlete_id", "season"]).copy()
    out["athlete_id"] = pd.to_numeric(out["athlete_id"], errors="coerce").astype("Int64")
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["athlete_name_norm"] = out["athlete_name"].map(normalize_name)
    out = out.drop_duplicates(subset=["athlete_id", "team_id", "season"]) 
    return out


def _fetch_cbd_roster(team_name: str, season: int, api_key: str) -> Tuple[list[dict], Optional[str]]:
    base = "https://api.collegebasketballdata.com/teams/roster"
    headers = {"Authorization": f"Bearer {api_key}"}
    tries = [
        {"team": team_name, "season": int(season)},
        {"school": team_name, "season": int(season)},
    ]
    for params in tries:
        try:
            r = requests.get(base, params=params, headers=headers, timeout=20)
            if r.status_code in (400, 404):
                continue
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                return data, None
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            continue
    return [], locals().get("err", "no_data")


def _records_from_cbd(universe: pd.DataFrame, api_key: Optional[str], run_id: str, failures: list[dict]) -> list[PhysicalRecord]:
    if not api_key:
        logger.warning("CBD API key missing; skipping cbd provider")
        return []
    rows: list[PhysicalRecord] = []
    for r in universe.itertuples(index=False):
        season = int(r.season)
        team_id = int(r.team_id) if pd.notna(r.team_id) else None
        team_name = str(r.team_name) if pd.notna(r.team_name) else None
        if not team_name:
            continue
        data, err = _fetch_cbd_roster(team_name, season, api_key)
        if err is not None:
            failures.append({
                "provider": "cbd",
                "season": season,
                "team_id": team_id,
                "team_name": team_name,
                "error": err,
            })
        for item in data:
            player_name = item.get("name") or item.get("player") or item.get("fullName") or ""
            if not player_name:
                continue
            rec = PhysicalRecord(
                provider="cbd",
                season=season,
                team_id=team_id,
                team_name=team_name,
                player_name_raw=str(player_name),
                player_id_raw=str(item.get("id") or item.get("athleteId") or "") or None,
                height_raw=(item.get("height") if item.get("height") is not None else item.get("heightInches")),
                weight_raw=(item.get("weight") if item.get("weight") is not None else item.get("weightPounds")),
                wingspan_raw=item.get("wingspan"),
                standing_reach_raw=item.get("standingReach"),
                source_url=f"https://api.collegebasketballdata.com/teams/roster?season={season}&team={team_name}",
                source_type="official_roster",
                confidence=0.96,
                is_measured_vs_listed="listed",
                payload_json=json.dumps(item, default=str),
                run_id=run_id,
                adapter_version=ADAPTER_VERSION,
            )
            rows.append(rec)
    return rows


def _records_from_cbbpy(universe: pd.DataFrame, run_id: str, failures: list[dict]) -> list[PhysicalRecord]:
    try:
        import cbbpy.mens_basketball as mbb  # type: ignore
    except Exception:
        logger.warning("cbbpy not available; skipping cbbpy provider")
        return []

    rows: list[PhysicalRecord] = []
    for r in universe.itertuples(index=False):
        season = int(r.season)
        team_id = int(r.team_id) if pd.notna(r.team_id) else None
        team_name = str(r.team_name) if pd.notna(r.team_name) else None
        if not team_name:
            continue
        slug = normalize_name(team_name).replace(" ", "-")
        if not slug:
            continue
        try:
            df = mbb.get_team_roster(slug, season)
        except Exception as exc:
            failures.append({
                "provider": "cbbpy",
                "season": season,
                "team_id": team_id,
                "team_name": team_name,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        if df is None or len(df) == 0:
            continue
        for _, rr in pd.DataFrame(df).iterrows():
            item = rr.to_dict()
            player_name = item.get("name") or item.get("player")
            if not player_name:
                continue
            rows.append(
                PhysicalRecord(
                    provider="cbbpy",
                    season=season,
                    team_id=team_id,
                    team_name=team_name,
                    player_name_raw=str(player_name),
                    player_id_raw=str(item.get("player_id") or item.get("id") or "") or None,
                    height_raw=item.get("height"),
                    weight_raw=item.get("weight"),
                    wingspan_raw=item.get("wingspan"),
                    standing_reach_raw=item.get("standing_reach"),
                    source_url=None,
                    source_type="official_roster",
                    confidence=0.9,
                    is_measured_vs_listed="listed",
                    payload_json=json.dumps(item, default=str),
                    run_id=run_id,
                    adapter_version=ADAPTER_VERSION,
                )
            )
    return rows


def _records_from_sportsipy(universe: pd.DataFrame, run_id: str, failures: list[dict]) -> list[PhysicalRecord]:
    try:
        from sportsipy.ncaab.teams import Teams  # type: ignore
    except Exception:
        logger.warning("sportsipy not available; skipping sportsipy provider")
        return []

    rows: list[PhysicalRecord] = []
    team_lookup = {(int(r.season), normalize_name(r.team_name)): (int(r.team_id) if pd.notna(r.team_id) else None, str(r.team_name) if pd.notna(r.team_name) else None) for r in universe.itertuples(index=False)}

    for season in sorted({int(s) for s in universe["season"].dropna().unique().tolist()}):
        try:
            teams = Teams(year=str(season))
        except Exception as exc:
            failures.append({
                "provider": "sportsipy",
                "season": season,
                "team_id": None,
                "team_name": None,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        for t in teams:
            t_name = getattr(t, "name", None) or getattr(t, "abbreviation", None)
            key = (season, normalize_name(t_name))
            team_id, team_name = team_lookup.get(key, (None, t_name))
            try:
                roster = t.roster.players
            except Exception:
                roster = []
            for p in roster:
                item = {
                    "name": getattr(p, "name", None),
                    "player_id": getattr(p, "player_id", None),
                    "height": getattr(p, "height", None),
                    "weight": getattr(p, "weight", None),
                }
                if not item["name"]:
                    continue
                rows.append(
                    PhysicalRecord(
                        provider="sportsipy",
                        season=season,
                        team_id=team_id,
                        team_name=team_name,
                        player_name_raw=str(item["name"]),
                        player_id_raw=str(item.get("player_id") or "") or None,
                        height_raw=item.get("height"),
                        weight_raw=item.get("weight"),
                        wingspan_raw=None,
                        standing_reach_raw=None,
                        source_url=None,
                        source_type="conference_roster",
                        confidence=0.82,
                        is_measured_vs_listed="listed",
                    payload_json=json.dumps(item, default=str),
                    run_id=run_id,
                    adapter_version=ADAPTER_VERSION,
                )
            )
    return rows


def _records_from_manual(manual_dir: Path, run_id: str) -> list[PhysicalRecord]:
    if not manual_dir.exists():
        logger.info("manual physicals directory missing (%s); skipping manual provider", manual_dir)
        return []

    rows: list[PhysicalRecord] = []
    files = sorted([p for p in manual_dir.rglob("*") if p.suffix.lower() in {".csv", ".parquet"}])
    for p in files:
        try:
            df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue

        # Flexible column mapping.
        colmap = {c.lower(): c for c in df.columns}
        season_col = colmap.get("season") or colmap.get("season_start_year") or colmap.get("year")
        team_id_col = colmap.get("team_id") or colmap.get("teamid")
        team_name_col = colmap.get("team_name") or colmap.get("team") or colmap.get("school")
        player_col = colmap.get("player_name") or colmap.get("name")
        player_id_col = colmap.get("player_id") or colmap.get("athlete_id")
        height_col = colmap.get("height") or colmap.get("height_raw") or colmap.get("height_inches")
        weight_col = colmap.get("weight") or colmap.get("weight_raw") or colmap.get("weight_pounds")
        wingspan_col = colmap.get("wingspan") or colmap.get("wingspan_raw")
        reach_col = colmap.get("standing_reach") or colmap.get("standing_reach_raw")
        source_col = colmap.get("source_url")
        if not (season_col and player_col):
            continue

        for _, rr in df.iterrows():
            try:
                season = int(pd.to_numeric(rr.get(season_col), errors="coerce"))
            except Exception:
                continue
            player = rr.get(player_col)
            if not player:
                continue
            payload = rr.to_dict()
            rows.append(
                PhysicalRecord(
                    provider="manual",
                    season=season,
                    team_id=int(rr.get(team_id_col)) if team_id_col and pd.notna(rr.get(team_id_col)) else None,
                    team_name=str(rr.get(team_name_col)) if team_name_col and pd.notna(rr.get(team_name_col)) else None,
                    player_name_raw=str(player),
                    player_id_raw=str(rr.get(player_id_col)) if player_id_col and pd.notna(rr.get(player_id_col)) else None,
                    height_raw=rr.get(height_col) if height_col else None,
                    weight_raw=rr.get(weight_col) if weight_col else None,
                    wingspan_raw=rr.get(wingspan_col) if wingspan_col else None,
                    standing_reach_raw=rr.get(reach_col) if reach_col else None,
                    source_url=str(rr.get(source_col)) if source_col and pd.notna(rr.get(source_col)) else None,
                    source_type="media_guide",
                    confidence=0.88,
                    is_measured_vs_listed="listed",
                    payload_json=json.dumps(payload, default=str),
                    run_id=run_id,
                    adapter_version=ADAPTER_VERSION,
                )
            )
    return rows


def _records_from_recruiting_fallback(con: duckdb.DuckDBPyConnection, start: int, end: int, run_id: str) -> list[PhysicalRecord]:
    q = """
    SELECT
      CAST(athleteId AS BIGINT) AS athlete_id,
      name,
      school,
      CAST(year AS BIGINT) AS recruit_year,
      CAST(heightInches AS DOUBLE) AS height_inches,
      CAST(weightPounds AS DOUBLE) AS weight_pounds
    FROM fact_recruiting_players
    WHERE athleteId IS NOT NULL
      AND year IS NOT NULL
      AND year BETWEEN ? AND ?
    """
    df = con.execute(q, [int(start - 1), int(end)]).df()
    rows: list[PhysicalRecord] = []
    for _, r in df.iterrows():
        season = int(r["recruit_year"]) + 1
        if season < start or season > end:
            continue
        payload = r.to_dict()
        rows.append(
            PhysicalRecord(
                provider="recruiting_fallback",
                season=season,
                team_id=None,
                team_name=str(r.get("school")) if pd.notna(r.get("school")) else None,
                player_name_raw=str(r.get("name") or ""),
                player_id_raw=str(int(r["athlete_id"])) if pd.notna(r.get("athlete_id")) else None,
                height_raw=str(r.get("height_inches")) if pd.notna(r.get("height_inches")) else None,
                weight_raw=str(r.get("weight_pounds")) if pd.notna(r.get("weight_pounds")) else None,
                wingspan_raw=None,
                standing_reach_raw=None,
                source_url=None,
                source_type="recruiting_fallback",
                confidence=0.35,
                is_measured_vs_listed="listed",
                payload_json=json.dumps(payload, default=str),
                run_id=run_id,
                adapter_version=ADAPTER_VERSION,
            )
        )
    return rows


def _records_from_nba_fallback(
    player_directory: pd.DataFrame,
    warehouse_v2_dir: Path,
    run_id: str,
) -> list[PhysicalRecord]:
    """
    Populate season-level college physical rows for NBA-mapped athletes from NBA
    physicals when college-side sources are unavailable.
    """
    xw_path = warehouse_v2_dir / "dim_player_nba_college_crosswalk.parquet"
    nba_path = warehouse_v2_dir / "dim_player_nba.parquet"
    if not xw_path.exists() or not nba_path.exists():
        logger.warning("nba_fallback skipped: missing crosswalk or nba dimension parquet")
        return []

    try:
        xw = pd.read_parquet(xw_path, columns=["athlete_id", "nba_id"]).drop_duplicates(subset=["athlete_id"])
        nba = pd.read_parquet(nba_path)
    except Exception as exc:
        logger.warning("nba_fallback skipped due source read failure: %s", exc)
        return []

    cols = [c for c in ["nba_id", "player_name", "ht_first", "ht_max", "wt_first", "wt_max"] if c in nba.columns]
    if "nba_id" not in cols:
        return []
    nba = nba[cols].drop_duplicates(subset=["nba_id"])

    link = xw.merge(nba, on="nba_id", how="left")
    if link.empty:
        return []

    pdx = player_directory[["athlete_id", "season", "team_id", "team_name"]].drop_duplicates()
    merged = pdx.merge(link, on="athlete_id", how="inner")
    if merged.empty:
        return []

    rows: list[PhysicalRecord] = []
    for r in merged.itertuples(index=False):
        h = r.ht_first if pd.notna(r.ht_first) else r.ht_max
        w = r.wt_first if pd.notna(r.wt_first) else r.wt_max
        if pd.isna(h) and pd.isna(w):
            continue
        height_raw = None if pd.isna(h) else str(float(h))
        if pd.isna(w):
            weight_raw = None
        else:
            wv = float(w)
            weight_raw = f"{wv} kg" if 50.0 <= wv <= 180.0 else f"{wv} lbs"

        payload = {
            "nba_id": getattr(r, "nba_id", None),
            "ht_first": getattr(r, "ht_first", None),
            "ht_max": getattr(r, "ht_max", None),
            "wt_first": getattr(r, "wt_first", None),
            "wt_max": getattr(r, "wt_max", None),
        }
        rows.append(
            PhysicalRecord(
                provider="nba_fallback",
                season=int(r.season),
                team_id=int(r.team_id) if pd.notna(r.team_id) else None,
                team_name=str(r.team_name) if pd.notna(r.team_name) else None,
                player_name_raw=str(getattr(r, "player_name", "") or ""),
                player_id_raw=str(getattr(r, "nba_id", "") or ""),
                height_raw=height_raw,
                weight_raw=weight_raw,
                wingspan_raw=None,
                standing_reach_raw=None,
                source_url=None,
                source_type="recruiting_fallback",
                confidence=0.35,
                is_measured_vs_listed="unknown",
                payload_json=json.dumps(payload, default=str),
                run_id=run_id,
                adapter_version=ADAPTER_VERSION,
            )
        )
    return rows


def _to_raw_df(records: list[PhysicalRecord]) -> pd.DataFrame:
    rows = []
    ts = now_utc_str()
    for r in records:
        rows.append(
            {
                "provider": r.provider,
                "season": int(r.season),
                "team_id": r.team_id,
                "team_name": r.team_name,
                "player_name_raw": r.player_name_raw,
                "player_id_raw": r.player_id_raw,
                "height_raw": r.height_raw,
                "weight_raw": r.weight_raw,
                "wingspan_raw": r.wingspan_raw,
                "standing_reach_raw": r.standing_reach_raw,
                "source_url": r.source_url,
                "source_type": r.source_type,
                "confidence": float(r.confidence),
                "is_measured_vs_listed": r.is_measured_vs_listed,
                "ingested_at_utc": ts,
                "payload_json": r.payload_json,
                "run_id": r.run_id,
                "adapter_version": r.adapter_version,
            }
        )
    return pd.DataFrame(rows)


def _ensure_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_team_roster_physical (
            provider VARCHAR,
            season BIGINT,
            team_id BIGINT,
            team_name VARCHAR,
            player_name_raw VARCHAR,
            player_id_raw VARCHAR,
            height_raw VARCHAR,
            weight_raw VARCHAR,
            wingspan_raw VARCHAR,
            standing_reach_raw VARCHAR,
            source_url VARCHAR,
            source_type VARCHAR,
            confidence DOUBLE,
            is_measured_vs_listed VARCHAR,
            ingested_at_utc VARCHAR,
            payload_json JSON
            ,run_id VARCHAR
            ,adapter_version VARCHAR
        )
        """
    )
    con.execute("ALTER TABLE raw_team_roster_physical ADD COLUMN IF NOT EXISTS run_id VARCHAR")
    con.execute("ALTER TABLE raw_team_roster_physical ADD COLUMN IF NOT EXISTS adapter_version VARCHAR")


def _append_raw(con: duckdb.DuckDBPyConnection, raw_df: pd.DataFrame) -> None:
    if raw_df.empty:
        return
    _ensure_tables(con)
    con.register("raw_phys_incoming", raw_df)
    con.execute(
        """
        INSERT INTO raw_team_roster_physical
        (
          provider, season, team_id, team_name, player_name_raw, player_id_raw,
          height_raw, weight_raw, wingspan_raw, standing_reach_raw,
          source_url, source_type, confidence, is_measured_vs_listed,
          ingested_at_utc, payload_json, run_id, adapter_version
        )
        SELECT
          provider, season, team_id, team_name, player_name_raw, player_id_raw,
          height_raw, weight_raw, wingspan_raw, standing_reach_raw,
          source_url, source_type, confidence, is_measured_vs_listed,
          ingested_at_utc, payload_json, run_id, adapter_version
        FROM raw_phys_incoming
        """
    )
    con.unregister("raw_phys_incoming")


def _resolve_identity(raw_df: pd.DataFrame, directory: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if raw_df.empty:
        return raw_df.copy(), pd.DataFrame(), pd.DataFrame()

    d = directory.copy()
    d["team_id"] = pd.to_numeric(d["team_id"], errors="coerce")
    d["season"] = pd.to_numeric(d["season"], errors="coerce")

    # Lookup maps.
    exact_map: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = {}
    season_team_map: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    season_name_map: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}

    for r in d.to_dict("records"):
        s = int(r["season"])
        t = int(r["team_id"]) if pd.notna(r.get("team_id")) else None
        n = normalize_name(r.get("athlete_name"))
        if t is not None and n:
            exact_map.setdefault((s, t, n), []).append(r)
            season_team_map.setdefault((s, t), []).append(r)
        if n:
            season_name_map.setdefault((s, n), []).append(r)

    resolved_rows: List[Dict[str, Any]] = []
    unresolved_rows: List[Dict[str, Any]] = []
    ambiguous_rows: List[Dict[str, Any]] = []

    for rr in raw_df.to_dict("records"):
        season = int(rr["season"])
        team_id = int(rr["team_id"]) if pd.notna(rr.get("team_id")) else None
        name_norm = normalize_name(rr.get("player_name_raw"))
        athlete_id: Optional[int] = None
        match_method = ""
        match_score = 0.0

        # 1) exact numeric player id bridge.
        pid_raw = rr.get("player_id_raw")
        if pid_raw is not None and str(pid_raw).strip() and str(pid_raw).strip().isdigit():
            pid = int(str(pid_raw).strip())
            cand = d[d["athlete_id"] == pid]
            if not cand.empty:
                # Prefer exact same season; then same team; else accept id exact.
                c = cand.copy()
                c["_score"] = 0
                c.loc[c["season"] == season, "_score"] += 2
                if team_id is not None:
                    c.loc[c["team_id"] == team_id, "_score"] += 1
                c = c.sort_values(["_score"], ascending=False)
                row = c.iloc[0].to_dict()
                athlete_id = int(row["athlete_id"])
                if pd.notna(row.get("team_id")):
                    team_id = int(row["team_id"])
                match_method = "player_id_exact"
                match_score = 1.0

        # 2) exact normalized name + team + season.
        if athlete_id is None and team_id is not None and name_norm:
            arr = exact_map.get((season, team_id, name_norm), [])
            if len(arr) == 1:
                athlete_id = int(arr[0]["athlete_id"])
                match_method = "name_team_season_exact"
                match_score = 0.99
            elif len(arr) > 1:
                ambiguous_rows.append({**rr, "reason": "multiple_exact_matches", "candidate_count": len(arr)})
                continue

        # 3) fuzzy normalized name + team + season.
        if athlete_id is None and team_id is not None and name_norm:
            arr = season_team_map.get((season, team_id), [])
            best = None
            second = None
            for c in arr:
                s = _name_score(name_norm, c.get("athlete_name"))
                if best is None or s > best[1]:
                    second = best
                    best = (c, s)
                elif second is None or s > second[1]:
                    second = (c, s)
            if best is not None and best[1] >= 0.93:
                if second is not None and abs(best[1] - second[1]) <= 0.01:
                    ambiguous_rows.append({**rr, "reason": "fuzzy_tie", "best_score": best[1], "second_score": second[1]})
                    continue
                athlete_id = int(best[0]["athlete_id"])
                match_method = "name_team_season_fuzzy"
                match_score = float(best[1])

        # 4) fallback exact name + season only.
        if athlete_id is None and name_norm:
            arr = season_name_map.get((season, name_norm), [])
            if len(arr) == 1:
                athlete_id = int(arr[0]["athlete_id"])
                if pd.notna(arr[0].get("team_id")):
                    team_id = int(arr[0]["team_id"])
                match_method = "name_season_exact"
                match_score = 0.9
            elif len(arr) > 1:
                ambiguous_rows.append({**rr, "reason": "name_season_multiple", "candidate_count": len(arr)})
                continue

        if athlete_id is None:
            unresolved_rows.append({**rr, "reason": "no_match"})
            continue

        resolved_rows.append(
            {
                **rr,
                "athlete_id": athlete_id,
                "team_id": team_id,
                "match_method": match_method,
                "match_score": match_score,
            }
        )

    resolved = pd.DataFrame(resolved_rows)
    unresolved = pd.DataFrame(unresolved_rows)
    ambiguous = pd.DataFrame(ambiguous_rows)
    return resolved, unresolved, ambiguous


def _canonicalize(resolved: pd.DataFrame) -> pd.DataFrame:
    if resolved.empty:
        cols = [
            "athlete_id", "season", "team_id", "height_in", "weight_lbs", "wingspan_in",
            "standing_reach_in", "wingspan_minus_height_in", "has_height", "has_weight", "has_wingspan",
            "source_type", "source_provider", "source_url", "confidence", "is_measured_vs_listed",
            "record_rank", "updated_at_utc",
        ]
        return pd.DataFrame(columns=cols)

    df = resolved.copy()
    df["height_in"] = df["height_raw"].map(parse_height_in)
    df["weight_lbs"] = df["weight_raw"].map(parse_weight_lbs)
    df["wingspan_in"] = df["wingspan_raw"].map(parse_optional_inches)
    df["standing_reach_in"] = df["standing_reach_raw"].map(parse_optional_inches)
    df = _validate_ranges(df)
    df["wingspan_minus_height_in"] = df["wingspan_in"] - df["height_in"]

    df["source_pri"] = df["source_type"].map(SOURCE_PRIORITY).fillna(0).astype(float)
    df["completeness"] = (
        df[["height_in", "weight_lbs", "wingspan_in", "standing_reach_in"]].notna().sum(axis=1)
    )
    df["score"] = (
        10.0 * df["source_pri"] +
        2.0 * pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0) +
        0.25 * df["completeness"] +
        pd.to_numeric(df["match_score"], errors="coerce").fillna(0.0)
    )

    # deterministic tiebreak.
    sort_cols = [
        "athlete_id", "season", "team_id", "score", "confidence", "completeness",
        "provider", "source_type", "player_name_raw",
    ]
    asc = [True, True, True, False, False, False, True, True, True]
    df = df.sort_values(sort_cols, ascending=asc)
    df["record_rank"] = df.groupby(["athlete_id", "season", "team_id"]).cumcount() + 1

    canon = df[df["record_rank"] == 1].copy()
    canon["has_height"] = canon["height_in"].notna().astype(int)
    canon["has_weight"] = canon["weight_lbs"].notna().astype(int)
    canon["has_wingspan"] = canon["wingspan_in"].notna().astype(int)
    canon["updated_at_utc"] = now_utc_str()
    canon = canon.rename(columns={
        "provider": "source_provider",
        "source_url": "source_url",
    })

    keep = [
        "athlete_id", "season", "team_id",
        "height_in", "weight_lbs", "wingspan_in", "standing_reach_in", "wingspan_minus_height_in",
        "has_height", "has_weight", "has_wingspan",
        "source_type", "source_provider", "source_url", "confidence", "is_measured_vs_listed",
        "record_rank", "updated_at_utc",
    ]
    return canon[keep]


def _slope_last_n(seasons: np.ndarray, values: np.ndarray, n: int = 3) -> float:
    mask = np.isfinite(seasons) & np.isfinite(values)
    seasons = seasons[mask]
    values = values[mask]
    if len(values) < n:
        return float("nan")
    seasons = seasons[-n:]
    values = values[-n:]
    x = seasons - seasons.mean()
    denom = float((x * x).sum())
    if denom <= 1e-9:
        return float("nan")
    y = values - values.mean()
    return float((x * y).sum() / denom)


def _build_trajectory(canon: pd.DataFrame) -> pd.DataFrame:
    if canon.empty:
        return pd.DataFrame(columns=[
            "athlete_id", "season", "height_in", "weight_lbs", "height_delta_yoy", "weight_delta_yoy",
            "height_slope_3yr", "weight_slope_3yr", "height_change_entry_to_final", "weight_change_entry_to_final",
            "trajectory_obs_count",
        ])
    out_rows = []
    for aid, g in canon.sort_values(["athlete_id", "season"]).groupby("athlete_id"):
        g = g.sort_values("season").copy()
        h = pd.to_numeric(g["height_in"], errors="coerce")
        w = pd.to_numeric(g["weight_lbs"], errors="coerce")
        seasons = pd.to_numeric(g["season"], errors="coerce").to_numpy(dtype=float)

        h_first = h.dropna().iloc[0] if h.notna().any() else np.nan
        h_last = h.dropna().iloc[-1] if h.notna().any() else np.nan
        w_first = w.dropna().iloc[0] if w.notna().any() else np.nan
        w_last = w.dropna().iloc[-1] if w.notna().any() else np.nan

        obs_count = int((h.notna() | w.notna()).sum())

        for i, row in g.iterrows():
            idx = g.index.get_loc(i)
            h_hist = h.iloc[: idx + 1].to_numpy(dtype=float)
            w_hist = w.iloc[: idx + 1].to_numpy(dtype=float)
            s_hist = seasons[: idx + 1]
            out_rows.append(
                {
                    "athlete_id": int(aid),
                    "season": int(row["season"]),
                    "height_in": float(row["height_in"]) if pd.notna(row["height_in"]) else np.nan,
                    "weight_lbs": float(row["weight_lbs"]) if pd.notna(row["weight_lbs"]) else np.nan,
                    "height_delta_yoy": float(h.iloc[idx] - h.iloc[idx - 1]) if idx >= 1 and pd.notna(h.iloc[idx]) and pd.notna(h.iloc[idx - 1]) else np.nan,
                    "weight_delta_yoy": float(w.iloc[idx] - w.iloc[idx - 1]) if idx >= 1 and pd.notna(w.iloc[idx]) and pd.notna(w.iloc[idx - 1]) else np.nan,
                    "height_slope_3yr": _slope_last_n(s_hist, h_hist, n=3),
                    "weight_slope_3yr": _slope_last_n(s_hist, w_hist, n=3),
                    "height_change_entry_to_final": float(h_last - h_first) if pd.notna(h_last) and pd.notna(h_first) else np.nan,
                    "weight_change_entry_to_final": float(w_last - w_first) if pd.notna(w_last) and pd.notna(w_first) else np.nan,
                    "trajectory_obs_count": obs_count,
                }
            )
    return pd.DataFrame(out_rows)


def _publish_tables(con: duckdb.DuckDBPyConnection, canon: pd.DataFrame, traj: pd.DataFrame) -> None:
    con.register("canon_in", canon)
    con.execute("CREATE OR REPLACE TABLE fact_college_player_physicals_by_season AS SELECT * FROM canon_in")
    con.unregister("canon_in")

    con.register("traj_in", traj)
    con.execute("CREATE OR REPLACE TABLE fact_college_player_physical_trajectory AS SELECT * FROM traj_in")
    con.unregister("traj_in")


def _export_parquet(canon: pd.DataFrame, traj: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    canon.to_parquet(out_dir / "fact_college_player_physicals_by_season.parquet", index=False)
    traj.to_parquet(out_dir / "fact_college_player_physical_trajectory.parquet", index=False)


def _write_audits(
    audit_dir: Path,
    raw_df: pd.DataFrame,
    resolved: pd.DataFrame,
    unresolved: pd.DataFrame,
    ambiguous: pd.DataFrame,
    canon: pd.DataFrame,
    provider_failures: pd.DataFrame,
) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)

    unresolved_path = audit_dir / "physicals_unresolved_identity.csv"
    ambiguous_path = audit_dir / "physicals_ambiguous_identity.csv"
    unresolved.to_csv(unresolved_path, index=False)
    ambiguous.to_csv(ambiguous_path, index=False)

    if canon.empty:
        cov = pd.DataFrame(columns=["season", "rows", "height_cov_pct", "weight_cov_pct", "wingspan_cov_pct"])
    else:
        cov = (
            canon.groupby("season", as_index=False)
            .agg(
                rows=("athlete_id", "count"),
                height_cov_pct=("has_height", lambda s: 100.0 * float(pd.to_numeric(s, errors="coerce").fillna(0).mean())),
                weight_cov_pct=("has_weight", lambda s: 100.0 * float(pd.to_numeric(s, errors="coerce").fillna(0).mean())),
                wingspan_cov_pct=("has_wingspan", lambda s: 100.0 * float(pd.to_numeric(s, errors="coerce").fillna(0).mean())),
            )
            .sort_values("season")
        )
    cov.to_csv(audit_dir / "physicals_coverage_by_season.csv", index=False)

    if raw_df.empty:
        mix = pd.DataFrame(columns=["provider", "season", "rows"])
    else:
        mix = raw_df.groupby(["provider", "season"], as_index=False).size().rename(columns={"size": "rows"})
        mix = mix.sort_values(["provider", "season"])
    mix.to_csv(audit_dir / "physicals_provider_mix.csv", index=False)

    linkage = pd.DataFrame([
        {
            "raw_rows": int(len(raw_df)),
            "resolved_rows": int(len(resolved)),
            "unresolved_rows": int(len(unresolved)),
            "ambiguous_rows": int(len(ambiguous)),
            "resolved_rate_pct": (100.0 * len(resolved) / len(raw_df)) if len(raw_df) else 0.0,
            "unresolved_rate_pct": (100.0 * len(unresolved) / len(raw_df)) if len(raw_df) else 0.0,
            "ambiguous_rate_pct": (100.0 * len(ambiguous) / len(raw_df)) if len(raw_df) else 0.0,
        }
    ])
    linkage.to_csv(audit_dir / "physicals_linkage_quality.csv", index=False)
    provider_failures.to_csv(audit_dir / "physicals_provider_failures.csv", index=False)


def run_pipeline(
    db_path: Path,
    audit_dir: Path,
    manual_dir: Path,
    start_season: int,
    end_season: int,
    providers: list[str],
    warehouse_v2_dir: Path,
) -> Dict[str, Any]:
    if load_dotenv is not None:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
        else:
            load_dotenv(override=False)

    con = duckdb.connect(str(db_path))

    team_universe = _team_season_universe(con, start_season, end_season)
    player_directory = _player_season_directory(con, start_season, end_season)

    logger.info("Team-season universe rows: %s", len(team_universe))
    logger.info("Player-season directory rows: %s", len(player_directory))

    records: list[PhysicalRecord] = []
    provider_failures: list[dict] = []
    api_key = os.getenv("CBD_API_KEY", "")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for p in providers:
        p = p.strip().lower()
        if not p:
            continue
        logger.info("Collecting provider=%s", p)
        if p == "cbd":
            records.extend(_records_from_cbd(team_universe, api_key, run_id, provider_failures))
        elif p == "cbbpy":
            records.extend(_records_from_cbbpy(team_universe, run_id, provider_failures))
        elif p == "sportsipy":
            records.extend(_records_from_sportsipy(team_universe, run_id, provider_failures))
        elif p == "manual":
            records.extend(_records_from_manual(manual_dir, run_id))
        elif p == "recruiting_fallback":
            records.extend(_records_from_recruiting_fallback(con, start_season, end_season, run_id))
        elif p == "nba_fallback":
            records.extend(_records_from_nba_fallback(player_directory, warehouse_v2_dir, run_id))
        else:
            logger.warning("Unknown provider skipped: %s", p)

    raw_df = _to_raw_df(records)
    if not raw_df.empty:
        raw_df["season"] = pd.to_numeric(raw_df["season"], errors="coerce").astype("Int64")
        raw_df["team_id"] = pd.to_numeric(raw_df["team_id"], errors="coerce").astype("Int64")
    _append_raw(con, raw_df)

    resolved, unresolved, ambiguous = _resolve_identity(raw_df, player_directory)
    canonical = _canonicalize(resolved)
    trajectory = _build_trajectory(canonical)
    _publish_tables(con, canonical, trajectory)
    _export_parquet(canonical, trajectory, warehouse_v2_dir)
    _write_audits(
        audit_dir,
        raw_df,
        resolved,
        unresolved,
        ambiguous,
        canonical,
        pd.DataFrame(provider_failures),
    )

    con.close()

    summary = {
        "raw_rows": int(len(raw_df)),
        "resolved_rows": int(len(resolved)),
        "canonical_rows": int(len(canonical)),
        "trajectory_rows": int(len(trajectory)),
        "unresolved_rows": int(len(unresolved)),
        "ambiguous_rows": int(len(ambiguous)),
        "providers": providers,
        "audit_dir": str(audit_dir),
        "db_path": str(db_path),
        "warehouse_v2_dir": str(warehouse_v2_dir),
        "run_id": run_id,
        "provider_failures": int(len(provider_failures)),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest season-by-season college physicals and build canonical/trajectory tables")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    ap.add_argument("--manual-dir", type=Path, default=DEFAULT_MANUAL)
    ap.add_argument("--start-season", type=int, default=2011)
    ap.add_argument("--end-season", type=int, default=datetime.now().year)
    ap.add_argument("--warehouse-v2-dir", type=Path, default=DEFAULT_WAREHOUSE_V2)
    ap.add_argument(
        "--providers",
        type=str,
        default="cbd,cbbpy,sportsipy,manual,recruiting_fallback,nba_fallback",
        help="Comma-separated provider order",
    )
    args = ap.parse_args()

    providers = [x.strip() for x in str(args.providers).split(",") if x.strip()]
    summary = run_pipeline(
        db_path=args.db,
        audit_dir=args.audit_dir,
        manual_dir=args.manual_dir,
        start_season=int(args.start_season),
        end_season=int(args.end_season),
        providers=providers,
        warehouse_v2_dir=args.warehouse_v2_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
