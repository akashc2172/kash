"""
Build NBA-College Crosswalk (Phase 4 Prerequisite)
==================================================
Purpose: Link `nba_id` (from Basketball-Excel/EPM) to `athlete_id` (from College Feature Store).
Source of Truth:
    - NBA: `dim_player_crosswalk` (ID, Name, Draft Year)
    - College: `warehouse.duckdb` (stg_shots.shooter_name -> athlete_id)

Method:
    1. Extract Dictionary `(athlete_id -> name, latest_season)` from College DB.
    2. Extract Dictionary `(nba_id -> name, draft_year)` from NBA Warehouse.
    3. Fuzzy Match with Draft Year Constraint:
       - Match Name Score > 0.85
       - `abs(Draft_Year - College_Final_Season) <= 2` (Allow for redshirting/gap years)

Output:
    - data/warehouse_v2/dim_player_nba_college_crosswalk.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import duckdb
import re
import unicodedata
from difflib import SequenceMatcher
import logging

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
WAREHOUSE_DIR = Path("data/warehouse_v2")
DB_PATH = 'data/warehouse.duckdb'
OUT_FILE = WAREHOUSE_DIR / "dim_player_nba_college_crosswalk.parquet"

SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b", flags=re.IGNORECASE)
PUNCT_RE = re.compile(r"[^a-z0-9\s]")
WS_RE = re.compile(r"\s+")


def _ascii_fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def normalize_name(name):
    """
    Normalize names for matching.
    Handles apostrophes/hyphens/diacritics and removes suffix noise.
    """
    if pd.isna(name):
        return ""
    n = _ascii_fold(str(name).lower())
    n = n.replace("-", " ").replace("'", " ")
    n = PUNCT_RE.sub(" ", n)
    n = SUFFIX_RE.sub(" ", n)
    n = WS_RE.sub(" ", n).strip()
    return n


def name_variants(name: str) -> set[str]:
    """Generate robust variants for non-exact matching."""
    base = normalize_name(name)
    if not base:
        return set()
    toks = [t for t in base.split(" ") if t]
    variants = {base, "".join(toks), " ".join(sorted(toks))}
    if len(toks) >= 2:
        first, last = toks[0], toks[-1]
        variants.add(f"{first} {last}")
        variants.add(f"{first[:1]} {last}")
        variants.add(f"{last} {first}")
    return {v.strip() for v in variants if v.strip()}


def name_score(nba_name: str, college_name: str) -> float:
    """Robust similarity across multiple normalized name variants."""
    v1 = name_variants(nba_name)
    v2 = name_variants(college_name)
    if not v1 or not v2:
        return 0.0
    best = 0.0
    for a in v1:
        for b in v2:
            s = SequenceMatcher(None, a, b).ratio()
            if s > best:
                best = s
    # Token overlap boost for hyphen/apostrophe or middle-name differences.
    t1 = set(normalize_name(nba_name).split())
    t2 = set(normalize_name(college_name).split())
    if t1 and t2:
        jacc = len(t1 & t2) / len(t1 | t2)
        best = max(best, 0.7 * best + 0.3 * jacc)
    return float(best)

def get_college_players():
    """Extract distinct athlete_id -> name from stg_shots."""
    logger.info("Extracting college player directory from DuckDB...")
    con = duckdb.connect(DB_PATH, read_only=True)
    
    # We take the most common name for an ID (handle variations)
    # Also get max season to help matching
    df = con.execute("""
        SELECT 
            shooterAthleteId as athlete_id,
            mode(shooter_name) as athlete_name,
            MAX(g.season) as final_season
        FROM stg_shots s
        JOIN dim_games g ON g.id = CAST(s.gameId AS BIGINT)
        WHERE shooterAthleteId IS NOT NULL
        GROUP BY 1
    """).df()
    
    df['norm_name'] = df['athlete_name'].apply(normalize_name)
    logger.info(f"Loaded {len(df):,} college athletes.")
    con.close()
    return df

def get_nba_players():
    """Extract NBA players with draft context."""
    logger.info("Loading NBA players...")
    crosswalk = pd.read_parquet(WAREHOUSE_DIR / "dim_player_crosswalk.parquet")
    dim_nba = pd.read_parquet(WAREHOUSE_DIR / "dim_player_nba.parquet")
    
    # Merge to get Draft Year
    df = pd.merge(crosswalk[['nba_id', 'player_name']], 
                  dim_nba[['nba_id', 'draft_year', 'rookie_season_year']], 
                  on='nba_id', how='inner')
    
    df['norm_name'] = df['player_name'].apply(normalize_name)
    
    # Use rookie_season - 1 as proxy for draft year if missing
    df['draft_year_proxy'] = df['draft_year'].fillna(df['rookie_season_year'] - 1)
    
    logger.info(f"Loaded {len(df):,} NBA players.")
    return df

def match_players(college_df, nba_df):
    logger.info("Running deterministic fuzzy+year matching...")

    # Block by first normalized token for speed.
    college_records = college_df.to_dict("records")
    block = {}
    for r in college_records:
        norm = r.get("norm_name", "")
        if not norm:
            continue
        key = norm.split(" ")[0]
        block.setdefault(key, []).append(r)

    candidates = []
    for _, nba in nba_df.iterrows():
        nba_name = nba.get("player_name", "")
        n_norm = nba.get("norm_name", "")
        n_draft = nba.get("draft_year_proxy", np.nan)
        n_id = nba.get("nba_id")
        if not n_norm or pd.isna(n_draft):
            continue

        key = n_norm.split(" ")[0]
        pool = block.get(key, [])
        if not pool:
            # fallback: same first letter
            first = key[:1]
            pool = [r for r in college_records if r.get("norm_name", "").startswith(first)]

        for c in pool:
            year_gap = abs(float(c["final_season"]) - float(n_draft))
            if year_gap > 2:
                continue
            s = name_score(nba_name, c["athlete_name"])
            if s < 0.84:
                continue
            # Mild year-gap prior: prefer closer season matches.
            s_adj = s - 0.02 * year_gap
            candidates.append({
                "nba_id": n_id,
                "athlete_id": int(c["athlete_id"]),
                "nba_name": nba_name,
                "college_name": c["athlete_name"],
                "match_score": float(s_adj),
                "name_score_raw": float(s),
                "year_gap": float(year_gap),
                "draft_year": float(n_draft),
                "college_final": float(c["final_season"]),
            })

    cand = pd.DataFrame(candidates)
    if cand.empty:
        return cand

    # Deterministic one-to-one resolution:
    # 1) keep best per nba_id
    # 2) if athlete maps to >1 nba, keep highest score
    cand = cand.sort_values(
        ["match_score", "name_score_raw", "year_gap", "nba_id", "athlete_id"],
        ascending=[False, False, True, True, True],
    )
    cand = cand.drop_duplicates(subset=["nba_id"], keep="first")
    cand = cand.sort_values(
        ["match_score", "name_score_raw", "year_gap", "athlete_id", "nba_id"],
        ascending=[False, False, True, True, True],
    )
    cand = cand.drop_duplicates(subset=["athlete_id"], keep="first")
    return cand.reset_index(drop=True)

def main():
    col_df = get_college_players()
    nba_df = get_nba_players()
    
    matches_df = match_players(col_df, nba_df)
    
    # Validation
    match_rate = len(matches_df) / len(nba_df) if len(nba_df) > 0 else 0
    logger.info(f"Matched {len(matches_df)} players ({match_rate:.1%} of NBA cohort).")
    
    # CURSOR NOTE: Validate match quality
    high_confidence = matches_df[matches_df['match_score'] >= 0.95]
    logger.info(f"  High confidence matches (score >= 0.95): {len(high_confidence)} ({len(high_confidence)/len(matches_df):.1%} of matches)")
    
    # Check for duplicate matches (one NBA player matched to multiple college players)
    dup_nba = matches_df[matches_df.duplicated(subset=['nba_id'], keep=False)]
    if len(dup_nba) > 0:
        logger.warning(f"  ⚠️  {len(dup_nba)} NBA players matched to multiple college athletes - review needed")
        logger.warning(f"      Example duplicates: {dup_nba[['nba_id', 'nba_name', 'college_name', 'match_score']].head(5).to_dict('records')}")
    
    # Check for duplicate college matches (one college player matched to multiple NBA players)
    dup_college = matches_df[matches_df.duplicated(subset=['athlete_id'], keep=False)]
    if len(dup_college) > 0:
        logger.warning(f"  ⚠️  {len(dup_college)} college athletes matched to multiple NBA players - review needed")
    
    # Save full match details for debugging, then save minimal crosswalk
    debug_file = WAREHOUSE_DIR / "dim_player_nba_college_crosswalk_debug.parquet"
    matches_df.to_parquet(debug_file, index=False)
    logger.info(f"Saved full match details to {debug_file}")
    
    # Save minimal crosswalk (just IDs and score)
    matches_df[['nba_id', 'athlete_id', 'match_score']].to_parquet(OUT_FILE, index=False)
    logger.info(f"Saved crosswalk to {OUT_FILE}")

if __name__ == "__main__":
    main()
