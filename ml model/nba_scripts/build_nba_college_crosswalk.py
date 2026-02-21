"""
Build NBA->NCAA crosswalk with deterministic draft-aware matching.

Outputs:
  - data/warehouse_v2/dim_player_nba_college_crosswalk.parquet
  - data/warehouse_v2/dim_player_nba_college_crosswalk_debug.parquet
  - data/audit/crosswalk_nba_to_college_coverage.csv
  - data/audit/crosswalk_ambiguity_catalog.csv
  - data/audit/crosswalk_unmatched_nba.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WAREHOUSE_DIR = Path("data/warehouse_v2")
AUDIT_DIR = Path("data/audit")
DB_PATH = Path("data/warehouse.duckdb")
BASKETBALL_EXCEL_ALL_PLAYERS = Path("data/basketball_excel/all_players.parquet")

OUT_FILE = WAREHOUSE_DIR / "dim_player_nba_college_crosswalk.parquet"
DEBUG_FILE = WAREHOUSE_DIR / "dim_player_nba_college_crosswalk_debug.parquet"

COVERAGE_FILE = AUDIT_DIR / "crosswalk_nba_to_college_coverage.csv"
AMBIG_FILE = AUDIT_DIR / "crosswalk_ambiguity_catalog.csv"
UNMATCHED_FILE = AUDIT_DIR / "crosswalk_unmatched_nba.csv"

SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b", flags=re.IGNORECASE)
PUNCT_RE = re.compile(r"[^a-z0-9\s]")
WS_RE = re.compile(r"\s+")

TIER_RANK = {
    "id_exact": 0,
    "draft_constrained_high": 1,
    "draft_constrained_medium": 2,
    "manual_review": 3,
}


def _ascii_fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def normalize_name(name: Any) -> str:
    if pd.isna(name):
        return ""
    n = _ascii_fold(str(name).lower())
    n = n.replace("-", " ").replace("'", " ")
    n = PUNCT_RE.sub(" ", n)
    n = SUFFIX_RE.sub(" ", n)
    n = WS_RE.sub(" ", n).strip()
    return n


def name_variants(name: str) -> set[str]:
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
    t1 = set(normalize_name(nba_name).split())
    t2 = set(normalize_name(college_name).split())
    if t1 and t2:
        jacc = len(t1 & t2) / len(t1 | t2)
        best = max(best, 0.7 * best + 0.3 * jacc)
    return float(best)


def _mode_numeric(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    m = s.mode()
    if m.empty:
        return None
    return float(m.iloc[0])


def dn_bucket(dn: Optional[float]) -> str:
    if dn is None or pd.isna(dn):
        return "undrafted_or_missing"
    d = int(dn)
    if 1 <= d <= 14:
        return "lottery"
    if 15 <= d <= 30:
        return "first_round"
    if 31 <= d <= 60:
        return "second_round"
    return "undrafted_or_missing"


def load_basketball_excel_profiles() -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    logger.info("Loading basketball-excel player universe...")
    if not BASKETBALL_EXCEL_ALL_PLAYERS.exists():
        raise FileNotFoundError(f"Missing basketball-excel source: {BASKETBALL_EXCEL_ALL_PLAYERS}")

    be = pd.read_parquet(BASKETBALL_EXCEL_ALL_PLAYERS)
    required = ["pid", "bbr_pid", "nm", "d_y", "d_n"]
    missing = [c for c in required if c not in be.columns]
    if missing:
        raise ValueError(f"basketball-excel missing required columns: {missing}")

    # Regular season rows are more stable for identity metadata; fallback to all rows if st unavailable.
    if "st" in be.columns:
        reg = be[pd.to_numeric(be["st"], errors="coerce") == 0].copy()
        if not reg.empty:
            be = reg

    be = be[["pid", "bbr_pid", "nm", "d_y", "d_n"]].copy()
    be["norm_name"] = be["nm"].apply(normalize_name)
    be["d_y"] = pd.to_numeric(be["d_y"], errors="coerce")
    be["d_n"] = pd.to_numeric(be["d_n"], errors="coerce")

    agg = (
        be.groupby(["pid", "bbr_pid", "norm_name"], dropna=False, as_index=False)
        .agg(
            nm=("nm", "first"),
            dy_mode=("d_y", _mode_numeric),
            dy_min=("d_y", lambda s: float(pd.to_numeric(s, errors="coerce").dropna().min()) if pd.to_numeric(s, errors="coerce").notna().any() else np.nan),
            dy_max=("d_y", lambda s: float(pd.to_numeric(s, errors="coerce").dropna().max()) if pd.to_numeric(s, errors="coerce").notna().any() else np.nan),
            dn_mode=("d_n", _mode_numeric),
            row_count=("nm", "count"),
        )
        .reset_index(drop=True)
    )
    agg["has_dy"] = agg["dy_mode"].notna().astype(int)
    agg["has_dn"] = agg["dn_mode"].notna().astype(int)

    by_bbr: Dict[str, Dict[str, Any]] = {}
    by_pid: Dict[str, Dict[str, Any]] = {}
    by_name: Dict[str, List[Dict[str, Any]]] = {}

    for r in agg.to_dict("records"):
        bbr = str(r.get("bbr_pid") or "").strip()
        pid = str(r.get("pid") or "").strip()
        nn = str(r.get("norm_name") or "").strip()
        if bbr:
            by_bbr[bbr] = r
        if pid:
            by_pid[pid] = r
        if nn:
            by_name.setdefault(nn, []).append(r)

    logger.info("Loaded basketball-excel profiles: %s", len(agg))
    return agg, by_bbr, by_pid, by_name


def get_college_players() -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
    logger.info("Extracting college player directory from DuckDB...")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        """
        SELECT
            shooterAthleteId AS athlete_id,
            mode(shooter_name) AS athlete_name,
            MAX(g.season) AS final_season,
            COUNT(*) AS shot_events
        FROM stg_shots s
        JOIN dim_games g ON g.id = CAST(s.gameId AS BIGINT)
        WHERE shooterAthleteId IS NOT NULL
        GROUP BY 1
        """
    ).df()
    con.close()

    df["norm_name"] = df["athlete_name"].apply(normalize_name)
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for r in df.to_dict("records"):
        nn = str(r.get("norm_name") or "")
        if nn:
            by_name.setdefault(nn, []).append(r)
    logger.info("Loaded %s college athletes.", len(df))
    return df, by_name


def get_nba_players() -> pd.DataFrame:
    logger.info("Loading NBA players...")
    crosswalk = pd.read_parquet(WAREHOUSE_DIR / "dim_player_crosswalk.parquet")
    dim_nba = pd.read_parquet(WAREHOUSE_DIR / "dim_player_nba.parquet")

    keep_cross = [c for c in ["nba_id", "player_name", "pid", "bbr_id"] if c in crosswalk.columns]
    keep_dim = [c for c in ["nba_id", "draft_year", "rookie_season_year"] if c in dim_nba.columns]
    df = pd.merge(crosswalk[keep_cross], dim_nba[keep_dim], on="nba_id", how="inner")
    df["norm_name"] = df["player_name"].apply(normalize_name)
    df["draft_year_proxy"] = pd.to_numeric(df.get("draft_year"), errors="coerce")
    rookie = pd.to_numeric(df.get("rookie_season_year"), errors="coerce")
    df["draft_year_proxy"] = df["draft_year_proxy"].where(df["draft_year_proxy"].notna(), rookie - 1)

    logger.info("Loaded %s NBA players.", len(df))
    return df


def _resolve_candidate_be_profile(
    cand_norm_name: str,
    nba_draft: float,
    be_by_name: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    arr = be_by_name.get(cand_norm_name, [])
    if not arr:
        return {}

    def keyfn(x: Dict[str, Any]) -> Tuple[float, int]:
        dy = x.get("dy_mode")
        dy_gap = abs(float(dy) - float(nba_draft)) if dy is not None and not pd.isna(dy) and not pd.isna(nba_draft) else 999.0
        return (dy_gap, -int(x.get("row_count") or 0))

    return sorted(arr, key=keyfn)[0]


def _draft_signal(dy_gap: Optional[float], dy_match: bool, bucket: str, degraded_year: bool) -> float:
    score = 0.0
    if dy_match:
        score += 0.08
    elif dy_gap is not None and not pd.isna(dy_gap):
        if dy_gap <= 1:
            score += 0.03
        elif dy_gap <= 2:
            score -= 0.01
        else:
            score -= 0.05

    bucket_bonus = {
        "lottery": 0.02,
        "first_round": 0.015,
        "second_round": 0.01,
        "undrafted_or_missing": 0.0,
    }
    score += bucket_bonus.get(bucket, 0.0)

    if degraded_year:
        score -= 0.02
    return float(score)


def generate_candidates(
    nba_df: pd.DataFrame,
    college_records: List[Dict[str, Any]],
    college_block: Dict[str, List[Dict[str, Any]]],
    college_by_name: Dict[str, List[Dict[str, Any]]],
    be_by_bbr: Dict[str, Dict[str, Any]],
    be_by_pid: Dict[str, Dict[str, Any]],
    be_by_name: Dict[str, List[Dict[str, Any]]],
    strict_year_window: int,
    fallback_year_window: int,
    min_name_score: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    candidates: List[Dict[str, Any]] = []
    unmatched_rows: List[Dict[str, Any]] = []

    for _, nba in nba_df.iterrows():
        nba_id = int(nba["nba_id"])
        nba_name = str(nba.get("player_name") or "")
        n_norm = str(nba.get("norm_name") or "")
        n_draft = pd.to_numeric(pd.Series([nba.get("draft_year_proxy")]), errors="coerce").iloc[0]
        n_pid = str(nba.get("pid") or "").strip()
        n_bbr = str(nba.get("bbr_id") or "").strip()

        seed = None
        seed_method = None
        if n_bbr and n_bbr in be_by_bbr:
            seed = be_by_bbr[n_bbr]
            seed_method = "bbr_id_link"
        elif n_pid and n_pid in be_by_pid:
            seed = be_by_pid[n_pid]
            seed_method = "pid_link"

        if not n_norm:
            unmatched_rows.append({
                "nba_id": nba_id,
                "nba_name": nba_name,
                "draft_year_proxy": n_draft,
                "reason": "missing_normalized_name",
            })
            continue

        key = n_norm.split(" ")[0]
        pool = list(college_block.get(key, []))
        if not pool:
            first = key[:1]
            if first:
                pool = [r for r in college_records if str(r.get("norm_name") or "").startswith(first)]

        # ID-seeded exact-name candidate expansion.
        if seed is not None:
            seed_norm = str(seed.get("norm_name") or "")
            if seed_norm and seed_norm in college_by_name:
                for c in college_by_name[seed_norm]:
                    if c not in pool:
                        pool.append(c)

        if not pool:
            unmatched_rows.append({
                "nba_id": nba_id,
                "nba_name": nba_name,
                "draft_year_proxy": n_draft,
                "reason": "no_college_candidates",
            })
            continue

        row_candidates = 0
        for c in pool:
            c_name = str(c.get("athlete_name") or "")
            c_norm = str(c.get("norm_name") or "")
            c_final = float(c.get("final_season")) if pd.notna(c.get("final_season")) else np.nan

            ns = name_score(nba_name, c_name)
            if seed is not None and c_norm and c_norm == str(seed.get("norm_name") or ""):
                ns = max(ns, 0.97)

            if ns < min_name_score:
                continue

            year_gap = abs(float(c_final) - float(n_draft)) if pd.notna(c_final) and pd.notna(n_draft) else np.nan
            if pd.notna(year_gap) and year_gap > fallback_year_window + 1:
                continue

            be_prof = _resolve_candidate_be_profile(c_norm, n_draft, be_by_name)
            dy_college = be_prof.get("dy_mode")
            dn_college = be_prof.get("dn_mode")
            dy_gap = abs(float(dy_college) - float(n_draft)) if dy_college is not None and not pd.isna(dy_college) and pd.notna(n_draft) else np.nan

            degraded_year = False
            if pd.notna(n_draft) and dy_college is not None and not pd.isna(dy_college):
                if dy_gap > fallback_year_window:
                    continue
                if dy_gap > strict_year_window:
                    degraded_year = True
            elif pd.notna(year_gap) and year_gap > fallback_year_window:
                continue

            dy_match = bool(pd.notna(dy_gap) and float(dy_gap) == 0.0)
            bucket = dn_bucket(dn_college)
            draft_signal_score = _draft_signal(dy_gap if pd.notna(dy_gap) else None, dy_match, bucket, degraded_year)
            shot_events = float(c.get("shot_events") or 0.0)
            # Mild volume prior to break same-name ties toward real rotation players.
            volume_bonus = min(np.log1p(max(shot_events, 0.0)) / 8.0, 1.0) * 0.02

            score = float(ns)
            if pd.notna(year_gap):
                score -= 0.02 * float(year_gap)
            score += draft_signal_score + volume_bonus
            score = float(np.clip(score, 0.0, 1.0))

            method = "name_draft_fuzzy"
            if seed_method and c_norm and c_norm == str(seed.get("norm_name") or ""):
                method = seed_method

            if method in {"bbr_id_link", "pid_link"} and (dy_match or (pd.notna(dy_gap) and dy_gap <= strict_year_window)):
                tier = "id_exact"
            elif ns >= 0.94 and not degraded_year:
                tier = "draft_constrained_high"
            elif ns >= 0.88:
                tier = "draft_constrained_medium"
            else:
                tier = "manual_review"

            candidates.append(
                {
                    "nba_id": nba_id,
                    "athlete_id": int(c["athlete_id"]),
                    "nba_name": nba_name,
                    "college_name": c_name,
                    "match_score": score,
                    "match_tier": tier,
                    "match_method": method,
                    "name_score_raw": float(ns),
                    "year_gap": float(year_gap) if pd.notna(year_gap) else np.nan,
                    "draft_year_proxy": float(n_draft) if pd.notna(n_draft) else np.nan,
                    "college_final": float(c_final) if pd.notna(c_final) else np.nan,
                    "dy_college": float(dy_college) if dy_college is not None and not pd.isna(dy_college) else np.nan,
                    "dn_college": float(dn_college) if dn_college is not None and not pd.isna(dn_college) else np.nan,
                    "dy_match": dy_match,
                    "dy_gap": float(dy_gap) if pd.notna(dy_gap) else np.nan,
                    "dn_bucket": bucket,
                    "draft_signal_score": float(draft_signal_score),
                    "shot_events": shot_events,
                    "degraded_year_window": int(degraded_year),
                }
            )
            row_candidates += 1

        if row_candidates == 0:
            unmatched_rows.append(
                {
                    "nba_id": nba_id,
                    "nba_name": nba_name,
                    "draft_year_proxy": n_draft,
                    "reason": "all_candidates_failed_filters",
                }
            )

    return pd.DataFrame(candidates), pd.DataFrame(unmatched_rows)


def build_ambiguity_catalog(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["nba_id", "nba_name", "n_candidates", "top_match_score", "second_match_score", "delta", "top_college", "second_college"])

    rows = []
    for nba_id, g in candidates.groupby("nba_id", sort=False):
        g2 = g.sort_values(["match_score", "name_score_raw"], ascending=[False, False]).reset_index(drop=True)
        if len(g2) < 2:
            continue
        top = g2.iloc[0]
        second = g2.iloc[1]
        delta = float(top["match_score"] - second["match_score"])
        if delta <= 0.02 or int((g2["match_score"] >= top["match_score"] - 0.01).sum()) > 1:
            rows.append(
                {
                    "nba_id": int(nba_id),
                    "nba_name": top["nba_name"],
                    "n_candidates": int(len(g2)),
                    "top_match_score": float(top["match_score"]),
                    "second_match_score": float(second["match_score"]),
                    "delta": delta,
                    "top_college": top["college_name"],
                    "second_college": second["college_name"],
                }
            )
    return pd.DataFrame(rows)


def resolve_one_to_one(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    c = candidates.copy()
    c["tier_rank"] = c["match_tier"].map(TIER_RANK).fillna(99)

    c = c.sort_values(
        ["tier_rank", "match_score", "name_score_raw", "dy_match", "year_gap", "shot_events", "nba_id", "athlete_id"],
        ascending=[True, False, False, False, True, False, True, True],
    )
    c = c.drop_duplicates(subset=["nba_id"], keep="first")

    c = c.sort_values(
        ["tier_rank", "match_score", "name_score_raw", "dy_match", "year_gap", "shot_events", "athlete_id", "nba_id"],
        ascending=[True, False, False, False, True, False, True, True],
    )
    c = c.drop_duplicates(subset=["athlete_id"], keep="first")

    return c.drop(columns=["tier_rank"], errors="ignore").reset_index(drop=True)


def build_coverage_report(nba_df: pd.DataFrame, selected: pd.DataFrame, unmatched: pd.DataFrame) -> pd.DataFrame:
    total_nba = int(len(nba_df))
    matched_nba = int(selected["nba_id"].nunique()) if not selected.empty else 0

    draft = pd.to_numeric(nba_df.get("draft_year_proxy"), errors="coerce")
    cohort_mask = draft.between(2011, 2024, inclusive="both")
    total_cohort = int(cohort_mask.sum())
    matched_ids = set(selected["nba_id"].tolist()) if not selected.empty else set()
    matched_cohort = int(nba_df.loc[cohort_mask, "nba_id"].isin(matched_ids).sum())

    tier_counts = selected["match_tier"].value_counts().to_dict() if not selected.empty else {}

    return pd.DataFrame(
        [
            {
                "total_nba": total_nba,
                "matched_nba": matched_nba,
                "match_rate_nba_all": (matched_nba / total_nba) if total_nba else 0.0,
                "total_nba_2011_2024": total_cohort,
                "matched_nba_2011_2024": matched_cohort,
                "match_rate_nba_2011_2024": (matched_cohort / total_cohort) if total_cohort else 0.0,
                "id_exact_count": int(tier_counts.get("id_exact", 0)),
                "draft_constrained_high_count": int(tier_counts.get("draft_constrained_high", 0)),
                "draft_constrained_medium_count": int(tier_counts.get("draft_constrained_medium", 0)),
                "manual_review_count": int(tier_counts.get("manual_review", 0)),
                "unmatched_count": int(len(unmatched)),
                "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            }
        ]
    )


def validate_and_gate(
    selected: pd.DataFrame,
    previous: pd.DataFrame,
    regression_tolerance: float,
) -> None:
    required = {
        "nba_id",
        "athlete_id",
        "match_score",
        "match_tier",
        "match_method",
        "name_score_raw",
        "year_gap",
        "draft_year_proxy",
        "dy_college",
        "dn_college",
        "dy_match",
        "dn_bucket",
        "draft_signal_score",
    }
    missing = sorted(required - set(selected.columns))
    if missing:
        raise RuntimeError(f"Crosswalk missing required columns: {missing}")

    dup_nba = int(selected.duplicated(subset=["nba_id"]).sum())
    dup_ath = int(selected.duplicated(subset=["athlete_id"]).sum())
    if dup_nba > 0:
        raise RuntimeError(f"duplicate nba_id rows in final crosswalk: {dup_nba}")
    if dup_ath > 0:
        raise RuntimeError(f"duplicate athlete_id rows in final crosswalk: {dup_ath}")

    if not previous.empty and "nba_id" in previous.columns:
        prev_high = set(previous.loc[pd.to_numeric(previous.get("match_score"), errors="coerce") >= 0.95, "nba_id"].astype(int).tolist())
        if not prev_high and "match_tier" in previous.columns:
            prev_high = set(
                previous.loc[previous["match_tier"].isin(["id_exact", "draft_constrained_high"]), "nba_id"].astype(int).tolist()
            )

        new_high = set(
            selected.loc[selected["match_tier"].isin(["id_exact", "draft_constrained_high"]), "nba_id"].astype(int).tolist()
        )

        if prev_high:
            lost = prev_high - new_high
            loss_rate = len(lost) / len(prev_high)
            if loss_rate > regression_tolerance:
                raise RuntimeError(
                    f"High-confidence regression too large: lost {len(lost)}/{len(prev_high)} ({loss_rate:.1%}) > {regression_tolerance:.1%}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NBA->NCAA crosswalk with draft-aware deterministic matching")
    parser.add_argument("--strict-year-window", type=int, default=1)
    parser.add_argument("--fallback-year-window", type=int, default=2)
    parser.add_argument("--min-name-score", type=float, default=0.84)
    parser.add_argument("--regression-tolerance", type=float, default=0.20)
    args = parser.parse_args()

    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    previous = pd.read_parquet(OUT_FILE) if OUT_FILE.exists() else pd.DataFrame()

    _, be_by_bbr, be_by_pid, be_by_name = load_basketball_excel_profiles()
    college_df, college_by_name = get_college_players()
    nba_df = get_nba_players()

    college_records = college_df.to_dict("records")
    block: Dict[str, List[Dict[str, Any]]] = {}
    for r in college_records:
        norm = str(r.get("norm_name") or "")
        if not norm:
            continue
        block.setdefault(norm.split(" ")[0], []).append(r)

    candidates_df, unmatched_df = generate_candidates(
        nba_df=nba_df,
        college_records=college_records,
        college_block=block,
        college_by_name=college_by_name,
        be_by_bbr=be_by_bbr,
        be_by_pid=be_by_pid,
        be_by_name=be_by_name,
        strict_year_window=args.strict_year_window,
        fallback_year_window=args.fallback_year_window,
        min_name_score=args.min_name_score,
    )

    ambiguity_df = build_ambiguity_catalog(candidates_df)
    selected_df = resolve_one_to_one(candidates_df)

    validate_and_gate(selected_df, previous, regression_tolerance=args.regression_tolerance)

    coverage_df = build_coverage_report(nba_df, selected_df, unmatched_df)

    # Persist audit/debug first.
    candidates_df.to_parquet(DEBUG_FILE, index=False)
    coverage_df.to_csv(COVERAGE_FILE, index=False)
    ambiguity_df.to_csv(AMBIG_FILE, index=False)
    unmatched_df.to_csv(UNMATCHED_FILE, index=False)

    final_cols = [
        "nba_id",
        "athlete_id",
        "match_score",
        "match_tier",
        "match_method",
        "name_score_raw",
        "year_gap",
        "draft_year_proxy",
        "dy_college",
        "dn_college",
        "dy_match",
        "dn_bucket",
        "draft_signal_score",
    ]
    selected_df[final_cols].to_parquet(OUT_FILE, index=False)

    logger.info("Saved crosswalk to %s", OUT_FILE)
    logger.info("Saved debug candidates to %s", DEBUG_FILE)
    logger.info("Saved audits: %s, %s, %s", COVERAGE_FILE, AMBIG_FILE, UNMATCHED_FILE)
    if not coverage_df.empty:
        r = coverage_df.iloc[0]
        logger.info(
            "Matched %s/%s NBA players (%.1f%%), cohort 2011-2024: %s/%s (%.1f%%)",
            int(r["matched_nba"]),
            int(r["total_nba"]),
            100 * float(r["match_rate_nba_all"]),
            int(r["matched_nba_2011_2024"]),
            int(r["total_nba_2011_2024"]),
            100 * float(r["match_rate_nba_2011_2024"]),
        )


if __name__ == "__main__":
    main()
