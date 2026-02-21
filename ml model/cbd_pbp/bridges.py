from __future__ import annotations

import csv
import difflib
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .schemas import Tables
from .warehouse import Warehouse


TEAM_TOKEN_RE = re.compile(r"[^a-z0-9]+")
PLAYER_EVENT_PATTERNS = [
    re.compile(r"^([^|]+?)\s+(Enters|Leaves)\s+Game", re.IGNORECASE),
    re.compile(r"^([^|]+?)\s+substitution\s+(in|out)", re.IGNORECASE),
    re.compile(r"substitution\s+(in|out)\s+by\s+([^|]+)$", re.IGNORECASE),
]


def _norm_team(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("&", " and ")
    s = TEAM_TOKEN_RE.sub(" ", s)
    s = re.sub(r"\b(university|college|state|st\.?)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_player(s: str) -> str:
    s = (s or "").upper().strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace(".", "").replace("'", "")
    return s


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=a, b=b).ratio()


@dataclass
class ScrapeGame:
    contest_id: int
    scrape_date: date
    scrape_home_team: str
    scrape_away_team: str
    csv_path: str


def _parse_scrape_games_from_file(csv_path: str) -> List[ScrapeGame]:
    """
    Parse one manual scrape CSV and extract ALL contest IDs in the file.
    Each date file contains many games; we need one bridge candidate per contest_id.
    """
    out: List[ScrapeGame] = []
    try:
        by_contest: Dict[int, Dict[str, object]] = {}
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid_raw = row.get("contest_id")
                dt_raw = row.get("date")
                raw = (row.get("raw_text") or "").strip()
                if not cid_raw or not dt_raw:
                    continue
                try:
                    cid = int(cid_raw)
                    d = date.fromisoformat(str(dt_raw))
                except Exception:
                    continue
                entry = by_contest.setdefault(
                    cid,
                    {"date": d, "home": None, "away": None},
                )
                if entry["home"] is None and "| Score |" in raw:
                    parts = [p.strip() for p in raw.split("|")]
                    if len(parts) >= 4:
                        entry["home"] = parts[1]
                        entry["away"] = parts[3]
        for cid, v in by_contest.items():
            home = v.get("home")
            away = v.get("away")
            d = v.get("date")
            if not home or not away or not isinstance(d, date):
                continue
            out.append(
                ScrapeGame(
                    contest_id=cid,
                    scrape_date=d,
                    scrape_home_team=str(home),
                    scrape_away_team=str(away),
                    csv_path=csv_path,
                )
            )
    except Exception:
        return []
    return out


def _load_manual_scrape_games(scrape_root: str, max_files: Optional[int] = None) -> List[ScrapeGame]:
    files: List[str] = []
    for root, _, names in os.walk(scrape_root):
        for n in names:
            if n.endswith(".csv") and n.startswith("ncaa_pbp_"):
                files.append(os.path.join(root, n))
    files = sorted(files)
    if max_files is not None:
        files = files[:max_files]

    out: List[ScrapeGame] = []
    for p in files:
        out.extend(_parse_scrape_games_from_file(p))
    return out


def _match_scrape_to_cbd(scrape_games: List[ScrapeGame], cbd_games: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    cbd_games = cbd_games.copy()
    cbd_games["cbd_date"] = pd.to_datetime(cbd_games["startDate"], utc=True).dt.date
    cbd_games["home_norm"] = cbd_games["homeTeam"].fillna("").map(_norm_team)
    cbd_games["away_norm"] = cbd_games["awayTeam"].fillna("").map(_norm_team)

    for sg in scrape_games:
        home_n = _norm_team(sg.scrape_home_team)
        away_n = _norm_team(sg.scrape_away_team)

        # Primary: same date; fallback: +/- 1 day to absorb timezone shifts.
        cands = cbd_games[cbd_games["cbd_date"] == sg.scrape_date]
        date_penalty = 0.0
        if cands.empty:
            plus = sg.scrape_date + timedelta(days=1)
            minus = sg.scrape_date - timedelta(days=1)
            cands = cbd_games[cbd_games["cbd_date"].isin([plus, minus])]
            date_penalty = 0.03
        best = None
        best_score = -1.0
        best_method = "unmatched"

        for _, c in cands.iterrows():
            score_direct = (_sim(home_n, c["home_norm"]) + _sim(away_n, c["away_norm"])) / 2.0
            score_swap = (_sim(home_n, c["away_norm"]) + _sim(away_n, c["home_norm"])) / 2.0
            if score_direct >= score_swap:
                s = score_direct
                method = "direct"
            else:
                s = score_swap
                method = "swapped"
            s_adj = s - date_penalty
            if s_adj > best_score:
                best_score = s_adj
                best = c
                best_method = method

        if best is None:
            continue
        if best_score < 0.70:
            continue

        rows.append(
            {
                "contest_id": sg.contest_id,
                "scrape_date": sg.scrape_date,
                "scrape_home_team": sg.scrape_home_team,
                "scrape_away_team": sg.scrape_away_team,
                "cbd_game_id": int(best["id"]),
                "cbd_date": best["cbd_date"],
                "cbd_home_team": best["homeTeam"],
                "cbd_away_team": best["awayTeam"],
                "match_method": best_method,
                "match_confidence": round(float(best_score), 4),
            }
        )

    if not rows:
        return pd.DataFrame(rows)
    out = pd.DataFrame(rows)
    # Deterministic one-to-one tie-breaks.
    out = out.sort_values(["match_confidence", "contest_id", "cbd_game_id"], ascending=[False, True, True])
    out = out.drop_duplicates(subset=["contest_id"], keep="first")
    out = out.drop_duplicates(subset=["cbd_game_id"], keep="first")
    return out


def _extract_scrape_players(csv_path: str) -> List[str]:
    players: List[str] = []
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = row.get("raw_text", "") or ""
                # First, explicit substitution/enter/leave patterns.
                matched = False
                for pat in PLAYER_EVENT_PATTERNS:
                    m = pat.search(raw)
                    if not m:
                        continue
                    # Pattern 3 captures name in group 2.
                    if pat.pattern.startswith("substitution"):
                        name = (m.group(2) or "").strip()
                    else:
                        name = (m.group(1) or "").strip()
                    if name:
                        players.append(name)
                    matched = True
                    break
                if matched:
                    continue

                # Generic fallback: parse player prefix before comma in event cells.
                parts = [p.strip() for p in raw.split("|")]
                for seg in parts:
                    if "," not in seg:
                        continue
                    prefix = seg.split(",", 1)[0].strip()
                    # Keep likely person names only.
                    if len(prefix) < 3:
                        continue
                    if any(tok in prefix.lower() for tok in ["time", "score", "period"]):
                        continue
                    players.append(prefix)
    except Exception:
        return []
    return list(sorted(set(players)))


def build_scrape_bridges(
    wh: Warehouse,
    scrape_root: str = "data/manual_scrapes",
    max_files: Optional[int] = None,
    include_player_bridge: bool = True,
):
    """
    Build persistent game/player bridge tables:
    - bridge_game_cbd_scrape
    - bridge_player_cbd_scrape
    """
    wh.init_schema({})
    wh.exec(
        """
        CREATE TABLE IF NOT EXISTS bridge_game_cbd_scrape (
            contest_id BIGINT,
            scrape_date DATE,
            scrape_home_team VARCHAR,
            scrape_away_team VARCHAR,
            cbd_game_id BIGINT,
            cbd_date DATE,
            cbd_home_team VARCHAR,
            cbd_away_team VARCHAR,
            match_method VARCHAR,
            match_confidence DOUBLE
        )
        """
    )
    wh.exec(
        """
        CREATE TABLE IF NOT EXISTS bridge_player_cbd_scrape (
            contest_id BIGINT,
            cbd_game_id BIGINT,
            scrape_player_name VARCHAR,
            scrape_player_norm VARCHAR,
            cbd_athlete_id BIGINT,
            cbd_athlete_name VARCHAR,
            cbd_athlete_norm VARCHAR,
            match_method VARCHAR,
            match_confidence DOUBLE
        )
        """
    )

    scrape_games = _load_manual_scrape_games(scrape_root=scrape_root, max_files=max_files)
    if not scrape_games:
        print(f"No scrape games found under {scrape_root}")
        return

    cbd_games = wh.query_df(
        f"""
        SELECT id, startDate, homeTeam, awayTeam
        FROM {Tables.GAMES}
        WHERE status = 'final'
        """
    )
    game_bridge = _match_scrape_to_cbd(scrape_games, cbd_games)
    if game_bridge.empty:
        print("No game bridges matched.")
        return

    wh.exec("DELETE FROM bridge_game_cbd_scrape")
    wh.ensure_table("bridge_game_cbd_scrape", game_bridge, pk=None)

    # Build player bridge using participants in matched CBD games (optional).
    if not include_player_bridge:
        print(
            f"Built bridges: games={len(game_bridge)}, "
            f"players=0 (skipped), scrape_root={scrape_root}"
        )
        return

    # Build player bridge using participants in matched CBD games.
    matched_ids = game_bridge["cbd_game_id"].astype(int).tolist()
    if not matched_ids:
        print("No matched CBD game ids for player bridge.")
        return

    id_list = ", ".join(str(x) for x in sorted(set(matched_ids)))
    participants = wh.query_df(
        f"""
        SELECT gameId, athleteId, athlete_name
        FROM stg_participants
        WHERE TRY_CAST(gameId AS BIGINT) IN ({id_list})
        """
    )
    participants["cbd_athlete_norm"] = participants["athlete_name"].fillna("").map(_norm_player)

    contest_to_csv = {g.contest_id: g.csv_path for g in scrape_games}
    player_rows: List[Dict] = []

    for _, r in game_bridge.iterrows():
        contest_id = int(r["contest_id"])
        game_id = int(r["cbd_game_id"])
        csv_path = contest_to_csv.get(contest_id)
        if not csv_path:
            continue
        scrape_players = _extract_scrape_players(csv_path)
        if not scrape_players:
            continue

        cands = participants[participants["gameId"].astype(str) == str(game_id)]
        if cands.empty:
            continue

        cand_pairs = list(cands[["athleteId", "athlete_name", "cbd_athlete_norm"]].drop_duplicates().itertuples(index=False, name=None))

        for sp in scrape_players:
            sp_norm = _norm_player(sp)
            best = None
            best_score = -1.0
            best_method = "unmatched"
            for aid, aname, anorm in cand_pairs:
                if sp_norm == anorm:
                    score = 1.0
                    method = "exact"
                else:
                    score = _sim(sp_norm, anorm)
                    method = "fuzzy"
                if score > best_score:
                    best_score = score
                    best = (aid, aname, anorm)
                    best_method = method

            if best is None:
                continue
            if best_score < 0.65:
                continue
            player_rows.append(
                {
                    "contest_id": contest_id,
                    "cbd_game_id": game_id,
                    "scrape_player_name": sp,
                    "scrape_player_norm": sp_norm,
                    "cbd_athlete_id": int(best[0]),
                    "cbd_athlete_name": best[1],
                    "cbd_athlete_norm": best[2],
                    "match_method": best_method,
                    "match_confidence": round(float(best_score), 4),
                }
            )

    wh.exec("DELETE FROM bridge_player_cbd_scrape")
    if player_rows:
        wh.ensure_table("bridge_player_cbd_scrape", pd.DataFrame(player_rows), pk=None)
    print(
        f"Built bridges: games={len(game_bridge)}, "
        f"players={len(player_rows)}, scrape_root={scrape_root}"
    )
