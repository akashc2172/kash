#!/usr/bin/env python3
"""
Reconstruct historical onFloor lineups from manual scrape text with strict QA.

Outputs:
  - data/fact_play_historical_combined_v2.parquet
  - data/audit/historical_lineup_quality_by_game.csv
  - data/audit/historical_lineup_quality_by_season.csv
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANUAL_ROOT = ROOT / "data" / "manual_scrapes"
OUT_COMBINED = ROOT / "data" / "fact_play_historical_combined_v2.parquet"
OUT_GAME_AUDIT = ROOT / "data" / "audit" / "historical_lineup_quality_by_game.csv"
OUT_SEASON_AUDIT = ROOT / "data" / "audit" / "historical_lineup_quality_by_season.csv"
WAREHOUSE_DB = ROOT / "data" / "warehouse.duckdb"

TEAM_EVENT_PREFIXES = (
    "TEAM,",
    "TEAM ",
    "BENCH ",
    "DEADBALL",
)

SUB_IN_MARKERS = ("ENTERS GAME", "SUBSTITUTION IN", "SUB IN")
SUB_OUT_MARKERS = ("LEAVES GAME", "SUBSTITUTION OUT", "SUB OUT")
EXPLICIT_LINEUP_MARKERS = ("TEAM FOR",)

ACTION_TOKENS = (
    "MADE",
    "MISSED",
    "REBOUND",
    "FOUL",
    "TURNOVER",
    "STEAL",
    "BLOCK",
    "ASSIST",
    "JUMPER",
    "LAYUP",
    "DUNK",
    "TIP-IN",
    "FREE THROW",
)


def normalize_name(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    s = raw.upper().strip()
    s = re.sub(r"^#?\d+\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" ,", ",").replace(", ", ",")
    s = re.sub(r"[^\w,\-'. ]+", "", s)
    s = s.strip(" ,.-")
    if s.startswith("TEAM"):
        return ""
    if s in {"", "UNKNOWN", "N/A"}:
        return ""
    # Prefer LAST,FIRST shape
    if "," not in s:
        parts = s.split(" ")
        if len(parts) >= 2:
            s = f"{parts[-1]},{' '.join(parts[:-1])}"
    s = s.replace(" ", "")
    return s


def is_team_event(event_text: str) -> bool:
    if not isinstance(event_text, str):
        return True
    t = event_text.strip().upper()
    if t == "":
        return True
    return t.startswith(TEAM_EVENT_PREFIXES)


def parse_clock_to_secs(clock_text: str) -> Optional[float]:
    if not isinstance(clock_text, str):
        return None
    c = clock_text.strip().upper()
    if c in {"", "TIME"}:
        return None
    if ":" in c:
        try:
            mm, ss = c.split(":")[:2]
            return float(mm) * 60.0 + float(ss)
        except Exception:
            return None
    m = re.match(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$", c)
    if m:
        mins = float(m.group(1)) if m.group(1) else 0.0
        secs = float(m.group(2)) if m.group(2) else 0.0
        return mins * 60.0 + secs
    return None


def parse_row(raw_text: str) -> Tuple[str, str, str, str]:
    if not isinstance(raw_text, str):
        return "", "", "", ""
    parts = [p.strip() for p in raw_text.split("|")]
    while len(parts) < 4:
        parts.append("")
    return parts[0], parts[1], parts[2], parts[3]


def parse_score(score_text: str) -> Tuple[int, int]:
    if not isinstance(score_text, str):
        return 0, 0
    m = re.search(r"(\d+)\s*-\s*(\d+)", score_text)
    if not m:
        return 0, 0
    return int(m.group(1)), int(m.group(2))


def extract_player_name(event_text: str) -> str:
    if not isinstance(event_text, str):
        return ""
    txt = event_text.upper().strip()
    if txt == "" or is_team_event(txt):
        return ""

    # Sub events: name appears before marker
    for marker in SUB_IN_MARKERS + SUB_OUT_MARKERS:
        idx = txt.find(marker)
        if idx > 0:
            return normalize_name(txt[:idx])

    # Explicit lineup lines handled elsewhere
    if any(marker in txt for marker in EXPLICIT_LINEUP_MARKERS):
        return ""

    # Typical event starts with player name
    m = re.match(r"^([A-Z][A-Z' .\-]*,[A-Z][A-Z' .\-]*)", txt)
    if m:
        return normalize_name(m.group(1))

    # Backup: only if action token present
    if any(tok in txt for tok in ACTION_TOKENS):
        head = txt.split(" ", 1)[0]
        return normalize_name(head)
    return ""


def parse_explicit_lineup(event_text: str) -> Set[str]:
    if not isinstance(event_text, str):
        return set()
    txt = event_text.upper()
    if "TEAM FOR" not in txt or ":" not in txt:
        return set()
    body = txt.split(":", 1)[1]
    chunks = [c.strip() for c in body.split("#") if c.strip()]
    names: Set[str] = set()
    for c in chunks:
        n = normalize_name(c)
        if n:
            names.add(n)
    return names


def is_sub_in(event_text: str) -> bool:
    if not isinstance(event_text, str):
        return False
    t = event_text.upper()
    return any(marker in t for marker in SUB_IN_MARKERS)


def is_sub_out(event_text: str) -> bool:
    if not isinstance(event_text, str):
        return False
    t = event_text.upper()
    return any(marker in t for marker in SUB_OUT_MARKERS)


def season_from_folder(folder_name: str) -> int:
    m = re.match(r"^(\d{4})-(\d{4})$", folder_name)
    if m:
        return int(m.group(1))
    if folder_name.isdigit() and len(folder_name) == 4:
        return int(folder_name)
    raise ValueError(f"Unexpected season folder: {folder_name}")


@dataclass
class Event:
    idx: int
    team: str
    event_type: str
    player: str


class GameReconstructor:
    def __init__(self, contest_id: str, rows: Sequence[str], season: int, date_str: str) -> None:
        self.contest_id = str(contest_id)
        self.rows = list(rows)
        self.season = int(season)
        self.date_str = date_str
        self.home_team = "HOME"
        self.away_team = "AWAY"
        self.events: List[Event] = []
        self.checkpoints: List[Tuple[int, str, Set[str]]] = []
        self.roster_home: Counter[str] = Counter()
        self.roster_away: Counter[str] = Counter()
        self.game_players: Set[str] = set()
        self.subs_detected = 0
        self.explicit_hits = 0

    def parse(self) -> None:
        for i, raw in enumerate(self.rows):
            clock, h_evt, score_text, a_evt = parse_row(raw)
            if clock.upper() == "TIME" and "SCORE" in score_text.upper():
                if h_evt:
                    self.home_team = h_evt
                if a_evt:
                    self.away_team = a_evt
                continue
            self._process_event(i, "HOME", h_evt)
            self._process_event(i, "AWAY", a_evt)

    def _process_event(self, idx: int, team_code: str, event_text: str) -> None:
        if not isinstance(event_text, str) or event_text.strip() == "":
            return
        roster = self.roster_home if team_code == "HOME" else self.roster_away

        explicit = parse_explicit_lineup(event_text)
        if explicit:
            self.checkpoints.append((idx, team_code, explicit))
            self.explicit_hits += 1
            for p in explicit:
                roster[p] += 2
                self.game_players.add(p)
            return

        player = extract_player_name(event_text)
        if player:
            roster[player] += 1
            self.game_players.add(player)

        if player and is_sub_in(event_text):
            self.events.append(Event(idx, team_code, "IN", player))
            self.subs_detected += 1
        elif player and is_sub_out(event_text):
            self.events.append(Event(idx, team_code, "OUT", player))
            self.subs_detected += 1
        elif player:
            self.events.append(Event(idx, team_code, "ACT", player))

    def _infer_starters_from_events(self, team_code: str) -> Set[str]:
        starters: Set[str] = set()
        seen_sub_in: Set[str] = set()
        for e in sorted([x for x in self.events if x.team == team_code], key=lambda x: x.idx):
            if e.event_type == "IN":
                seen_sub_in.add(e.player)
            elif e.event_type in {"OUT", "ACT"} and e.player not in seen_sub_in:
                starters.add(e.player)
            if len(starters) >= 5:
                break
        roster = self.roster_home if team_code == "HOME" else self.roster_away
        for p, _ in roster.most_common():
            if len(starters) >= 5:
                break
            if p:
                starters.add(p)
        return starters

    def _solve_starting_lineup(self, team_code: str) -> Set[str]:
        cps = [c for c in self.checkpoints if c[1] == team_code]
        if cps:
            cp_idx, _, cp_players = sorted(cps, key=lambda x: x[0])[0]
            starters = set(cp_players)
            for e in sorted(
                [x for x in self.events if x.team == team_code and x.idx < cp_idx],
                key=lambda x: x.idx,
                reverse=True,
            ):
                if e.event_type == "IN":
                    starters.discard(e.player)
                elif e.event_type == "OUT":
                    starters.add(e.player)
            return starters
        return self._infer_starters_from_events(team_code)

    @staticmethod
    def _ghost_fill(curr: Set[str], roster: Counter[str]) -> Set[str]:
        out = set([p for p in curr if p])
        if len(out) > 5:
            ranked = sorted(list(out), key=lambda x: roster.get(x, 0), reverse=True)
            out = set(ranked[:5])
        if len(out) < 5:
            for p, _ in roster.most_common():
                if p and p not in out:
                    out.add(p)
                if len(out) == 5:
                    break
        return out

    def reconstruct(self) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        self.parse()
        curr_home = self._solve_starting_lineup("HOME")
        curr_away = self._solve_starting_lineup("AWAY")

        rows_out: List[Dict[str, object]] = []
        len10 = 0
        placeholders = 0
        unique_lineups: Set[str] = set()

        events_by_idx: Dict[int, List[Event]] = defaultdict(list)
        for e in self.events:
            events_by_idx[e.idx].append(e)

        for i, raw in enumerate(self.rows):
            clock, h_evt, score_text, a_evt = parse_row(raw)
            hs, ass = parse_score(score_text)

            # Normalize lineup before emitting this event.
            curr_home = self._ghost_fill(curr_home, self.roster_home)
            curr_away = self._ghost_fill(curr_away, self.roster_away)

            combined = []
            for p in sorted(curr_home):
                combined.append({"id": None, "name": p, "team": self.home_team})
            for p in sorted(curr_away):
                combined.append({"id": None, "name": p, "team": self.away_team})

            names = [x["name"] for x in combined]
            is_len10 = len(combined) == 10
            has_placeholder = any((n.startswith("TEAM") or n == "") for n in names)
            if is_len10:
                len10 += 1
            if has_placeholder:
                placeholders += 1

            if is_len10 and not has_placeholder:
                lineup_conf = 1.0
                quality_flag = "valid_10"
            elif is_len10:
                lineup_conf = 0.55
                quality_flag = "placeholder_team"
            elif len(combined) >= 6:
                lineup_conf = 0.35
                quality_flag = "partial"
            else:
                lineup_conf = 0.1
                quality_flag = "insufficient_events"

            unique_lineups.add(json.dumps(combined, sort_keys=True))
            rows_out.append(
                {
                    "gameSourceId": self.contest_id,
                    "season": self.season,
                    "date": self.date_str,
                    "clock": clock,
                    "playText": raw,
                    "homeScore": hs,
                    "awayScore": ass,
                    "onFloor": json.dumps(combined),
                    "lineup_source": "manual_reconstructed",
                    "lineup_confidence": lineup_conf,
                    "lineup_quality_flag": quality_flag,
                }
            )

            # Apply row substitutions for next row state.
            for e in events_by_idx.get(i, []):
                target = curr_home if e.team == "HOME" else curr_away
                if e.event_type == "IN":
                    target.add(e.player)
                elif e.event_type == "OUT":
                    target.discard(e.player)

        n_rows = len(rows_out)
        pct_len10 = (len10 / n_rows) if n_rows else 0.0
        pct_placeholder = (placeholders / n_rows) if n_rows else 1.0
        confidence_mean = float(pd.Series([r["lineup_confidence"] for r in rows_out]).mean()) if n_rows else 0.0
        unique_players = len(self.game_players)
        game_pass = pct_len10 >= 0.80 and pct_placeholder <= 0.05 and unique_players >= 10
        summary = {
            "season": self.season,
            "gameSourceId": self.contest_id,
            "rows": n_rows,
            "pct_len10": pct_len10,
            "pct_placeholder": pct_placeholder,
            "subs_detected": self.subs_detected,
            "explicit_lineup_hits": self.explicit_hits,
            "confidence_mean": confidence_mean,
            "lineup_switches": len(unique_lineups),
            "unique_players_game": unique_players,
            "gate_pass": bool(game_pass),
        }
        return rows_out, summary


def _date_from_filename(name: str) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    return m.group(1) if m else "1900-01-01"


def _load_manual_rows(folder: Path) -> List[Tuple[str, pd.DataFrame, str]]:
    out: List[Tuple[str, pd.DataFrame, str]] = []
    for f in sorted(folder.glob("*.csv")):
        df = pd.read_csv(f)
        if "contest_id" not in df.columns or "raw_text" not in df.columns:
            continue
        for contest_id, gdf in df.groupby("contest_id", sort=False):
            out.append((str(contest_id), gdf.copy(), _date_from_filename(f.name)))
    return out


def _append_api_rows(rows: List[Dict[str, object]], min_season: int) -> List[Dict[str, object]]:
    if not WAREHOUSE_DB.exists():
        return rows
    con = duckdb.connect(str(WAREHOUSE_DB))
    tables = set(con.execute("SHOW TABLES").fetchdf()["name"].tolist())
    if "fact_play_raw" not in tables:
        con.close()
        return rows
    api_df = con.execute(
        """
        SELECT
          CAST(gameId AS VARCHAR) AS gameSourceId,
          CAST(season AS INTEGER) AS season,
          CAST(COALESCE(gameStartDate, '') AS VARCHAR) AS date,
          CAST(clock AS VARCHAR) AS clock,
          CAST(playText AS VARCHAR) AS playText,
          CAST(COALESCE(homeScore, 0) AS INTEGER) AS homeScore,
          CAST(COALESCE(awayScore, 0) AS INTEGER) AS awayScore,
          CAST(onFloor AS VARCHAR) AS onFloor
        FROM fact_play_raw
        WHERE season >= ?
        """,
        [min_season],
    ).df()
    con.close()
    if api_df.empty:
        return rows

    for _, r in api_df.iterrows():
        onf = r["onFloor"]
        confidence = 0.2
        flag = "api_missing_onfloor"
        if isinstance(onf, str) and onf.strip():
            try:
                arr = json.loads(onf)
                if isinstance(arr, list):
                    if len(arr) == 10:
                        confidence = 0.95
                        flag = "valid_10"
                    elif len(arr) >= 6:
                        confidence = 0.4
                        flag = "partial"
                    else:
                        flag = "insufficient_events"
            except Exception:
                flag = "parse_fail"
        rows.append(
            {
                "gameSourceId": str(r["gameSourceId"]),
                "season": int(r["season"]),
                "date": str(r["date"])[:10] if r["date"] else "1900-01-01",
                "clock": str(r["clock"]) if r["clock"] is not None else "",
                "playText": str(r["playText"]) if r["playText"] is not None else "",
                "homeScore": int(r["homeScore"]) if pd.notna(r["homeScore"]) else 0,
                "awayScore": int(r["awayScore"]) if pd.notna(r["awayScore"]) else 0,
                "onFloor": onf if isinstance(onf, str) else "[]",
                "lineup_source": "api",
                "lineup_confidence": confidence,
                "lineup_quality_flag": flag,
            }
        )
    return rows


def _build_season_audit(game_audit: pd.DataFrame) -> pd.DataFrame:
    if game_audit.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "games",
                "pct_games_pass",
                "pct_rows_len10",
                "pct_rows_placeholder",
                "avg_unique_players_game",
                "gate_pass",
            ]
        )
    s = (
        game_audit.groupby("season")
        .agg(
            games=("gameSourceId", "nunique"),
            pct_games_pass=("gate_pass", "mean"),
            pct_rows_len10=("pct_len10", "mean"),
            pct_rows_placeholder=("pct_placeholder", "mean"),
            avg_unique_players_game=("unique_players_game", "mean"),
        )
        .reset_index()
    )
    s["gate_pass"] = (
        (s["pct_rows_len10"] >= 0.80)
        & (s["pct_rows_placeholder"] <= 0.05)
        & (s["pct_games_pass"] >= 0.80)
        & (s["avg_unique_players_game"] >= 10)
    )
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconstruct historical onFloor with strict QA.")
    ap.add_argument("--manual-root", default=str(MANUAL_ROOT))
    ap.add_argument("--output-combined", default=str(OUT_COMBINED))
    ap.add_argument("--output-game-audit", default=str(OUT_GAME_AUDIT))
    ap.add_argument("--output-season-audit", default=str(OUT_SEASON_AUDIT))
    ap.add_argument("--start-season", type=int, default=2011)
    ap.add_argument("--end-season", type=int, default=2024)
    ap.add_argument("--include-api-from-season", type=int, default=2025)
    ap.add_argument("--no-api-append", action="store_true")
    ap.add_argument("--max-games-per-season", type=int, default=0)
    args = ap.parse_args()

    manual_root = Path(args.manual_root)
    if not manual_root.exists():
        raise FileNotFoundError(f"Manual scrape root missing: {manual_root}")

    all_rows: List[Dict[str, object]] = []
    game_summaries: List[Dict[str, object]] = []

    folders = sorted([p for p in manual_root.iterdir() if p.is_dir()])
    for folder in folders:
        try:
            season = season_from_folder(folder.name)
        except Exception:
            continue
        if season < args.start_season or season > args.end_season:
            continue
        contest_groups = _load_manual_rows(folder)
        if args.max_games_per_season and args.max_games_per_season > 0:
            contest_groups = contest_groups[: args.max_games_per_season]
        print(f"[reconstruct] season={season} games={len(contest_groups)}")
        for contest_id, gdf, date_str in contest_groups:
            rows = gdf["raw_text"].astype(str).tolist()
            recon = GameReconstructor(contest_id, rows, season, date_str)
            out_rows, summary = recon.reconstruct()
            all_rows.extend(out_rows)
            game_summaries.append(summary)

    if not args.no_api_append:
        all_rows = _append_api_rows(all_rows, args.include_api_from_season)

    out_df = pd.DataFrame(all_rows)
    game_df = pd.DataFrame(game_summaries)
    season_df = _build_season_audit(game_df)

    Path(args.output_combined).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_game_audit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_season_audit).parent.mkdir(parents=True, exist_ok=True)

    out_df.to_parquet(args.output_combined, index=False)
    game_df.to_csv(args.output_game_audit, index=False)
    season_df.to_csv(args.output_season_audit, index=False)

    print(f"combined_rows={len(out_df)} output={args.output_combined}")
    print(f"game_audit_rows={len(game_df)} output={args.output_game_audit}")
    print(f"season_audit_rows={len(season_df)} output={args.output_season_audit}")


if __name__ == "__main__":
    main()
