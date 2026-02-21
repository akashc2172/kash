#!/usr/bin/env python3
"""
Derive player minutes/games/turnovers from historical cleaned PBP using substitution text.

This is a fallback when `onFloor` contains team placeholders and cannot be used
for player-level exposure.

Input:
  data/fact_play_historical_combined.parquet

Output:
  data/warehouse_v2/fact_player_season_stats_backfill_manual_subs.parquet
  with columns:
    season, team_name, player_name, minutes_derived, turnovers_derived, games_derived

Notes:
- Historical `season` here remains start-year (e.g., 2021 for 2021-22). The
  downstream join layer maps to end-year (+1) for feature-surface parity.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from collections import defaultdict

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "fact_play_historical_combined.parquet"
OUT = ROOT / "data" / "warehouse_v2" / "fact_player_season_stats_backfill_manual_subs.parquet"

TURNOVER_KEYWORDS = (
    "TURNOVER",
    "TRAVELING",
    "BAD PASS",
    "OFFENSIVE FOUL",
    "3 SECOND",
    "SHOT CLOCK",
    "OUT OF BOUNDS",
    "PALMING",
    "DOUBLE DRIBBLE",
    "CARRYING",
    "BACKCOURT",
)


def parse_clock_to_secs(clock_text: str) -> float | None:
    if not isinstance(clock_text, str):
        return None
    c = clock_text.strip()
    if not c or c.upper() == "TIME":
        return None
    if ":" not in c:
        return None
    parts = c.split(":")
    try:
        mm = float(parts[0])
        ss = float(parts[1]) if len(parts) > 1 else 0.0
        return mm * 60.0 + ss
    except Exception:
        return None


def split_play_text(raw: str) -> tuple[str, str, str, str]:
    if not isinstance(raw, str):
        return "", "", "", ""
    parts = [p.strip() for p in raw.split("|")]
    while len(parts) < 4:
        parts.append("")
    return parts[0], parts[1], parts[2], parts[3]


def extract_player_name(event_text: str) -> str | None:
    if not isinstance(event_text, str):
        return None
    e = event_text.strip()
    if not e:
        return None
    if "," in e:
        name = e.split(",", 1)[0].strip()
        return name if name else None
    m = re.match(r"^([A-Za-z'\\-\\. ]+)", e)
    if m:
        name = m.group(1).strip()
        return name if name else None
    return None


def is_sub_in(event_text: str) -> bool:
    return isinstance(event_text, str) and "SUBSTITUTION IN" in event_text.upper()


def is_sub_out(event_text: str) -> bool:
    return isinstance(event_text, str) and "SUBSTITUTION OUT" in event_text.upper()


def is_turnover_event(event_text: str) -> bool:
    if not isinstance(event_text, str):
        return False
    up = event_text.upper()
    return any(k in up for k in TURNOVER_KEYWORDS)


def build_rows(df: pd.DataFrame) -> pd.DataFrame:
    stats = defaultdict(lambda: {"seconds": 0.0, "turnovers": 0, "games": 0})

    for game_id, g in df.groupby("gameSourceId", sort=False):
        g = g.copy()
        # Preserve original row order from parquet load.
        g["_row"] = range(len(g))
        g = g.sort_values("_row")

        season = int(pd.to_numeric(g["season"], errors="coerce").dropna().iloc[0])
        players_seen = set()

        # Track side-to-team from header row.
        home_team = "HOME"
        away_team = "AWAY"
        for raw in g["playText"].astype(str):
            c, h, _, a = split_play_text(raw)
            if c.upper() == "TIME" and h and a:
                home_team, away_team = h, a
                break

        on_home = set()
        on_away = set()

        # Parse rows once.
        parsed = []
        for raw in g["playText"].astype(str):
            clock, h_evt, _, a_evt = split_play_text(raw)
            secs = parse_clock_to_secs(clock)
            parsed.append((secs, h_evt, a_evt))

        # Durations to next timed event.
        next_secs = [None] * len(parsed)
        nxt = None
        for i in range(len(parsed) - 1, -1, -1):
            s = parsed[i][0]
            next_secs[i] = nxt
            if s is not None:
                nxt = s

        for i, (secs, h_evt, a_evt) in enumerate(parsed):
            # Period boundary hints.
            row_txt = (h_evt + " " + a_evt).upper()
            if "PERIOD START" in row_txt:
                on_home.clear()
                on_away.clear()

            # Apply home/away events to on-court sets.
            for side_evt, on_set in ((h_evt, on_home), (a_evt, on_away)):
                name = extract_player_name(side_evt)
                if name and name.upper() != "TEAM":
                    players_seen.add((home_team if on_set is on_home else away_team, name))

                if is_sub_in(side_evt) and name and name.upper() != "TEAM":
                    on_set.add(name)
                elif is_sub_out(side_evt) and name and name.upper() != "TEAM":
                    if name in on_set:
                        on_set.remove(name)
                else:
                    # Starter inference: first actions before complete lineup set.
                    if name and name.upper() != "TEAM" and len(on_set) < 5:
                        on_set.add(name)

                if is_turnover_event(side_evt) and name and name.upper() != "TEAM":
                    key = (
                        season,
                        home_team if on_set is on_home else away_team,
                        name,
                    )
                    stats[key]["turnovers"] += 1

            # Allocate interval to current on-court players.
            if secs is None:
                continue
            nxt = next_secs[i]
            if nxt is None:
                dur = max(secs, 0.0)
            elif nxt <= secs:
                dur = max(secs - nxt, 0.0)
            else:
                # clock reset implies period boundary.
                dur = max(secs, 0.0)

            if dur <= 0:
                continue

            for p in on_home:
                key = (season, home_team, p)
                stats[key]["seconds"] += dur
            for p in on_away:
                key = (season, away_team, p)
                stats[key]["seconds"] += dur

        for team, p in players_seen:
            key = (season, team, p)
            stats[key]["games"] += 1

    rows = []
    for (season, team_name, player_name), val in stats.items():
        rows.append(
            {
                "season": int(season),
                "team_name": str(team_name),
                "player_name": str(player_name),
                "minutes_derived": round(float(val["seconds"]) / 60.0, 2),
                "turnovers_derived": int(val["turnovers"]),
                "games_derived": int(val["games"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(INPUT))
    ap.add_argument("--output", default=str(OUT))
    ap.add_argument("--start-season", type=int, default=None)
    ap.add_argument("--end-season", type=int, default=None)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    con = duckdb.connect()
    where = []
    if args.start_season is not None:
        where.append(f"season >= {int(args.start_season)}")
    if args.end_season is not None:
        where.append(f"season <= {int(args.end_season)}")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    q = f"""
        SELECT gameSourceId, season, playText
        FROM read_parquet('{in_path.as_posix()}')
        {where_sql}
    """
    df = con.execute(q).df()
    con.close()

    if df.empty:
        print("No rows selected; nothing to do.")
        return

    out = build_rows(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"saved={out_path} rows={len(out)}")


if __name__ == "__main__":
    main()

