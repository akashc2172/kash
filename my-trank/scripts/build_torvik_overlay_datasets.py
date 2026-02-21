#!/usr/bin/env python3
"""Build static Torvik overlay datasets from international CSV sources.

Outputs JSON files to public/torvik-datasets/{year}.json and updates manifest.json.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
INTL_DIR = ROOT / "public" / "data" / "international_stat_history"
OUT_DIR = ROOT / "public" / "torvik-datasets"

ARCHIVE_FILE = INTL_DIR / "internationalplayerarchive.csv"
FILE_2026 = INTL_DIR / "2026_records.csv"


def _to_int(value: str) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _to_float(value: str):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def load_rows(path: Path, source: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            season = _to_int(raw.get("torvik_year", ""))
            player = (raw.get("key") or "").strip()
            team = (raw.get("team") or "").strip()
            if not season or not player:
                continue

            row = {
                "season": season,
                "player": player,
                "team": team,
                "conf": (raw.get("conf") or "").strip(),
                "g": _to_int(raw.get("g", "")),
                "mpg": _to_float(raw.get("mpg", "")),
                "ppg": _to_float(raw.get("ppg", "")),
                "usg": _to_float(raw.get("usg", "")),
                "ts": _to_float(raw.get("ts", "")),
                "efg": _to_float(raw.get("efg", "")),
                "porpag": _to_float(raw.get("porpag", "")),
                "dporpag": _to_float(raw.get("dporpag", "")),
                "bpm": _to_float(raw.get("bpm", "")),
                "obpm": _to_float(raw.get("obpm", "")),
                "dbpm": _to_float(raw.get("dbpm", "")),
                "per": _to_float(raw.get("per", "")),
                "threepa_100": _to_float(raw.get("3pa/100", "")),
                "rimfga_100": _to_float(raw.get("rimfga/100", "")),
                "midfga_100": _to_float(raw.get("midfga/100", "")),
                "stops_100": _to_float(raw.get("stops/100", "")),
                "total_rapm": _to_float(raw.get("total RAPM", "")),
                "source": source,
            }
            rows.append(row)
    return rows


def dedupe_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    # Prefer explicit 2026 file for season 2026 when collisions exist.
    priority = {"intl_2026": 2, "intl_archive": 1}
    best: Dict[Tuple[int, str, str], Dict[str, object]] = {}

    for row in rows:
        season = int(row["season"])
        key = (season, _norm(str(row.get("player", ""))), _norm(str(row.get("team", ""))))
        existing = best.get(key)
        if not existing:
            best[key] = row
            continue
        if priority.get(str(row.get("source", "")), 0) >= priority.get(str(existing.get("source", "")), 0):
            best[key] = row

    output = list(best.values())
    output.sort(key=lambda r: (int(r["season"]), str(r["player"]), str(r.get("team", ""))))
    return output


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    archive_rows = load_rows(ARCHIVE_FILE, "intl_archive")
    rows_2026 = load_rows(FILE_2026, "intl_2026")

    merged = dedupe_rows([*archive_rows, *rows_2026])

    by_season: Dict[int, List[Dict[str, object]]] = {}
    for row in merged:
        season = int(row["season"])
        by_season.setdefault(season, []).append(row)

    datasets = []
    for season in sorted(by_season):
        season_rows = by_season[season]
        out_file = OUT_DIR / f"{season}.json"
        with out_file.open("w", encoding="utf-8") as fh:
            json.dump(season_rows, fh, ensure_ascii=True, indent=2)
            fh.write("\n")

        datasets.append(
            {
                "season": season,
                "name": f"CTS Intl Overlay {season}",
                "url": f"/torvik-datasets/{season}.json",
                "match": {
                    "primary": ["player", "team"],
                    "fallback": ["player"],
                },
                "inject_columns": [
                    "conf",
                    "usg",
                    "ts",
                    "porpag",
                    "bpm",
                    "per",
                ],
                "rows": len(season_rows),
            }
        )

    manifest = {
        "version": "2026.02.21",
        "sources": [
            str(ARCHIVE_FILE.relative_to(ROOT)),
            str(FILE_2026.relative_to(ROOT)),
        ],
        "datasets": datasets,
    }

    with (OUT_DIR / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=True, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    build()
