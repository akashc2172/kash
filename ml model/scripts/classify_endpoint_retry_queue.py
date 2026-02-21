#!/usr/bin/env python3
"""
Classify endpoint retry queue rows as actionable vs source-limited.

Reads:
  - data/audit/reingest_manifest_subs.csv
  - data/audit/reingest_manifest_lineups.csv

Writes:
  - data/audit/endpoint_retry_queue_classification.csv
  - data/audit/endpoint_retry_queue_summary.json
  - optionally updates data/audit/retry_policy_cache.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

API_BASE = "https://api.collegebasketballdata.com"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def endpoint_url(endpoint: str, game_id: str) -> str:
    if endpoint == "subs":
        return f"{API_BASE}/substitutions/game/{game_id}"
    if endpoint == "lineups":
        return f"{API_BASE}/lineups/game/{game_id}"
    raise ValueError(f"Unsupported endpoint: {endpoint}")


def classify_status(status: int, payload: Any) -> str:
    if status == 200:
        if isinstance(payload, list) and len(payload) == 0:
            return "provider_empty"
        return "actionable_has_data"
    if status in (400, 404):
        return "provider_empty"
    if status == 429:
        return "retryable_rate_limited"
    if 500 <= status < 600:
        return "retryable_server_error"
    return "retryable_unknown_status"


def probe_endpoint(
    game_id: str,
    endpoint: str,
    api_key: str,
    timeout_s: int,
    max_retries: int,
) -> tuple[int | None, str, int]:
    url = endpoint_url(endpoint, game_id)
    headers = {"Authorization": f"Bearer {api_key}"}
    tries = 0
    last_status = None
    for attempt in range(max_retries):
        tries += 1
        try:
            resp = requests.get(url, headers=headers, timeout=timeout_s)
            last_status = int(resp.status_code)
            payload = []
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except Exception:
                    payload = []
            cls = classify_status(resp.status_code, payload)
            if cls.startswith("retryable") and resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(3, 1 + attempt))
                continue
            return last_status, cls, tries
        except requests.RequestException:
            time.sleep(min(3, 1 + attempt))
            continue
    return last_status, "retryable_request_error", tries


def load_manifest(path: Path, endpoint: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["game_id", "season", "season_type", "endpoint"])
    df = pd.read_csv(path, dtype={"game_id": str})
    if df.empty:
        return pd.DataFrame(columns=["game_id", "season", "season_type", "endpoint"])
    keep = [c for c in ["game_id", "season", "season_type"] if c in df.columns]
    out = df[keep].copy()
    out["endpoint"] = endpoint
    return out.drop_duplicates(subset=["game_id", "endpoint"]).reset_index(drop=True)


def update_retry_cache(cache_path: Path, classified: pd.DataFrame) -> None:
    cache = {"updated_at": now_iso(), "entries": {}}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            cache.setdefault("entries", {})
        except Exception:
            cache = {"updated_at": now_iso(), "entries": {}}

    entries = cache["entries"]
    for _, row in classified.iterrows():
        key = f"{row['game_id']}|{row['endpoint']}"
        cls = str(row["classification"])
        prev = entries.get(key, {})
        attempts = int(prev.get("attempts", 0)) + int(row.get("tries", 1) or 1)
        if cls in {"provider_empty", "source_limited_after_retry"}:
            entries[key] = {
                "game_id": str(row["game_id"]),
                "endpoint": str(row["endpoint"]),
                "state": "terminal",
                "last_result": cls,
                "attempts": attempts,
                "last_checked_at": str(row["checked_at"]),
                "cooldown_until": None,
            }
        elif cls == "actionable_has_data":
            entries[key] = {
                "game_id": str(row["game_id"]),
                "endpoint": str(row["endpoint"]),
                "state": "retryable",
                "last_result": "actionable_has_data",
                "attempts": attempts,
                "last_checked_at": str(row["checked_at"]),
                "cooldown_until": None,
            }
        else:
            # keep retryable with a short cooldown; closure can pick these up later
            entries[key] = {
                "game_id": str(row["game_id"]),
                "endpoint": str(row["endpoint"]),
                "state": "retryable",
                "last_result": cls,
                "attempts": attempts,
                "last_checked_at": str(row["checked_at"]),
                "cooldown_until": None,
            }

    cache["updated_at"] = now_iso()
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify subs/lineups retry queue rows.")
    parser.add_argument("--audit-dir", default="data/audit")
    parser.add_argument("--max-per-endpoint", type=int, default=1000)
    parser.add_argument("--timeout-sec", type=int, default=20)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--db", default="data/warehouse.duckdb")
    parser.add_argument("--update-retry-cache", action="store_true")
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()
    api_key = os.getenv("CBD_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("CBD_API_KEY is required for queue classification.")

    repo_root = Path(__file__).resolve().parents[1]
    audit_dir = (repo_root / args.audit_dir).resolve() if not Path(args.audit_dir).is_absolute() else Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)

    subs = load_manifest(audit_dir / "reingest_manifest_subs.csv", "subs").head(args.max_per_endpoint)
    lineups = load_manifest(audit_dir / "reingest_manifest_lineups.csv", "lineups").head(args.max_per_endpoint)
    queue = pd.concat([subs, lineups], ignore_index=True)
    if queue.empty:
        print("No queue rows to classify.")
        return

    rows = []

    def _probe_row(r: dict) -> dict:
        gid = str(r["game_id"])
        endpoint = str(r["endpoint"])
        status, cls, tries = probe_endpoint(
            game_id=gid,
            endpoint=endpoint,
            api_key=api_key,
            timeout_s=args.timeout_sec,
            max_retries=args.retries,
        )
        return {
            "game_id": gid,
            "season": int(r["season"]) if pd.notna(r.get("season")) else None,
            "season_type": r.get("season_type"),
            "endpoint": endpoint,
            "http_status": status,
            "classification": cls,
            "tries": tries,
            "checked_at": now_iso(),
        }

    row_dicts = queue.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = [ex.submit(_probe_row, r) for r in row_dicts]
        for f in as_completed(futs):
            rows.append(f.result())

    out_df = pd.DataFrame(rows)
    # Fallback classifier: if probing failed at request layer for all rows,
    # classify via observed DB coverage after retries.
    if not out_df.empty and (out_df["classification"] == "retryable_request_error").all():
        repo_root = Path(__file__).resolve().parents[1]
        db_path = (repo_root / args.db).resolve() if not Path(args.db).is_absolute() else Path(args.db)
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            have_subs = set(
                con.execute("SELECT DISTINCT CAST(gameId AS VARCHAR) AS game_id FROM stg_subs")
                .fetchdf()["game_id"]
                .astype(str)
                .tolist()
            )
            have_lineups = set(
                con.execute("SELECT DISTINCT CAST(gameId AS VARCHAR) AS game_id FROM stg_lineups")
                .fetchdf()["game_id"]
                .astype(str)
                .tolist()
            )
        finally:
            con.close()

        def db_classify(r: pd.Series) -> str:
            gid = str(r["game_id"])
            ep = str(r["endpoint"])
            if ep == "subs":
                return "actionable_resolved_in_db" if gid in have_subs else "source_limited_after_retry"
            if ep == "lineups":
                return "actionable_resolved_in_db" if gid in have_lineups else "source_limited_after_retry"
            return "retryable_request_error"

        out_df["classification"] = out_df.apply(db_classify, axis=1)

    out_csv = audit_dir / "endpoint_retry_queue_classification.csv"
    out_df.to_csv(out_csv, index=False)

    summary = (
        out_df.groupby(["endpoint", "classification"], as_index=False)
        .agg(rows=("game_id", "count"))
        .sort_values(["endpoint", "rows"], ascending=[True, False])
    )
    summary_json = {
        "generated_at": now_iso(),
        "rows": int(len(out_df)),
        "by_endpoint": {
            ep: grp[["classification", "rows"]].to_dict(orient="records")
            for ep, grp in summary.groupby("endpoint")
        },
    }
    (audit_dir / "endpoint_retry_queue_summary.json").write_text(
        json.dumps(summary_json, indent=2), encoding="utf-8"
    )

    if args.update_retry_cache:
        update_retry_cache(audit_dir / "retry_policy_cache.json", out_df)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {audit_dir / 'endpoint_retry_queue_summary.json'}")
    if args.update_retry_cache:
        print(f"Updated: {audit_dir / 'retry_policy_cache.json'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
