from __future__ import annotations
import os
import time
import requests
import pandas as pd
from typing import Any, Optional
from tqdm import tqdm
from dotenv import load_dotenv

import cbbd
from cbbd.rest import ApiException

from .warehouse import Warehouse

load_dotenv()

# API base URL for direct HTTP calls (bypasses Pydantic validation)
API_BASE = "https://api.collegebasketballdata.com"

# --- API Call Counter ---
_CALL_COUNT = 0
_CALL_COUNT_FILE = os.path.join(os.path.dirname(__file__), "..", "api_call_count.txt")
_CALL_LIMIT = 120000  # Safety buffer

def _increment_call_count(n=1):
    """Increment the global call counter and persist periodically."""
    global _CALL_COUNT
    _CALL_COUNT += n
    # Persist every 50 calls for efficiency
    if _CALL_COUNT % 50 == 0:
        try:
            with open(_CALL_COUNT_FILE, "w") as f:
                f.write(str(_CALL_COUNT))
        except:
            pass
    # Safety check
    if _CALL_COUNT >= _CALL_LIMIT:
        print(f"[QUOTA] Reached {_CALL_COUNT} calls. Approaching limit!")

def _load_call_count():
    """Load persisted call count on module import."""
    global _CALL_COUNT
    try:
        if os.path.exists(_CALL_COUNT_FILE):
            with open(_CALL_COUNT_FILE) as f:
                _CALL_COUNT = int(f.read().strip())
    except:
        _CALL_COUNT = 0

_load_call_count()
# --- End Call Counter ---

API_BASE = "https://api.collegebasketballdata.com"

def _fetch_lineups_raw(game_id: int, api_key: str) -> pd.DataFrame:
    """
    Fetch lineup stats directly via HTTP, bypassing SDK's Pydantic validation.
    This allows null values in offenseRating/netRating etc.
    """
    url = f"{API_BASE}/lineups/game"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"gameId": game_id}
    
    for attempt in range(5):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:
                wait = min(60, 2 ** attempt)
                print(f"[429] Rate limit on lineups. Waiting {wait}s...")
                time.sleep(wait)
                continue
            elif resp.status_code == 400:
                return pd.DataFrame()
            resp.raise_for_status()
            data = resp.json()
            if not data:
                return pd.DataFrame()
            return pd.DataFrame(data)
        except Exception as e:
            wait = min(30, 2 ** attempt)
            print(f"[Lineup HTTP Error] {e}. Retrying in {wait}s...")
            time.sleep(wait)
    print(f"[Lineup] Max retries exceeded for game {game_id}")
    return pd.DataFrame()

def _models_to_df(models: list) -> pd.DataFrame:
    """Convert a list of cbbd model objects to a DataFrame."""
    if not models:
        return pd.DataFrame()
    return pd.DataFrame([m.to_dict() for m in models])

def _safe_api_call(func, *args, max_retries=10, **kwargs):
    """Wrapper with exponential backoff for rate limits."""
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            _increment_call_count()  # Count successful call
            return result
        except ApiException as e:
            if e.status == 429:
                wait = min(60, 2 ** attempt)
                print(f"[429] Rate limit. Waiting {wait}s...")
                time.sleep(wait)
                continue
            elif e.status == 400:
                print(f"[400] Bad Request: {e.reason}")
                return None
            raise
        except Exception as e:
            # Check for Pydantic/Validation errors which should skip, not retry
            if "validation error" in str(e).lower() or type(e).__name__ == "ValidationError":
                print(f"[Skip] Validation Error: {e}")
                return None
            
            wait = min(30, 2 ** attempt)
            print(f"[Error] {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Max retries exceeded for {func}")

def get_client() -> cbbd.ApiClient:
    """Create a configured cbbd API client."""
    config = cbbd.Configuration(
        host="https://api.collegebasketballdata.com",
        access_token=os.getenv("CBD_API_KEY")
    )
    return cbbd.ApiClient(config)

def ingest_static(wh: Warehouse):
    """Ingest static dimension tables (teams, conferences, venues, etc.)."""
    with get_client() as client:
        # Teams
        api = cbbd.TeamsApi(client)
        df = _models_to_df(_safe_api_call(api.get_teams))
        if not df.empty:
            wh.ensure_table("dim_teams", df, pk=["id"])

        # Conferences
        api = cbbd.ConferencesApi(client)
        df = _models_to_df(_safe_api_call(api.get_conferences))
        if not df.empty:
            wh.ensure_table("dim_conferences", df, pk=["id"])
        
        # Conference History
        df = _models_to_df(_safe_api_call(api.get_conference_history))
        if not df.empty:
            wh.ensure_table("dim_conference_history", df, pk=None)

        # Venues
        api = cbbd.VenuesApi(client)
        df = _models_to_df(_safe_api_call(api.get_venues))
        if not df.empty:
            wh.ensure_table("dim_venues", df, pk=["id"])

        # Draft
        api = cbbd.DraftApi(client)
        df = _models_to_df(_safe_api_call(api.get_draft_positions))
        if not df.empty:
            wh.ensure_table("dim_draft_positions", df, pk=["name"])
        df = _models_to_df(_safe_api_call(api.get_draft_teams))
        if not df.empty:
            wh.ensure_table("dim_draft_teams", df, pk=["id"])

        # Lines Providers
        api = cbbd.LinesApi(client)
        df = _models_to_df(_safe_api_call(api.get_providers))
        if not df.empty:
            wh.ensure_table("dim_lines_providers", df, pk=["id"])

def ingest_season(wh: Warehouse, season: int, season_type: str = "regular"):
    """Ingest a full season's data."""
    with get_client() as client:
        # 1) Games
        games_api = cbbd.GamesApi(client)
        games = _models_to_df(_safe_api_call(games_api.get_games, season=season, season_type=season_type))
        if games.empty:
            print(f"No games found for {season} {season_type}")
            return
        wh.ensure_table("dim_games", games, pk=["id"])
        print(f"Ingested {len(games)} games for {season} {season_type}")

        # 2) Ratings / Rankings / Stats (bulk endpoints)
        ratings_api = cbbd.RatingsApi(client)
        rankings_api = cbbd.RankingsApi(client)
        stats_api = cbbd.StatsApi(client)
        recruiting_api = cbbd.RecruitingApi(client)
        draft_api = cbbd.DraftApi(client)
        lines_api = cbbd.LinesApi(client)

        bulk_tables = [
            ("fact_ratings_adjusted", lambda: ratings_api.get_adjusted_efficiency(season=season)),
            ("fact_ratings_srs", lambda: ratings_api.get_srs(season=season)),
            ("fact_rankings", lambda: rankings_api.get_rankings(season=season)),
            ("fact_recruiting_players", lambda: recruiting_api.get_recruits(year=season)),
            ("fact_draft_picks", lambda: draft_api.get_draft_picks(year=season)),
            ("fact_lines", lambda: lines_api.get_lines(season=season)),
            ("fact_team_season_stats", lambda: stats_api.get_team_season_stats(season=season)),
            ("fact_player_season_stats", lambda: stats_api.get_player_season_stats(season=season)),
        ]
        for table, fetch_func in bulk_tables:
            try:
                result = _safe_api_call(fetch_func)
                df = _models_to_df(result) if result else pd.DataFrame()
                if not df.empty:
                    wh.ensure_table(table, df, pk=None)
                    print(f"  {table}: {len(df)} rows")
            except Exception as e:
                print(f"  Warning: {table} failed: {e}")

        # 3) Per-game: Lineups + Subs + Plays
        try:
            existing = wh.query_df("SELECT DISTINCT gameId FROM fact_play_raw")
            done_ids = set(existing["gameId"].astype(str).tolist()) if not existing.empty else set()
        except Exception:
            done_ids = set()

        game_ids = games["id"].astype(str).tolist()
        todo = [g for g in game_ids if g not in done_ids]
        print(f"Resuming per-game ingest: {len(done_ids)} done, {len(todo)} to go.")

        lineups_api = cbbd.LineupsApi(client)
        plays_api = cbbd.PlaysApi(client)

        for gid in tqdm(todo, desc="Per-game ingest"):
            gid_int = int(gid)
            
            # Lineups
            try:
                lineups = _safe_api_call(lineups_api.get_lineup_stats_by_game, game_id=gid_int)
                ldf = _models_to_df(lineups) if lineups else pd.DataFrame()
                if not ldf.empty:
                    ldf["gameId"] = gid
                    wh.ensure_table("fact_lineup_stint_raw", ldf, pk=None)
            except Exception as e:
                print(f"[{gid}] Lineups error: {e}")

            # Substitutions
            try:
                subs = _safe_api_call(plays_api.get_substitutions_by_game, game_id=gid_int)
                sdf = _models_to_df(subs) if subs else pd.DataFrame()
                if not sdf.empty:
                    sdf["gameId"] = gid
                    wh.ensure_table("fact_substitution_raw", sdf, pk=None)
            except Exception as e:
                print(f"[{gid}] Subs error: {e}")

            # Plays
            try:
                plays = _safe_api_call(plays_api.get_plays, game_id=gid_int)
                pdf = _models_to_df(plays) if plays else pd.DataFrame()
                if not pdf.empty:
                    pdf["gameId"] = gid
                    wh.ensure_table("fact_play_raw", pdf, pk=None)
            except Exception as e:
                print(f"[{gid}] Plays error: {e}")

def ingest_play_types(wh: Warehouse):
    """Ingest play type definitions."""
    with get_client() as client:
        api = cbbd.PlaysApi(client)
        df = _models_to_df(_safe_api_call(api.get_play_types))
        if not df.empty:
            wh.ensure_table("dim_play_types", df, pk=None)

def ingest_games_only(wh: Warehouse, season: int, season_type: str = "regular"):
    """Resume per-game ingest only (skips bulk tables)."""
    with get_client() as client:
        # Get game IDs from existing dim_games table
        try:
            games = wh.query_df("SELECT id FROM dim_games")
            game_ids = games["id"].astype(str).tolist()
        except Exception as e:
            print(f"Error: dim_games not found. Run full ingest first. {e}")
            return

        # Get already-done games
        try:
            existing = wh.query_df("SELECT DISTINCT gameId FROM fact_play_raw")
            done_ids = set(existing["gameId"].astype(str).tolist()) if not existing.empty else set()
        except Exception:
            done_ids = set()

        todo = [g for g in game_ids if g not in done_ids]
        print(f"Resuming per-game ingest: {len(done_ids)} done, {len(todo)} to go.")

        lineups_api = cbbd.LineupsApi(client)
        plays_api = cbbd.PlaysApi(client)

        for gid in tqdm(todo, desc="Per-game ingest"):
            gid_int = int(gid)
            
            # Lineups (use raw HTTP to preserve null ratings)
            try:
                # api_key = os.getenv("CBD_API_KEY")
                # ldf = _fetch_lineups_raw(gid_int, api_key)
                ldf = pd.DataFrame() # SKIP LINEUPS to fix 30s timeout/hang
                if not ldf.empty:
                    ldf["gameId"] = gid
                    wh.ensure_table("fact_lineup_stint_raw", ldf, pk=None)
            except Exception as e:
                print(f"[{gid}] Lineups error: {e}")

            # Substitutions
            try:
                subs = _safe_api_call(plays_api.get_substitutions_by_game, game_id=gid_int)
                sdf = _models_to_df(subs) if subs else pd.DataFrame()
                if not sdf.empty:
                    sdf["gameId"] = gid
                    wh.ensure_table("fact_substitution_raw", sdf, pk=None)
            except Exception as e:
                print(f"[{gid}] Subs error: {e}")

            # Plays
            try:
                plays = _safe_api_call(plays_api.get_plays, game_id=gid_int)
                pdf = _models_to_df(plays) if plays else pd.DataFrame()
                if not pdf.empty:
                    pdf["gameId"] = gid
                    wh.ensure_table("fact_play_raw", pdf, pk=None)
            except Exception as e:
                print(f"[{gid}] Plays error: {e}")

