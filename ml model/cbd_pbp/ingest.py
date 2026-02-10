"""
CBD Play-by-Play Data Ingest Pipeline
======================================
Fetches game data from the College Basketball Data API and stores in DuckDB.
"""

from __future__ import annotations
import os
import time
import requests
import duckdb
import pandas as pd
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from tqdm import tqdm
from dotenv import load_dotenv

import cbbd
from cbbd.rest import ApiException

from .warehouse import Warehouse
from .schemas import Tables, CORE_DDL, TABLE_PRIMARY_KEYS

load_dotenv()

# API base URL for direct HTTP calls (bypasses Pydantic validation)
API_BASE = "https://api.collegebasketballdata.com"


# -----------------------------------------------------------------------------
# Quota Manager (DB-backed)
# -----------------------------------------------------------------------------

class QuotaManager:
    """Track API call usage in DuckDB instead of file system."""
    
    LIMIT = 120000  # Daily limit with safety buffer
    
    def __init__(self, wh: Warehouse):
        self.wh = wh
        self._ensure_table()
        self._count = self._load_count()
    
    def _ensure_table(self):
        """Create meta_api_usage table if not exists."""
        try:
            self.wh.exec("""
                CREATE TABLE IF NOT EXISTS meta_api_usage (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    call_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Insert initial row if empty
            self.wh.exec("""
                INSERT INTO meta_api_usage (id, call_count) 
                SELECT 1, 0 WHERE NOT EXISTS (SELECT 1 FROM meta_api_usage WHERE id = 1)
            """)
        except duckdb.Error:
            pass
    
    def _load_count(self) -> int:
        try:
            result = self.wh.query_df("SELECT call_count FROM meta_api_usage WHERE id = 1")
            return int(result.iloc[0, 0]) if not result.empty else 0
        except Exception:
            return 0
    
    def increment(self, n: int = 1):
        self._count += n
        # Persist periodically
        if self._count % 50 == 0:
            self._persist()
        if self._count >= self.LIMIT:
            print(f"[QUOTA] Reached {self._count} calls. Approaching limit!")
    
    def _persist(self):
        try:
            self.wh.exec(
                "UPDATE meta_api_usage SET call_count = ?, last_updated = CURRENT_TIMESTAMP WHERE id = 1",
                {"1": self._count}
            )
        except Exception:
            pass
    
    @property
    def count(self) -> int:
        return self._count


# Global quota manager (initialized lazily)
_quota_manager: Optional[QuotaManager] = None


def _get_quota_manager(wh: Warehouse) -> QuotaManager:
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = QuotaManager(wh)
    return _quota_manager


# -----------------------------------------------------------------------------
# API Helpers
# -----------------------------------------------------------------------------

def _fetch_lineups_raw(game_id: int, api_key: str, quota: Optional[QuotaManager] = None) -> pd.DataFrame:
    """
    Fetch lineup stats directly via HTTP, bypassing SDK's Pydantic validation.
    This allows null values in offenseRating/netRating etc.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    urls = [
        f"{API_BASE}/lineups/game/{game_id}",
        f"{API_BASE}/lineups/game",
    ]
    
    for attempt in range(5):
        try:
            data = None
            for idx, url in enumerate(urls):
                params = {} if idx == 0 else {"gameId": game_id}
                resp = requests.get(url, headers=headers, params=params, timeout=30)
                if resp.status_code == 404:
                    continue
                if resp.status_code == 429:
                    wait = min(60, 2 ** attempt)
                    print(f"[429] Rate limit on lineups. Waiting {wait}s...")
                    time.sleep(wait)
                    data = None
                    break
                if resp.status_code == 400:
                    return pd.DataFrame()
                resp.raise_for_status()
                data = resp.json()
                if quota:
                    quota.increment()
                break
            if data is None:
                continue
            if not data:
                return pd.DataFrame()
            return pd.DataFrame(data)
        except requests.RequestException as e:
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


def _safe_api_call(func, *args, quota: Optional[QuotaManager] = None, max_retries: int = 10, **kwargs):
    """Wrapper with exponential backoff for rate limits."""
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            if quota:
                quota.increment()
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
        except (ValueError, TypeError) as e:
            # Pydantic validation errors - skip, don't retry
            if "validation error" in str(e).lower():
                print(f"[Skip] Validation Error: {e}")
                return None
            raise
    raise RuntimeError(f"Max retries exceeded for {func}")


def get_client() -> cbbd.ApiClient:
    """Create a configured cbbd API client."""
    config = cbbd.Configuration(
        host="https://api.collegebasketballdata.com",
        access_token=os.getenv("CBD_API_KEY")
    )
    return cbbd.ApiClient(config)


# -----------------------------------------------------------------------------
# Data Transformations (Pure Functions)
# -----------------------------------------------------------------------------

def _sanitize_plays_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize plays DataFrame to match warehouse schema."""
    if df.empty:
        return df
    
    # Fix wallclock: DB expects BIGINT (Unix timestamp), but SDK returns datetime
    if "wallclock" in df.columns:
        df["wallclock"] = df["wallclock"].apply(
            lambda x: int(x.timestamp()) if pd.notnull(x) else None
        )
        df["wallclock"] = df["wallclock"].astype("Int64")
    
    return df


# -----------------------------------------------------------------------------
# Game Processing (DRY - Single Source of Truth)
# -----------------------------------------------------------------------------

@dataclass
class GameResult:
    """Result of processing a single game."""
    game_id: str
    lineups: List[Dict]
    subs: List[Dict]
    plays: List[Dict]
    errors: List[str]


def _process_single_game(
    gid: str,
    plays_api,
    quota: QuotaManager,
    include_lineups: bool,
    need_plays: bool,
    need_subs: bool,
    need_lineups: bool,
) -> GameResult:
    """
    Process a single game: fetch lineups, subs, plays from API.
    Returns raw dict data (not DataFrames) for batching efficiency.
    """
    gid_int = int(gid)
    result = GameResult(game_id=gid, lineups=[], subs=[], plays=[], errors=[])
    
    # Lineups
    if need_lineups and include_lineups:
        try:
            api_key = os.getenv("CBD_API_KEY")
            ldf = _fetch_lineups_raw(gid_int, api_key, quota) if api_key else pd.DataFrame()
            if not ldf.empty:
                ldf["gameId"] = gid
                result.lineups = ldf.to_dict("records")
        except (ApiException, requests.RequestException, duckdb.Error) as e:
            result.errors.append(f"lineups: {e}")
    
    # Substitutions
    if need_subs:
        try:
            subs = _safe_api_call(plays_api.get_substitutions_by_game, game_id=gid_int, quota=quota)
            sdf = _models_to_df(subs) if subs else pd.DataFrame()
            if not sdf.empty:
                sdf["gameId"] = gid
                result.subs = sdf.to_dict("records")
        except (ApiException, requests.RequestException, duckdb.Error) as e:
            result.errors.append(f"subs: {e}")
    
    # Plays
    if need_plays:
        try:
            plays = _safe_api_call(plays_api.get_plays, game_id=gid_int, quota=quota)
            pdf = _models_to_df(plays) if plays else pd.DataFrame()
            if not pdf.empty:
                pdf = _sanitize_plays_df(pdf)
                pdf["gameId"] = gid
                result.plays = pdf.to_dict("records")
        except (ApiException, requests.RequestException, duckdb.Error) as e:
            result.errors.append(f"plays: {e}")
    
    return result


def _log_ingest_failure(
    wh: Warehouse, game_id: str, season: int, season_type: str, endpoint: str, error: str
):
    """Log a failed ingestion attempt."""
    row = pd.DataFrame([{
        "gameId": str(game_id),
        "season": int(season),
        "seasonType": str(season_type),
        "endpoint": str(endpoint),
        "error": str(error)[:2000],
        "loggedAt": pd.Timestamp.utcnow(),
    }])
    wh.ensure_table(Tables.INGEST_FAILURES, row, pk=None)


# -----------------------------------------------------------------------------
# Static Dimension Ingest
# -----------------------------------------------------------------------------

def ingest_static(wh: Warehouse):
    """Ingest static dimension tables (teams, conferences, venues, etc.)."""
    quota = _get_quota_manager(wh)
    
    with get_client() as client:
        # Teams
        api = cbbd.TeamsApi(client)
        df = _models_to_df(_safe_api_call(api.get_teams, quota=quota))
        if not df.empty:
            wh.ensure_table(Tables.TEAMS, df, pk=TABLE_PRIMARY_KEYS.get(Tables.TEAMS))

        # Conferences
        api = cbbd.ConferencesApi(client)
        df = _models_to_df(_safe_api_call(api.get_conferences, quota=quota))
        if not df.empty:
            wh.ensure_table(Tables.CONFERENCES, df, pk=TABLE_PRIMARY_KEYS.get(Tables.CONFERENCES))
        
        # Conference History
        df = _models_to_df(_safe_api_call(api.get_conference_history, quota=quota))
        if not df.empty:
            wh.ensure_table(Tables.CONFERENCE_HISTORY, df, pk=None)

        # Venues
        api = cbbd.VenuesApi(client)
        df = _models_to_df(_safe_api_call(api.get_venues, quota=quota))
        if not df.empty:
            wh.ensure_table(Tables.VENUES, df, pk=TABLE_PRIMARY_KEYS.get(Tables.VENUES))

        # Draft
        api = cbbd.DraftApi(client)
        df = _models_to_df(_safe_api_call(api.get_draft_positions, quota=quota))
        if not df.empty:
            wh.ensure_table(Tables.DRAFT_POSITIONS, df, pk=TABLE_PRIMARY_KEYS.get(Tables.DRAFT_POSITIONS))
        df = _models_to_df(_safe_api_call(api.get_draft_teams, quota=quota))
        if not df.empty:
            wh.ensure_table(Tables.DRAFT_TEAMS, df, pk=TABLE_PRIMARY_KEYS.get(Tables.DRAFT_TEAMS))

        # Lines Providers
        api = cbbd.LinesApi(client)
        df = _models_to_df(_safe_api_call(api.get_providers, quota=quota))
        if not df.empty:
            wh.ensure_table(Tables.LINES_PROVIDERS, df, pk=TABLE_PRIMARY_KEYS.get(Tables.LINES_PROVIDERS))


def ingest_play_types(wh: Warehouse):
    """Ingest play type definitions."""
    quota = _get_quota_manager(wh)
    with get_client() as client:
        api = cbbd.PlaysApi(client)
        df = _models_to_df(_safe_api_call(api.get_play_types, quota=quota))
        if not df.empty:
            wh.ensure_table(Tables.PLAY_TYPES, df, pk=None)


def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def _normalize_player_season_stats_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize nested player-season payload to a flat schema.
    This avoids map/struct cast failures in DuckDB and preserves all values.
    """
    out = df.copy()

    # Ensure seasonType is explicit on every row.
    if "seasonType" not in out.columns:
        out["seasonType"] = None

    def extract(col: str, key: str, target: str):
        out[target] = out[col].apply(lambda x: _safe_get(x, key)) if col in out.columns else None

    extract("fieldGoals", "made", "fg_made")
    extract("fieldGoals", "attempted", "fg_attempted")
    extract("fieldGoals", "pct", "fg_pct")

    extract("twoPointFieldGoals", "made", "two_made")
    extract("twoPointFieldGoals", "attempted", "two_attempted")
    extract("twoPointFieldGoals", "pct", "two_pct")

    extract("threePointFieldGoals", "made", "three_made")
    extract("threePointFieldGoals", "attempted", "three_attempted")
    extract("threePointFieldGoals", "pct", "three_pct")

    extract("freeThrows", "made", "ft_made")
    extract("freeThrows", "attempted", "ft_attempted")
    extract("freeThrows", "pct", "ft_pct")

    extract("rebounds", "offensive", "oreb")
    extract("rebounds", "defensive", "dreb")
    extract("rebounds", "total", "reb")

    extract("winShares", "offensive", "ws_offensive")
    extract("winShares", "defensive", "ws_defensive")
    extract("winShares", "total", "ws_total")
    extract("winShares", "totalPer40", "ws_totalPer40")

    keep_cols = [
        "season", "seasonType", "seasonLabel", "teamId", "team", "conference",
        "athleteId", "athleteSourceId", "name", "position", "games", "starts",
        "minutes", "points", "turnovers", "fouls", "assists", "steals", "blocks",
        "offensiveRating", "defensiveRating", "netRating", "PORPAG", "usage",
        "assistsTurnoverRatio", "offensiveReboundPct", "freeThrowRate",
        "effectiveFieldGoalPct", "trueShootingPct",
        "fg_made", "fg_attempted", "fg_pct",
        "two_made", "two_attempted", "two_pct",
        "three_made", "three_attempted", "three_pct",
        "ft_made", "ft_attempted", "ft_pct",
        "oreb", "dreb", "reb",
        "ws_offensive", "ws_defensive", "ws_total", "ws_totalPer40",
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    return out[keep_cols]


def ingest_player_season_stats(
    wh: Warehouse,
    season: int,
    season_type: str = "regular",
    team: Optional[str] = None,
    conference: Optional[str] = None,
):
    """
    Ingest player season stats and persist to:
    1) legacy fact_player_season_stats (best effort, now with seasonType)
    2) normalized fact_player_season_stats_norm (authoritative)
    """
    quota = _get_quota_manager(wh)
    wh.init_schema(CORE_DDL)

    with get_client() as client:
        api = cbbd.StatsApi(client)
        rows = _safe_api_call(
            api.get_player_season_stats,
            season=season,
            season_type=season_type,
            team=team,
            conference=conference,
            quota=quota,
        )
        df = _models_to_df(rows) if rows else pd.DataFrame()
        if df.empty:
            print(f"No player season stats returned for {season} {season_type}")
            return

        df["seasonType"] = season_type
        norm_df = _normalize_player_season_stats_df(df)

        # Ensure legacy table can carry seasonType going forward.
        try:
            wh.exec(f'ALTER TABLE {Tables.PLAYER_SEASON_STATS} ADD COLUMN IF NOT EXISTS seasonType VARCHAR')
        except duckdb.Error:
            pass

        # Legacy insert (drop winShares nested object to avoid map->struct cast failure).
        legacy_df = df.drop(columns=["winShares"], errors="ignore")
        try:
            wh.ensure_table(Tables.PLAYER_SEASON_STATS, legacy_df, pk=None)
        except Exception as e:
            print(f"Warning: legacy insert into {Tables.PLAYER_SEASON_STATS} partially failed: {e}")

        # Authoritative normalized table.
        wh.ensure_table(Tables.PLAYER_SEASON_STATS_NORM, norm_df, pk=None)
        print(
            f"Ingested player season stats: season={season}, seasonType={season_type}, "
            f"rows={len(df)}, normalized_rows={len(norm_df)}"
        )


# -----------------------------------------------------------------------------
# Query Helpers
# -----------------------------------------------------------------------------

def _existing_game_ids_for_table(
    wh: Warehouse, table_name: str, season: int, season_type: str
) -> Set[str]:
    """Get game IDs already ingested for a table/season."""
    try:
        df = wh.query_df(f"""
            SELECT DISTINCT f.gameId
            FROM {table_name} f
            JOIN {Tables.GAMES} g ON g.id = TRY_CAST(f.gameId AS BIGINT)
            WHERE g.season = {season} AND g.seasonType = '{season_type}'
        """)
        if df.empty:
            return set()
        return set(df["gameId"].astype(str).tolist())
    except duckdb.Error:
        return set()


# -----------------------------------------------------------------------------
# Season Ingest (Main Entry Points)
# -----------------------------------------------------------------------------

def fetch_and_ingest_season(
    wh: Warehouse, season: int, season_type: str = "regular", include_lineups: bool = False
):
    """
    Fetch games for a season/type, add them to dim_games, then ingest per-game data.
    """
    quota = _get_quota_manager(wh)
    
    # Initialize core schema (ensures tables exist with correct types)
    wh.init_schema(CORE_DDL)
    
    with get_client() as client:
        # 1) Fetch and add games to dim_games
        games_api = cbbd.GamesApi(client)
        games = _models_to_df(_safe_api_call(games_api.get_games, season=season, season_type=season_type, quota=quota))
        if games.empty:
            print(f"No games found for {season} {season_type}")
            return
        wh.ensure_table(Tables.GAMES, games, pk=TABLE_PRIMARY_KEYS.get(Tables.GAMES))
        print(f"Added/updated {len(games)} games for {season} {season_type} in {Tables.GAMES}")

        # 2) Determine what's missing
        game_ids = set(games["id"].astype(str).tolist())
        done_play = _existing_game_ids_for_table(wh, Tables.PLAY_RAW, season, season_type)
        done_subs = _existing_game_ids_for_table(wh, Tables.SUBSTITUTION_RAW, season, season_type)
        done_lineups = (
            _existing_game_ids_for_table(wh, Tables.LINEUP_STINT_RAW, season, season_type)
            if include_lineups
            else game_ids  # Skip if not including
        )

        missing_play = game_ids - done_play
        missing_subs = game_ids - done_subs
        missing_lineups = game_ids - done_lineups
        todo = sorted(missing_play | missing_subs | missing_lineups, key=lambda x: int(x))
        
        print(
            f"Per-game queue for {season} {season_type}: "
            f"plays={len(missing_play)}, subs={len(missing_subs)}, "
            f"lineups={len(missing_lineups)} (include={include_lineups}), "
            f"total={len(todo)}"
        )

        # 3) Process games
        plays_api = cbbd.PlaysApi(client)
        
        for gid in tqdm(todo, desc="Per-game ingest"):
            result = _process_single_game(
                gid=gid,
                plays_api=plays_api,
                quota=quota,
                include_lineups=include_lineups,
                need_plays=gid in missing_play,
                need_subs=gid in missing_subs,
                need_lineups=gid in missing_lineups,
            )
            
            # Write results
            if result.lineups:
                wh.ensure_table(Tables.LINEUP_STINT_RAW, pd.DataFrame(result.lineups), pk=None)
            if result.subs:
                wh.ensure_table(Tables.SUBSTITUTION_RAW, pd.DataFrame(result.subs), pk=None)
            if result.plays:
                wh.ensure_table(Tables.PLAY_RAW, pd.DataFrame(result.plays), pk=None)
            
            # Log errors
            for err in result.errors:
                endpoint, msg = err.split(": ", 1) if ": " in err else ("unknown", err)
                _log_ingest_failure(wh, gid, season, season_type, endpoint, msg)


def ingest_games_only(
    wh: Warehouse, season: int, season_type: str = "regular", include_lineups: bool = False
):
    """Resume per-game ingest only (skips bulk tables). Uses existing dim_games."""
    quota = _get_quota_manager(wh)
    wh.init_schema(CORE_DDL)
    
    with get_client() as client:
        # Get game IDs from existing dim_games
        try:
            games = wh.query_df(f"""
                SELECT id FROM {Tables.GAMES}
                WHERE season = {season} AND seasonType = '{season_type}'
            """)
            game_ids = set(games["id"].astype(str).tolist())
        except duckdb.Error as e:
            print(f"Error: {Tables.GAMES} not found. Run full ingest first. {e}")
            return

        # Determine missing
        done_play = _existing_game_ids_for_table(wh, Tables.PLAY_RAW, season, season_type)
        done_subs = _existing_game_ids_for_table(wh, Tables.SUBSTITUTION_RAW, season, season_type)
        done_lineups = (
            _existing_game_ids_for_table(wh, Tables.LINEUP_STINT_RAW, season, season_type)
            if include_lineups
            else game_ids
        )

        missing_play = game_ids - done_play
        missing_subs = game_ids - done_subs
        missing_lineups = game_ids - done_lineups
        todo = sorted(missing_play | missing_subs | missing_lineups, key=lambda x: int(x))
        
        print(
            f"Resume queue for {season} {season_type}: "
            f"plays={len(missing_play)}, subs={len(missing_subs)}, "
            f"lineups={len(missing_lineups)} (include={include_lineups}), "
            f"total={len(todo)}"
        )

        plays_api = cbbd.PlaysApi(client)
        
        for gid in tqdm(todo, desc="Per-game ingest"):
            result = _process_single_game(
                gid=gid,
                plays_api=plays_api,
                quota=quota,
                include_lineups=include_lineups,
                need_plays=gid in missing_play,
                need_subs=gid in missing_subs,
                need_lineups=gid in missing_lineups,
            )
            
            if result.lineups:
                wh.ensure_table(Tables.LINEUP_STINT_RAW, pd.DataFrame(result.lineups), pk=None)
            if result.subs:
                wh.ensure_table(Tables.SUBSTITUTION_RAW, pd.DataFrame(result.subs), pk=None)
            if result.plays:
                wh.ensure_table(Tables.PLAY_RAW, pd.DataFrame(result.plays), pk=None)
            
            for err in result.errors:
                endpoint, msg = err.split(": ", 1) if ": " in err else ("unknown", err)
                _log_ingest_failure(wh, gid, season, season_type, endpoint, msg)
