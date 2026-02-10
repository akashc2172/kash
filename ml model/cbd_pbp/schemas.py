"""
CBD Warehouse Schema Definitions
================================
Explicit table names, column types, and DDL for the DuckDB warehouse.
Prevents schema drift and "magic string" bugs.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional


# -----------------------------------------------------------------------------
# Table Name Constants
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Tables:
    """Central registry of all table names."""
    # Dimensions
    GAMES = "dim_games"
    TEAMS = "dim_teams"
    CONFERENCES = "dim_conferences"
    CONFERENCE_HISTORY = "dim_conference_history"
    VENUES = "dim_venues"
    DRAFT_POSITIONS = "dim_draft_positions"
    DRAFT_TEAMS = "dim_draft_teams"
    LINES_PROVIDERS = "dim_lines_providers"
    PLAY_TYPES = "dim_play_types"
    
    # Facts (raw)
    PLAY_RAW = "fact_play_raw"
    SUBSTITUTION_RAW = "fact_substitution_raw"
    LINEUP_STINT_RAW = "fact_lineup_stint_raw"
    
    # Facts (aggregated)
    RATINGS_ADJUSTED = "fact_ratings_adjusted"
    RATINGS_SRS = "fact_ratings_srs"
    RANKINGS = "fact_rankings"
    RECRUITING_PLAYERS = "fact_recruiting_players"
    DRAFT_PICKS = "fact_draft_picks"
    LINES = "fact_lines"
    TEAM_SEASON_STATS = "fact_team_season_stats"
    PLAYER_SEASON_STATS = "fact_player_season_stats"
    PLAYER_SEASON_STATS_NORM = "fact_player_season_stats_norm"
    BRIDGE_GAME_CBD_SCRAPE = "bridge_game_cbd_scrape"
    BRIDGE_PLAYER_CBD_SCRAPE = "bridge_player_cbd_scrape"
    
    # Meta
    INGEST_FAILURES = "ingest_failures"
    API_USAGE = "meta_api_usage"


# -----------------------------------------------------------------------------
# DDL Definitions (Core Tables)
# -----------------------------------------------------------------------------

CORE_DDL: Dict[str, str] = {
    Tables.PLAY_RAW: """
        CREATE TABLE IF NOT EXISTS fact_play_raw (
            id BIGINT,
            gameId VARCHAR,
            gameStartDate TIMESTAMP WITH TIME ZONE,
            season INTEGER,
            seasonType VARCHAR,
            gameType VARCHAR,
            tournament VARCHAR,
            playType VARCHAR,
            isHomeTeam BOOLEAN,
            teamId INTEGER,
            team VARCHAR,
            conference VARCHAR,
            opponentId INTEGER,
            opponent VARCHAR,
            opponentConference VARCHAR,
            period INTEGER,
            clock VARCHAR,
            secondsRemaining INTEGER,
            homeScore INTEGER,
            awayScore INTEGER,
            homeWinProbability DOUBLE,
            scoringPlay BOOLEAN,
            shootingPlay BOOLEAN,
            scoreValue INTEGER,
            playText VARCHAR,
            participants JSON,
            onFloor JSON,
            shotInfo JSON,
            teamSeed INTEGER,
            opponentSeed INTEGER,
            wallclock BIGINT
        )
    """,
    
    Tables.SUBSTITUTION_RAW: """
        CREATE TABLE IF NOT EXISTS fact_substitution_raw (
            gameId VARCHAR,
            period INTEGER,
            clock VARCHAR,
            secondsRemaining INTEGER,
            homeScore INTEGER,
            awayScore INTEGER,
            teamId INTEGER,
            team VARCHAR,
            playerId INTEGER,
            player VARCHAR,
            type VARCHAR
        )
    """,
    
    Tables.LINEUP_STINT_RAW: """
        CREATE TABLE IF NOT EXISTS fact_lineup_stint_raw (
            gameId VARCHAR,
            teamId INTEGER,
            team VARCHAR,
            conference VARCHAR,
            players JSON,
            seconds DOUBLE,
            possessions DOUBLE,
            teamPoints INTEGER,
            opponentPoints INTEGER,
            offenseRating DOUBLE,
            defenseRating DOUBLE,
            netRating DOUBLE
        )
    """,
    
    Tables.INGEST_FAILURES: """
        CREATE TABLE IF NOT EXISTS ingest_failures (
            gameId VARCHAR,
            season INTEGER,
            seasonType VARCHAR,
            endpoint VARCHAR,
            error VARCHAR,
            loggedAt TIMESTAMP
        )
    """,
    
    Tables.API_USAGE: """
        CREATE TABLE IF NOT EXISTS meta_api_usage (
            id INTEGER PRIMARY KEY,
            call_count INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    Tables.PLAYER_SEASON_STATS_NORM: """
        CREATE TABLE IF NOT EXISTS fact_player_season_stats_norm (
            season INTEGER,
            seasonType VARCHAR,
            seasonLabel VARCHAR,
            teamId BIGINT,
            team VARCHAR,
            conference VARCHAR,
            athleteId BIGINT,
            athleteSourceId VARCHAR,
            name VARCHAR,
            position VARCHAR,
            games INTEGER,
            starts INTEGER,
            minutes INTEGER,
            points INTEGER,
            turnovers INTEGER,
            fouls INTEGER,
            assists INTEGER,
            steals INTEGER,
            blocks INTEGER,
            offensiveRating DOUBLE,
            defensiveRating DOUBLE,
            netRating DOUBLE,
            PORPAG DOUBLE,
            usage DOUBLE,
            assistsTurnoverRatio DOUBLE,
            offensiveReboundPct DOUBLE,
            freeThrowRate DOUBLE,
            effectiveFieldGoalPct DOUBLE,
            trueShootingPct DOUBLE,
            fg_made INTEGER,
            fg_attempted INTEGER,
            fg_pct DOUBLE,
            two_made INTEGER,
            two_attempted INTEGER,
            two_pct DOUBLE,
            three_made INTEGER,
            three_attempted INTEGER,
            three_pct DOUBLE,
            ft_made INTEGER,
            ft_attempted INTEGER,
            ft_pct DOUBLE,
            oreb INTEGER,
            dreb INTEGER,
            reb INTEGER,
            ws_offensive DOUBLE,
            ws_defensive DOUBLE,
            ws_total DOUBLE,
            ws_totalPer40 DOUBLE
        )
    """,
    Tables.BRIDGE_GAME_CBD_SCRAPE: """
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
    """,
    Tables.BRIDGE_PLAYER_CBD_SCRAPE: """
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
}


# -----------------------------------------------------------------------------
# Primary Key Definitions
# -----------------------------------------------------------------------------

TABLE_PRIMARY_KEYS: Dict[str, Optional[List[str]]] = {
    Tables.GAMES: ["id"],
    Tables.TEAMS: ["id"],
    Tables.CONFERENCES: ["id"],
    Tables.VENUES: ["id"],
    Tables.DRAFT_POSITIONS: ["name"],
    Tables.DRAFT_TEAMS: ["id"],
    Tables.LINES_PROVIDERS: ["id"],
    # Facts have no PK (append-only)
    Tables.PLAY_RAW: None,
    Tables.SUBSTITUTION_RAW: None,
    Tables.LINEUP_STINT_RAW: None,
    Tables.INGEST_FAILURES: None,
    Tables.API_USAGE: ["id"],
}
