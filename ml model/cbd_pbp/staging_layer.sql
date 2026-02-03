-- =============================================================================
-- CBD PBP Feature Store: SQL Staging Layer (V3.4 - LOCKED DEFINITIONS)
-- 
-- PROFILING RESULTS (Locked 2025-01-25):
--   - participants: STRUCT(id INTEGER, name VARCHAR)[] → Use UNNEST, access .id
--   - athletes: STRUCT(id INTEGER, name VARCHAR)[] → Use UNNEST, access .id
--   - shot_range values: 'three_pointer', 'rim', 'free_throw', 'jumper'
-- =============================================================================

-- #############################################################################
-- SECTION 1: STAGING VIEWS (Lightweight, always current)
-- #############################################################################

-- -----------------------------------------------------------------------------
-- 1A. stg_plays: Base play event view
--     Grain: Event (1 row per play)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW stg_plays AS
SELECT
    p.gameId,
    p.id AS playId,
    p.teamId,
    p.opponentId,
    p.playType,
    p.scoringPlay,
    p.shootingPlay,
    p.scoreValue,
    p.period,
    p.secondsRemaining,
    p.homeScore,
    p.awayScore,
    p.homeWinProbability,
    p.playText,
    p.shotInfo,
    p.participants
FROM fact_play_raw p;

-- -----------------------------------------------------------------------------
-- 1B. stg_shots: Shot events with shotInfo parsed + context flags
--     Grain: Shot Event
--     LOCKED: IDs use json_extract_string → CAST (handles string IDs)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW stg_shots AS
SELECT
    p.gameId,
    p.playId,
    p.teamId,
    p.opponentId,
    p.period,
    p.secondsRemaining,
    p.homeScore,
    p.awayScore,
    p.homeWinProbability,
    COALESCE(p.scoreValue, 0) AS scoreValue,
    
    CAST(json_extract_string(p.shotInfo, '$.shooter.id') AS INT) AS shooterAthleteId,
    json_extract_string(p.shotInfo, '$.shooter.name') AS shooter_name,
    CAST(json_extract(p.shotInfo, '$.made') AS BOOLEAN) AS made,
    json_extract_string(p.shotInfo, '$.range') AS shot_range,
    CAST(json_extract(p.shotInfo, '$.assisted') AS BOOLEAN) AS assisted,
    CAST(json_extract_string(p.shotInfo, '$.assistedBy.id') AS INT) AS assistAthleteId,
    -- NOTE: use TRY_CAST because older seasons may contain JSON null/non-numeric placeholders.
    -- TRY_CAST yields SQL NULL (clean missingness) rather than NaN / cast errors.
    TRY_CAST(json_extract(p.shotInfo, '$.location.x') AS FLOAT) AS loc_x,
    TRY_CAST(json_extract(p.shotInfo, '$.location.y') AS FLOAT) AS loc_y,
    
    CASE 
        WHEN p.period >= 2 AND p.secondsRemaining < 300 AND ABS(p.homeScore - p.awayScore) <= 5 
        THEN TRUE ELSE FALSE 
    END AS is_high_leverage,
    CASE 
        WHEN p.period >= 2 AND ABS(p.homeScore - p.awayScore) > 20 
        THEN TRUE ELSE FALSE 
    END AS is_garbage
    
FROM stg_plays p
WHERE p.shotInfo IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 1C. stg_participants: Explode participants array
--     Grain: Event × Player
--     LOCKED: participants is STRUCT(id, name)[] - use UNNEST, access .id
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW stg_participants AS
SELECT
    p.gameId,
    p.playId,
    part.id AS athleteId,
    part.name AS athlete_name
FROM stg_plays p,
LATERAL UNNEST(p.participants) AS part
WHERE p.participants IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 1D. stg_lineups: Lineup stint view
--     Grain: Stint
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW stg_lineups AS
SELECT
    l.gameId,
    l.teamId,
    l.conference,
    l.idHash AS lineupHash,
    l.totalSeconds,
    l.pace,
    l.offenseRating,
    l.defenseRating,
    l.netRating,
    l.teamStats,
    l.opponentStats,
    l.athletes
FROM fact_lineup_stint_raw l;

-- -----------------------------------------------------------------------------
-- 1E. bridge_lineup_athletes: Explode athletes array
--     Grain: Stint × Player (exactly 5 per lineup)
--     LOCKED: athletes is STRUCT(id, name)[] - use UNNEST, access .id
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW bridge_lineup_athletes AS
SELECT
    l.gameId,
    l.lineupHash,
    l.teamId,
    l.totalSeconds,
    l.pace,
    l.offenseRating,
    l.defenseRating,
    l.netRating,
    ath.id AS athleteId
FROM stg_lineups l,
LATERAL UNNEST(l.athletes) AS ath;

-- -----------------------------------------------------------------------------
-- 1F. stg_subs: Substitution events flattened
--     Grain: Sub Event
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW stg_subs AS
SELECT
    s.gameId,
    s.teamId,
    s.athleteId,
    s.athlete AS athlete_name,
    CAST(json_extract(s.subIn, '$.secondsRemaining') AS INT) AS subIn_secondsRemaining,
    CAST(json_extract(s.subIn, '$.period') AS INT) AS subIn_period,
    CAST(json_extract(s.subIn, '$.teamPoints') AS INT) AS subIn_teamPoints,
    CAST(json_extract(s.subIn, '$.opponentPoints') AS INT) AS subIn_opponentPoints,
    CAST(json_extract(s.subOut, '$.secondsRemaining') AS INT) AS subOut_secondsRemaining,
    CAST(json_extract(s.subOut, '$.period') AS INT) AS subOut_period,
    CAST(json_extract(s.subOut, '$.teamPoints') AS INT) AS subOut_teamPoints,
    CAST(json_extract(s.subOut, '$.opponentPoints') AS INT) AS subOut_opponentPoints
FROM fact_substitution_raw s;

-- #############################################################################
-- SECTION 2: MAPPING TABLES (LOCKED after profiling)
-- #############################################################################

-- LOCKED: Shot range values from profiling: three_pointer, rim, free_throw, jumper
CREATE TABLE IF NOT EXISTS dim_shot_range_map(range_value VARCHAR, range_bucket VARCHAR);
DELETE FROM dim_shot_range_map;
INSERT INTO dim_shot_range_map VALUES
  ('rim', 'rim'),
  ('three_pointer', 'three'),
  ('jumper', 'mid'),
  ('free_throw', 'ft');

-- #############################################################################
-- SECTION 3: MATERIALIZED FACT TABLES
-- #############################################################################

-- -----------------------------------------------------------------------------
-- 3A. fact_player_game_shots_by_range: Shot stats by range (long form)
--     Grain: Player-Game-Range
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE fact_player_game_shots_by_range AS
SELECT
    gameId,
    teamId,
    shooterAthleteId AS athleteId,
    shot_range,
    COUNT(*) AS fga,
    SUM(CASE WHEN made THEN 1 ELSE 0 END) AS fgm,
    SUM(scoreValue) AS pts,
    SUM(CASE WHEN assisted THEN 1 ELSE 0 END) AS assisted_att,
    SUM(CASE WHEN assisted AND made THEN 1 ELSE 0 END) AS assisted_made,
    SUM(CASE WHEN is_high_leverage THEN 1 ELSE 0 END) AS high_lev_att,
    SUM(CASE WHEN is_high_leverage AND made THEN 1 ELSE 0 END) AS high_lev_made,
    SUM(CASE WHEN is_garbage THEN 1 ELSE 0 END) AS garbage_att
FROM stg_shots
WHERE shooterAthleteId IS NOT NULL
GROUP BY gameId, teamId, shooterAthleteId, shot_range;

-- -----------------------------------------------------------------------------
-- 3B. fact_player_game_shots_bucketed: Shot stats by bucket (uses mapping)
--     Grain: Player-Game-Bucket
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE fact_player_game_shots_bucketed AS
SELECT
    s.gameId,
    s.teamId,
    s.shooterAthleteId AS athleteId,
    COALESCE(m.range_bucket, 'unknown') AS range_bucket,
    COUNT(*) AS att,
    SUM(CASE WHEN s.made THEN 1 ELSE 0 END) AS made,
    SUM(s.scoreValue) AS pts,
    SUM(CASE WHEN s.assisted THEN 1 ELSE 0 END) AS assisted_att
FROM stg_shots s
LEFT JOIN dim_shot_range_map m ON s.shot_range = m.range_value
WHERE s.shooterAthleteId IS NOT NULL
GROUP BY 1, 2, 3, 4;

-- -----------------------------------------------------------------------------
-- 3C. fact_player_game_shots: Collapsed to Player-Game grain
--     Grain: Player-Game
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE fact_player_game_shots AS
SELECT
    gameId,
    teamId,
    athleteId,
    SUM(fga) AS fga,
    SUM(fgm) AS fgm,
    SUM(pts) AS pts,
    SUM(assisted_att) AS assisted_att,
    SUM(assisted_made) AS assisted_made,
    SUM(high_lev_att) AS high_lev_att,
    SUM(high_lev_made) AS high_lev_made,
    SUM(garbage_att) AS garbage_att,
    SUM(assisted_att)::FLOAT / NULLIF(SUM(fga), 0) AS assisted_share
FROM fact_player_game_shots_by_range
GROUP BY gameId, teamId, athleteId;

-- -----------------------------------------------------------------------------
-- 3D. fact_player_game_impact: On-court impact per player-game
--     Grain: Player-Game (Seconds-weighted)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE fact_player_game_impact AS
SELECT
    gameId,
    athleteId,
    teamId,
    SUM(totalSeconds) AS seconds_on,
    SUM(netRating * totalSeconds) / NULLIF(SUM(totalSeconds), 0) AS on_net_rating,
    SUM(offenseRating * totalSeconds) / NULLIF(SUM(totalSeconds), 0) AS on_ortg,
    SUM(defenseRating * totalSeconds) / NULLIF(SUM(totalSeconds), 0) AS on_drtg
FROM bridge_lineup_athletes
GROUP BY gameId, athleteId, teamId;

-- -----------------------------------------------------------------------------
-- 3E. fact_player_game: Master join of Shots + Impact
--     Grain: Player-Game
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE fact_player_game AS
SELECT
    COALESCE(s.gameId, i.gameId) AS gameId,
    COALESCE(s.athleteId, i.athleteId) AS athleteId,
    COALESCE(i.teamId, s.teamId) AS teamId,
    s.fga, s.fgm, s.pts,
    s.assisted_att, s.assisted_made,
    s.high_lev_att, s.high_lev_made, s.garbage_att,
    s.assisted_share,
    i.seconds_on, i.on_net_rating, i.on_ortg, i.on_drtg
FROM fact_player_game_shots s
FULL OUTER JOIN fact_player_game_impact i
    ON s.gameId = i.gameId AND s.athleteId = i.athleteId;

-- -----------------------------------------------------------------------------
-- 3F. fact_team_game: Team-level totals from stints
--     Grain: Team-Game
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE fact_team_game AS
SELECT
    gameId,
    teamId,
    SUM(totalSeconds) AS seconds_game,
    SUM(pace * totalSeconds) / NULLIF(SUM(totalSeconds), 0) AS pace,
    SUM(offenseRating * totalSeconds) / NULLIF(SUM(totalSeconds), 0) AS offenseRating,
    SUM(defenseRating * totalSeconds) / NULLIF(SUM(totalSeconds), 0) AS defenseRating,
    SUM(netRating * totalSeconds) / NULLIF(SUM(totalSeconds), 0) AS netRating
FROM stg_lineups
GROUP BY gameId, teamId;

-- #############################################################################
-- SECTION 4: SANITY-CHECK ASSERTIONS
-- #############################################################################

-- 4A. Lineup Expansion: Each (gameId, lineupHash) MUST have exactly 5 athletes
-- SELECT gameId, lineupHash, COUNT(*) AS n FROM bridge_lineup_athletes GROUP BY 1, 2 HAVING n <> 5;

-- 4B. Stint Seconds: should sum to ~2400s per team per game
-- SELECT gameId, teamId, SUM(totalSeconds) AS sec FROM stg_lineups GROUP BY 1, 2 ORDER BY sec;

-- 4C. Shot Rows: stg_shots must not be empty
-- SELECT COUNT(*) FROM stg_shots;

-- 4D. Join Cardinality
-- SELECT COUNT(*) FROM fact_player_game;

-- =============================================================================
-- END OF STAGING LAYER (V3.4 - LOCKED)
-- =============================================================================
