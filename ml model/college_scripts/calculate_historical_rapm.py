import pandas as pd
import numpy as np
import json
import os
import scipy.sparse as sparse
from scipy.sparse.linalg import cg
import logging
import difflib
import re
import argparse
from typing import Tuple, Dict, List, Optional

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INPUT_PARQUET = 'data/fact_play_historical_combined_v2.parquet'
OUTPUT_CSV = 'data/historical_rapm_results_enhanced.csv'
OUTPUT_DIAG_CSV = 'data/audit/historical_rapm_split_quality.csv'
LINEUP_SEASON_AUDIT_CSV = 'data/audit/historical_lineup_quality_by_season.csv'
LAMBDA = 1000.0

DEFAULT_MIN_STINTS = 2000
DEFAULT_MIN_VALID_5V5_RATE = 0.80
DEFAULT_MAX_UNRESOLVED_HOME_RATE = 0.20
DEFAULT_MAX_PARSE_FAIL_RATE = 0.05
DEFAULT_MIN_UNIQUE_PLAYERS = 700

# Leverage thresholds (based on pbpstats research)
LEVERAGE_THRESHOLDS = {
    'garbage': 0.05,      # Below this = garbage time
    'low': 0.10,          # Low leverage
    'medium': 0.20,       # Medium leverage  
    'high': 0.35,         # High leverage
    'very_high': 0.50     # Very high leverage (crunch time)
}

def parse_clock(clock_str: str) -> int:
    """Parse clock string to seconds remaining.

    Supports:
    - MM:SS
    - MM:SS.s
    - PTMMMS / PTMMMS.S (ISO-like provider format)
    """
    try:
        if clock_str is None:
            return 0
        s = str(clock_str).strip().upper()
        if s == "":
            return 0
        if ":" in s:
            parts = s.split(":")
            if len(parts) >= 2:
                m = int(float(parts[0]))
                sec = int(float(parts[1]))
                return max(0, m * 60 + sec)
        # ISO-like: PT19M32S or PT19M32.1S
        m = re.match(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$", s)
        if m:
            mins = int(m.group(1)) if m.group(1) else 0
            secs = int(float(m.group(2))) if m.group(2) else 0
            return max(0, mins * 60 + secs)
        # plain numeric string
        if re.match(r"^\d+(\.\d+)?$", s):
            return int(float(s))
        return 0
    except:
        return 0


def compute_win_probability(margin: int, seconds_remaining: int, 
                            total_game_seconds: int = 2400) -> float:
    """
    Estimate win probability for home team using logistic model.
    
    Based on empirical NBA/NCAA research:
    - At game start (t=2400s), margin has minimal impact
    - At game end (t=0s), margin is decisive
    - Uses time-weighted logistic function
    
    Args:
        margin: Home team lead (positive = home winning)
        seconds_remaining: Seconds left in game
        total_game_seconds: Total game length (2400 for 40-min college)
    
    Returns:
        Win probability for home team [0, 1]
    """
    if seconds_remaining <= 0:
        return 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
    
    # Time factor: how much margin matters (0 at start, 1 at end)
    time_factor = 1.0 - (seconds_remaining / total_game_seconds)
    time_factor = max(0.0, min(1.0, time_factor))  # Clamp to [0, 1]
    
    # Effective margin scales with time remaining
    # At game start, need huge lead to matter; at end, small lead decisive
    # Empirical: ~0.15 points per second of game time for significance
    effective_margin = margin * (1.0 + 2.0 * time_factor)
    
    # Logistic function with empirically-tuned scale
    # Scale of ~4 means 8-point lead at halftime ≈ 75% win prob
    scale = 4.0 + 6.0 * time_factor  # More decisive late
    wp = 1.0 / (1.0 + np.exp(-effective_margin / scale))
    
    return wp


def compute_leverage_index(margin: int, seconds_remaining: int,
                           total_game_seconds: int = 2400) -> float:
    """
    Compute leverage index for a game state.
    
    Leverage = expected win probability swing from possession outcome.
    Based on pbpstats.com methodology:
    - Calculate WP change for each possible outcome (0-4 pts)
    - Weight by outcome frequency
    - Take sqrt of weighted sum of squared changes
    
    Args:
        margin: Home team lead
        seconds_remaining: Seconds remaining
        total_game_seconds: Total game length
    
    Returns:
        Leverage index [0, 1] where higher = more impactful
    """
    if seconds_remaining <= 0:
        return 0.0
    
    # Current win probability
    wp_current = compute_win_probability(margin, seconds_remaining, total_game_seconds)
    
    # Assume possession takes ~18 seconds on average
    time_after = max(0, seconds_remaining - 18)
    
    # Possible outcomes and their frequencies (empirical from NCAA data)
    outcomes = [
        (0, 0.51),   # No score: 51%
        (1, 0.09),   # 1 point (FT): 9%
        (2, 0.28),   # 2 points: 28%
        (3, 0.11),   # 3 points: 11%
        (4, 0.01),   # 4 points (and-1 or 2 FTs): 1%
    ]
    
    # Calculate weighted sum of squared WP changes
    weighted_sq_sum = 0.0
    for pts, freq in outcomes:
        # WP after scoring pts (from home perspective, assume home has ball)
        wp_after = compute_win_probability(margin + pts, time_after, total_game_seconds)
        wp_change = wp_after - wp_current
        weighted_sq_sum += freq * (wp_change ** 2)
    
    leverage = np.sqrt(weighted_sq_sum)
    
    # Normalize to [0, 1] range (max theoretical leverage ~0.5)
    leverage_normalized = min(1.0, leverage / 0.5)
    
    return leverage_normalized


def classify_leverage(leverage: float) -> str:
    """
    Classify leverage into buckets for filtering/analysis.
    
    Returns: 'garbage', 'low', 'medium', 'high', or 'very_high'
    """
    if leverage < LEVERAGE_THRESHOLDS['garbage']:
        return 'garbage'
    elif leverage < LEVERAGE_THRESHOLDS['low']:
        return 'low'
    elif leverage < LEVERAGE_THRESHOLDS['medium']:
        return 'medium'
    elif leverage < LEVERAGE_THRESHOLDS['high']:
        return 'high'
    else:
        return 'very_high'


def compute_rubber_band_adjustment(margin_at_start: int, 
                                   expected_regression_rate: float = 0.02) -> float:
    """
    Compute expected margin regression for rubber band effect.
    
    Teams ahead tend to coast; teams behind try harder.
    This creates systematic bias in raw +/- that we can adjust for.
    
    Args:
        margin_at_start: Score differential at stint start
        expected_regression_rate: Points of regression per point of lead per minute
    
    Returns:
        Expected margin change due to rubber band effect (negative when ahead)
    """
    return -expected_regression_rate * margin_at_start

def solve_rapm(season_stints: pd.DataFrame, lambda_val: float,
               weight_col: str = 'poss',
               use_leverage_weights: bool = False,
               use_rubber_band: bool = False) -> pd.DataFrame:
    """
    Solve RAPM using ridge regression with configurable weighting.
    
    Args:
        season_stints: DataFrame with stint data
        lambda_val: Ridge regularization parameter
        weight_col: Column to use for base weights ('poss' or 'leverage_weight')
        use_leverage_weights: If True, multiply weights by leverage index
        use_rubber_band: If True, adjust target for rubber band effect
    
    Returns:
        DataFrame with player RAPM values
    """
    all_players = set()
    for _, row in season_stints.iterrows():
        all_players.update(row['home_players'])
        all_players.update(row['away_players'])
    
    players = sorted(list(all_players))
    p_map = {p: i for i, p in enumerate(players)}
    n_players = len(players)
    
    n_stints = len(season_stints)
    row_ind, col_ind, data = [], [], []
    
    season_stints = season_stints.reset_index(drop=True)
    
    for i, row in season_stints.iterrows():
        for pid in row['home_players']:
            row_ind.append(i)
            col_ind.append(p_map[pid])
            data.append(1.0)
        for pid in row['away_players']:
            row_ind.append(i)
            col_ind.append(p_map[pid])
            data.append(-1.0)
    
    X = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n_stints, n_players))
    
    # Target: Point differential per 100 possessions
    poss = season_stints['poss'].values
    margin_diff = season_stints['margin_diff'].values.copy()
    
    # Apply rubber band adjustment if requested
    if use_rubber_band and 'margin_start' in season_stints.columns:
        rubber_band_adj = season_stints['margin_start'].apply(
            lambda m: compute_rubber_band_adjustment(m)
        ).values
        # Adjust for stint duration (in minutes)
        duration_mins = season_stints['duration'].values / 60.0
        margin_diff = margin_diff - (rubber_band_adj * duration_mins)
    
    pts_per_100 = (margin_diff / np.maximum(poss, 0.1)) * 100.0
    
    # Compute weights
    w = poss.copy()
    if use_leverage_weights and 'leverage' in season_stints.columns:
        # Multiply by leverage (higher leverage = more weight)
        leverage = season_stints['leverage'].values
        w = w * (0.5 + leverage)  # Floor of 0.5 to not completely ignore low-leverage
    
    W = sparse.diags(w)
    
    # Center target
    y_mean = np.average(pts_per_100, weights=w)
    y = pts_per_100 - y_mean
    
    # Normal Equations: (X'WX + lambda*I)c = X'Wy
    XTW = X.T @ W
    XTWX = XTW @ X
    XTWy = XTW @ y
    
    A = XTWX + lambda_val * sparse.eye(n_players)
    b = XTWy
    
    coef, _ = cg(A, b, rtol=1e-5)
    
    res = pd.DataFrame({
        'player_name': players,
        'rapm': coef,
        'poss_total': np.abs(X).T @ w
    })
    return res


def solve_rapm_split(season_stints: pd.DataFrame, lambda_val: float,
                     use_leverage_weights: bool = False) -> pd.DataFrame:
    """
    Solve for Offensive and Defensive RAPM separately.
    
    O-RAPM: Points scored per 100 possessions (home perspective)
    D-RAPM: Points allowed per 100 possessions (home perspective, inverted)
    
    Args:
        season_stints: DataFrame with stint data including pts_scored_home, pts_allowed_home
        lambda_val: Ridge regularization parameter
        use_leverage_weights: If True, weight by leverage index
    
    Returns:
        DataFrame with player O-RAPM, D-RAPM, and Net RAPM
    """
    all_players = set()
    for _, row in season_stints.iterrows():
        all_players.update(row['home_players'])
        all_players.update(row['away_players'])
    
    players = sorted(list(all_players))
    p_map = {p: i for i, p in enumerate(players)}
    n_players = len(players)
    n_stints = len(season_stints)
    
    # Build design matrix (same for O and D)
    row_ind, col_ind, data = [], [], []
    season_stints = season_stints.reset_index(drop=True)
    
    for i, row in season_stints.iterrows():
        for pid in row['home_players']:
            row_ind.append(i)
            col_ind.append(p_map[pid])
            data.append(1.0)
        for pid in row['away_players']:
            row_ind.append(i)
            col_ind.append(p_map[pid])
            data.append(-1.0)
    
    X = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n_stints, n_players))
    
    poss = season_stints['poss'].values
    w = poss.copy()
    if use_leverage_weights and 'leverage' in season_stints.columns:
        leverage = season_stints['leverage'].values
        w = w * (0.5 + leverage)
    
    W = sparse.diags(w)
    
    def solve_single_target(target_values):
        pts_per_100 = (target_values / np.maximum(poss, 0.1)) * 100.0
        y_mean = np.average(pts_per_100, weights=w)
        y = pts_per_100 - y_mean
        
        XTW = X.T @ W
        XTWX = XTW @ X
        XTWy = XTW @ y
        
        A = XTWX + lambda_val * sparse.eye(n_players)
        b = XTWy
        
        coef, _ = cg(A, b, rtol=1e-5)
        return coef
    
    # O-RAPM: Points scored by home team in stint
    # For home players: positive = good offense
    # For away players: their offense is home's defense, so sign flips
    if 'pts_scored_home' in season_stints.columns:
        o_coef = solve_single_target(season_stints['pts_scored_home'].values)
        d_coef = solve_single_target(-season_stints['pts_allowed_home'].values)  # Negative = good defense
    else:
        # Fall back to margin-based approximation
        # Assume roughly equal split of margin to O and D
        margin = season_stints['margin_diff'].values
        o_coef = solve_single_target(margin / 2)
        d_coef = solve_single_target(margin / 2)
    
    res = pd.DataFrame({
        'player_name': players,
        'o_rapm': o_coef,
        'd_rapm': d_coef,
        'rapm_net': o_coef + d_coef,
        'poss_total': np.abs(X).T @ w
    })
    return res


def compute_on_off_stats(season_stints: pd.DataFrame) -> pd.DataFrame:
    from collections import defaultdict
    game_stats = season_stints.groupby('gameId').agg({
        'poss': 'sum',
        'pts_scored_home': 'sum',
        'pts_allowed_home': 'sum'
    }).rename(columns={'poss': 'game_poss'})
    
    pg_stats = defaultdict(lambda: {'side': None, 'on_poss': 0.0, 'on_scored': 0.0, 'on_allowed': 0.0})
    
    for _, row in season_stints.iterrows():
        gid = row['gameId']
        poss = row['poss']
        h_scored = row.get('pts_scored_home', 0)
        a_scored = row.get('pts_allowed_home', 0)
        
        for p in row['home_players']:
            st = pg_stats[(gid, p)]
            st['side'] = 'home'
            st['on_poss'] += poss
            st['on_scored'] += h_scored
            st['on_allowed'] += a_scored
            
        for p in row['away_players']:
            st = pg_stats[(gid, p)]
            st['side'] = 'away'
            st['on_poss'] += poss
            st['on_scored'] += a_scored
            st['on_allowed'] += h_scored
            
    player_season = defaultdict(lambda: {'on_poss': 0.0, 'on_scored': 0.0, 'on_allowed': 0.0,
                                         'off_poss': 0.0, 'off_scored': 0.0, 'off_allowed': 0.0})
                                         
    for (gid, p), st in pg_stats.items():
        if gid not in game_stats.index:
            continue
        g_poss = game_stats.at[gid, 'game_poss']
        if st['side'] == 'home':
            g_scored = game_stats.at[gid, 'pts_scored_home']
            g_allowed = game_stats.at[gid, 'pts_allowed_home']
        else:
            g_scored = game_stats.at[gid, 'pts_allowed_home']
            g_allowed = game_stats.at[gid, 'pts_scored_home']
            
        off_poss = max(0.0, g_poss - st['on_poss'])
        off_scored = max(0.0, g_scored - st['on_scored'])
        off_allowed = max(0.0, g_allowed - st['on_allowed'])
        
        pst = player_season[p]
        pst['on_poss'] += st['on_poss']
        pst['on_scored'] += st['on_scored']
        pst['on_allowed'] += st['on_allowed']
        pst['off_poss'] += off_poss
        pst['off_scored'] += off_scored
        pst['off_allowed'] += off_allowed
        
    rows = []
    for p, pst in player_season.items():
        on_ortg = (pst['on_scored'] / pst['on_poss'] * 100.0) if pst['on_poss'] > 0 else np.nan
        on_drtg = (pst['on_allowed'] / pst['on_poss'] * 100.0) if pst['on_poss'] > 0 else np.nan
        off_ortg = (pst['off_scored'] / pst['off_poss'] * 100.0) if pst['off_poss'] > 0 else np.nan
        off_drtg = (pst['off_allowed'] / pst['off_poss'] * 100.0) if pst['off_poss'] > 0 else np.nan
        
        rows.append({
            'player_name': p,
            'on_ortg': on_ortg,
            'on_drtg': on_drtg,
            'on_net_rating': on_ortg - on_drtg if (pd.notna(on_ortg) and pd.notna(on_drtg)) else np.nan,
            'off_ortg': off_ortg,
            'off_drtg': off_drtg,
            'off_net_rating': off_ortg - off_drtg if (pd.notna(off_ortg) and pd.notna(off_drtg)) else np.nan,
        })
        
    return pd.DataFrame(rows)


def _parse_season_list(raw: str) -> Optional[List[int]]:
    if raw is None or str(raw).strip() == "":
        return None
    out: List[int] = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return sorted(set(out))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Calculate historical RAPM with split-quality diagnostics and hard gates.")
    p.add_argument("--input-parquet", default=INPUT_PARQUET)
    p.add_argument("--output-csv", default=OUTPUT_CSV)
    p.add_argument("--diagnostics-csv", default=OUTPUT_DIAG_CSV)
    p.add_argument("--lineup-season-audit-csv", default=LINEUP_SEASON_AUDIT_CSV)
    p.add_argument("--require-lineup-gate", action="store_true", default=True)
    p.add_argument("--no-require-lineup-gate", dest="require_lineup_gate", action="store_false")
    p.add_argument("--lambda-val", type=float, default=LAMBDA)
    p.add_argument("--diagnostics-only", action="store_true")
    p.add_argument("--strict-gates", action="store_true", default=True)
    p.add_argument("--no-strict-gates", dest="strict_gates", action="store_false")
    p.add_argument("--include-seasons", type=str, default="")
    p.add_argument("--exclude-seasons", type=str, default="")
    p.add_argument("--min-stints", type=int, default=DEFAULT_MIN_STINTS)
    p.add_argument("--min-valid-5v5-rate", type=float, default=DEFAULT_MIN_VALID_5V5_RATE)
    p.add_argument("--max-unresolved-home-rate", type=float, default=DEFAULT_MAX_UNRESOLVED_HOME_RATE)
    p.add_argument("--max-parse-fail-rate", type=float, default=DEFAULT_MAX_PARSE_FAIL_RATE)
    p.add_argument("--min-unique-players", type=int, default=DEFAULT_MIN_UNIQUE_PLAYERS)
    return p


def _split_players_with_quality(on_floor_json: str, home_name: str):
    try:
        data = json.loads(on_floor_json)
    except Exception:
        return [], [], False, 0

    if not isinstance(data, list):
        return [], [], False, 0

    home_norm = _norm_team_name(home_name)
    h_players: List[str] = []
    a_players: List[str] = []
    for p in data:
        if not isinstance(p, dict):
            continue
        player_name = p.get('name')
        if not isinstance(player_name, str) or player_name.strip() == "":
            continue
        team_name = p.get('team', '')
        t_norm = _norm_team_name(team_name)
        is_home = False
        if home_norm and t_norm:
            if t_norm == home_norm or t_norm in home_norm or home_norm in t_norm:
                is_home = True
        elif isinstance(team_name, str) and isinstance(home_name, str) and team_name == home_name:
            is_home = True
        if is_home:
            h_players.append(player_name)
        else:
            a_players.append(player_name)

    # de-dupe but preserve order
    h_players = list(dict.fromkeys(h_players))
    a_players = list(dict.fromkeys(a_players))
    return h_players, a_players, True, len(data)


def _canonical_lineup_hash(on_floor_json: str) -> int:
    """
    Build an order-invariant lineup hash from onFloor JSON.
    Raw JSON ordering can jitter between events even when the 10 players are unchanged.
    """
    try:
        data = json.loads(on_floor_json)
    except Exception:
        return hash(on_floor_json)
    if not isinstance(data, list):
        return hash(on_floor_json)
    tokens: List[str] = []
    for p in data:
        if not isinstance(p, dict):
            continue
        name = p.get("name", "")
        team = p.get("team", "")
        player_id = p.get("id", p.get("athleteId", ""))
        token = f"{_norm_team_name(str(team))}|{str(player_id).strip()}|{str(name).strip().upper()}"
        tokens.append(token)
    tokens.sort()
    return hash("::".join(tokens))


def _season_gate_row(
    season: int,
    metrics: Dict[str, float],
    include_override: Optional[List[int]],
    exclude_override: Optional[List[int]],
    min_stints: int,
    min_valid_5v5_rate: float,
    max_unresolved_home_rate: float,
    max_parse_fail_rate: float,
    min_unique_players: int,
) -> Dict[str, object]:
    reasons: List[str] = []
    forced_exclude = exclude_override is not None and season in exclude_override

    if metrics["n_stints"] < min_stints:
        reasons.append(f"n_stints<{min_stints}")
    if metrics["valid_5v5_rate"] < min_valid_5v5_rate:
        reasons.append(f"valid_5v5_rate<{min_valid_5v5_rate:.2f}")
    if metrics["unresolved_home_rate"] > max_unresolved_home_rate:
        reasons.append(f"unresolved_home_rate>{max_unresolved_home_rate:.2f}")
    if metrics["parse_fail_rate"] > max_parse_fail_rate:
        reasons.append(f"parse_fail_rate>{max_parse_fail_rate:.2f}")
    if metrics["unique_players_5v5"] < min_unique_players:
        reasons.append(f"unique_players_5v5<{min_unique_players}")

    gate_pass = len(reasons) == 0
    if forced_exclude:
        gate_pass = False
        reasons = ["forced_exclude"]

    return {
        "season": int(season),
        **metrics,
        "gate_pass": bool(gate_pass),
        "gate_reason": ";".join(reasons) if reasons else "pass",
        "forced_include": False,
        "forced_exclude": bool(forced_exclude),
    }


def _norm_team_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = re.sub(r"[^A-Za-z0-9]+", "", name).upper().strip()
    return s


def _parse_home_from_score_header(play_text: str) -> str:
    if not isinstance(play_text, str):
        return ""
    # Header format: "Time | HOME TEAM | Score | AWAY TEAM" or other variations.
    parts = [p.strip() for p in play_text.split("|")]
    
    # Filter out known non-team labels
    teams = [p for p in parts if p.lower() not in ["", "time", "score"]]
    
    # If we found at least one team, return the first one as our guess for the reference team
    if len(teams) > 0:
        return teams[0]
    return ""


def main():
    args = _build_parser().parse_args()
    include_override = _parse_season_list(args.include_seasons)
    exclude_override = _parse_season_list(args.exclude_seasons)

    if not os.path.exists(args.input_parquet):
        logger.error(f"Input file not found: {args.input_parquet}")
        return

    logger.info(f"Loading {args.input_parquet}...")
    df = pd.read_parquet(args.input_parquet)
    if "season" not in df.columns:
        logger.error("Input parquet missing required 'season' column.")
        return
    if include_override:
        df = df[df["season"].isin(include_override)].copy()
    if exclude_override:
        df = df[~df["season"].isin(exclude_override)].copy()
    logger.info("Rows after include/exclude filtering: %d", len(df))
    if df.empty:
        logger.error("No rows remain after include/exclude filtering.")
        return

    # Pre-RAPM season lineup gate (from reconstruction audit).
    lineup_gate_by_season: Dict[int, bool] = {}
    lineup_gate_reason: Dict[int, str] = {}
    if args.require_lineup_gate:
        if not os.path.exists(args.lineup_season_audit_csv):
            logger.error("Required lineup season audit missing: %s", args.lineup_season_audit_csv)
            return
        season_audit = pd.read_csv(args.lineup_season_audit_csv)
        if "season" not in season_audit.columns or "gate_pass" not in season_audit.columns:
            logger.error("Lineup season audit missing required columns (season, gate_pass).")
            return
        for _, r in season_audit.iterrows():
            season = int(r["season"])
            lineup_gate_by_season[season] = bool(r["gate_pass"])
            reason_bits = []
            if "pct_rows_len10" in season_audit.columns and pd.notna(r.get("pct_rows_len10", np.nan)):
                reason_bits.append(f"pct_rows_len10={float(r['pct_rows_len10']):.3f}")
            if "pct_rows_placeholder" in season_audit.columns and pd.notna(r.get("pct_rows_placeholder", np.nan)):
                reason_bits.append(f"pct_rows_placeholder={float(r['pct_rows_placeholder']):.3f}")
            if "pct_games_pass" in season_audit.columns and pd.notna(r.get("pct_games_pass", np.nan)):
                reason_bits.append(f"pct_games_pass={float(r['pct_games_pass']):.3f}")
            if "avg_unique_players_game" in season_audit.columns and pd.notna(r.get("avg_unique_players_game", np.nan)):
                reason_bits.append(f"avg_unique_players_game={float(r['avg_unique_players_game']):.2f}")
            lineup_gate_reason[season] = ";".join(reason_bits) if reason_bits else "lineup_gate_eval"
    
    # 1. Period Detection & Clock Normalization
    logger.info("Parsing clocks and detecting periods...")
    df['seconds_rem'] = df['clock'].apply(parse_clock)
    
    # Logic: If clock increases by > 60s from prev row in same game, it's a new period
    df['clock_diff'] = df.groupby('gameSourceId')['seconds_rem'].diff()
    df['new_period'] = (df['clock_diff'] > 60).fillna(False)
    df['period'] = df.groupby('gameSourceId')['new_period'].cumsum() + 1
    
    def abs_elapsed(period, sec_rem):
        if period == 1: return 1200 - sec_rem
        return 1200 + (period - 1) * 1200 + (1200 - sec_rem) # Simple 20m periods logic
    
    df['abs_time'] = df.apply(lambda r: abs_elapsed(r['period'], r['seconds_rem']), axis=1)
    
    # 2. Stint Building
    logger.info("Detecting stints...")
    # onFloor can reorder players while lineup is unchanged; canonical hash prevents false stint breaks.
    df['lineup_hash'] = df['onFloor'].apply(_canonical_lineup_hash)
    
    df['stint_change'] = (
        (df['gameSourceId'] != df['gameSourceId'].shift()) |
        (df['lineup_hash'] != df['lineup_hash'].shift())
    )
    df['stint_id'] = df['stint_change'].cumsum()
    
    # Aggregate Stints
    stint_agg = df.groupby('stint_id').agg({
        'gameSourceId': 'first',
        'season': 'first',
        'abs_time': ['min', 'max'],
        'homeScore': ['first', 'last'],
        'awayScore': ['first', 'last'],
        'onFloor': 'first'
    })
    stint_agg.columns = ['gameId', 'season', 't0', 't1', 'h0', 'h1', 'a0', 'a1', 'onFloor']
    
    # Calculate Stint Stats
    stint_agg['duration'] = stint_agg['t1'] - stint_agg['t0']
    stint_agg['margin_diff'] = (stint_agg['h1'] - stint_agg['h0']) - (stint_agg['a1'] - stint_agg['a0'])
    stint_agg['pts_scored_home'] = stint_agg['h1'] - stint_agg['h0']
    stint_agg['pts_allowed_home'] = stint_agg['a1'] - stint_agg['a0']
    
    # Possessions Estimate: (duration / 2400) * 68 (avg pace)
    stint_agg['poss'] = (stint_agg['duration'] / 2400.0) * 68.0
    
    # Compute margin at stint start (for leverage and rubber band)
    stint_agg['margin_start'] = stint_agg['h0'] - stint_agg['a0']
    
    # Compute seconds remaining at stint start
    # t0 is absolute elapsed time, so seconds_remaining = 2400 - t0
    stint_agg['seconds_remaining'] = 2400 - stint_agg['t0']
    stint_agg['seconds_remaining'] = stint_agg['seconds_remaining'].clip(lower=0)
    
    # Compute leverage index for each stint
    logger.info("Computing leverage indices...")
    stint_agg['leverage'] = stint_agg.apply(
        lambda row: compute_leverage_index(
            int(row['margin_start']), 
            int(row['seconds_remaining'])
        ), axis=1
    )
    stint_agg['leverage_bucket'] = stint_agg['leverage'].apply(classify_leverage)
    
    # Filter valid stints
    stints = stint_agg[stint_agg['duration'] >= 0].copy()
    
    logger.info("Partitioning players into Home/Away...")
    score_rows = df[df['playText'].str.contains(r"\|\s*Score\s*\|", na=False)].copy()
    score_rows = score_rows.sort_values(['gameSourceId', 'abs_time'])
    header_home_map = (
        score_rows.groupby('gameSourceId')['playText']
        .first()
        .apply(_parse_home_from_score_header)
        .to_dict()
    )

    # Build candidate team labels from onFloor itself (per game).
    game_team_candidates: Dict[str, List[str]] = {}
    for gid, gdf in df[['gameSourceId', 'onFloor']].dropna().groupby('gameSourceId'):
        team_counts: Dict[str, int] = {}
        for onf in gdf['onFloor'].head(200):
            try:
                players = json.loads(onf)
            except Exception:
                continue
            for p in players:
                t = p.get('team')
                if isinstance(t, str) and t.strip():
                    team_counts[t.strip()] = team_counts.get(t.strip(), 0) + 1
        if team_counts:
            ordered = sorted(team_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            game_team_candidates[gid] = [t for t, _ in ordered[:2]]

    def resolve_home_team_name(game_id: str) -> str:
        header_home = header_home_map.get(game_id, "")
        candidates = game_team_candidates.get(game_id, [])
        if not candidates:
            return header_home
        if not header_home:
            return candidates[0]

        h_norm = _norm_team_name(header_home)
        best = candidates[0]
        best_score = -1.0
        for cand in candidates:
            c_norm = _norm_team_name(cand)
            if not c_norm:
                continue
            if c_norm == h_norm:
                score = 3.0
            elif c_norm in h_norm or h_norm in c_norm:
                score = 2.0
            else:
                score = difflib.SequenceMatcher(None, c_norm, h_norm).ratio()
            if score > best_score:
                best = cand
                best_score = score
        return best

    resolved_home_team = {gid: resolve_home_team_name(gid) for gid in stints['gameId'].dropna().unique()}

    # Apply split
    # This might be slow on 1.5M rows, but we only have ~100k stints.
    logger.info(f"Processing {len(stints)} stints...")
    
    h_p_list = []
    a_p_list = []
    unresolved_home = 0
    parse_fail = 0
    split_valid_5v5 = 0
    split_nonempty = 0
    quality_rows: List[Dict[str, object]] = []
    for i, row in stints.iterrows():
        home_name = resolved_home_team.get(row['gameId'], "")
        if not home_name:
            unresolved_home += 1
        hp, ap, parse_ok, roster_count = _split_players_with_quality(row['onFloor'], home_name)
        if not parse_ok:
            parse_fail += 1
        if len(hp) > 0 and len(ap) > 0:
            split_nonempty += 1
        if len(hp) == 5 and len(ap) == 5:
            split_valid_5v5 += 1
        quality_rows.append({
            "season": int(row["season"]),
            "gameId": row["gameId"],
            "parse_ok": bool(parse_ok),
            "home_resolved": bool(home_name),
            "home_players_n": int(len(hp)),
            "away_players_n": int(len(ap)),
            "is_valid_5v5": bool(len(hp) == 5 and len(ap) == 5),
            "is_nonempty_split": bool(len(hp) > 0 and len(ap) > 0),
            "duration": float(row["duration"]),
            "poss": float(row["poss"]),
            "onfloor_roster_count": int(roster_count),
        })
        h_p_list.append(hp)
        a_p_list.append(ap)
    
    stints['home_players'] = h_p_list
    stints['away_players'] = a_p_list
    if unresolved_home > 0:
        logger.warning("Unresolved home team for %d stints; fallback split may be noisy.", unresolved_home)
    if parse_fail > 0:
        logger.warning("Failed to parse onFloor for %d stints.", parse_fail)

    # Season-level split quality diagnostics + hard gates.
    quality_df = pd.DataFrame(quality_rows)
    quality_df = quality_df[quality_df["poss"] > 0.1].copy()
    diagnostics: List[Dict[str, object]] = []
    for season, q in quality_df.groupby("season"):
        unique_players_5v5: set = set()
        valid_rows = q[q["is_valid_5v5"]]
        season_rows = stints[stints["season"] == season]
        for _, r in season_rows.iterrows():
            if len(r["home_players"]) == 5 and len(r["away_players"]) == 5 and r["poss"] > 0.1:
                unique_players_5v5.update(r["home_players"])
                unique_players_5v5.update(r["away_players"])
        metrics = {
            "n_stints": int(len(q)),
            "valid_5v5_rate": float(q["is_valid_5v5"].mean()) if len(q) else 0.0,
            "nonempty_split_rate": float(q["is_nonempty_split"].mean()) if len(q) else 0.0,
            "unresolved_home_rate": float((~q["home_resolved"]).mean()) if len(q) else 1.0,
            "parse_fail_rate": float((~q["parse_ok"]).mean()) if len(q) else 1.0,
            "avg_home_players_n": float(q["home_players_n"].mean()) if len(q) else 0.0,
            "avg_away_players_n": float(q["away_players_n"].mean()) if len(q) else 0.0,
            "unique_players_5v5": int(len(unique_players_5v5)),
            "n_valid_5v5_stints": int(len(valid_rows)),
        }
        row = _season_gate_row(
            season=int(season),
            metrics=metrics,
            include_override=include_override,
            exclude_override=exclude_override,
            min_stints=args.min_stints,
            min_valid_5v5_rate=args.min_valid_5v5_rate,
            max_unresolved_home_rate=args.max_unresolved_home_rate,
            max_parse_fail_rate=args.max_parse_fail_rate,
            min_unique_players=args.min_unique_players,
        )
        lineup_gate_pass = True
        lineup_reason = "lineup_gate_not_required"
        if args.require_lineup_gate:
            lineup_gate_pass = bool(lineup_gate_by_season.get(int(season), False))
            lineup_reason = lineup_gate_reason.get(int(season), "season_missing_in_lineup_audit")
            if not lineup_gate_pass:
                row["gate_pass"] = False
                if row["gate_reason"] == "pass":
                    row["gate_reason"] = f"lineup_gate_fail:{lineup_reason}"
                else:
                    row["gate_reason"] = f"{row['gate_reason']};lineup_gate_fail:{lineup_reason}"
        row["lineup_gate_pass"] = bool(lineup_gate_pass)
        row["lineup_gate_reason"] = lineup_reason
        diagnostics.append(row)

    diag_df = pd.DataFrame(diagnostics).sort_values("season")
    os.makedirs(os.path.dirname(args.diagnostics_csv), exist_ok=True)
    diag_df.to_csv(args.diagnostics_csv, index=False)
    logger.info("Wrote split-quality diagnostics to %s", args.diagnostics_csv)
    if not diag_df.empty:
        logger.info(
            "Season gate summary: %s",
            diag_df[["season", "gate_pass", "gate_reason", "valid_5v5_rate", "unresolved_home_rate", "unique_players_5v5"]]
            .to_dict("records"),
        )
    
    # Log leverage distribution
    logger.info(f"Leverage distribution: {stints['leverage_bucket'].value_counts().to_dict()}")
    
    # 3. Solve per Season with multiple RAPM variants
    all_rapm = []
    unique_seasons = sorted(stints['season'].unique())
    logger.info(f"Target seasons found: {unique_seasons}")
    included_by_gate: Dict[int, bool] = {}
    if not diag_df.empty:
        included_by_gate = {int(r["season"]): bool(r["gate_pass"]) for _, r in diag_df.iterrows()}
    
    for season in unique_seasons:
        if args.strict_gates and season in included_by_gate and not included_by_gate[season]:
            logger.warning("Skipping season %s due to failing split-quality gate.", season)
            continue
        if args.diagnostics_only:
            logger.info("Diagnostics-only mode enabled; skipping RAPM solve for season %s.", season)
            continue
        logger.info(f"Solving RApM variants for {season}...")
        season_stints = stints[
            (stints['season'] == season)
            & (stints['poss'] > 0.1)
            & (stints['home_players'].apply(len) == 5)
            & (stints['away_players'].apply(len) == 5)
        ]
        if len(season_stints) == 0:
            logger.warning(f"No data for {season}")
            continue
        
        # 1. Standard RAPM (possession-weighted)
        res_standard = solve_rapm(season_stints, args.lambda_val, use_leverage_weights=False)
        res_standard = res_standard.rename(columns={'rapm': 'rapm_standard'})
        
        # 2. Leverage-weighted RAPM
        res_leverage = solve_rapm(season_stints, args.lambda_val, use_leverage_weights=True)
        res_leverage = res_leverage[['player_name', 'rapm']].rename(columns={'rapm': 'rapm_leverage_weighted'})
        
        # 3. High-leverage only RAPM (filter to high/very_high stints)
        high_lev_stints = season_stints[season_stints['leverage_bucket'].isin(['high', 'very_high'])]
        if len(high_lev_stints) > 100:  # Need minimum sample
            res_high_lev = solve_rapm(high_lev_stints, args.lambda_val)
            res_high_lev = res_high_lev[['player_name', 'rapm']].rename(columns={'rapm': 'rapm_high_leverage'})
        else:
            res_high_lev = pd.DataFrame({'player_name': res_standard['player_name'], 'rapm_high_leverage': np.nan})
        
        # 4. Non-garbage RAPM (exclude garbage time)
        non_garbage_stints = season_stints[season_stints['leverage_bucket'] != 'garbage']
        res_non_garbage = solve_rapm(non_garbage_stints, args.lambda_val)
        res_non_garbage = res_non_garbage[['player_name', 'rapm']].rename(columns={'rapm': 'rapm_non_garbage'})
        
        # 5. O/D Split RAPM
        res_split = solve_rapm_split(season_stints, args.lambda_val, use_leverage_weights=False)
        res_split = res_split[['player_name', 'o_rapm', 'd_rapm']]
        
        # 6. Rubber-band adjusted RAPM
        res_rubber = solve_rapm(season_stints, args.lambda_val, use_rubber_band=True)
        res_rubber = res_rubber[['player_name', 'rapm']].rename(columns={'rapm': 'rapm_rubber_adj'})
        
        # 7. On/Off Stats
        res_on_off = compute_on_off_stats(season_stints)
        
        # Merge all variants
        res = res_standard.merge(res_leverage, on='player_name', how='left')
        res = res.merge(res_high_lev, on='player_name', how='left')
        res = res.merge(res_non_garbage, on='player_name', how='left')
        res = res.merge(res_split, on='player_name', how='left')
        res = res.merge(res_rubber, on='player_name', how='left')
        if not res_on_off.empty:
            res = res.merge(res_on_off, on='player_name', how='left')
        
        res['season'] = season
        
        all_rapm.append(res)
        
    if all_rapm:
        final_df = pd.concat(all_rapm)
        final_df.to_csv(args.output_csv, index=False)
        logger.info(f"Results saved to {args.output_csv}")
        
        # Summary statistics
        logger.info(f"Total players: {len(final_df)}")
        logger.info(f"Columns: {list(final_df.columns)}")
        
        # Display top 10 for most recent season
        latest_season = final_df['season'].max()
        print(f"\n--- TOP 10 RApM {latest_season} (Standard) ---")
        print(final_df[final_df['season'] == latest_season].sort_values('rapm_standard', ascending=False).head(10)[
            ['player_name', 'rapm_standard', 'rapm_leverage_weighted', 'o_rapm', 'd_rapm', 'poss_total']
        ])
        
        # Correlation analysis between RAPM variants
        print("\n--- RAPM Variant Correlations ---")
        rapm_cols = ['rapm_standard', 'rapm_leverage_weighted', 'rapm_non_garbage', 'rapm_rubber_adj']
        print(final_df[rapm_cols].corr().round(3))
    else:
        logger.error("No results generated.")

if __name__ == "__main__":
    main()
