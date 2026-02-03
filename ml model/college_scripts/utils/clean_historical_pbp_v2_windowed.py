"""
Windowed Activity Ghost Fill (Improved Version)
===============================================

This is an improved version of the Ghost Fill algorithm that uses "windowed activity"
instead of "global most active" to reduce errors in blowout situations.

Problem with Current Approach:
- Current `ensure_five()` uses global game-total activity (roster_counter)
- In blowouts, starters sit in 2nd half, bench plays more
- At minute 35, we might incorrectly fill with a starter who hasn't played since minute 20

Solution: Windowed Activity
- Track activity in rolling windows (e.g., last 10 minutes of game time)
- When filling a ghost at time T, prioritize players active in the window [T-10, T]
- This better reflects substitution patterns (starters early, bench late in blowouts)

Author: cursor
Date: 2026-01-29
Based on: antigravity's proposal in ml_model_master_plan.md Section 1.2
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
import json
import re


def normalize_name(n):
    """
    Normalize player name to match format used throughout the pipeline.
    
    Args:
        n: Raw name (e.g., "01 Jackson, Warren" or "Jackson, Warren")
    
    Returns:
        Normalized name (e.g., "JACKSON,WARREN")
    """
    n = re.sub(r'#\d+\s*', '', n)  # Remove #01
    n = re.sub(r'^\d+\s*', '', n)   # Remove leading numbers
    return n.strip().upper().replace(" ", "")


class WindowedActivityTracker:
    """
    Tracks player activity in rolling time windows.
    
    This allows us to prioritize players who were recently active when filling
    ghost lineups, rather than using global game-total activity.
    """
    
    def __init__(self, window_size_seconds: int = 600):
        """
        Args:
            window_size_seconds: Size of rolling window (default 600 = 10 minutes)
        """
        self.window_size = window_size_seconds
        # Track activity events: {player_name: [list of timestamps]}
        self.activity_events: Dict[str, List[int]] = defaultdict(list)
        # Track current game time (in seconds from start)
        self.current_time = 0
        # Track period transitions (clock resets)
        self.period_start_times: List[int] = [0]  # [0, 1200, 2400, ...]
        self.current_period = 1
    
    def add_activity(self, player_name: str, game_time_seconds: int):
        """
        Record that a player was active at a specific game time.
        
        Args:
            player_name: Normalized player name
            game_time_seconds: Game time in seconds (0 = start of game)
        """
        self.activity_events[player_name].append(game_time_seconds)
        self.current_time = game_time_seconds
    
    def detect_period_transition(self, clock_seconds: int):
        """
        Detect when clock resets (new period starts).
        
        Args:
            clock_seconds: Seconds remaining in current period
        """
        # If clock jumps up significantly, new period started
        # Example: Was at 0:05 (5s remaining), now at 20:00 (1200s remaining)
        # This means we're in a new period
        if clock_seconds > 1000:  # Clock reset (new period)
            self.current_period += 1
            # Calculate absolute game time
            # Period 1: 0-1200s, Period 2: 1200-2400s, etc.
            period_start = (self.current_period - 1) * 1200
            self.period_start_times.append(period_start)
    
    def get_windowed_activity(self, current_time: int) -> Counter:
        """
        Get activity counts for players in the rolling window [current_time - window, current_time].
        
        Args:
            current_time: Current game time in seconds
        
        Returns:
            Counter of {player_name: activity_count} for players active in window
        """
        window_start = max(0, current_time - self.window_size)
        window_activity = Counter()
        
        for player, timestamps in self.activity_events.items():
            # Count activities in window
            in_window = [t for t in timestamps if window_start <= t <= current_time]
            window_activity[player] = len(in_window)
        
        return window_activity
    
    def get_global_activity(self) -> Counter:
        """
        Get global game-total activity (fallback for early game when window is small).
        
        Returns:
            Counter of {player_name: total_activity_count}
        """
        return Counter({player: len(timestamps) for player, timestamps in self.activity_events.items()})


class WindowedGameSolver:
    """
    Improved game solver using windowed activity for ghost fill.
    
    This replaces the global activity approach in clean_historical_pbp_v2.py
    with a time-aware windowed approach.
    """
    
    def __init__(self, game_id, rows, h_team, a_team, season, window_size_seconds: int = 600):
        """
        Args:
            game_id: Game identifier
            rows: List of raw PBP text rows
            h_team: Home team identifier
            a_team: Away team identifier
            season: Season year
            window_size_seconds: Size of rolling window for activity tracking (default 600 = 10 min)
        """
        self.game_id = game_id
        self.rows = rows
        self.h_team = h_team
        self.a_team = a_team
        self.season = season
        self.window_size = window_size_seconds
        
        # Activity trackers for each team
        self.tracker_home = WindowedActivityTracker(window_size_seconds)
        self.tracker_away = WindowedActivityTracker(window_size_seconds)
        
        # Event tracking (same as original)
        self.events = []
        self.checkpoints = []
        self.on_floor_history = []
        
        # Current lineups
        self.curr_h = set()
        self.curr_a = set()
    
    def parse_rows(self):
        """
        Pass 1: Parse all rows to extract events, checkpoints, and activity.
        
        This is similar to the original, but we also track activity timestamps
        for windowed analysis.
        """
        for i, row in enumerate(self.rows):
            parts = [p.strip() for p in row.split("|")]
            if len(parts) < 4:
                self.events.append(None)
                continue
            
            h_evt, a_evt = parts[1], parts[3]
            
            # Parse clock to get game time
            clock_str = parts[0] if len(parts) > 0 else "00:00"
            clock_seconds = self._parse_clock(clock_str)
            
            # Detect period transitions
            if clock_seconds:
                self.tracker_home.detect_period_transition(clock_seconds)
                self.tracker_away.detect_period_transition(clock_seconds)
            
            # Calculate absolute game time (accounting for periods)
            # Period 1: 0-1200s, Period 2: 1200-2400s, etc.
            absolute_time = self._get_absolute_time(clock_seconds, i)
            
            row_evts = []
            
            def process_text(text, team_label, tracker):
                if not text:
                    return
                
                # Check for explicit lineup
                if "TEAM For" in text:
                    raw = text.split(":", 1)[-1]
                    ps = {normalize_name(x) for x in raw.split("#") if x.strip()}
                    self.checkpoints.append((i, team_label, ps, absolute_time))
                    for p in ps:
                        tracker.add_activity(p, absolute_time)
                    return
                
                # Check for substitutions
                if "Enters" in text:
                    p = normalize_name(text.split("Enters")[0])
                    tracker.add_activity(p, absolute_time)
                    row_evts.append({'type': 'IN', 'team': team_label, 'player': p, 'time': absolute_time})
                elif "Leaves" in text:
                    p = normalize_name(text.split("Leaves")[0])
                    tracker.add_activity(p, absolute_time)
                    row_evts.append({'type': 'OUT', 'team': team_label, 'player': p, 'time': absolute_time})
                else:
                    # Stat event (player was active)
                    first = text.split(" ")[0]
                    if "," in first:
                        p = normalize_name(first)
                        tracker.add_activity(p, absolute_time)
            
            process_text(h_evt, 'HOME', self.tracker_home)
            process_text(a_evt, 'AWAY', self.tracker_away)
            
            self.events.append(row_evts)
    
    def _parse_clock(self, clock_str: str) -> Optional[int]:
        """Parse clock string to seconds remaining in period."""
        if not clock_str or ":" not in clock_str:
            return None
        try:
            parts = clock_str.split(":")
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        except:
            return None
    
    def _get_absolute_time(self, clock_seconds: Optional[int], row_index: int) -> int:
        """
        Convert clock + row index to absolute game time.
        
        This is approximate - we use row index as a proxy for game progression
        when clock is missing.
        """
        if clock_seconds is None:
            # Fallback: estimate from row index (assumes ~1 play per 5 seconds)
            return row_index * 5
        
        # Use period start times to calculate absolute time
        period = self.tracker_home.current_period
        period_start = (period - 1) * 1200
        period_time = 1200 - clock_seconds  # Time elapsed in current period
        return period_start + period_time
    
    def solve_timeline(self):
        """
        Pass 2/3: Solve initial lineups and propagate forward.
        
        Uses windowed activity for ghost fill instead of global activity.
        """
        # Get initial lineups (same logic as original)
        def get_initial_lineup(team_label, tracker):
            cps = [c for c in self.checkpoints if c[1] == team_label]
            if cps:
                # Use first checkpoint
                idx, _, current, time = cps[0]
                starter_set = set(current)
                # Backpropagate from checkpoint
                for r in range(idx-1, -1, -1):
                    evts = self.events[r]
                    if not evts:
                        continue
                    for e in reversed(evts):
                        if e['team'] != team_label:
                            continue
                        if e['type'] == 'IN':
                            starter_set.discard(e['player'])
                        elif e['type'] == 'OUT':
                            starter_set.add(e['player'])
                return starter_set
            else:
                # Inference: use windowed activity at game start
                # At time 0, window is small, so fall back to global
                windowed = tracker.get_global_activity()
                starters = set([p for p, _ in windowed.most_common(5)])
                return starters
        
        self.curr_h = get_initial_lineup('HOME', self.tracker_home)
        self.curr_a = get_initial_lineup('AWAY', self.tracker_away)
        
        # Propagate forward with windowed ghost fill
        for i, evts in enumerate(self.events):
            # Get current game time for windowed activity
            clock_str = self.rows[i].split("|")[0] if "|" in self.rows[i] else "00:00"
            clock_seconds = self._parse_clock(clock_str)
            current_time = self._get_absolute_time(clock_seconds, i)
            
            # Windowed ghost fill BEFORE recording state
            self.ensure_five_windowed(
                self.curr_h, 
                self.tracker_home, 
                current_time,
                'HOME'
            )
            self.ensure_five_windowed(
                self.curr_a, 
                self.tracker_away, 
                current_time,
                'AWAY'
            )
            
            # Record state
            self.on_floor_history.append((set(self.curr_h), set(self.curr_a)))
            
            # Apply substitutions
            if evts:
                for e in evts:
                    tgt = self.curr_h if e['team'] == 'HOME' else self.curr_a
                    if e['type'] == 'IN':
                        tgt.add(e['player'])
                    elif e['type'] == 'OUT':
                        tgt.discard(e['player'])
    
    def ensure_five_windowed(
        self, 
        lineup_set: Set[str], 
        tracker: WindowedActivityTracker, 
        current_time: int,
        team_label: str
    ):
        """
        Ensure exactly 5 players on floor using windowed activity.
        
        This is the improved version that uses rolling window activity
        instead of global game-total activity.
        
        Args:
            lineup_set: Current set of players on floor
            tracker: Activity tracker for this team
            current_time: Current absolute game time
            team_label: 'HOME' or 'AWAY' (for logging)
        """
        if len(lineup_set) == 5:
            return
        
        # If > 5, remove least active in window
        if len(lineup_set) > 5:
            windowed_activity = tracker.get_windowed_activity(current_time)
            # Sort by windowed activity (least active first)
            sorted_p = sorted(
                list(lineup_set), 
                key=lambda x: windowed_activity.get(x, 0)
            )
            # Remove least active players
            to_remove = sorted_p[:len(lineup_set) - 5]
            for p in to_remove:
                lineup_set.discard(p)
            return
        
        # If < 5, ADD most active in window (not already in set)
        if len(lineup_set) < 5:
            windowed_activity = tracker.get_windowed_activity(current_time)
            
            # If window is too small (early game), fall back to global
            if current_time < self.window_size:
                windowed_activity = tracker.get_global_activity()
            
            # Candidates: In roster, not in set, sorted by windowed activity
            candidates = [
                (p, count) 
                for p, count in windowed_activity.most_common() 
                if p not in lineup_set
            ]
            
            # Add most active candidates until we have 5
            for player, _ in candidates:
                lineup_set.add(player)
                if len(lineup_set) == 5:
                    break
    
    def export_rows(self):
        """
        Export rows with reconstructed lineups (same format as original).
        """
        output = []
        for i, row_text in enumerate(self.rows):
            h_set, a_set = self.on_floor_history[i]
            
            # Format onFloor JSON
            on_floor = []
            for p in h_set:
                on_floor.append({'id': None, 'name': p, 'team': self.h_team})
            for p in a_set:
                on_floor.append({'id': None, 'name': p, 'team': self.a_team})
            
            # Parse clock/score
            parts = [p.strip() for p in row_text.split("|")]
            clock = parts[0] if len(parts) > 0 else "00:00"
            score = parts[2] if len(parts) > 2 else "0-0"
            if "-" in score:
                hs, as_ = score.split("-")
            else:
                hs, as_ = 0, 0
            
            output.append({
                "gameSourceId": str(self.game_id),
                "season": self.season,
                "clock": clock,
                "playText": row_text,
                "homeScore": hs,
                "awayScore": as_,
                "onFloor": json.dumps(on_floor)
            })
        return output


# Example usage function (to be integrated into clean_historical_pbp_v2.py)
def process_game_windowed(contest_id, df, team_df, window_size_seconds: int = 600):
    """
    Process a single game using windowed activity ghost fill.
    
    This is a drop-in replacement for the game processing logic in
    clean_historical_pbp_v2.py that uses windowed activity instead of global.
    
    Args:
        contest_id: Game identifier
        df: DataFrame with raw PBP rows
        team_df: Team mapping DataFrame
        window_size_seconds: Size of rolling window (default 600 = 10 minutes)
    
    Returns:
        List of output rows (same format as original)
    """
    # Parse header to get teams (same as original)
    h_raw, a_raw = None, None
    for text in df['raw_text']:
        if "| Score |" in text:
            parts = [p.strip() for p in text.split("|")]
            if len(parts) >= 4:
                h_raw, a_raw = parts[1], parts[3]
                break
    
    if not h_raw:
        return None
    
    # Match teams (same as original - would use fuzzy matching)
    # For now, simplified
    h_team = h_raw
    a_team = a_raw
    
    rows = df['raw_text'].tolist()
    
    # Create solver with windowed activity
    solver = WindowedGameSolver(
        contest_id, 
        rows, 
        h_team, 
        a_team, 
        season=2015,  # Extract from filename
        window_size_seconds=window_size_seconds
    )
    
    # Solve
    solver.parse_rows()
    solver.solve_timeline()
    
    # Export
    return solver.export_rows()
