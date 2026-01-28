from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
import re, math

def safe_get(d: Dict[str,Any], *keys, default=None):
    cur = d
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def normalize_bool(x):
    if x is None: return None
    if isinstance(x, bool): return x
    if isinstance(x, str): return x.strip().lower() in ("true","1","yes","y")
    return bool(x)

def wp_bin(wp: float, bins=(0.05,0.2,0.8,0.95)):
    # returns 0..len(bins)
    if wp is None or (isinstance(wp,float) and math.isnan(wp)): return None
    for i,b in enumerate(bins):
        if wp < b: return i
    return len(bins)

def score_state(homeScore:int, awayScore:int, homePossession:bool, team_is_home:bool):
    # score from team perspective
    if homeScore is None or awayScore is None: return None
    teamScore = homeScore if team_is_home else awayScore
    oppScore  = awayScore if team_is_home else homeScore
    return teamScore - oppScore

def is_high_leverage(seconds_remaining:int, period:int, wp: float):
    # default: final 5 minutes in regulation + WP in [0.2,0.8]
    if seconds_remaining is None or period is None or wp is None: return False
    if period < 2: # college halves; if quarters adjust later
        return False
    return (seconds_remaining <= 300) and (0.2 <= wp <= 0.8)

def is_garbage(seconds_remaining:int, period:int, margin:int):
    if seconds_remaining is None or period is None or margin is None: return False
    # simple: final 4 min and margin >= 15
    return (seconds_remaining <= 240) and (abs(margin) >= 15)

def shot_family_from_play(playType: str, shotInfo: Optional[dict]=None):
    # prefer shotInfo.range when present, then playType string
    pt = (playType or "").lower()
    if "dunk" in pt: return "dunk"
    if "tip" in pt: return "tipin"
    if "layup" in pt: return "layup"
    # treat remaining as jumpers
    if ("3" in pt and "pt" in pt) or "three" in pt:
        return "three_pt_jumper"
    return "two_pt_jumper"

def zone_from_shotinfo(shotInfo: Optional[dict]):
    # Flexible: if API returns categorical location string, use it.
    # If returns x/y, map to coarse zones.
    if not shotInfo: return None
    loc = shotInfo.get("location")
    if loc is None:
        return None
    if isinstance(loc, str):
        return loc.strip().lower()
    if isinstance(loc, dict):
        x = loc.get("x")
        y = loc.get("y")
        if x is None or y is None:
            # maybe fields named differently
            x = loc.get("X") or loc.get("cx")
            y = loc.get("Y") or loc.get("cy")
        try:
            x = float(x); y = float(y)
        except Exception:
            return None
        # VERY rough half-court mapping; tune once you inspect coordinate system
        r = math.hypot(x, y)
        if r < 4:
            return "restricted"
        if r < 8:
            return "paint_nonra"
        if r < 20:
            return "midrange"
        # threes
        if abs(x) > 20:
            return "corner3"
        return "above_break3"
    return None
