import pandas as pd
import numpy as np

PARQUET_PATH = "data/fact_play_historical_combined.parquet"

def parse_clock(clock_str):
    try:
        parts = clock_str.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 0
    except:
        return 0

def main():
    print(f"Reading {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)
    
    # Add seconds column
    # Vectorized parse is faster? or apply
    # Most clocks are MM:SS. Some might be empty.
    print("Parsing clocks...")
    df['seconds_remaining'] = df['clock'].astype(str).apply(parse_clock)
    
    # Sort just in case (though we rely on file order) - actually we want to TEST file order.
    # So we group by game and Iterate.
    
    games = df.groupby('gameSourceId')
    total_games = len(games)
    
    perfect_time_order = 0
    perfect_score_order = 0
    
    print(f"Auditing {total_games} games...")
    
    for game_id, group in games:
        # Check Time Monotonicity (Should generally decrease)
        # Note: NCAA 2 halves. 20:00 -> 00:00, then 20:00 -> 00:00.
        # So it's not strictly monotonic for the WHOLE game.
        # But within a half it should be.
        # We don't have 'period' parsed yet in this file? 
        # Actually checking 'seconds' diff. If diff is positive (time went up), it MIGHT be a new period.
        # Let's look for "impossible" jumps, e.g. 15:00 -> 18:00 without a "Half" marker?
        
        # Simpler check: Score Monotonicity. Score can never go down.
        # (Unless correction).
        
        h_scores = group['homeScore'].values
        a_scores = group['awayScore'].values
        
        # Check diffs
        h_diff = np.diff(h_scores)
        a_diff = np.diff(a_scores)
        
        # Valid if all diffs >= 0
        if np.all(h_diff >= 0) and np.all(a_diff >= 0):
            perfect_score_order += 1
        elif perfect_score_order < 3: # Print first 3 failures
            print(f"\nExample Failure (Game {game_id}):")
            # Find index where it drops
            bad_h = np.where(h_diff < 0)[0]
            bad_a = np.where(a_diff < 0)[0]
            
            if len(bad_h) > 0:
                idx = bad_h[0]
                print(f"  Home Score Drop at Row {idx}: {h_scores[idx]} -> {h_scores[idx+1]}")
                print(f"  Rows: {group.iloc[idx]['playText']}  -->  {group.iloc[idx+1]['playText']}")
                
            if len(bad_a) > 0:
                idx = bad_a[0]
                print(f"  Away Score Drop at Row {idx}: {a_scores[idx]} -> {a_scores[idx+1]}")
                print(f"  Rows: {group.iloc[idx]['playText']}  -->  {group.iloc[idx+1]['playText']}")
            
        # Check Time: Count how many times time goes UP.
        # Should happen ~1-2 times (Halftime, OT).
        time_vals = group['seconds_remaining'].values
        time_diff = np.diff(time_vals) # Next - Curr.
        # If time decreases, diff is NEGATIVE.
        # If time goes UP (new half), diff is POSITIVE.
        
        jumps = np.sum(time_diff > 0)
        if jumps <= 4: # Allow for 1st half, 2nd half, maybe OTs. 
            perfect_time_order += 1
            
    print("-" * 30)
    print(f"Games Audited: {total_games}")
    print(f"Perfect Score Linearity: {perfect_score_order} ({perfect_score_order/total_games:.1%})")
    print(f"Valid Time Flow (<=4 jumps): {perfect_time_order} ({perfect_time_order/total_games:.1%})")
    print("-" * 30)
    
    if perfect_score_order < total_games:
        print("Note: Score drops usually imply data corrections or parsing glitches in raw text.")
        
if __name__ == "__main__":
    main()
