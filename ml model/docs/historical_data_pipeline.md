# Historical PBP Reconstruction Pipeline

This document outlines the logic used to bridge the gap between "Lossy" historical NCAA Play-by-Play (2010-2023) and "Rich" modern CBD data (2024+).

## 1. The Challenge of "Ghost Players"
In raw PBP, players who play long stretches without recording a stat (shot, foul, rebound) or subbing do not appear in the text timeline. Sequential reconstruction (tracking IN/OUT) results in lineups with < 5 players.

## 2. The Holistic Solver Algorithm
To ensure **5-on-Floor Consistency**, we use a multi-pass constraint solver:

### Pass 1: Global Game Sweep
- **Roster Discovery**: Scan every row in the game. Extract every name mentioned in stats, subs, or lineup checks. 
- **Bench Logic**: Mark a player as "Known Bench" for a time range [T1, T2] if we see `P1 Leaves Game` at T1 and `P1 Enters Game` at T2.

### Pass 2: Checkpoint Anchoring
- Locate `TEAM For [Team]: #XX Player...` blocks. These are "Absolute Truth" timestamps.
- Use these to initialize the `on_floor` state at specific indices.

### Pass 3: Bidirectional Propagation
- **Backwards**: From the first Absolute Truth block, walk back to 0:00, reversing substitutions.
- **Forwards**: From the first/last checkpoint, walk through the game, applying substitutions.

### Pass 4: Participation Gap Filling (The "Ghost" Fix)
If at any row `len(on_floor) < 5`:
- Identify `Potential Ghosts`: Players seen in Pass 1 who are NOT currently in the `on_floor` set and NOT in the "Known Bench" state.
- **Fill**: Insert the most frequently appearing `Potential Ghosts` until `len == 5`.

## 3. Data Integration
The resulting dataset is exported as `fact_play_historical.parquet`, mimicking the schema of `fact_play_raw` (2025). This allows the RApM engine to treat 2015 and 2025 data as identical inputs.
## 4. Advanced Feature Handling
 
+### Spatial Normalization
+*   **Coordinate Space**: All PBP-derived shots are mapped to a canonical **940 x 500** coordinate system (0.1 ft units).
+*   **Historical Missingness**: For 2010-2018 where raw text lacks X,Y, the pipeline exports `loc_x=NULL`, ensuring Tier 2 features (`avg_shot_dist`) stay `NaN` without breaking the regression.
+
+### Volume & Usage
+*   **Minutes Reconstruction**: Successfully derived from PBP text via sub-event timestamps for specific validation blocks (2015, 2017).
+*   **Volume Proxy**: `poss_total` (total possessions where player was on floor) is used as the universal volume denominator for Usage Rates across all eras, providing a robust box-score-independent metric.
+
