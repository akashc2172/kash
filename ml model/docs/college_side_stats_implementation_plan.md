# College Side Stats Feeding Plan (NCAA) — Executable Implementation Spec
Prepared by: **cursor**  
Last updated: 2026-01-29  

This is the **college-side** implementation plan for feeding the Generative Prospect Model’s amateur evidence \(x\). It is derived from:
- The extracted spec text from `cbd_pbp/NBA_Generative_Prospect_Model_Proposal_v12_nba_variable_registry.docx`
- Current repo architecture (`cbd_pbp/`, `cbd_pbp/staging_layer.sql`, `college_scripts/build_college_feature_store_v1.py`, `college_scripts/build_prospect_career_store*.py`)
- Practical constraints discovered during Phase 4 (historical box-score gaps in `fact_player_season_stats`)

The goal is **not** to “stuff features into a parquet”. The goal is to create a stable, leakage-safe, interpretable set of observables \(x_{i,t}\) that make the latent trait \(z_i\) identifiable and transferable across eras and contexts.

---

## 0. Modeling contract (what the college side must provide)

### 0.1 Variable roles
- **Amateur observables** \(x_{i,t}\): college PBP-derived features + lineup/on-court context + opponent strength + (optional) box-score derived rates.
- **Latent traits** \(z_i\): inferred by the model, not computed here.
- **NBA outcomes** \(a, y\): never used as inputs for prospects; used only in training on historical NBA players (already implemented on NBA side).

### 0.2 Strict anti-leakage rules (college side)
- **No post-draft info** in \(x\). For NCAA, that means:
  - Only use games **through the end of the player’s final college season** (or “as-of” snapshots if doing in-season inference).
  - When computing “final season” snapshots used in Phase 4, do not include any college games after declared draft date (if we later model declared dates).
- **No circular labels**: do not mix NBA targets back into college features.
- **No cross-player inference-time features** that implicitly encode outcomes (e.g., “NBA draft pick” is *not* an input).

---

## 1. Current repo state (what exists today)

### 1.1 College PBP warehouse (DuckDB)
Primary DB: `ml model/data/warehouse.duckdb`

Key staging and materialized facts (per `cbd_pbp/staging_layer.sql` and proposal):
- **Raw / staging**
  - `stg_plays`: play events
  - `stg_shots`: shot events with `shot_range ∈ {three_pointer, rim, free_throw, jumper}`, `made`, `assisted`, leverage flags
  - `stg_lineups`, `bridge_lineup_athletes`: lineup stints exploded to 5 athletes
  - `stg_subs`: substitution events
- **Materialized facts**
  - `fact_player_game_shots_by_range` (player–game–range)
  - `fact_player_game_shots_bucketed` (player–game–bucket)
  - `fact_player_game_shots` (player–game)
  - `fact_player_game_impact` (player–game seconds-weighted on-court ratings)
  - `fact_player_game` (FULL OUTER join shots + impact at player–game)
  - `fact_team_game` (team–game pace/off/def/net from stints)

### 1.2 College feature store v1
Outputs in `ml model/data/college_feature_store/`:
- `college_features_v1.parquet` (**athlete-season-split**, 20 split rows + baseline)
  - Confirmed: ~1.25M rows, ~46 columns, includes shot counts by range, team pace, conference, etc.
- `prospect_career_v1.parquet` (**athlete**, longitudinal aggregates)
  - Confirmed: includes `final_trueShootingPct`, `final_usage` (from v2 build)
- `college_impact_v1.parquet` (impact proxy; implementation-dependent)

### 1.3 Known gaps
- `fact_player_season_stats` in `warehouse.duckdb` currently has **only seasons 2005 and 2025**.
  - This blocks historical usage/box-score features for 2006–2024 unless we ingest those seasons.
  - PBP-derived features remain usable for seasons where `stg_shots` exists.

---

## 2. Desired end-state feature tables (what we must produce)

We will standardize on three “feature products” that serve different modeling modes.

### 2.1 Product A — `feat_college_season_split` (training/inference default)
**Grain**: `(athlete_id, season, split_id)`  
**Source**: `college_features_v1.parquet` (with targeted extensions)

This is the default \(x_{i,t}\) block for prospect inference.

### 2.2 Product B — `feat_college_asof_windows` (optional in-season inference)
**Grain**: `(athlete_id, season, teamId, asOfGameId, window_id)`  
**Source**: `cbd_pbp/windows.py` producing `fact_player_window` (long format)

Use-case: “draft board mid-season”, “value of information”, and ASTz-like snapshot normalization.

### 2.3 Product C — `feat_college_career_summary` (joining layer / Phase 4 baseline)
**Grain**: `(athlete_id)`  
**Source**: `prospect_career_v1.parquet` (v2)

Use-case: bridging and “final season” baselines for adaptation gaps.

---

## 3. Input datasets (exact fields and how we use them)

### 3.1 `stg_shots` (PBP shot events)
**Fields used**:
- Identity: `gameId`, `teamId`, `opponentId` (if present), `shooterAthleteId`
- Shot taxonomy: `shot_range`, `scoreValue`, `made`
- Assistance: `assisted`, `assistAthleteId`
- Context: `is_high_leverage`, `is_garbage`
- Optional spatial: `loc_x`, `loc_y` (for corner-3 bins, shot distance proxies)

**Used to compute**:
- Bucket counts: FGA/FGM/PTS by bucket (rim/mid/three/ft)
- Attempt shares: rim_share, three_share, mid_share, ft_rate
- Efficiency: FG% by bucket; TS%/eFG% proxies
- Assisted structure: assisted_share overall and by bucket
- Creation proxies: unassisted_rim_rate, unassisted_3_rate
- Leverage-only variants: same stats filtered to high leverage, and downweight/flag garbage

### 3.2 `bridge_lineup_athletes` + `stg_lineups` (lineup stints)
**Fields used**:
- `gameId`, `teamId`, `athleteId`, `totalSeconds`
- `pace`, `offenseRating`, `defenseRating`, `netRating`

**Used to compute**:
- On-court ratings per player-game: `on_net_rating`, `on_ortg`, `on_drtg`
- Exposure: `seconds_on`
- Team-adjusted: `on_net_rating - team_net_rating` (team from `fact_team_game`)
- Opponent-adjusted: weight games by opponent strength proxy before season aggregation
- Shrinkage: impact values shrink toward 0 with variance inversely related to seconds

### 3.3 `dim_games` + derived views (season & home/away/neutral)
Used to assign season, and split context (home/away/neutral) for competition splits.

### 3.4 `fact_team_season_stats` (team context)
Used for:
- `team_pace`
- `conference`, `is_power_conf`
- optionally team offensive/defensive context features

### 3.5 `fact_player_season_stats` (box score totals; currently partial coverage)
Used for:
- `minutes_total`, `tov_total`, `ast_total`, `stl_total`, `blk_total`, rebounds
- Enabling derived advanced rates:
  - USG proxy \(\frac{FGA + 0.44·FTA + TOV}{poss}\) where poss is derived from team pace and minutes share
  - AST/TO, TO%, STL%, BLK% (requires possessions/minutes)

**Reality check**: this table needs backfill for 2006–2024.

---

## 4. Feature families (and how they map to latent traits)

We keep features grouped by “trait family” so the model can learn interpretable subspaces.

### 4.1 Shooting profile (shot diet + touch + efficiency)
**Counts** (per season & split):
- `rim_att`, `mid_att`, `three_att`, `ft_att`, `fga_total`, `shots_total`
- `rim_made`, `mid_made`, `three_made`, `ft_made`

**Rates**:
- `rim_fg_pct`, `mid_fg_pct`, `three_fg_pct`, `ft_pct`
- `rim_share`, `mid_share`, `three_share` (attempt share of FGA)
- `ft_rate` = `ft_att / fga_total`
- `assisted_share_*` and `unassisted_share_*`

**Use in model**:
- spacing/shot selection traits, touch quality, shooting skill calibration across contexts

### 4.2 Creation vs finishing (assisted/unassisted structure)
**Key features**:
- `assisted_share_rim`, `assisted_share_three`, `assisted_share_mid`
- `unassisted_3_rate`, `unassisted_rim_rate` (creation proxies)

**Use in model**:
- separates “self-creator” vs “play finisher” archetypes without hardcoding positions

### 4.3 Leverage sensitivity and garbage filtering
**Features**:
- `high_lev_att_rate`, `high_lev_fg_pct`
- `garbage_att_rate`

**Use in model**:
- uncertainty inflation (clutch inference is noisy), and to avoid garbage-time stat bias

### 4.4 Lineup/on-court impact (college proxy for +/-)
**Features**:
- `seconds_on`, `on_net_rating`, `on_ortg`, `on_drtg`
- team-adjusted variants
- shrinkage weights (seconds-based)

**Use in model**:
- gives early signal for “impact without usage” / connectors / defense value

### 4.5 Advanced rate stats & normalization (ASTz, USG, TO, etc.)
**AST% calibration (ASTz)** from proposal:
- normalize freshman AST% across seasons and role buckets (StudentT hierarchical baseline)
- operational approximation (Phase 1): z-score within `(season, role_bucket)` using robust median/MAD; store uncertainty proxies

**Important**: true hierarchical StudentT is a modeling-layer concern; the feature store should export:
- raw rate (AST proxy)
- season/role baseline stats (mean/std or median/MAD)
- computed standardized value + sample size used

---

## 5. Temporal aggregation, splits, and weighting

### 5.1 Splits (already implemented in v1)
**Split axis 1: leverage**: `ALL`, `HIGH_LEVERAGE`, `LOW_LEVERAGE`, `GARBAGE`  
**Split axis 2: opponent strength**: `ALL`, `VS_TOP50`, `VS_TOP100`, `VS_OTHERS`, `VS_UNKNOWN`

**Implementation**: `college_scripts/build_college_feature_store_v1.py` via `v_shots_augmented` and SRS proxy.

### 5.2 Windows (optional; recommended)
Use `cbd_pbp/windows.py`:
- `season_to_date`, `rolling5`, `rolling10`, `rolling15`

**Adds**:
- “as-of” features for scouting mid-season
- enables ASTz snapshot logic (“same date each season”) in a principled way

### 5.3 Recency weighting (proposal recommendation)
Add a recency-weighted season aggregate:
- Exponential weights over games with a tuned half-life (15–20 games)

**Implementation approach**:
- Build `fact_player_game` ordered by `startTime`
- Precompute EWMA aggregates per athlete-season and store as an additional “window_id”

---

## 6. Transforms & stabilization (critical)

### 6.1 Rate stabilization (empirical Bayes)
For noisy rates (FG%, assisted shares, etc.):
- store both numerator + denominator
- compute stabilized rate in model layer **or** export stabilized rate using Beta prior:
  - \(\hat{p} = \frac{made + \alpha}{att + \alpha + \beta}\)
  - choose \((\alpha,\beta)\) per bucket using league-wide priors per season

### 6.2 Era normalization
For rates with season drift (AST%, pace environments, 3P rates):
- export:
  - raw value
  - season baseline mean/std (or robust equivalents)
  - standardized z value
  - sample size used

### 6.3 Missingness indicators (MAR handling)
For any feature that can be missing due to data coverage:
- add `*_missing` boolean columns
- do not silently fill with 0 unless the semantics are “true zero”

---

## 7. Execution plan (commands + artifacts)

### Step 0 — Ensure DuckDB is built
Modern CBD seasons:
- `python -m cbd_pbp.cli ingest-season --season <YEAR> --season-type regular --out data/warehouse.duckdb`
- `python -m cbd_pbp.cli build-derived --season <YEAR> --season-type regular --out data/warehouse.duckdb`

Historical NCAA reconstruction (if needed):
- `python college_scripts/scrapers/scrape_ncaa_master.py ...`
- `python college_scripts/utils/clean_historical_pbp_v2.py ...`
- `python college_scripts/calculate_historical_rapm.py ...`

### Step 1 — Build/refresh feature store v1
- `python college_scripts/build_college_feature_store_v1.py`
Outputs:
- `data/college_feature_store/college_features_v1.parquet`
- `data/college_feature_store/college_impact_v1.parquet`
- `data/college_feature_store/coverage_report_v1.csv`

### Step 2 — Build career store (baseline joining layer)
- `python college_scripts/build_prospect_career_store.py` (v1)
- or Phase 4 ready:
  - `python college_scripts/build_prospect_career_store_v2.py`
Output:
- `data/college_feature_store/prospect_career_v1.parquet` (contains `final_trueShootingPct`, `final_usage`)

### Step 3 (optional) — Build windowed “as-of” tables
- `python -m cbd_pbp.cli build-windows --season <YEAR> --season-type regular --out data/warehouse.duckdb --windows season_to_date,rolling10,rolling15`
Output:
- `fact_player_window` / `fact_team_window` in DuckDB

---

## 8. Quality gates (must-pass checks)

### 8.1 Uniqueness
`college_features_v1.parquet` must be unique on:
- `(athlete_id, season, split_id)`

If duplicates exist:
- investigate join cardinality in `build_college_feature_store_v1.py`
- enforce dedup at write-time (keep max exposure row) and log count

### 8.2 Range checks
- all shares/pcts ∈ \([0,1]\)
- `shots_total >= fga_total + ft_att` (or defined relationship per schema)
- `assisted_made_* <= *_made`

### 8.3 Coverage checks
- per season: fraction of athletes with non-null PBP features should exceed threshold
- for `final_usage`: verify seasons where `minutes_total` and `tov_total` exist; otherwise mark missing and do not compute usage gap targets

---

## 9. Backlog (highest-value improvements)

### 9.1 Backfill `fact_player_season_stats` (2006–2024)
This unlocks:
- historical `minutes_total`, `tov_total`, etc.
- stable usage proxies and many advanced rate stats
- Phase 4 `gap_usg_legacy` coverage

### 9.2 ASTz implementation (feature-side scaffolding)
Export:
- `ast_proxy_raw`
- `ast_proxy_baseline_mean/std` (or robust)
- `ast_proxy_z`
- `ast_proxy_n` (sample size)

### 9.3 Recency-weighted aggregates & “best 15 game window”
Adds scouting realism and supports “stress test” features from proposal.

---

## 10. What to tell the model team (interface guarantees)

When this plan is implemented, the model training code can assume:
- `college_features_v1.parquet` provides leakage-safe \(x_{i,t}\) at season+split grain
- `prospect_career_v1.parquet` provides stable “final season” baselines for bridging tasks
- any missingness is explicit and safe to mask (no silent zeros)

— **cursor** (implementation plan author)

---

## 11. Spatial Data Strategy (Phase 4.1 Hybrid Resolution)

Addressing heterogenous spatial data availability (numeric X,Y is **not** universally populated).

### 11.1 Problem Statement (validated against current DuckDB)
- **Text is not usable for corner-vs-above-break**: `playText` almost never contains “corner” (single-digit to teens per season), so we should not attempt text heuristics for corner 3s.
- **Heterogeneity in numeric X,Y**:
  - In `fact_play_raw.shotInfo.location`, older seasons (2010–2018) frequently contain JSON null / non-numeric placeholders.
  - Numeric X,Y appears in earnest starting 2019 and rises through 2025.
  - Practical implication: Tier-2 spatial features must be computed only from **shots with numeric X,Y**, and missingness must be explicit.
- **Bias Risk**: model can overfit to presence of X,Y (learning “modern players have coords”), overpowering universal signals.

### 11.2 Feature Tiers
We define two tiers of spatial features to strictly separate "Universal" from "Modern Enhanced".

**Tier 1: Universal Low-Res (2010-2025)**
*Backbone features available for ALL historical training.*
- `rim_freq`, `mid_freq`, `three_freq`, `ft_freq` (from `shot_range`)
- `rim_fg_pct`, `mid_fg_pct`, `three_fg_pct`

**Tier 2: Modern High-Res (2019+)**
*Auxiliary features for modern precision.*
- `avg_shot_dist` (numeric)
- `shot_dist_var` (variance of distance - dispersion measure)
- `corner_3_rate` (corner 3 att / xy 3 att)
- `corner_3_pct`
- `deep_3_rate` (3s > 27ft / xy 3 att)
- `rim_purity` (Rim attempts < 4ft / xy rim att)

### 11.3 Integration Rules
1. **Clean missingness at ingestion**: when extracting `loc_x/loc_y` from `shotInfo`, use `TRY_CAST` so non-numeric/JSON-null becomes SQL NULL (not NaN). (Implemented in `cbd_pbp/staging_layer.sql`.)
2. **Coverage-aware feature construction**:
   - Always export `xy_shots` (count of shots with numeric X,Y) and `xy_coverage = xy_shots / total_shots`.
   - **Gating**:
     - General stats: `xy_shots >= 25`.
     - 3PT stats (Corner, Deep): `xy_3_shots >= 15`.
     - Rim stats (Purity): `xy_rim_shots >= 20`.
   - Else set Tier-2 features to NULL and rely on Tier-1.
3. **Explicit Missingness**: Tier-2 features MUST be NULL/NaN (not 0) when not supported by the data (older seasons, or low `xy_shots`).
4. **Feature Masking (Dropout)**: During training on 2019+ data, randomly mask Tier-2 features (set to NULL) ~20–30% of the time (or stratify by season) to prevent coordinate dependency.
### 11.4 Validated Geometry Spec
- **Source Scale**: `stg_shots` coordinates are in **0.1 ft** units (e.g., `x=940` means 94ft).
- **Coordinate System**:
  - X Range: 0 to 940 (Length)
  - Y Range: 0 to 500 (Width, 50ft)
  - Hoops: Located at X ~52.5 and ~887.5 (5.25ft and 88.75ft in normalized space).
- **Implemented Normalization**:
  - `loc_x, loc_y` divided by 10.0 to get Feet.
  - Corner 3 Bounding Box (NCAA): `abs(y_ft - 25) > 21` (within 4ft of sideline) AND `x_ft < 14` (short corner).

### 11.5 Implementation Status
- Tier 2 Features (`avg_shot_dist`, `corner_3_rate`, `deep_3_rate`, `rim_purity`) are implemented in `college_features_v1`.
- **Gating**: Validated gating logic implemented `xy_3_shots >= 15` etc.
- **Result**: Robust spatial traits for 2019-2025; clean missingness for history.

