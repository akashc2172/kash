# Latent Model Input Plan (Multi-Season Career Handling)

**Date**: 2026-02-03
**Status**: Proposed plan for sequence inputs into the latent model

## Goal

Use multi-season college careers without confusing career length with NBA longevity. The input should represent **college trajectory** only, while treating short careers (one-and-done, transfers) and long careers (senior seasons, late bloomers) fairly.

## Data Sources

- Long career table: `data/college_feature_store/prospect_career_long_v1.parquet`
- Wide career table: `data/college_feature_store/prospect_career_v1.parquet`
- Feature store: `data/college_feature_store/college_features_v1.parquet` (ALL__ALL split)

## Core Representation

Each player is represented by a sequence of seasons:

```
Player i: [Season_1, Season_2, ..., Season_T]
```

Where each `Season_t` contains:
- Rates (TS%, usage, 3PT%, rim_fg_pct, etc.)
- Volume (minutes, games, poss_total)
- Spatial (Tier 2) with explicit `NaN` and mask
- Context (team pace, SRS proxy, role bucket)

## Handling Variable Career Lengths (Cam Johnson vs Rob Dillingham)

We treat **season count as context**, not as signal of quality. The model sees T via:

- `season_index` (1..T)
- `career_years` (T)
- `age_at_season` or `class_year` (if available)
- `transfer_flag` and `team_change_flag`

The model is **not allowed to treat longer careers as better** by default.

## Late/Early Breakout (Update)

We add *continuous* breakout timing signals that the model can learn to use
without hard-coded rules:

- `breakout_timing_volume`: where along the career the player's volume peaks
- `breakout_timing_usage`: where along the career the player's usage peaks
- `breakout_timing_eff`: where along the career the player's efficiency peaks
- `breakout_timing_avg`: summary index

These are normalized to `[0, 1]` by season rank: `0 = early`, `1 = late`.

### Two Different "Late Breakout" Notions (Clarification)

There are two timing axes that can both matter:

1. **Career-stage breakout (year-to-year):** “Did you peak early in your college *career* or late?”
   - This is what the `breakout_timing_*` features represent.
   - Normalization is by **season rank within career** (freshman=early, senior=late).

2. **Within-season breakout (“star run”):** “Did you surge late in the *season* (e.g. conference play / March)?”
   - This is represented by the `ws_*` (within-season window) features like:
     `final_ws_pps_last5`, `final_ws_delta_pps_last5_minus_prev5`,
     `final_ws_breakout_timing_eff`.
   - These are normalized within the **single season timeline**, not the career timeline.
   - If we don’t have the necessary game logs, these stay `NaN` and are gated by masks like `final_has_ws_last5`
     (we never treat missing as 0).

The “1/3 through freshman year” vs “1/3 through a 4-year career” concern is exactly why we keep these as
two separate axes instead of collapsing them into one timing number.

1. **Career-stage breakout** (what we implement now)
   - *Question*: did the player peak early in their college career or late?
   - *Normalization*: by `season_rank` within the player's college career.
   - This captures “4-year late bloomers vs one-and-dones”, *but it cannot* distinguish
     “March breakout” inside a season.

2. **Within-season breakout** (planned next)
   - *Question*: did the player improve late in the season (conference play / March),
     or were they strong all year?
   - *Normalization*: by game index/date within a season (0 = early season, 1 = late season).
   - This captures “only got hot in March” vs “consistently good”.

These are complementary, and we should not force one to stand in for the other.

### Why This Isn't Hardcoding

- We do not label players as “late breakout = bad”.
- The model learns the effect size from data.
- Breakout features are *contextual*, not targets.

### Making It Prototype-Dependent (Archetype-Conditioned)

Breakout timing plausibly matters differently by archetype (e.g., rim-running bigs
vs shot creators). To let the model learn this:

1. Feed breakout timing features into the encoder (so `z` can reflect them).
2. In decoders, optionally concatenate archetype probabilities:
   - `y_hat = Decoder([z, archetype_probs, breakout_feats])`
3. Or use a light Mixture-of-Experts head:
   - experts correspond to learned archetypes; gating uses archetype_probs.

This lets the model learn “late breakout is worse *for archetype A* but neutral
or even positive *for archetype B*” without manual rules.

### Missing Data / Era Safety

- If a player-season lacks the required inputs (minutes/games, windows), the corresponding breakout
  features should be `NaN` and accompanied by masks (e.g., `has_within_season_windows = 0`).
- Do **not** impute missing to 0. Zero is only valid when the player truly recorded 0 for that stat.

## Reliability Weighting (Exposure-Aware)

Season observations are noisy when minutes/games are low. We weight season inputs by exposure:

```
reliability_t = min(1, minutes_t / 800) * min(1, games_t / 20)
```

These weights are used in two ways:

1. **Feature shrinkage** (empirical Bayes):
   - `x_t_shrunk = w_t * x_t + (1 - w_t) * career_mean`

2. **Attention pooling**: higher exposure seasons get more weight when the model aggregates.

## Recommended Encoder Design

### 1. Season Encoder (Per-Season MLP)
Encodes each season into a season embedding `h_t`.

### 2. Sequence Encoder (Time-Aware)
Two options (can start with A and upgrade to B):

A) **Attention Pooling (Fast + Stable)**
- Attention weights are a function of `h_t`, `season_index`, and `reliability_t`
- Aggregates into a single career embedding

B) **Transformer / GRU (Advanced)**
- Uses time embeddings (`season_index`, `age_at_season`)
- Allows the model to see progression and role shifts

### 3. Gated Fusion with Final Season
The final season is the strongest signal for draft projection. We gate between:

```
career_embedding = g * final_season_embedding + (1 - g) * pooled_embedding
```

Where `g` is learned from career length, exposure, and variance. This prevents
long careers from overpowering a strong final season (e.g., Cam Johnson late leap)
while still allowing multi-year stability to inform the latent embedding.

## Transfer & Role Change Handling

We explicitly add indicators:
- `team_change_flag` per season
- `role_shift_flag` (usage or minutes jump above threshold)
- `coaching_change_flag` (if available)

The sequence encoder can learn change-points. A simple rule can be used to tag
`role_shift_flag` when usage or minutes changes by > 25% YoY.

## Bayesian / Hierarchical Extension (Advanced)

For small sample seasons, use partial pooling within each player:

```
x_t ~ Normal(theta_i, sigma_t)

sigma_t = sigma0 / sqrt(minutes_t)

theta_i ~ Normal(mu_league, sigma_player)
```

This allows unstable seasons to shrink toward the player-level mean instead
of being treated as equally precise as high-minute seasons.

## Suggested Inputs to Latent Encoder

Baseline input vector per season:
- `trueShootingPct`, `usage`, `rim_fg_pct`, `three_fg_pct`, `ft_pct`
- `ast_total`, `tov_total`, `stl_total`, `blk_total`
- `minutes_per_game`, `poss_per_game`, `games_played`
- `avg_shot_dist`, `corner_3_rate`, `deep_3_rate`, `rim_purity`, `shot_dist_var`
- `has_spatial_data` mask
- `season_index`, `career_years`, `age_at_season` (if available)

## Output

A single latent embedding `z` per player, constructed from a sequence-aware
representation of the college career.

## Next Implementation Step

1. Generate trajectory stub from `prospect_career_long_v1.parquet`
2. Build a simple attention pooling encoder over seasons
3. Add reliability weighting based on minutes/games
4. Evaluate against baseline using final-season-only features
