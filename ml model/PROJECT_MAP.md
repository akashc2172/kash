# PROMPT_CONTEXT: Project Atlas & Architecture 🗺️✨

> **SYSTEM INSTRUCTION FOR AI AGENTS (Cursor, Claude, GPT):**
> When starting a session in this repository, **READ THIS FILE FIRST**. It contains the comprehensive mental model, directory structure, data schemas, and operational constraints required to modify the code safely.

---

## 1. 🎯 The Mission: "Deep History RApM"
We are building the world's first **15-Year Regularized Adjusted Plus-Minus (RApM)** model for NCAA Basketball (2010–2025) to predict NBA prospect success. 

**The Core Challenge**: Modern data (2024+) is rich. Historical data (2010-2023) is just raw text. We have built a custom "Ghost Fill" engine to reconstruct historical lineups and unify them into a single dataset.

---

## 2. 🏗️ Architecture & Data Flow

### **Zone A: The Data Lake (`data/`)**
*   `data/warehouse.duckdb`: The central relational source.
    *   **`dim_teams`**: The source of truth for Team IDs.
    *   **`fact_play_raw`**: The 2025 modern PBP table.
*   `data/manual_scrapes/{YEAR}/`: Landing zone for raw historical CSVs (2012-2013, 2015-2016, 2017-2018 active).
*   `data/fact_play_historical_combined.parquet`: The **Output Artifact**. 1.5M lines of reconstructed history matching the 2025 schema.
*   `data/historical_rapm_results_enhanced.csv`: The **Output Artifact** of the enhanced RApM calculation (includes 7 RAPM variants).

### **Zone B: The Engine Room (`college_scripts/`)**
*   **`scrapers/scrape_ncaa_master.py`**: The tool to fetch new history.
*   **`utils/clean_historical_pbp_v2.py`**: The Cleaning Engine. Scans `data/manual_scrapes/`, handles "Ghost Fill", and outputs Parquet.
    *   *Note*: A new "Windowed Activity" version (`clean_historical_pbp_v2_windowed.py`) has been created to better handle substitutions in blowouts.
*   **`derive_minutes_from_historical_pbp.py`**: **Phase 1.2 Backfill Engine**. Derives minutes/turnovers from PBP text to fill missing box scores (2010-2018).
*   **`calculate_historical_rapm.py`**: The Impact Solver. Consumes historical Parquet, deduces stints, and solves Ridge RApM (lambda=1000).
    *   **Enhanced (Jan 2025)**: Now computes 7 RAPM variants:
        - `rapm_standard`: Possession-weighted (original)
        - `rapm_leverage_weighted`: Weights by leverage index (clutch signal)
        - `rapm_high_leverage`: High/very_high leverage stints only
        - `rapm_non_garbage`: Excludes garbage time
        - `o_rapm` / `d_rapm`: Offensive/Defensive split
        - `rapm_rubber_adj`: Rubber-band effect correction
    *   **Win Probability Model**: `compute_win_probability()` for leverage calculation
    *   **Leverage Index**: `compute_leverage_index()` based on pbpstats methodology

### **Zone C: The Model Lab (`analysis/`)**
*   `build_rapm_targets.py`: Calculates the RApM labels (Ridge Regression) for NBA players.
*   **Concept Update**: "Gap" = NBA metric (translation) - College metric. See `docs/zone_c_gap_concept_clarification.md`.

---

## 3. 🧬 Critical Schemas (Memory Injection)

**Common Play Format (The "Bridge" Schema)**
Whether 2015 or 2025, all modeling data converges to this structure:
```json
{
  "gameSourceId": "239560", 
  "season": 2015,
  "homeScore": 31,
  "awayScore": 28,
  "playText": "JACKSON,WARREN made Layup",
  "onFloor": [ // The "Ghost Fill" Algorithm guarantees exactly 10 items here
    {"id": null, "name": "JACKSON,WARREN", "team": "HOME"},
    {"id": null, "name": "JONES,ANDRE", "team": "HOME"},
    ... (3 more home, 5 away)
  ]
}
```

---

## 4. 🧠 Conceptual Dictionary

*   **Ghost Player**: A player who is on the court but doesn't record a stat/sub for long periods. Our solver infers their presence by checking global game activity.
*   **CBD**: "College Basketball Data" - our modern vendor source (2024+).
*   **Stint**: A period of game time where the 10 players on the floor are constant. RApM is calculated on *Stints*, not raw plays.
*   **Leverage Index**: Expected win probability swing from a possession outcome. High leverage = close game, late. Low leverage = blowout.
*   **Rubber Band Effect**: Systematic bias where leading teams coast and trailing teams try harder, creating noise in raw +/- metrics.
*   **O-RAPM / D-RAPM**: Offensive and Defensive RAPM computed separately by regressing on points scored vs points allowed.

---

## 5. ⚠️ Operational Constraints (Read Before Writing Code)

1.  **Do Not Break the Bridge**: Any change to `clean_historical_pbp_v2.py` MUST output the `onFloor` JSON struct exactly as defined above. The unified model depends on it.
6.  **Team IDs**: Always use fuzzy matching against `dim_teams` (DuckDB) before inventing new team IDs.
7.  **No "Partial" Lineups**: If a lineup has 4 players, the RApM regression crashes. The solver must strictly enforce `len(lineup) == 5`.

---

## 6. 📐 Spatial Data Dictionary (Phase 4 Updates)

> **Detailed Implementation Plans**:
> *   Phase 2: [Feature Store Hardening](docs/phase2_feature_store_hardening.md)
> *   Phase 3: [Model Training](docs/phase3_model_training.md)
> *   Phase 4: [Validation](docs/phase4_validation.md)

**Coordinate System (Confirmed)**
*   **Source**: `stg_shots.loc_x` / `loc_y` (from `fact_play_raw` JSON).
*   **Scale**: **0.1 feet**. Value `940` = 94.0 feet.
*   **Orientation**:
    *   **X**: 0 (Baseline A) to 940 (Baseline B).
    *   **Y**: 0 (Sideline) to 500 (Sideline).
    *   **Hoops**: Located at `X=52.5` (5.25ft) and `X=887.5` (88.75ft).

**Feature Tiers (Bias Mitigation)**
*   **Tier 1 (Universal)**: Shot Zones (Rim, Mid, 3, FT). Available 2010-2025. Derived from text/range.
*   **Tier 2 (Modern High-Res)**: Spatial features derived from X,Y coordinates.
    *   **Features**: `avg_shot_dist`, `shot_dist_var`, `corner_3_rate`, `corner_3_pct`, `deep_3_rate`, `rim_purity`
    *   **Availability**: Partial 2019+ (19.7% coverage), Robust 2020+ (51%+ coverage). Missing for 2010-2018.
    *   **Precision Gating**:
        *   General stats (avg_dist, variance): `xy_shots >= 25`
        *   3PT stats (corner, deep): `xy_3_shots >= 15`
        *   Rim stats (purity): `xy_rim_shots >= 20`
    *   **Missingness**: Explicit `NaN` (not 0) when gating thresholds not met.

**Feature Blocks (Jan 2025 Additions)**
*   **Impact Block**: 7 RAPM variants (standard, leverage-weighted, high-leverage, non-garbage, O/D split, rubber-adjusted)
*   **Athleticism Block**: `dunk_rate`, `dunk_freq`, `putback_rate`, `transition_freq`, `transition_eff`, `rim_pressure_index`
*   **Defense Activity Block**: `deflection_proxy`, `contest_proxy` (blocks without fouling)
*   **Decision Discipline Block**: `pressure_handle_proxy`, `clutch_shooting_delta`
*   **Shot Creation Block**: `self_creation_rate`, `self_creation_eff`
*   **Context Block**: `leverage_poss_share` (clutch usage)

---

## 7. 🚀 Training Pipeline (Feb 2025 Additions)

> **Visual DAG**: See `docs/model_architecture_dag.md` for complete pipeline diagram.

### Zone D: The Training Lab (`nba_scripts/`)

*   **`build_unified_training_table.py`**: Merges college features + NBA targets into single training matrix.
    *   Inputs: `college_features_v1.parquet`, `prospect_career_v1.parquet`, crosswalk, gaps
    *   Output: `data/training/unified_training_table.parquet`
    *   Features: 60+ columns (Tier 1 + Tier 2 + Career + Masks)
*   **`train_baseline_xgboost.py`**: XGBoost baseline model with walk-forward validation.
    *   Walk-Forward: Train (2010-2017), Val (2018-2019), Test (2020-2022)
    *   Multi-Task: `y_peak_ovr`, `gap_ts_legacy`, `year1_epm_tot`, `made_nba`
    *   Output: `models/xgboost_baseline_{DATE}/` (model, feature importance, metrics)
*   **`run_training_pipeline.py`**: Pipeline orchestrator with prerequisite checks.
    *   Usage: `python3 run_training_pipeline.py --all`

### Data Flow Summary

```
CBD API + NCAA Scrapes → warehouse.duckdb
                              ↓
                    build_college_feature_store_v1.py
                              ↓
                    college_features_v1.parquet
                              ↓
                    build_prospect_career_store_v2.py
                              ↓
                    prospect_career_v1.parquet
                              ↓
    NBA Targets ──────────────┼─────────── Crosswalk
                              ↓
                    build_unified_training_table.py
                              ↓
                    unified_training_table.parquet
                              ↓
                    train_baseline_xgboost.py
                              ↓
                    models/xgboost_baseline_*/
```

### Current Status (2026-02-01)

| Component | Status | Notes |
|-----------|--------|-------|
| Training Pipeline Scripts | ✅ Complete | All scripts created |
| XGBoost Baseline | ✅ Ready | Awaiting data files |
| **Latent Space Model** | ✅ Complete | Archetypes + multi-task heads |
| Historical Scraping | ⏳ In Progress | Minutes/rotation data for RAPM |
| Data Files | ❌ Missing | Need to generate or copy to worktree |

---

## 8. 🧠 Latent Space Model (Feb 2025)

> **Architecture Doc**: See `docs/latent_space_architecture.md` for full design.

### Why Latent Space?

XGBoost answers "**what** predicts success" but not "**why**". The latent space model discovers:

- **Player archetypes** (rim-runner, 3-and-D wing, shot creator, etc.)
- **Feature interactions** that define career pathways
- **Player similarity** for comparisons
- **Narrative generation** for scouting reports

### Components (`models/`)

| Module | Purpose |
|--------|---------|
| `player_encoder.py` | Tier 1/2/Career → 32-dim latent `z` |
| `prospect_model.py` | Multi-head decoder (RAPM, survival, archetypes) |
| `archetype_analyzer.py` | Post-training cluster interpretation |

### Training

```bash
python3 nba_scripts/train_latent_model.py --epochs 100 --latent-dim 32 --n-archetypes 8
```

### Usage

```python
from models import ProspectModel, ArchetypeAnalyzer

model = ProspectModel(latent_dim=32, n_archetypes=8)
model.load_state_dict(torch.load('model.pt'))

# Analyze a prospect
analysis = analyzer.analyze_player(tier1, tier2, career)
print(analysis.narrative)
# Output:
# **Cooper Flagg** projects as a **Two-Way Wing** (85% confidence).
# Projection: 2.1 ± 0.8 peak RAPM (All-Star caliber)
# Comparisons: Jaylen Brown (78%), OG Anunoby (72%)
```
<<<<<<< /Users/akashc/my-trankcopy/ml model/PROJECT_MAP.md
>>>>>>> /Users/akashc/.windsurf/worktrees/my-trankcopy/my-trankcopy-2f05fd79/ml model/PROJECT_MAP.md
=======
>>>>>>> /Users/akashc/.windsurf/worktrees/my-trankcopy/my-trankcopy-2f05fd79/ml model/PROJECT_MAP.md
