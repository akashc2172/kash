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
*   `data/manual_scrapes/{YEAR}/`: Landing zone for raw historical CSVs.
*   `data/fact_play_historical_combined.parquet`: The **Output Artifact**. 1.5M lines of reconstructed history matching the 2025 schema.
*   `data/historical_rapm_results_lambda1000.csv`: The **Output Artifact** of the RApM calculation.

### **Zone B: The Engine Room (`college_scripts/`)**
*   **`scrapers/scrape_ncaa_master.py`**: The tool to fetch new history.
*   **`utils/clean_historical_pbp_v2.py`**: The Cleaning Engine. Scans `data/manual_scrapes/`, handles "Ghost Fill", and outputs Parquet.
*   **`calculate_historical_rapm.py`**: The Impact Solver. Consumes historical Parquet, deduces stints, and solves Ridge RApM (lambda=1000).

### **Zone C: The Model Lab (`analysis/`)**
*   `build_rapm_targets.py`: Calculates the RApM labels (Ridge Regression) for NBA players.

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

---

## 5. ⚠️ Operational Constraints (Read Before Writing Code)

1.  **Do Not Break the Bridge**: Any change to `clean_historical_pbp_v2.py` MUST output the `onFloor` JSON struct exactly as defined above. The unified model depends on it.
2.  **Team IDs**: Always use fuzzy matching against `dim_teams` (DuckDB) before inventing new team IDs.
3.  **No "Partial" Lineups**: If a lineup has 4 players, the RApM regression crashes. The solver must strictly enforce `len(lineup) == 5`.
