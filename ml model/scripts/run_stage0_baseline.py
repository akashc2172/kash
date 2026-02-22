#!/usr/bin/env python3
"""Stage 0: Baseline Freeze for 2026 Model Stack.
Captures baseline metrics from the supervised training surface (not inference pool).
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent.parent
SUPERVISED_PATH = BASE / "data" / "training" / "unified_training_table_supervised.parquet"
OUT_JSON = BASE / "data" / "audit" / "stack2026_stage0_baseline.json"

def main():
    if not SUPERVISED_PATH.exists():
        print(f"Error: {SUPERVISED_PATH} not found. Cannot capture baseline.")
        return

    df = pd.read_parquet(SUPERVISED_PATH)
    
    total_rows = len(df)  # true supervised surface count
    
    # 2. Target-space spread from primary target proxy (y_peak_epm_window)
    score_col = "y_peak_epm_window" if "y_peak_epm_window" in df.columns else ("y_peak_ovr" if "y_peak_ovr" in df.columns else None)
    if score_col is not None:
        scores = pd.to_numeric(df[score_col], errors="coerce").dropna()
        score_std = scores.std()
        score_iqr = np.percentile(scores, 75) - np.percentile(scores, 25) if len(scores) > 0 else 0
        min_score, max_score = scores.min(), scores.max()
    else:
        score_std = 0
        score_iqr = 0
        min_score = 0
        max_score = 0

    # 3. Target Coverage from real supervised columns
    peak_epm_cov = float(pd.to_numeric(df.get("y_peak_epm_window"), errors="coerce").notna().mean()) if "y_peak_epm_window" in df.columns else 0.0
    year1_epm_cov = float(pd.to_numeric(df.get("year1_epm_tot"), errors="coerce").notna().mean()) if "year1_epm_tot" in df.columns else 0.0
    peak_rapm_cov = float(pd.to_numeric(df.get("y_peak_ovr"), errors="coerce").notna().mean()) if "y_peak_ovr" in df.columns else 0.0

    baseline = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "supervised_row_count": int(total_rows),
        "target_coverage_peak_epm_window": peak_epm_cov,
        "target_coverage_year1_epm": year1_epm_cov,
        "target_coverage_peak_rapm": peak_rapm_cov,
        "score_source_col": score_col,
        "score_std": float(score_std),
        "score_iqr": float(score_iqr),
        "score_min": float(min_score),
        "score_max": float(max_score),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"Baseline saved to {OUT_JSON}")
    print(json.dumps(baseline, indent=2))

if __name__ == "__main__":
    main()
