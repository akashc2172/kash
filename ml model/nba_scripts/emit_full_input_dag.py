#!/usr/bin/env python3
"""
Emit Full Input DAG + Explicit Column Lists
==========================================
Generates a markdown report that enumerates *every* actual input column used
by the models (Tier1/Tier2/Career/Within), plus targets and masks.

Output:
  docs/full_input_columns.md
"""

from __future__ import annotations

from pathlib import Path
import datetime as dt
import pandas as pd
import pyarrow.parquet as pq
import sys


def main() -> None:
    base = Path(__file__).parent.parent
    sys.path.insert(0, str(base))
    training_path = base / "data/training/unified_training_table.parquet"
    out_path = base / "docs/full_input_columns.md"

    from models import TIER1_COLUMNS, TIER2_COLUMNS, CAREER_BASE_COLUMNS, WITHIN_COLUMNS

    if not training_path.exists():
        raise FileNotFoundError(f"Missing: {training_path}")

    schema = pq.read_schema(training_path)
    cols = set(schema.names)
    frame = pd.read_parquet(training_path)

    # Targets and key masks we care about.
    targets = [
        "y_peak_ovr", "y_peak_off", "y_peak_def",
        "year1_epm_tot", "year1_epm_off", "year1_epm_def",
        "gap_ts_legacy", "gap_usg_legacy",
        "made_nba",
    ]
    masks = [
        "has_spatial_data",
        "final_has_ws_last5", "final_has_ws_last10",
        "final_has_ws_breakout_timing_eff",
    ]

    def present_list(xs: list[str]) -> list[str]:
        return [x for x in xs if x in cols]

    missing_tier1 = [c for c in TIER1_COLUMNS if c not in cols]
    missing_tier2 = [c for c in TIER2_COLUMNS if c not in cols]
    missing_career = [c for c in CAREER_BASE_COLUMNS if c not in cols]
    missing_within = [c for c in WITHIN_COLUMNS if c not in cols]
    dead_inputs = []
    for c in TIER1_COLUMNS + TIER2_COLUMNS + CAREER_BASE_COLUMNS + WITHIN_COLUMNS:
        if c in frame.columns:
            s = pd.to_numeric(frame[c], errors="coerce").fillna(0.0)
            non_zero = float((s != 0).mean())
            if non_zero < 0.001:
                dead_inputs.append((c, non_zero))
    allowed_dead = {
        "final_has_ws_last10",
        "final_ws_minutes_last10",
        "final_ws_pps_last10",
        "final_ws_delta_pps_last5_minus_prev5",
        "final_has_ws_breakout_timing_eff",
        "final_ws_breakout_timing_eff",
        "has_within_window_data",
    }
    disallowed_dead = [(c, nz) for c, nz in dead_inputs if c not in allowed_dead]

    md = []
    md.append("# Full Input Columns (Verified)\n")
    md.append(f"**Generated**: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append(f"**Source Table**: `data/training/unified_training_table.parquet`\n")
    md.append(f"**Total Columns**: {len(schema.names)}\n")

    md.append("## Model Inputs\n")

    md.append("### Tier 1 (Universal)\n")
    for c in TIER1_COLUMNS:
        md.append(f"- `{c}`" + ("" if c in cols else " (MISSING)"))

    md.append("\n### Tier 2 (Spatial)\n")
    for c in TIER2_COLUMNS:
        md.append(f"- `{c}`" + ("" if c in cols else " (MISSING)"))

    md.append("\n### Career (Progression)\n")
    for c in CAREER_BASE_COLUMNS:
        md.append(f"- `{c}`" + ("" if c in cols else " (MISSING)"))

    md.append("\n### Within-Season Windows (Star Run)\n")
    for c in WITHIN_COLUMNS:
        md.append(f"- `{c}`" + ("" if c in cols else " (MISSING)"))

    md.append("\n## Targets (Labels Only)\n")
    for c in targets:
        md.append(f"- `{c}`" + ("" if c in cols else " (MISSING)"))

    md.append("\n## Masks / Coverage Flags\n")
    for c in masks:
        md.append(f"- `{c}`" + ("" if c in cols else " (MISSING)"))

    md.append("\n## Notes\n")
    md.append("- Columns marked `(MISSING)` are not present in the current training table and would be silently imputed by naive loaders; fix by adding them upstream or removing them from the model column lists.\n")
    if dead_inputs:
        md.append("- Columns listed in `Dead Inputs` are present but effectively constant/zero and should be treated as unwired until source population is fixed.\n")
    md.append("- NBA data is used as targets only; see `docs/end_to_end_wiring.md` and `nba_scripts/nba_data_loader.py` for leakage rules.\n")
    if dead_inputs:
        md.append("\n## Dead Inputs (non-zero < 0.1%)\n")
        for c, nz in sorted(dead_inputs):
            md.append(f"- `{c}` (non-zero rate={nz:.4f})")

    out_path.write_text("\n".join(md))

    # Hard fail if anything is missing (explicit verification).
    if missing_tier1 or missing_tier2 or missing_career or missing_within or disallowed_dead:
        lines = ["Missing required input columns:"]
        lines += [f"Tier1: {c}" for c in missing_tier1]
        lines += [f"Tier2: {c}" for c in missing_tier2]
        lines += [f"Career: {c}" for c in missing_career]
        lines += [f"Within: {c}" for c in missing_within]
        if disallowed_dead:
            lines.append("Dead (present but near-zero) input columns:")
            lines += [f"{c} (non-zero={nz:.4f})" for c, nz in sorted(disallowed_dead)]
        raise SystemExit("\n".join(lines))


if __name__ == "__main__":
    main()
