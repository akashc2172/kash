"""Shared exposure selection logic with provenance for train/serve parity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_valid_games(series: pd.Series, lo: float = 1.0, hi: float = 45.0) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return x.where((x >= lo) & (x <= hi), np.nan)


def _as_valid_minutes(series: pd.Series, lo: float = 1.0, hi: float = 2000.0) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return x.where((x >= lo) & (x <= hi), np.nan)


def select_games_played_with_provenance(
    df: pd.DataFrame,
    *,
    api_col: str = "games_played",
    backfill_col: str = "backfill_games_played",
    hist_col: str = "hist_games_played_text",
    derived_col: str = "college_games_played",
    prefer_gap: float = 5.0,
    conflict_gap: float = 8.0,
    backfill_variant_col: str = "backfill_alignment_variant",
    hist_variant_col: str = "hist_alignment_variant",
) -> pd.DataFrame:
    """
    Select canonical games_played using deterministic source priority + undercoverage overrides.

    Priority with override:
    1) API/base games (api_col)
    2) Backfill if missing OR exceeds existing by >= prefer_gap
    3) Historical text if missing OR exceeds existing by >= prefer_gap
    4) Derived fallback when still missing
    """
    out = df.copy()
    idx = out.index

    api = _as_valid_games(out.get(api_col, pd.Series(index=idx, dtype=float)))
    backfill = _as_valid_games(out.get(backfill_col, pd.Series(index=idx, dtype=float)))
    hist = _as_valid_games(out.get(hist_col, pd.Series(index=idx, dtype=float)))
    derived = _as_valid_games(out.get(derived_col, pd.Series(index=idx, dtype=float)))

    selected = api.copy()
    source = pd.Series(np.where(selected.notna(), "api_box", "missing"), index=idx, dtype=object)
    align = pd.Series("none", index=idx, dtype=object)

    backfill_variant = out.get(backfill_variant_col, pd.Series("none", index=idx)).astype(str)
    hist_variant = out.get(hist_variant_col, pd.Series("none", index=idx)).astype(str)

    use_backfill = backfill.notna() & (selected.isna() | ((backfill - selected) >= prefer_gap))
    selected = selected.where(~use_backfill, backfill)
    source = source.where(~use_backfill, "manual_subs_backfill")
    align = align.where(~use_backfill, backfill_variant)

    use_hist = hist.notna() & (selected.isna() | ((hist - selected) >= prefer_gap))
    selected = selected.where(~use_hist, hist)
    source = source.where(~use_hist, "manual_text_backfill")
    align = align.where(~use_hist, hist_variant)

    use_derived = selected.isna() & derived.notna()
    selected = selected.where(~use_derived, derived)
    source = source.where(~use_derived, "derived_proxy")

    # Conflict flag: top-two valid candidates differ materially.
    cand = pd.concat(
        [api.rename("api"), backfill.rename("backfill"), hist.rename("hist"), derived.rename("derived")],
        axis=1,
    )

    def _conflict(row: pd.Series) -> int:
        vals = pd.to_numeric(row, errors="coerce").dropna().sort_values(ascending=False).to_numpy()
        if vals.size < 2:
            return 0
        return int((vals[0] - vals[1]) >= conflict_gap)

    conflict = cand.apply(_conflict, axis=1)

    source_rank_map = {
        "api_box": 1,
        "manual_subs_backfill": 2,
        "manual_text_backfill": 3,
        "derived_proxy": 4,
        "missing": 99,
    }
    source_rank = source.map(source_rank_map).fillna(99).astype(int)

    out["college_games_played_candidate_api"] = api
    out["college_games_played_candidate_backfill"] = backfill
    out["college_games_played_candidate_hist_text"] = hist
    out["college_games_played_candidate_derived"] = derived
    out["college_games_played"] = selected
    out["games_played"] = selected
    out["college_games_played_source"] = source
    out["college_games_played_source_rank"] = source_rank
    out["college_games_played_conflict_flag"] = conflict.astype(int)
    out["college_games_played_alignment_variant"] = align
    return out


def games_source_mix_by_season(
    df: pd.DataFrame,
    season_col: str = "college_final_season",
) -> pd.DataFrame:
    """Season-level source mix summary."""
    if season_col not in df.columns or "college_games_played_source" not in df.columns:
        return pd.DataFrame()
    g = (
        df.groupby([season_col, "college_games_played_source"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    tot = g.groupby(season_col)["rows"].transform("sum")
    g["share"] = np.where(tot > 0, g["rows"] / tot, np.nan)
    return g.sort_values([season_col, "rows"], ascending=[True, False]).reset_index(drop=True)


def select_minutes_with_provenance(
    df: pd.DataFrame,
    *,
    api_col: str = "minutes_total",
    backfill_col: str = "backfill_minutes_total",
    hist_col: str = "hist_minutes_total",
    derived_col: str = "derived_minutes_total_candidate",
    prefer_gap: float = 50.0,
    conflict_gap: float = 150.0,
    backfill_variant_col: str = "backfill_alignment_variant",
    hist_variant_col: str = "hist_alignment_variant",
) -> pd.DataFrame:
    """
    Select canonical minutes_total using deterministic source priority + undercoverage overrides.
    Minutes <= 0 are treated as missing.
    """
    out = df.copy()
    idx = out.index

    api = _as_valid_minutes(out.get(api_col, pd.Series(index=idx, dtype=float)))
    backfill = _as_valid_minutes(out.get(backfill_col, pd.Series(index=idx, dtype=float)))
    hist = _as_valid_minutes(out.get(hist_col, pd.Series(index=idx, dtype=float)))
    derived = _as_valid_minutes(out.get(derived_col, pd.Series(index=idx, dtype=float)))

    selected = api.copy()
    source = pd.Series(np.where(selected.notna(), "api_box", "missing"), index=idx, dtype=object)
    align = pd.Series("none", index=idx, dtype=object)

    backfill_variant = out.get(backfill_variant_col, pd.Series("none", index=idx)).astype(str)
    hist_variant = out.get(hist_variant_col, pd.Series("none", index=idx)).astype(str)

    use_backfill = backfill.notna() & (selected.isna() | ((backfill - selected) >= prefer_gap))
    selected = selected.where(~use_backfill, backfill)
    source = source.where(~use_backfill, "manual_subs_backfill")
    align = align.where(~use_backfill, backfill_variant)

    use_hist = hist.notna() & (selected.isna() | ((hist - selected) >= prefer_gap))
    selected = selected.where(~use_hist, hist)
    source = source.where(~use_hist, "manual_text_backfill")
    align = align.where(~use_hist, hist_variant)

    use_derived = selected.isna() & derived.notna()
    selected = selected.where(~use_derived, derived)
    source = source.where(~use_derived, "derived_proxy")

    cand = pd.concat(
        [api.rename("api"), backfill.rename("backfill"), hist.rename("hist"), derived.rename("derived")],
        axis=1,
    )

    def _conflict(row: pd.Series) -> int:
        vals = pd.to_numeric(row, errors="coerce").dropna().sort_values(ascending=False).to_numpy()
        if vals.size < 2:
            return 0
        return int((vals[0] - vals[1]) >= conflict_gap)

    conflict = cand.apply(_conflict, axis=1)

    source_rank_map = {
        "api_box": 1,
        "manual_subs_backfill": 2,
        "manual_text_backfill": 3,
        "derived_proxy": 4,
        "missing": 99,
    }
    source_rank = source.map(source_rank_map).fillna(99).astype(int)

    out["college_minutes_total_candidate_api"] = api
    out["college_minutes_total_candidate_backfill"] = backfill
    out["college_minutes_total_candidate_hist_text"] = hist
    out["college_minutes_total_candidate_derived"] = derived
    out["college_minutes_total"] = selected
    out["minutes_total"] = selected
    out["college_minutes_total_source"] = source
    out["college_minutes_total_source_rank"] = source_rank
    out["college_minutes_total_conflict_flag"] = conflict.astype(int)
    out["college_minutes_total_alignment_variant"] = align
    return out


def minutes_source_mix_by_season(
    df: pd.DataFrame,
    season_col: str = "college_final_season",
) -> pd.DataFrame:
    """Season-level minutes source mix summary."""
    if season_col not in df.columns or "college_minutes_total_source" not in df.columns:
        return pd.DataFrame()
    g = (
        df.groupby([season_col, "college_minutes_total_source"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    tot = g.groupby(season_col)["rows"].transform("sum")
    g["share"] = np.where(tot > 0, g["rows"] / tot, np.nan)
    return g.sort_values([season_col, "rows"], ascending=[True, False]).reset_index(drop=True)
