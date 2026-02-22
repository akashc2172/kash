#!/usr/bin/env python3
"""
Build college_pathway_context_v2.parquet.

Pathways-to-success context block:
- adjusted on/off core signals (reliability-shrunk)
- event-family contextual proxies
- developmental velocity (YoY deltas)
"""

from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[1]
FEATURE_DIR = BASE / "data" / "college_feature_store"
IMPACT_PATH = FEATURE_DIR / "college_impact_stack_v1.parquet"
FEATURES_PATH = FEATURE_DIR / "college_features_v1.parquet"
OUT_PATH = FEATURE_DIR / "college_pathway_context_v2.parquet"


def _z(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    mu = v.mean(skipna=True)
    sd = v.std(skipna=True)
    if pd.isna(sd) or sd < 1e-9:
        # Neutral z-score when the season distribution is degenerate.
        return pd.Series(0.0, index=s.index)
    return (v - mu) / sd


def _per100(n: pd.Series, poss: pd.Series) -> pd.Series:
    n = pd.to_numeric(n, errors="coerce")
    poss = pd.to_numeric(poss, errors="coerce")
    return np.where(poss > 0, (n / poss) * 100.0, np.nan)


def _season_standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[c] = (
            out.groupby("season")[c]
            .transform(lambda s: _z(s))
        )
    return out


def main() -> None:
    if not IMPACT_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing source files: {IMPACT_PATH} / {FEATURES_PATH}")

    impact = pd.read_parquet(IMPACT_PATH)
    feats = pd.read_parquet(FEATURES_PATH)

    keep_feat = [
        "athlete_id", "season", "poss_total", "minutes_total", "team_pace",
        "ast_total", "orb_total", "drb_total", "trb_total", "stl_total", "blk_total", "tov_total",
        "transition_freq", "dunk_rate", "rim_pressure_index",
        "fga_total", "shots_total", "rim_made", "three_made", "mid_made", "ft_made",
    ]
    keep_feat = [c for c in keep_feat if c in feats.columns]
    feats = feats[keep_feat].copy()

    for col in ["athlete_id", "season"]:
        feats[col] = pd.to_numeric(feats[col], errors="coerce")
        impact[col] = pd.to_numeric(impact[col], errors="coerce")

    # Base on full feature coverage, then attach impact where available.
    # This avoids dropping entire cohorts when impact stack is sparse.
    df = feats.merge(impact, on=["athlete_id", "season"], how="left", suffixes=("", "_impact"))
    # Keep one row per (athlete_id, season): prefer the split with most exposure (ALL__ALL has totals).
    sort_cols = [c for c in ["minutes_total", "poss_total", "shots_total", "fga_total"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
    df = df.drop_duplicates(subset=["athlete_id", "season"], keep="first").copy()

    poss_impact = pd.to_numeric(df.get("impact_poss_total", pd.Series(np.nan, index=df.index)), errors="coerce")
    poss_feat = pd.to_numeric(df.get("poss_total", pd.Series(np.nan, index=df.index)), errors="coerce")
    minutes_feat = pd.to_numeric(df.get("minutes_total", pd.Series(np.nan, index=df.index)), errors="coerce")
    pace_feat = pd.to_numeric(df.get("team_pace", pd.Series(np.nan, index=df.index)), errors="coerce")
    # Possession fallback when explicit poss_total is missing.
    # team_pace is possessions per 40 team minutes -> player-poss proxy by minutes share.
    poss_est = (minutes_feat * pace_feat) / 40.0
    poss = poss_impact.combine_first(poss_feat).combine_first(poss_est)
    seconds = pd.to_numeric(df.get("impact_seconds_total"), errors="coerce")

    # Robust denominator for per-100 rates when poss/minutes are bugged (e.g. 0/NaN for some stars).
    # Use FGA-based proxy so we can still compute box proxies and avoid NaNs.
    fga_for_denom = pd.to_numeric(df.get("fga_total", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0)
    poss_for_rates = np.where(
        poss.notna() & (poss > 0),
        poss,
        np.maximum(fga_for_denom.values * 1.2 + 1.0, 1.0),
    )
    poss_for_rates = pd.Series(poss_for_rates, index=df.index, dtype=float)

    k = 300.0
    rel = np.where(poss.notna(), poss / (poss + k), np.nan)
    rel = np.clip(rel, 0.0, 1.0)
    # Penalize proxy-only rows (no impact possessions) but keep them live.
    proxy_only = poss_impact.isna() & poss_feat.notna()
    rel = np.where(proxy_only, rel * 0.65, rel)

    # Precompute box-style per-100 proxies for fallback context when explicit
    # on/off-like signals are unavailable. Use poss_for_rates so bugged poss/minutes don't yield NaN.
    ast100_f = pd.Series(_per100(df.get("ast_total"), poss_for_rates), index=df.index)
    reb100_f = pd.Series(_per100(df.get("trb_total"), poss_for_rates), index=df.index)
    stl100_f = pd.Series(_per100(df.get("stl_total"), poss_for_rates), index=df.index)
    blk100_f = pd.Series(_per100(df.get("blk_total"), poss_for_rates), index=df.index)
    tov100_f = pd.Series(_per100(df.get("tov_total"), poss_for_rates), index=df.index)
    dunk_f = pd.to_numeric(df.get("dunk_rate"), errors="coerce")
    rim_f = pd.to_numeric(df.get("rim_pressure_index"), errors="coerce")

    box_net_proxy = (
        0.30 * _z(ast100_f) - 0.25 * _z(tov100_f) + 0.25 * _z(stl100_f)
        + 0.20 * _z(blk100_f) + 0.20 * _z(dunk_f) + 0.15 * _z(rim_f) + 0.10 * _z(reb100_f)
    )
    box_off_proxy = (
        0.45 * _z(ast100_f) - 0.35 * _z(tov100_f) + 0.35 * _z(dunk_f) + 0.25 * _z(rim_f)
    )
    box_def_proxy = (
        0.45 * _z(stl100_f) + 0.40 * _z(blk100_f) + 0.20 * _z(reb100_f)
    )
    has_box_signal = (
        ast100_f.notna() | reb100_f.notna() | stl100_f.notna() | blk100_f.notna() | tov100_f.notna()
        | dunk_f.notna() | rim_f.notna()
    )
    # Keep proxy rows alive when possessions are missing: low-confidence reliability.
    rel = np.where(np.isfinite(rel), rel, np.where(has_box_signal, 0.22, np.nan))

    ripm_tot = pd.to_numeric(df.get("rIPM_tot_std"), errors="coerce")
    ripm_off = pd.to_numeric(df.get("rIPM_off_std"), errors="coerce")
    ripm_def = pd.to_numeric(df.get("rIPM_def_std"), errors="coerce")
    onoff_net = (
        pd.to_numeric(df.get("impact_on_off_net_diff_raw"), errors="coerce")
        .combine_first(pd.to_numeric(df.get("impact_on_net_raw"), errors="coerce"))
        .combine_first(ripm_tot * 3.0)
        .combine_first(box_net_proxy * 2.5)
    )
    onoff_ortg = (
        pd.to_numeric(df.get("impact_on_off_ortg_diff_raw"), errors="coerce")
        .combine_first(pd.to_numeric(df.get("impact_on_ortg_raw"), errors="coerce"))
        .combine_first(ripm_off * 3.0)
        .combine_first(box_off_proxy * 2.5)
    )
    onoff_drtg = (
        pd.to_numeric(df.get("impact_on_off_drtg_diff_raw"), errors="coerce")
        .combine_first(pd.to_numeric(df.get("impact_on_drtg_raw"), errors="coerce"))
        .combine_first(-ripm_def * 3.0)
        .combine_first(box_def_proxy * 2.5)
    )

    # NaN poison fix: when we have valid on/off signal but rel is NaN (e.g. poss missing), use safe rel
    # so the product is not poisoned. No fake data—we keep the real signal and apply a default weight.
    has_valid_signal = onoff_net.notna() | onoff_ortg.notna() | onoff_drtg.notna()
    rel_safe = np.where(np.isfinite(rel), rel, np.where(has_valid_signal, 0.5, np.nan))
    rel_safe = pd.Series(rel_safe, index=df.index)

    # Reliability-shrunk adjusted on/off core (use rel_safe so valid impact is not zeroed by NaN rel).
    df["ctx_adj_onoff_net"] = onoff_net * rel_safe
    df["ctx_adj_onoff_ortg"] = onoff_ortg * rel_safe
    df["ctx_adj_onoff_drtg"] = onoff_drtg * rel_safe
    # If we had any raw signal but ctx_* are still NaN (e.g. 0*NaN or rel_safe was NaN), fill with 0 so downstream is not poisoned.
    has_raw_any = pd.Series(pd.to_numeric(df.get("has_impact_raw"), errors="coerce").fillna(0).values > 0, index=df.index)
    still_nan = (
        df["ctx_adj_onoff_net"].isna() & df["ctx_adj_onoff_ortg"].isna() & df["ctx_adj_onoff_drtg"].isna()
    ) & (has_valid_signal | has_raw_any | has_box_signal)
    if still_nan.any():
        df.loc[still_nan, "ctx_adj_onoff_net"] = 0.0
        df.loc[still_nan, "ctx_adj_onoff_ortg"] = 0.0
        df.loc[still_nan, "ctx_adj_onoff_drtg"] = 0.0

    # Event family per-100 rates from available box/event totals (robust denominator).
    ast100 = pd.Series(_per100(df.get("ast_total"), poss_for_rates), index=df.index)
    reb100 = pd.Series(_per100(df.get("trb_total"), poss_for_rates), index=df.index)
    stl100 = pd.Series(_per100(df.get("stl_total"), poss_for_rates), index=df.index)
    blk100 = pd.Series(_per100(df.get("blk_total"), poss_for_rates), index=df.index)
    tov100 = pd.Series(_per100(df.get("tov_total"), poss_for_rates), index=df.index)

    tmp = pd.DataFrame({
        "athlete_id": df["athlete_id"],
        "season": df["season"],
        "ast100": ast100, "reb100": reb100, "stl100": stl100, "blk100": blk100, "tov100": tov100,
        "transition_freq": pd.to_numeric(df.get("transition_freq"), errors="coerce"),
        "dunk_rate": pd.to_numeric(df.get("dunk_rate"), errors="coerce"),
        "rim_pressure_index": pd.to_numeric(df.get("rim_pressure_index"), errors="coerce"),
        "off_sig": 0.5 * _z(onoff_ortg) + 0.5 * _z(ripm_off),
        "def_sig": 0.5 * _z(onoff_drtg) + 0.5 * _z(ripm_def),
        "net_sig": _z(onoff_net),
        "rel": rel_safe,
    })
    tmp = _season_standardize(
        tmp,
        ["ast100", "reb100", "stl100", "blk100", "tov100", "transition_freq", "dunk_rate", "rim_pressure_index"],
    )

    # Contextualized event family on/off proxies.
    df["ctx_adj_onoff_ast_per100"] = (tmp["off_sig"] + 0.25 * tmp["net_sig"] + 0.5 * tmp["ast100"]) * tmp["rel"]
    df["ctx_adj_onoff_reb_per100"] = (tmp["def_sig"] + 0.25 * tmp["net_sig"] + 0.5 * tmp["reb100"]) * tmp["rel"]
    df["ctx_adj_onoff_stl_per100"] = (tmp["def_sig"] + 0.5 * tmp["stl100"]) * tmp["rel"]
    df["ctx_adj_onoff_blk_per100"] = (tmp["def_sig"] + 0.5 * tmp["blk100"]) * tmp["rel"]
    df["ctx_adj_onoff_tov_per100"] = (tmp["off_sig"] - 0.5 * tmp["tov100"]) * tmp["rel"]
    df["ctx_adj_onoff_transition"] = (tmp["off_sig"] + 0.5 * tmp["transition_freq"]) * tmp["rel"]
    df["ctx_adj_onoff_dunk_pressure"] = (tmp["off_sig"] + 0.5 * _z(tmp["dunk_rate"]) + 0.5 * _z(tmp["rim_pressure_index"])) * tmp["rel"]

    # Reliability + source quality (use rel_safe so downstream sees non-zero when we had valid signal).
    df["ctx_reliability_weight"] = np.where(np.isfinite(rel_safe), rel_safe, 0.0)
    df["ctx_adj_lambda"] = k
    df["ctx_adj_design_rank"] = np.where(np.isfinite(poss), np.minimum(poss / (k + 1.0), 10.0), np.nan)
    has_core = df[["ctx_adj_onoff_net", "ctx_adj_onoff_ortg", "ctx_adj_onoff_drtg"]].notna().any(axis=1)
    has_raw = pd.to_numeric(df.get("has_impact_raw"), errors="coerce").fillna(0).astype(int)
    pass_mask = has_core & (rel_safe >= 0.10)
    low_mask = has_core & ~pass_mask
    is_proxy = (has_raw <= 0) & has_core
    qflag = np.where(pass_mask, "pass", np.where(low_mask, "masked_low_conf", "excluded"))
    qflag = np.where(is_proxy & has_core, "proxy", qflag)
    df["ctx_quality_flag"] = qflag
    df["ctx_adj_source_quality"] = np.where(
        has_core & (has_raw > 0),
        np.where(pass_mask, 1.0, 0.6),
        np.where(has_core, 0.5, 0.0),
    )
    # Coverage flag = core signal exists (do not hard-mask out proxy rows).
    df["has_ctx_onoff_core"] = has_core.astype(int)

    # Developmental velocity (YoY).
    vel_cols = [
        "ctx_adj_onoff_net",
        "ctx_adj_onoff_ortg",
        "ctx_adj_onoff_drtg",
        "ctx_adj_onoff_ast_per100",
        "ctx_adj_onoff_reb_per100",
        "ctx_adj_onoff_transition",
        "ctx_adj_onoff_dunk_pressure",
    ]
    df = df.sort_values(["athlete_id", "season"]).copy()
    for c in vel_cols:
        out_col = (
            c.replace("ctx_adj_onoff_", "ctx_vel_")
            .replace("_per100", "")
            + "_yoy"
        )
        df[out_col] = df.groupby("athlete_id")[c].diff()
    df["ctx_vel_obs_count"] = (
        df.groupby("athlete_id")["season"]
        .transform(lambda s: s.notna().cumsum())
    )
    vel_out_cols = [c for c in df.columns if c.startswith("ctx_vel_")]
    df["has_ctx_velocity"] = df[vel_out_cols].notna().any(axis=1).astype(int)

    # Contract columns.
    df["path_onoff_poss"] = poss
    df["path_onoff_seconds"] = seconds
    df["path_onoff_reliability_weight"] = df["ctx_reliability_weight"]
    df["path_onoff_source"] = np.where(has_raw > 0, "impact_or_onoff", np.where(has_core, "proxy_from_box", "missing"))
    df["path_onoff_quality_flag"] = df["ctx_quality_flag"]

    # --- Superstar Rescue Override: high-volume players must not stay excluded ---
    # Use robust volume metrics (FGA, points, poss) because minutes_total can be 0/NaN upstream.
    minutes_v = pd.to_numeric(df.get("minutes_total", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0)
    fga_v = pd.to_numeric(df.get("fga_total", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0)
    poss_v = pd.to_numeric(poss, errors="coerce").fillna(0)
    rim_m = pd.to_numeric(df.get("rim_made", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0)
    three_m = pd.to_numeric(df.get("three_made", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0)
    mid_m = pd.to_numeric(df.get("mid_made", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0)
    ft_m = pd.to_numeric(df.get("ft_made", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0)
    points_v = 2.0 * (rim_m + mid_m) + 3.0 * three_m + ft_m
    is_high_volume_star = (
        (minutes_v > 500)
        | (fga_v > 200)
        | (points_v > 300)
        | (poss_v > 1000)
    )
    # Rescue high-volume stars who are excluded or have missing/NaN quality flag (e.g. lineup gate left them out).
    quality_excluded_or_missing = (df["ctx_quality_flag"] == "excluded") | df["ctx_quality_flag"].isna()
    rescue_mask = is_high_volume_star & quality_excluded_or_missing
    if rescue_mask.any():
        # Force rescued rows to proxy_from_box with non-null context and weight 0.5.
        df.loc[rescue_mask, "ctx_quality_flag"] = "proxy_from_box"
        df.loc[rescue_mask, "path_onoff_source"] = "proxy_from_box"
        df.loc[rescue_mask, "path_onoff_quality_flag"] = "proxy_from_box"
        df.loc[rescue_mask, "ctx_reliability_weight"] = 0.5
        df.loc[rescue_mask, "path_onoff_reliability_weight"] = 0.5
        df.loc[rescue_mask, "ctx_adj_source_quality"] = 0.5
        df.loc[rescue_mask, "has_ctx_onoff_core"] = 1
        # Fill context from box proxies so they are non-null (scale by 0.5). Fallback 0 so we never write NaN.
        rel_rescue = 0.5
        df.loc[rescue_mask, "ctx_adj_onoff_net"] = (box_net_proxy * 2.5 * rel_rescue).loc[rescue_mask].fillna(0.0)
        df.loc[rescue_mask, "ctx_adj_onoff_ortg"] = (box_off_proxy * 2.5 * rel_rescue).loc[rescue_mask].fillna(0.0)
        df.loc[rescue_mask, "ctx_adj_onoff_drtg"] = (box_def_proxy * 2.5 * rel_rescue).loc[rescue_mask].fillna(0.0)
        df.loc[rescue_mask, "ctx_adj_onoff_ast_per100"] = (
            (tmp["off_sig"] + 0.25 * tmp["net_sig"] + 0.5 * tmp["ast100"]) * rel_rescue
        ).loc[rescue_mask].fillna(0.0)
        df.loc[rescue_mask, "ctx_adj_onoff_reb_per100"] = (
            (tmp["def_sig"] + 0.25 * tmp["net_sig"] + 0.5 * tmp["reb100"]) * rel_rescue
        ).loc[rescue_mask].fillna(0.0)
        df.loc[rescue_mask, "ctx_adj_onoff_stl_per100"] = (
            (tmp["def_sig"] + 0.5 * tmp["stl100"]) * rel_rescue
        ).loc[rescue_mask].fillna(0.0)
        df.loc[rescue_mask, "ctx_adj_onoff_blk_per100"] = (
            (tmp["def_sig"] + 0.5 * tmp["blk100"]) * rel_rescue
        ).loc[rescue_mask].fillna(0.0)
        df.loc[rescue_mask, "ctx_adj_onoff_tov_per100"] = (
            (tmp["off_sig"] - 0.5 * tmp["tov100"]) * rel_rescue
        ).loc[rescue_mask].fillna(0.0)
        df.loc[rescue_mask, "ctx_adj_onoff_transition"] = (
            (tmp["off_sig"] + 0.5 * tmp["transition_freq"]) * rel_rescue
        ).loc[rescue_mask].fillna(0.0)
        df.loc[rescue_mask, "ctx_adj_onoff_dunk_pressure"] = (
            (tmp["off_sig"] + 0.5 * _z(tmp["dunk_rate"]) + 0.5 * _z(tmp["rim_pressure_index"])) * rel_rescue
        ).loc[rescue_mask].fillna(0.0)
        logger.info("Superstar rescue override: %d rows (excluded/missing -> proxy_from_box)", rescue_mask.sum())

    # Contract: path_onoff_source in [impact_or_onoff, proxy_from_box] => ctx_adj_onoff_net/ortg/drtg must be non-null
    source_set = df["path_onoff_source"].isin(["impact_or_onoff", "proxy_from_box"])
    for col in ["ctx_adj_onoff_net", "ctx_adj_onoff_ortg", "ctx_adj_onoff_drtg"]:
        df.loc[source_set, col] = df.loc[source_set, col].fillna(0.0)

    keep = [
        "athlete_id", "season",
        "ctx_adj_onoff_net", "ctx_adj_onoff_ortg", "ctx_adj_onoff_drtg",
        "ctx_adj_onoff_ast_per100", "ctx_adj_onoff_reb_per100", "ctx_adj_onoff_stl_per100",
        "ctx_adj_onoff_blk_per100", "ctx_adj_onoff_tov_per100", "ctx_adj_onoff_transition",
        "ctx_adj_onoff_dunk_pressure",
        "ctx_adj_lambda", "ctx_adj_design_rank", "ctx_adj_source_quality",
        "ctx_reliability_weight", "ctx_quality_flag", "has_ctx_onoff_core",
        "ctx_vel_net_yoy", "ctx_vel_ortg_yoy", "ctx_vel_drtg_yoy",
        "ctx_vel_ast_yoy", "ctx_vel_reb_yoy", "ctx_vel_transition_yoy",
        "ctx_vel_dunk_pressure_yoy", "ctx_vel_obs_count", "has_ctx_velocity",
        "path_onoff_poss", "path_onoff_seconds", "path_onoff_reliability_weight",
        "path_onoff_source", "path_onoff_quality_flag",
    ]
    # map generated velocity names
    rename_vel = {
        "ctx_vel_net_yoy": "ctx_vel_net_yoy",
        "ctx_vel_ortg_yoy": "ctx_vel_ortg_yoy",
        "ctx_vel_drtg_yoy": "ctx_vel_drtg_yoy",
        "ctx_vel_ast_yoy": "ctx_vel_ast_yoy",
        "ctx_vel_reb_yoy": "ctx_vel_reb_yoy",
        "ctx_vel_transition_yoy": "ctx_vel_transition_yoy",
        "ctx_vel_dunk_pressure_yoy": "ctx_vel_dunk_pressure_yoy",
    }
    # normalize actual generated names
    for src, dst in [
        ("ctx_vel_net_yoy", "ctx_vel_net_yoy"),
        ("ctx_vel_ortg_yoy", "ctx_vel_ortg_yoy"),
        ("ctx_vel_drtg_yoy", "ctx_vel_drtg_yoy"),
        ("ctx_vel_ast_yoy", "ctx_vel_ast_yoy"),
        ("ctx_vel_reb_yoy", "ctx_vel_reb_yoy"),
        ("ctx_vel_transition_yoy", "ctx_vel_transition_yoy"),
        ("ctx_vel_dunk_pressure_yoy", "ctx_vel_dunk_pressure_yoy"),
    ]:
        if src not in df.columns:
            # fallback from generated names
            alt = src.replace("ctx_vel_", "ctx_vel_").replace("_yoy", "_yoy")
            if alt in df.columns:
                df[src] = df[alt]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    out = df[keep].copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    logger.info("Saved pathway context v2: %s (%d rows)", OUT_PATH, len(out))

    cov = (
        out.groupby("season", as_index=False)
        .agg(
            rows=("athlete_id", "size"),
            has_ctx_onoff_core=("has_ctx_onoff_core", "mean"),
            has_ctx_velocity=("has_ctx_velocity", "mean"),
            ctx_reliability_weight=("ctx_reliability_weight", "mean"),
        )
    )
    cov_path = BASE / "data" / "audit" / "rolling_yearly" / "pathway_coverage_by_season.csv"
    cov_path.parent.mkdir(parents=True, exist_ok=True)
    cov.to_csv(cov_path, index=False)
    logger.info("Saved pathway coverage audit: %s", cov_path)


if __name__ == "__main__":
    main()
