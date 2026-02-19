#!/usr/bin/env python3
"""
Build NBA Development-Rate Fact Table (Y1->Y3)
==============================================

Creates a latent Bayesian development-rate target using Year 1-3 EPM and
3-year RAPM windows. The latent per-player state is:

    theta_it = alpha_i + d_i * (t - 1)

with observations:

    y_it ~ Normal(theta_it, sigma_epm^2 / w_epm_it)
    r_i  ~ Normal((theta_i1 + theta_i2 + theta_i3) / 3, sigma_rapm^2 / w_rapm_i)

where w_epm_it = max(minutes_it, m0) and w_rapm_i = max(off_poss_i, p0).

The prior is empirical-Bayes:

    [alpha_i, d_i] ~ Normal(B * x_i, Sigma_player)

fitted via EM with closed-form E-step per player.

Output:
- data/warehouse_v2/fact_player_development_rate.parquet
- data/warehouse_v2/audit/fact_player_development_rate_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
NBA_MERGED_PATH = BASE_DIR / "data/nba_merged/nba_player_season_merged_2004_2025.parquet"
RAPM_PATH = BASE_DIR / "data/nba_six_factor_rapm.csv"
DIM_PATH = BASE_DIR / "data/warehouse_v2/dim_player_nba.parquet"
OUT_PATH = BASE_DIR / "data/warehouse_v2/fact_player_development_rate.parquet"
AUDIT_DIR = BASE_DIR / "data/warehouse_v2/audit"

MODEL_VERSION = "dev_rate_y1_y3_latent_eb_v1"
P10_Z = -1.2815515655446004
P90_Z = 1.2815515655446004


@dataclass
class ModelConfig:
    m0: float = 150.0
    p0: float = 1500.0
    w_min: float = 0.05
    w_max: float = 250.0
    max_iter: int = 40
    tol: float = 1e-4
    ridge: float = 1e-5
    min_sigma2: float = 1e-3


def _first_existing(columns: Sequence[str], candidates: Sequence[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    if required:
        raise KeyError(f"None of required columns found: {candidates}")
    return None


def _to_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").round().astype("Int64")


def _standardize_with_missing_indicator(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    out = pd.DataFrame(index=df.index)
    used: List[str] = []

    for col in cols:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        missing = x.isna().astype(float)
        mean = x.mean(skipna=True)
        std = x.std(skipna=True)
        if not np.isfinite(mean):
            mean = 0.0
        if not np.isfinite(std) or std <= 0:
            std = 1.0

        z = (x - mean) / std
        z = z.fillna(0.0)

        z_col = f"basis_{col}_z"
        m_col = f"basis_{col}_missing"
        out[z_col] = z.astype(float)
        out[m_col] = missing.astype(float)
        used.extend([z_col, m_col])

    return out, used


def load_dim_player(dim_path: Path) -> pd.DataFrame:
    dim = pd.read_parquet(dim_path)
    if "nba_id" not in dim.columns:
        raise KeyError("dim_player_nba.parquet missing 'nba_id'")

    dim = dim.copy()
    dim["nba_id"] = _to_int_series(dim["nba_id"])
    dim = dim.dropna(subset=["nba_id"]).copy()
    dim["nba_id"] = dim["nba_id"].astype(int)
    dim = dim.drop_duplicates(subset=["nba_id"], keep="first")

    # Wingspan scaffolding: ensure columns exist for downstream schema compatibility.
    for c in ["wingspan_in", "standing_reach_in", "wingspan_minus_height_in", "has_wingspan"]:
        if c not in dim.columns:
            dim[c] = np.nan if c != "has_wingspan" else 0.0

    if "has_wingspan" in dim.columns:
        dim["has_wingspan"] = pd.to_numeric(dim["has_wingspan"], errors="coerce").fillna(0.0)

    return dim


def build_epm_observations(nba_merged_path: Path) -> Tuple[pd.DataFrame, Dict[int, int]]:
    df = pd.read_parquet(nba_merged_path)

    id_col = _first_existing(df.columns, ["nba_id", "nid", "nba_player_id", "epm__player_id"])
    season_col = _first_existing(df.columns, ["season_year", "epm__season"])
    rookie_col = _first_existing(df.columns, ["rookie_season_year", "epm__rookie_year"])
    epm_col = _first_existing(df.columns, ["epm__tot", "year1_epm_tot"])
    mp_col = _first_existing(df.columns, ["minutes", "epm__mp", "mp"], required=False)

    d = pd.DataFrame(
        {
            "nba_id": _to_int_series(df[id_col]),
            "season": _to_int_series(df[season_col]),
            "rookie_anchor": pd.to_numeric(df[rookie_col], errors="coerce"),
            "epm_tot": pd.to_numeric(df[epm_col], errors="coerce"),
        }
    )
    if mp_col is not None:
        d["minutes"] = pd.to_numeric(df[mp_col], errors="coerce")
    else:
        d["minutes"] = np.nan

    d = d.dropna(subset=["nba_id", "season", "epm_tot", "rookie_anchor"]).copy()
    d["nba_id"] = d["nba_id"].astype(int)
    d["season"] = d["season"].astype(int)

    # Existing pipeline stores `epm__rookie_year` as draft year, so tenure is season-rookie_year.
    if rookie_col == "epm__rookie_year":
        d["tenure_year"] = d["season"] - d["rookie_anchor"]
        rookie_season = (d["rookie_anchor"] + 1).round()
    else:
        d["tenure_year"] = d["season"] - d["rookie_anchor"] + 1
        rookie_season = d["rookie_anchor"].round()

    d["rookie_season_year"] = pd.to_numeric(rookie_season, errors="coerce").astype("Int64")
    d = d[(d["tenure_year"] >= 1) & (d["tenure_year"] <= 3)].copy()
    d["tenure_year"] = d["tenure_year"].astype(int)

    # Dedup to one row per player-tenure via minutes-weighted average EPM.
    d["minutes"] = d["minutes"].fillna(0.0)
    d["weighted_epm"] = d["epm_tot"] * d["minutes"]

    grouped = (
        d.groupby(["nba_id", "tenure_year"], as_index=False)
        .agg(
            epm_num=("weighted_epm", "sum"),
            minutes=("minutes", "sum"),
            epm_fallback=("epm_tot", "mean"),
            rookie_season_year=("rookie_season_year", "min"),
        )
    )
    grouped["epm_tot"] = np.where(
        grouped["minutes"] > 0,
        grouped["epm_num"] / grouped["minutes"],
        grouped["epm_fallback"],
    )

    grouped = grouped[["nba_id", "tenure_year", "epm_tot", "minutes", "rookie_season_year"]]
    grouped = grouped.sort_values(["nba_id", "tenure_year"]).reset_index(drop=True)

    rookie_map = (
        grouped[["nba_id", "rookie_season_year"]]
        .dropna(subset=["rookie_season_year"])
        .drop_duplicates(subset=["nba_id"], keep="first")
    )
    rookie_map_dict = {int(r.nba_id): int(r.rookie_season_year) for r in rookie_map.itertuples(index=False)}

    logger.info("EPM observations: %d player-tenure rows", len(grouped))
    return grouped, rookie_map_dict


def build_rapm_observations(rapm_path: Path, rookie_lookup: Dict[int, int]) -> pd.DataFrame:
    rapm = pd.read_csv(rapm_path)

    year_interval_col = _first_existing(rapm.columns, ["Year_Interval", "Interval"])
    latest_col = _first_existing(rapm.columns, ["Latest_Year"])
    poss_col = _first_existing(rapm.columns, ["Off_Poss"])
    rapm_col = _first_existing(rapm.columns, ["OVR_RAPM"])

    d = pd.DataFrame(
        {
            "nba_id": _to_int_series(rapm["nba_id"]),
            "year_interval": rapm[year_interval_col].astype(str),
            "latest_year": _to_int_series(rapm[latest_col]),
            "off_poss": pd.to_numeric(rapm[poss_col], errors="coerce"),
            "rapm_ovr": pd.to_numeric(rapm[rapm_col], errors="coerce"),
        }
    )

    d = d.dropna(subset=["nba_id", "latest_year", "off_poss", "rapm_ovr"]).copy()
    d["nba_id"] = d["nba_id"].astype(int)
    d["latest_year"] = d["latest_year"].astype(int)

    d = d[d["year_interval"].str.upper() == "3Y"].copy()
    d["rookie_season_year"] = d["nba_id"].map(rookie_lookup)
    d["target_latest_year"] = d["rookie_season_year"] + 2
    d = d[d["rookie_season_year"].notna()]
    d = d[d["latest_year"] == d["target_latest_year"]]

    # Deduplicate by highest possession sample if multiple records exist.
    d = d.sort_values(["nba_id", "off_poss"], ascending=[True, False]).drop_duplicates(subset=["nba_id"], keep="first")
    d = d[["nba_id", "rapm_ovr", "off_poss", "latest_year"]].reset_index(drop=True)

    logger.info("RAPM observations (3Y rookie+2 windows): %d players", len(d))
    return d


def _posterior_for_player(
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    obs_rows: List[Tuple[np.ndarray, float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute posterior for [alpha, d] given Gaussian prior and weighted observations.

    obs_rows contains tuples (h_row, y, obs_variance) where h_row is shape (2,).
    """
    if not obs_rows:
        return prior_mean, prior_cov

    inv_prior = np.linalg.inv(prior_cov)
    precision = inv_prior.copy()
    rhs = inv_prior @ prior_mean

    for h, y, var in obs_rows:
        inv_var = 1.0 / max(var, 1e-9)
        precision += inv_var * np.outer(h, h)
        rhs += inv_var * h * y

    post_cov = np.linalg.inv(precision)
    post_mean = post_cov @ rhs
    return post_mean, post_cov


def fit_empirical_bayes(
    player_ids: np.ndarray,
    X: np.ndarray,
    epm_obs: Dict[int, List[Tuple[int, float, float]]],
    rapm_obs: Dict[int, Tuple[float, float]],
    cfg: ModelConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, int, bool]:
    """
    Fit EM for latent [alpha, d] model.

    Returns:
      post_mean [N,2], post_cov [N,2,2], B [2,P], Sigma [2,2], sigma_epm2, sigma_rapm2
    """
    n, p = X.shape

    # Initialize priors and noise scales.
    B = np.zeros((2, p), dtype=float)
    Sigma = np.array([[1.0, 0.0], [0.0, 0.25]], dtype=float)

    # Heuristic initialization from observed target scales.
    epm_values = [y for rows in epm_obs.values() for (_, y, _) in rows]
    rapm_values = [v[0] for v in rapm_obs.values()]
    sigma_epm2 = max(np.nanvar(epm_values) if epm_values else 1.0, cfg.min_sigma2)
    sigma_rapm2 = max(np.nanvar(rapm_values) if rapm_values else sigma_epm2, cfg.min_sigma2)

    post_mean = np.zeros((n, 2), dtype=float)
    post_cov = np.repeat(Sigma[None, :, :], n, axis=0)

    converged = False
    for it in range(1, cfg.max_iter + 1):
        prev_B = B.copy()
        prev_Sigma = Sigma.copy()
        prev_sig_epm2 = sigma_epm2
        prev_sig_rapm2 = sigma_rapm2

        # E-step
        epm_weighted_resid = 0.0
        epm_obs_count = 0
        rapm_weighted_resid = 0.0
        rapm_obs_count = 0

        for i, pid in enumerate(player_ids):
            mu_i = B @ X[i]

            obs_rows: List[Tuple[np.ndarray, float, float]] = []

            for t, y, w in epm_obs.get(int(pid), []):
                h = np.array([1.0, float(t - 1)], dtype=float)
                w_eff = max(w, cfg.m0)
                var = sigma_epm2 / max(w_eff, 1e-6)
                obs_rows.append((h, y, var))

            if int(pid) in rapm_obs:
                r, w_r = rapm_obs[int(pid)]
                h_r = np.array([1.0, 1.0], dtype=float)  # mean over t=1..3 -> alpha + d
                w_r_eff = max(w_r, cfg.p0)
                var_r = sigma_rapm2 / max(w_r_eff, 1e-6)
                obs_rows.append((h_r, r, var_r))

            m_i, S_i = _posterior_for_player(mu_i, Sigma, obs_rows)
            post_mean[i] = m_i
            post_cov[i] = S_i

            for t, y, w in epm_obs.get(int(pid), []):
                h = np.array([1.0, float(t - 1)], dtype=float)
                err2 = float((y - h @ m_i) ** 2 + h @ S_i @ h)
                epm_weighted_resid += max(w, cfg.m0) * err2
                epm_obs_count += 1

            if int(pid) in rapm_obs:
                r, w_r = rapm_obs[int(pid)]
                h_r = np.array([1.0, 1.0], dtype=float)
                err2_r = float((r - h_r @ m_i) ** 2 + h_r @ S_i @ h_r)
                rapm_weighted_resid += max(w_r, cfg.p0) * err2_r
                rapm_obs_count += 1

        # M-step: B via ridge regression on posterior means.
        XtX = X.T @ X
        XtX.flat[:: XtX.shape[0] + 1] += cfg.ridge
        B_t = np.linalg.solve(XtX, X.T @ post_mean)
        B = B_t.T

        # M-step: Sigma from posterior second moments.
        prior_means = X @ B.T
        centered = post_mean - prior_means
        Sigma_accum = np.zeros((2, 2), dtype=float)
        for i in range(n):
            Sigma_accum += post_cov[i] + np.outer(centered[i], centered[i])
        Sigma = Sigma_accum / max(n, 1)
        Sigma.flat[::3] += cfg.ridge

        # M-step: observation noise scales.
        if epm_obs_count > 0:
            sigma_epm2 = max(epm_weighted_resid / epm_obs_count, cfg.min_sigma2)
        if rapm_obs_count > 0:
            sigma_rapm2 = max(rapm_weighted_resid / rapm_obs_count, cfg.min_sigma2)

        delta = max(
            float(np.max(np.abs(B - prev_B))),
            float(np.max(np.abs(Sigma - prev_Sigma))),
            abs(float(sigma_epm2 - prev_sig_epm2)),
            abs(float(sigma_rapm2 - prev_sig_rapm2)),
        )

        logger.info(
            "EM iter %d/%d | delta=%.6f | sigma_epm2=%.4f | sigma_rapm2=%.4f",
            it,
            cfg.max_iter,
            delta,
            sigma_epm2,
            sigma_rapm2,
        )

        if delta < cfg.tol:
            converged = True
            break

    return post_mean, post_cov, B, Sigma, sigma_epm2, sigma_rapm2, it, converged


def build_development_fact(
    nba_merged_path: Path,
    rapm_path: Path,
    dim_path: Path,
    output_path: Path,
    cfg: ModelConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    dim = load_dim_player(dim_path)
    epm_rows, rookie_from_epm = build_epm_observations(nba_merged_path)

    # Rookie lookup priority: dim_player_nba first, then EPM-derived fallback.
    rookie_lookup = {
        int(r.nba_id): int(r.rookie_season_year)
        for r in dim[["nba_id", "rookie_season_year"]].dropna().itertuples(index=False)
    }
    for k, v in rookie_from_epm.items():
        rookie_lookup.setdefault(k, v)

    rapm_rows = build_rapm_observations(rapm_path, rookie_lookup)

    # Observation maps.
    epm_obs: Dict[int, List[Tuple[int, float, float]]] = {}
    for r in epm_rows.itertuples(index=False):
        epm_obs.setdefault(int(r.nba_id), []).append((int(r.tenure_year), float(r.epm_tot), float(r.minutes)))

    rapm_obs: Dict[int, Tuple[float, float]] = {
        int(r.nba_id): (float(r.rapm_ovr), float(r.off_poss))
        for r in rapm_rows.itertuples(index=False)
    }

    # Universe of players to score.
    universe = sorted(set(dim["nba_id"].astype(int).tolist()) | set(epm_obs.keys()) | set(rapm_obs.keys()))
    players = pd.DataFrame({"nba_id": universe})

    # Pre-NBA basis terms from dim (schema-safe if columns are missing).
    basis_source = players.merge(dim, on="nba_id", how="left")
    basis_base = pd.DataFrame(index=basis_source.index)
    basis_base["basis_intercept"] = 1.0

    numeric_basis_candidates = [
        "ht_first",
        "wt_first",
        "draft_year",
        "rookie_season_year",
        "wingspan_in",
        "standing_reach_in",
        "wingspan_minus_height_in",
    ]
    basis_numeric, used_numeric = _standardize_with_missing_indicator(basis_source, numeric_basis_candidates)
    basis_base = pd.concat([basis_base, basis_numeric], axis=1)

    if "has_wingspan" in basis_source.columns:
        basis_base["basis_has_wingspan"] = pd.to_numeric(basis_source["has_wingspan"], errors="coerce").fillna(0.0)

    basis_cols = basis_base.columns.tolist()
    X = basis_base.to_numpy(dtype=float)

    player_ids = players["nba_id"].to_numpy(dtype=int)
    post_mean, post_cov, B, Sigma, sigma_epm2, sigma_rapm2, iters, converged = fit_empirical_bayes(
        player_ids=player_ids,
        X=X,
        epm_obs=epm_obs,
        rapm_obs=rapm_obs,
        cfg=cfg,
    )

    dev_mean = post_mean[:, 1]
    dev_sd = np.sqrt(np.clip(post_cov[:, 1, 1], 1e-12, None))

    out = pd.DataFrame(
        {
            "nba_id": player_ids,
            "dev_rate_y1_y3_mean": dev_mean,
            "dev_rate_y1_y3_sd": dev_sd,
            "dev_rate_y1_y3_p10": dev_mean + P10_Z * dev_sd,
            "dev_rate_y1_y3_p50": dev_mean,
            "dev_rate_y1_y3_p90": dev_mean + P90_Z * dev_sd,
            "dev_rate_quality_weight": np.clip(1.0 / np.clip(dev_sd ** 2, 1e-9, None), cfg.w_min, cfg.w_max),
            "dev_has_y1": [int(any(t == 1 for (t, _, _) in epm_obs.get(int(pid), []))) for pid in player_ids],
            "dev_has_y2": [int(any(t == 2 for (t, _, _) in epm_obs.get(int(pid), []))) for pid in player_ids],
            "dev_has_y3": [int(any(t == 3 for (t, _, _) in epm_obs.get(int(pid), []))) for pid in player_ids],
            "dev_has_rapm3y": [int(int(pid) in rapm_obs) for pid in player_ids],
            "dev_obs_epm_count": [int(len(epm_obs.get(int(pid), []))) for pid in player_ids],
            "dev_obs_rapm_count": [int(1 if int(pid) in rapm_obs else 0) for pid in player_ids],
            "dev_model_version": MODEL_VERSION,
        }
    )

    out = out.sort_values("nba_id").reset_index(drop=True)

    # Diagnostics and QA checks.
    out_non_null = out["dev_rate_y1_y3_mean"].notna().mean()
    uncertainty_non_null = out["dev_rate_y1_y3_sd"].notna().mean()

    overlap = epm_rows.merge(rapm_rows[["nba_id"]], on="nba_id", how="inner")
    if "rookie_season_year" in epm_rows.columns:
        overlap_by_rookie = (
            overlap[["nba_id", "rookie_season_year"]]
            .dropna()
            .groupby("rookie_season_year")
            .size()
            .sort_index()
            .to_dict()
        )
    else:
        overlap_by_rookie = {}

    diagnostics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "config": {
            "m0": cfg.m0,
            "p0": cfg.p0,
            "w_min": cfg.w_min,
            "w_max": cfg.w_max,
            "max_iter": cfg.max_iter,
            "tol": cfg.tol,
            "ridge": cfg.ridge,
        },
        "source_rows": {
            "dim_player_nba": int(len(dim)),
            "epm_player_tenure_rows": int(len(epm_rows)),
            "rapm_y1_to_y3_rows": int(len(rapm_rows)),
        },
        "coverage": {
            "players_scored": int(len(out)),
            "dev_rate_non_null_rate": float(out_non_null),
            "dev_uncertainty_non_null_rate": float(uncertainty_non_null),
            "dev_has_y1_rate": float(out["dev_has_y1"].mean()),
            "dev_has_y2_rate": float(out["dev_has_y2"].mean()),
            "dev_has_y3_rate": float(out["dev_has_y3"].mean()),
            "dev_has_rapm3y_rate": float(out["dev_has_rapm3y"].mean()),
        },
        "distribution": {
            "mean": float(out["dev_rate_y1_y3_mean"].mean()),
            "std": float(out["dev_rate_y1_y3_mean"].std()),
            "p05": float(out["dev_rate_y1_y3_mean"].quantile(0.05)),
            "p50": float(out["dev_rate_y1_y3_mean"].quantile(0.50)),
            "p95": float(out["dev_rate_y1_y3_mean"].quantile(0.95)),
            "uncertainty_p50": float(out["dev_rate_y1_y3_sd"].quantile(0.50)),
            "uncertainty_p90": float(out["dev_rate_y1_y3_sd"].quantile(0.90)),
        },
        "em_fit": {
            "iterations": int(iters),
            "converged": bool(converged),
            "sigma_epm2": float(sigma_epm2),
            "sigma_rapm2": float(sigma_rapm2),
            "Sigma_player": Sigma.tolist(),
            "basis_columns": basis_cols,
            "B_shape": list(B.shape),
        },
        "drift_checks": {
            "epm_rapm_overlap_players": int(overlap["nba_id"].nunique()),
            "epm_rapm_overlap_by_rookie_season": {str(k): int(v) for k, v in overlap_by_rookie.items()},
        },
        "quality_gate": {
            "min_non_null_rate": 0.60,
            "passed": bool(out_non_null >= 0.60 and uncertainty_non_null >= 0.60),
            "non_critical_notes": [
                "gap_usg_legacy is intentionally excluded from critical readiness for this module.",
            ],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit_latest = AUDIT_DIR / "fact_player_development_rate_diagnostics.json"
    audit_timestamped = AUDIT_DIR / f"fact_player_development_rate_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    audit_latest.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    audit_timestamped.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    return out, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Y1->Y3 latent development-rate fact table")
    parser.add_argument("--nba-merged", type=Path, default=NBA_MERGED_PATH)
    parser.add_argument("--rapm", type=Path, default=RAPM_PATH)
    parser.add_argument("--dim", type=Path, default=DIM_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--m0", type=float, default=150.0, help="Minutes floor for EPM reliability weights")
    parser.add_argument("--p0", type=float, default=1500.0, help="Possession floor for RAPM reliability weights")
    parser.add_argument("--w-min", type=float, default=0.05, help="Minimum clipped quality weight")
    parser.add_argument("--w-max", type=float, default=250.0, help="Maximum clipped quality weight")
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--tol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ModelConfig(
        m0=args.m0,
        p0=args.p0,
        w_min=args.w_min,
        w_max=args.w_max,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    logger.info("Building fact_player_development_rate...")
    logger.info("Inputs: nba_merged=%s | rapm=%s | dim=%s", args.nba_merged, args.rapm, args.dim)

    out, diagnostics = build_development_fact(
        nba_merged_path=args.nba_merged,
        rapm_path=args.rapm,
        dim_path=args.dim,
        output_path=args.out,
        cfg=cfg,
    )

    logger.info("Wrote %s (%d rows)", args.out, len(out))
    logger.info(
        "Coverage: y1=%.2f%% y2=%.2f%% y3=%.2f%% rapm3y=%.2f%%",
        100.0 * diagnostics["coverage"]["dev_has_y1_rate"],
        100.0 * diagnostics["coverage"]["dev_has_y2_rate"],
        100.0 * diagnostics["coverage"]["dev_has_y3_rate"],
        100.0 * diagnostics["coverage"]["dev_has_rapm3y_rate"],
    )
    logger.info("Quality gate passed: %s", diagnostics["quality_gate"]["passed"])


if __name__ == "__main__":
    main()
