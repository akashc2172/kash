"""
Build college development-rate labels from multi-season college panel.

Outputs one row per athlete final college season with posterior-like slope summaries.
"""

from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
CAREER_LONG_PATH = BASE_DIR / "data/college_feature_store/prospect_career_long_v1.parquet"
IMPACT_STACK_PATH = BASE_DIR / "data/college_feature_store/college_impact_stack_v1.parquet"
OUT_PATH = BASE_DIR / "data/college_feature_store/fact_player_college_development_rate.parquet"
VERSION = "college_dev_rate_eb_v1"


def _series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    """Return an index-aligned Series even if a column is missing."""
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index, dtype="float64")


def weighted_slope_and_sd(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted OLS slope and approximate standard error."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[mask], y[mask], w[mask]
    n = len(x)
    if n < 2:
        return np.nan, np.nan

    sw = np.sum(w)
    x_bar = np.sum(w * x) / sw
    y_bar = np.sum(w * y) / sw
    sxx = np.sum(w * (x - x_bar) ** 2)
    if sxx <= 1e-10:
        return np.nan, np.nan
    sxy = np.sum(w * (x - x_bar) * (y - y_bar))
    slope = sxy / sxx

    # Approximate slope uncertainty.
    y_hat = y_bar + slope * (x - x_bar)
    rss = np.sum(w * (y - y_hat) ** 2)
    dof = max(n - 2, 1)
    sigma2 = rss / dof
    se = np.sqrt(max(sigma2 / sxx, 1e-12))
    return float(slope), float(se)


def build_labels(career: pd.DataFrame, impact: pd.DataFrame | None) -> pd.DataFrame:
    df = career.copy()
    df = df.sort_values(["athlete_id", "season"]).reset_index(drop=True)

    # ---------------------------------------------------------------------------
    # Derive columns that are missing/sparse in the upstream career-long panel
    # ---------------------------------------------------------------------------

    # usage: approximate from (fga + 0.44*fta + tov) / poss
    if "usage" not in df.columns:
        df["usage"] = np.nan
    if {"fga_total", "tov_total", "poss_total"}.issubset(df.columns):
        fga = pd.to_numeric(df["fga_total"], errors="coerce")
        ft_att_raw = _series(df, "ft_att")
        if ft_att_raw.isna().all():
            ft_att_raw = _series(df, "ft_att_total", default=0.0)
        ft_att = pd.to_numeric(ft_att_raw, errors="coerce").fillna(0)
        tov = pd.to_numeric(df["tov_total"], errors="coerce")
        poss = pd.to_numeric(df["poss_total"], errors="coerce")
        usg_approx = (fga + 0.44 * ft_att + tov) / poss.clip(lower=1)
        can_usg = df["usage"].isna() & fga.notna() & tov.notna() & poss.notna() & (poss > 0)
        df.loc[can_usg, "usage"] = usg_approx[can_usg]

    if impact is not None and not impact.empty:
        keep = ["athlete_id", "season", "rIPM_tot_std", "impact_reliability_weight"]
        keep = [c for c in keep if c in impact.columns]
        df = df.merge(impact[keep], on=["athlete_id", "season"], how="left")

    out_rows = []
    for athlete_id, g in df.groupby("athlete_id", sort=False):
        g = g.sort_values("season").reset_index(drop=True)
        tenure = np.arange(len(g), dtype=float)

        # Exposure weights: prefer minutes, then possessions, then flat.
        minutes = pd.to_numeric(_series(g, "minutes_total"), errors="coerce").values
        poss = pd.to_numeric(_series(g, "poss_total"), errors="coerce").values
        w_base = np.where(np.isfinite(minutes) & (minutes > 0), minutes, poss)
        w_base = np.where(np.isfinite(w_base) & (w_base > 0), w_base, 1.0)

        usage = pd.to_numeric(_series(g, "usage"), errors="coerce").values
        ts = pd.to_numeric(_series(g, "trueShootingPct"), errors="coerce").values
        ast = pd.to_numeric(_series(g, "ast_total"), errors="coerce").values
        tov_arr = pd.to_numeric(_series(g, "tov_total"), errors="coerce").values
        # Creation rate per 100 poss.  minutes_total is zero for all pre-2025
        # rows, so we use poss_total (universally populated) as the denominator.
        creation = np.where(
            np.isfinite(poss) & (poss > 0),
            (ast - tov_arr) / poss * 100.0,
            np.nan,
        )

        impact_y = pd.to_numeric(_series(g, "rIPM_tot_std"), errors="coerce").values
        impact_w = pd.to_numeric(_series(g, "impact_reliability_weight"), errors="coerce").values
        impact_w = np.where(np.isfinite(impact_w) & (impact_w > 0), impact_w, w_base)

        s_off, se_off = weighted_slope_and_sd(tenure, usage, w_base)
        s_eff, se_eff = weighted_slope_and_sd(tenure, ts, w_base)
        s_creation, se_creation = weighted_slope_and_sd(tenure, creation, w_base)
        s_impact, se_impact = weighted_slope_and_sd(tenure, impact_y, impact_w)

        means = np.array([s_off, s_eff, s_creation, s_impact], dtype=float)
        sds = np.array([se_off, se_eff, se_creation, se_impact], dtype=float)
        valid = np.isfinite(means) & np.isfinite(sds)

        if np.any(valid):
            dev_mean = float(np.nanmean(means[valid]))
            dev_sd = float(np.sqrt(np.nanmean(np.square(sds[valid]))))
        else:
            dev_mean, dev_sd = np.nan, np.nan

        # Transfer marker from team changes.
        has_transfer = int(g["teamId"].dropna().nunique() > 1) if "teamId" in g.columns else 0
        final_season = int(g["season"].max())

        quality = np.nan
        if np.isfinite(dev_sd):
            quality = float(np.clip(1.0 / max(dev_sd * dev_sd, 1e-6), 0.05, 10.0))

        out_rows.append(
            {
                "athlete_id": int(athlete_id),
                "final_college_season": final_season,
                "college_dev_rate_off_mean": s_off,
                "college_dev_rate_off_sd": se_off,
                "college_dev_rate_eff_mean": s_eff,
                "college_dev_rate_eff_sd": se_eff,
                "college_dev_rate_creation_mean": s_creation,
                "college_dev_rate_creation_sd": se_creation,
                "college_dev_rate_impact_mean": s_impact,
                "college_dev_rate_impact_sd": se_impact,
                "college_dev_rate_phys_mean": np.nan,
                "college_dev_rate_phys_sd": np.nan,
                "college_dev_p10": dev_mean - 1.2816 * dev_sd if np.isfinite(dev_mean) and np.isfinite(dev_sd) else np.nan,
                "college_dev_p50": dev_mean,
                "college_dev_p90": dev_mean + 1.2816 * dev_sd if np.isfinite(dev_mean) and np.isfinite(dev_sd) else np.nan,
                "college_dev_quality_weight": quality,
                "college_dev_obs_years": int(len(g)),
                "college_dev_has_transfer": has_transfer,
                "college_dev_model_version": VERSION,
            }
        )

    return pd.DataFrame(out_rows)


def main() -> None:
    if not CAREER_LONG_PATH.exists():
        raise FileNotFoundError(f"Missing input: {CAREER_LONG_PATH}")
    career = pd.read_parquet(CAREER_LONG_PATH)
    impact = pd.read_parquet(IMPACT_STACK_PATH) if IMPACT_STACK_PATH.exists() else None
    out = build_labels(career, impact)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    logger.info("Saved %s (%d rows)", OUT_PATH, len(out))
    if not out.empty:
        cov = out["college_dev_p50"].notna().mean() * 100.0
        logger.info("college_dev coverage: %.1f%%", cov)


if __name__ == "__main__":
    main()
