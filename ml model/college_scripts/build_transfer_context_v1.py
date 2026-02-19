"""
Build transfer context features from career-long college panel.
"""

from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
IN_PATH = BASE_DIR / "data/college_feature_store/prospect_career_long_v1.parquet"
FEATURES_PATH = BASE_DIR / "data/college_feature_store/college_features_v1.parquet"
OUT_PATH = BASE_DIR / "data/college_feature_store/fact_player_transfer_context.parquet"
VERSION = "transfer_context_v1"


def _series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    """Return an index-aligned Series even if the column is missing."""
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index, dtype="float64")


def build_transfer_context(df: pd.DataFrame) -> pd.DataFrame:
    req = {"athlete_id", "season", "teamId"}
    if not req.issubset(df.columns):
        missing = sorted(req - set(df.columns))
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values(["athlete_id", "season"]).copy()

    # Enrich with season-level context from the canonical feature store when available.
    # This fills conf/pace metadata that may be absent in career_long for older seasons.
    if FEATURES_PATH.exists():
        base = pd.read_parquet(FEATURES_PATH, columns=["athlete_id", "season", "split_id", "is_power_conf", "team_pace"])
        base = base[base["split_id"] == "ALL__ALL"].copy()
        base = base.sort_values(["athlete_id", "season"]).drop_duplicates(["athlete_id", "season"], keep="last")
        enrich = base[["athlete_id", "season", "is_power_conf", "team_pace"]]
        for c in ["is_power_conf", "team_pace"]:
            if c in df.columns:
                df = df.drop(columns=[c])
        df = df.merge(enrich, on=["athlete_id", "season"], how="left")

    # ---------------------------------------------------------------------------
    # Derive columns the original code assumed existed but are mostly/entirely null
    # ---------------------------------------------------------------------------

    # Pace proxy:
    # 1) Prefer team_pace when present (possessions per game scale).
    # 2) Fallback ONLY if we can keep the same scale (poss_total / games_played).
    #    Do not fallback to raw poss_total, which is season-exposure scale and
    #    causes large artificial transfer_pace_delta values.
    team_pace = pd.to_numeric(_series(df, "team_pace"), errors="coerce")
    poss_total = pd.to_numeric(_series(df, "poss_total"), errors="coerce")
    games_played = pd.to_numeric(_series(df, "games_played"), errors="coerce")
    pace_fallback = np.where(games_played > 0, poss_total / games_played, np.nan)
    # Last fallback: map poss_total into pace scale using robust ratio calibration.
    # Ratio should be ~0.02-0.12 (pace~65-75, poss_total~600-3000).
    # Previous wide clipping could inflate scale and create unrealistic deltas.
    overlap = team_pace.notna() & poss_total.notna() & (poss_total > 0)
    if overlap.any():
        ratio = (team_pace[overlap] / poss_total[overlap]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio) > 0:
            scale = float(np.clip(ratio.median(), 0.01, 0.2))
        else:
            scale = np.nan
    else:
        scale = np.nan
    poss_scaled = poss_total * scale if np.isfinite(scale) else np.nan
    pace_proxy = np.where(np.isfinite(pace_fallback), pace_fallback, poss_scaled)
    df["pace_proxy"] = np.where(team_pace.notna(), team_pace, pace_proxy)

    # usage: approximate from (fga + 0.44*fta + tov) / poss when null
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

    rows = []

    for athlete_id, g in df.groupby("athlete_id", sort=False):
        g = g.sort_values("season").reset_index(drop=True)
        if len(g) < 2:
            continue

        for i in range(1, len(g)):
            prev = g.iloc[i - 1]
            cur = g.iloc[i]

            if pd.isna(prev["teamId"]) or pd.isna(cur["teamId"]):
                continue
            if int(prev["teamId"]) == int(cur["teamId"]):
                continue

            # Pace delta uses calibrated pace_proxy so pre-2025 seasons are usable
            # without mixing raw season-volume scales.
            pace_prev = prev.get("pace_proxy", np.nan)
            pace_cur = cur.get("pace_proxy", np.nan)
            usage_prev = prev.get("usage", np.nan)
            usage_cur = cur.get("usage", np.nan)
            ts_prev = prev.get("trueShootingPct", np.nan)
            ts_cur = cur.get("trueShootingPct", np.nan)

            # Conference metadata is enriched from college_features_v1 when present.
            # If still unavailable for a row, delta remains NaN.
            conf_prev = prev.get("is_power_conf", np.nan)
            conf_cur = cur.get("is_power_conf", np.nan)

            transfer_conf_delta = np.nan
            if pd.notna(conf_prev) and pd.notna(conf_cur):
                transfer_conf_delta = float(conf_cur) - float(conf_prev)

            transfer_pace_delta = pace_cur - pace_prev if pd.notna(pace_prev) and pd.notna(pace_cur) else np.nan
            transfer_role_delta = usage_cur - usage_prev if pd.notna(usage_prev) and pd.notna(usage_cur) else np.nan
            transfer_perf_delta_raw = ts_cur - ts_prev if pd.notna(ts_prev) and pd.notna(ts_cur) else np.nan

            # Context-adjusted delta: remove a simple pace/conf component from raw perf jump.
            transfer_perf_delta_context_adj = transfer_perf_delta_raw
            if pd.notna(transfer_perf_delta_raw):
                pace_term = 0.0 if pd.isna(transfer_pace_delta) else 0.001 * float(transfer_pace_delta)
                conf_term = 0.0 if pd.isna(transfer_conf_delta) else 0.005 * float(transfer_conf_delta)
                transfer_perf_delta_context_adj = float(transfer_perf_delta_raw) - pace_term - conf_term

            z_terms = []
            for v, s in [
                (transfer_role_delta, 0.08),
                (transfer_perf_delta_context_adj, 0.06),
                (transfer_pace_delta, 8.0),
                (transfer_conf_delta, 1.0),
            ]:
                if pd.notna(v):
                    z_terms.append((float(v) / s) ** 2)
            transfer_shock_score = float(np.sqrt(sum(z_terms))) if z_terms else np.nan

            rows.append(
                {
                    "athlete_id": int(athlete_id),
                    "season_from": int(prev["season"]),
                    "season_to": int(cur["season"]),
                    "team_from": int(prev["teamId"]),
                    "team_to": int(cur["teamId"]),
                    "transfer_conf_delta": transfer_conf_delta,
                    "transfer_pace_delta": transfer_pace_delta,
                    "transfer_pace_proxy_flag": int(
                        not (
                            pd.notna(prev.get("team_pace", np.nan))
                            and pd.notna(cur.get("team_pace", np.nan))
                        )
                    ),
                    "transfer_role_delta": transfer_role_delta,
                    "transfer_perf_delta_raw": transfer_perf_delta_raw,
                    "transfer_perf_delta_context_adj": transfer_perf_delta_context_adj,
                    "transfer_shock_score": transfer_shock_score,
                    "transfer_model_version": VERSION,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input missing: {IN_PATH}")
    df = pd.read_parquet(IN_PATH)
    out = build_transfer_context(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    logger.info("Saved %s (%d rows)", OUT_PATH, len(out))


if __name__ == "__main__":
    main()
