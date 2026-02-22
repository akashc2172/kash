#!/usr/bin/env python3
"""
Error Decomposition & Miss Analysis
====================================
Aggregates yearly rolling predictions and slices residuals to find
systematic model biases. Produces:

1. Worst misses table (top overpredicted + underpredicted per year)
2. Bias by cohort slice (age, conference, recruiting, position proxy, class year)
3. Top-K hit rates (how many true top-K are in predicted top-K)
4. Year-over-year stability check

Usage:
    python run_error_decomposition.py
    python run_error_decomposition.py --years 2019 2020 2021 2022 2023 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
ROLLING_DIR = BASE / "data" / "inference" / "rolling_yearly"
SUPERVISED_PATH = BASE / "data" / "training" / "unified_training_table_supervised.parquet"
AUDIT_DIR = BASE / "data" / "audit"
DECOMP_DIR = AUDIT_DIR / "error_decomposition"


def load_yearly_rankings(years: list) -> pd.DataFrame:
    """Load and concat all yearly nba_mapped rankings."""
    frames = []
    for year in years:
        p = ROLLING_DIR / str(year) / f"rankings_{year}_nba_mapped.csv"
        if p.exists():
            df = pd.read_csv(p)
            df['eval_year'] = year
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_features_for_slicing() -> pd.DataFrame:
    """Load the supervised table for slicing dimensions."""
    if not SUPERVISED_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(SUPERVISED_PATH)
    # Keep only slicing columns + identifiers
    keep = ['athlete_id', 'nba_id', 'draft_year_proxy', 'college_final_season']
    slice_cols = [
        'college_is_power_conf', 'college_recruiting_stars', 'college_recruiting_rank',
        'career_years', 'class_year', 'age_at_season',
        'college_height_in', 'college_weight_lbs',
        'usage', 'college_minutes_total',
        'college_poss_proxy',
        'is_transfer',
        'college_rim_fg_pct', 'college_three_fg_pct', 'college_ft_pct',
    ]
    keep += [c for c in slice_cols if c in df.columns]
    return df[keep].copy()


def compute_hit_rates(df: pd.DataFrame, k_values=[10, 25, 50]) -> dict:
    """Compute top-K hit rates: how many true top-K appear in predicted top-K."""
    results = {}
    has_actual = df['actual_target'].notna()
    df_eval = df[has_actual].copy()
    if len(df_eval) < 10:
        return results

    for k in k_values:
        if len(df_eval) < k:
            continue
        true_top_k = set(df_eval.nlargest(k, 'actual_target')['athlete_id'].values)
        pred_top_k = set(df_eval.nsmallest(k, 'pred_rank')['athlete_id'].values)
        hits = len(true_top_k & pred_top_k)
        results[f'top_{k}_hits'] = hits
        results[f'top_{k}_hit_rate'] = hits / k
        results[f'top_{k}_total'] = k
    return results


def compute_ndcg(df: pd.DataFrame, k: int = 25) -> float:
    """Compute NDCG@k for ranking quality."""
    has_actual = df['actual_target'].notna()
    df_eval = df[has_actual].copy()
    if len(df_eval) < k:
        return float('nan')

    # Sort by predicted rank (ascending = best first)
    df_eval = df_eval.sort_values('pred_rank', ascending=True).head(k)

    # Relevance = actual target value (higher = more relevant)
    relevance = df_eval['actual_target'].values
    # Shift to non-negative
    relevance = relevance - relevance.min() + 1e-6

    # DCG
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(relevance))

    # Ideal DCG (sorted by actual)
    ideal_rel = np.sort(relevance)[::-1]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_rel))

    return dcg / idcg if idcg > 0 else 0.0


def worst_misses(df: pd.DataFrame, n: int = 15) -> tuple:
    """Get worst overpredictions and underpredictions."""
    has_both = df['actual_target'].notna() & df['rank_error'].notna()
    df_eval = df[has_both].copy()
    if len(df_eval) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Overpredicted = big positive rank_error (we ranked them too high, they busted)
    overpred = df_eval.nlargest(n, 'rank_error')
    # Underpredicted = big negative rank_error (we ranked them too low, they were good)
    underpred = df_eval.nsmallest(n, 'rank_error')

    show_cols = ['player_name', 'eval_year', 'pred_rank', 'actual_rank', 'rank_error',
                 'pred_mu', 'actual_target', 'pred_sd']
    show_cols = [c for c in show_cols if c in df_eval.columns]
    return overpred[show_cols], underpred[show_cols]


def slice_bias(df: pd.DataFrame, slice_col: str, bins=None) -> pd.DataFrame:
    """Compute mean residual bias by a categorical/binned slice."""
    has_actual = df['actual_target'].notna() & df['pred_mu'].notna()
    df_eval = df[has_actual].copy()
    if len(df_eval) == 0 or slice_col not in df_eval.columns:
        return pd.DataFrame()

    df_eval['residual'] = df_eval['pred_mu'] - df_eval['actual_target']
    df_eval['abs_residual'] = df_eval['residual'].abs()

    if bins is not None:
        df_eval[f'{slice_col}_bin'] = pd.cut(pd.to_numeric(df_eval[slice_col], errors='coerce'), bins=bins)
        group_col = f'{slice_col}_bin'
    else:
        group_col = slice_col

    agg = df_eval.groupby(group_col).agg(
        n=('residual', 'count'),
        mean_residual=('residual', 'mean'),
        std_residual=('residual', 'std'),
        mean_abs_residual=('abs_residual', 'mean'),
        mean_rank_error=('rank_error', 'mean'),
        mean_pred_mu=('pred_mu', 'mean'),
        mean_actual=('actual_target', 'mean'),
    ).reset_index()
    agg['bias_direction'] = np.where(agg['mean_residual'] > 0.2, 'OVER', np.where(agg['mean_residual'] < -0.2, 'UNDER', 'ok'))
    return agg.sort_values('mean_abs_residual', ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', type=int, nargs='+', default=list(range(2013, 2025)))
    args = parser.parse_args()

    DECOMP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load all yearly rankings
    df_rank = load_yearly_rankings(args.years)
    if df_rank.empty:
        logger.error("No yearly rankings found!")
        return
    logger.info(f"Loaded {len(df_rank):,} ranked rows across {df_rank['eval_year'].nunique()} years")

    # 2. Join slicing features
    df_feat = load_features_for_slicing()
    if not df_feat.empty:
        # Match on athlete_id + draft_year_proxy
        merge_cols = ['athlete_id']
        if 'draft_year_proxy' in df_feat.columns and 'draft_year_proxy' in df_rank.columns:
            merge_cols.append('draft_year_proxy')
        slice_cols_to_add = [c for c in df_feat.columns if c not in df_rank.columns and c not in merge_cols]
        if slice_cols_to_add:
            df_rank = df_rank.merge(df_feat[merge_cols + slice_cols_to_add].drop_duplicates(subset=merge_cols),
                                    on=merge_cols, how='left')
            logger.info(f"  Joined {len(slice_cols_to_add)} slicing dimensions")

    # Compute residuals
    df_rank['pred_mu'] = pd.to_numeric(df_rank.get('pred_mu'), errors='coerce')
    df_rank['actual_target'] = pd.to_numeric(df_rank.get('actual_target'), errors='coerce')
    df_rank['pred_rank'] = pd.to_numeric(df_rank.get('pred_rank'), errors='coerce')
    df_rank['actual_rank'] = pd.to_numeric(df_rank.get('actual_rank'), errors='coerce')
    df_rank['rank_error'] = pd.to_numeric(df_rank.get('rank_error'), errors='coerce')
    df_rank['residual'] = df_rank['pred_mu'] - df_rank['actual_target']

    # =========================================================================
    # 3. WORST MISSES (global across all years)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("WORST MISSES (All Years Combined)")
    logger.info("=" * 60)
    overpred, underpred = worst_misses(df_rank, n=25)
    if not overpred.empty:
        overpred.to_csv(DECOMP_DIR / 'worst_overpredictions.csv', index=False)
        logger.info(f"\nTOP OVERPREDICTIONS (Busts — we loved them, NBA didn't):")
        logger.info(overpred.head(10).to_string(index=False))
    if not underpred.empty:
        underpred.to_csv(DECOMP_DIR / 'worst_underpredictions.csv', index=False)
        logger.info(f"\nTOP UNDERPREDICTIONS (Sleepers — NBA loved them, we missed):")
        logger.info(underpred.head(10).to_string(index=False))

    # =========================================================================
    # 4. PER-YEAR WORST MISSES
    # =========================================================================
    per_year_misses = []
    for year in sorted(df_rank['eval_year'].unique()):
        yr_df = df_rank[df_rank['eval_year'] == year]
        op, up = worst_misses(yr_df, n=5)
        if not op.empty:
            op['miss_type'] = 'overpredicted'
            per_year_misses.append(op)
        if not up.empty:
            up['miss_type'] = 'underpredicted'
            per_year_misses.append(up)
    if per_year_misses:
        pd.concat(per_year_misses).to_csv(DECOMP_DIR / 'per_year_worst_misses.csv', index=False)

    # =========================================================================
    # 5. HIT RATES & NDCG PER YEAR
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TOP-K HIT RATES & NDCG BY YEAR")
    logger.info("=" * 60)

    yr_metrics = []
    for year in sorted(df_rank['eval_year'].unique()):
        yr_df = df_rank[df_rank['eval_year'] == year]
        hits = compute_hit_rates(yr_df, k_values=[5, 10, 25])
        ndcg = compute_ndcg(yr_df, k=10)
        from scipy.stats import spearmanr
        has_both = yr_df['actual_target'].notna() & yr_df['pred_mu'].notna()
        sp = spearmanr(yr_df.loc[has_both, 'pred_mu'], yr_df.loc[has_both, 'actual_target'])[0] if has_both.sum() > 3 else float('nan')

        metrics = {'year': year, 'n_with_target': int(has_both.sum()),
                   'spearman_rho': float(sp), 'ndcg_at_10': float(ndcg)}
        metrics.update(hits)
        yr_metrics.append(metrics)
        logger.info(f"  {year}: ρ={sp:.3f} | NDCG@10={ndcg:.3f} | top-10 hits: {hits.get('top_10_hits', '?')}/{hits.get('top_10_total', '?')}")

    metrics_df = pd.DataFrame(yr_metrics)
    metrics_df.to_csv(DECOMP_DIR / 'yearly_ranking_quality.csv', index=False)

    # =========================================================================
    # 6. BIAS SLICES
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SYSTEMATIC BIAS ANALYSIS")
    logger.info("=" * 60)

    slices = {
        'college_is_power_conf': None,
        'college_recruiting_stars': None,
        'career_years': None,
        'class_year': None,
        'age_at_season': [18, 19, 20, 21, 22, 23, 25],
        'college_height_in': [70, 74, 78, 82, 86, 90],
        'usage': [0, 0.15, 0.20, 0.25, 0.30, 0.50],
    }

    for slice_col, bins in slices.items():
        if slice_col not in df_rank.columns:
            continue
        bias = slice_bias(df_rank, slice_col, bins=bins)
        if not bias.empty:
            bias.to_csv(DECOMP_DIR / f'bias_by_{slice_col}.csv', index=False)
            flagged = bias[bias['bias_direction'] != 'ok']
            if not flagged.empty:
                logger.info(f"\n  ⚠️  BIAS DETECTED in [{slice_col}]:")
                logger.info(flagged[['n', slice_col if bins is None else f'{slice_col}_bin',
                                     'mean_residual', 'bias_direction']].to_string(index=False))
            else:
                logger.info(f"  ✅ No significant bias in [{slice_col}]")

    # =========================================================================
    # 7. ARCHETYPE ANALYSIS (create simple archetypes)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PLAYER ARCHETYPE BIAS")
    logger.info("=" * 60)

    has_actual = df_rank['actual_target'].notna()
    df_eval = df_rank[has_actual].copy()

    if 'college_height_in' in df_eval.columns and 'usage' in df_eval.columns:
        h = pd.to_numeric(df_eval['college_height_in'], errors='coerce')
        u = pd.to_numeric(df_eval['usage'], errors='coerce')
        df_eval['archetype'] = 'unknown'
        df_eval.loc[(h >= 80) & (u < 0.22), 'archetype'] = 'rim_protector'
        df_eval.loc[(h >= 80) & (u >= 0.22), 'archetype'] = 'scoring_big'
        df_eval.loc[(h < 80) & (h >= 76) & (u >= 0.22), 'archetype'] = 'scoring_wing'
        df_eval.loc[(h < 80) & (h >= 76) & (u < 0.22), 'archetype'] = 'role_wing'
        df_eval.loc[(h < 76) & (u >= 0.25), 'archetype'] = 'lead_guard'
        df_eval.loc[(h < 76) & (u < 0.25), 'archetype'] = 'role_guard'

        arch_bias = slice_bias(df_eval, 'archetype')
        if not arch_bias.empty:
            arch_bias.to_csv(DECOMP_DIR / 'bias_by_archetype.csv', index=False)
            logger.info(arch_bias[['archetype', 'n', 'mean_residual', 'mean_abs_residual', 'bias_direction']].to_string(index=False))

    # =========================================================================
    # 8. SUMMARY REPORT
    # =========================================================================
    overall_spearman = metrics_df['spearman_rho'].mean() if not metrics_df.empty else 0
    overall_ndcg = metrics_df['ndcg_at_10'].mean() if not metrics_df.empty else 0

    summary = {
        'total_ranked_rows': int(len(df_rank)),
        'years_evaluated': sorted([int(y) for y in df_rank['eval_year'].unique()]),
        'avg_spearman_rho': float(overall_spearman),
        'avg_ndcg_at_10': float(overall_ndcg),
        'total_with_actual': int(df_rank['actual_target'].notna().sum()),
    }
    with open(DECOMP_DIR / 'decomposition_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info(f"ERROR DECOMPOSITION COMPLETE")
    logger.info(f"  Avg Spearman ρ: {overall_spearman:.3f}")
    logger.info(f"  Avg NDCG@10:    {overall_ndcg:.3f}")
    logger.info(f"  Output dir:     {DECOMP_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
