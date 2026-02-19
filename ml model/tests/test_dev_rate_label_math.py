#!/usr/bin/env python3
"""
Synthetic math checks for dev-rate label generation.

Covers:
1. Posterior slope recovery for known synthetic d_i.
2. Uncertainty ordering: higher-exposure players should have lower posterior sd.
"""

from __future__ import annotations

import numpy as np


def test_dev_rate_posterior_recovery_and_uncertainty_ordering() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from nba_scripts.build_fact_nba_development_rate import ModelConfig, fit_empirical_bayes

    rng = np.random.default_rng(7)
    n = 200

    # Design matrix with intercept + two pre-NBA features.
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2]).astype(float)

    # True prior mapping + latent player parameters.
    B_true = np.array([[0.8, 0.2, -0.1], [0.15, 0.25, 0.05]])
    mu_true = X @ B_true.T
    eps = rng.multivariate_normal(mean=[0.0, 0.0], cov=np.array([[0.08, 0.01], [0.01, 0.05]]), size=n)
    alpha_true = mu_true[:, 0] + eps[:, 0]
    d_true = mu_true[:, 1] + eps[:, 1]

    # Exposure strata: first half low exposure, second half high exposure.
    low = np.arange(n // 2)
    high = np.arange(n // 2, n)

    minutes = np.full(n, 220.0)
    minutes[high] = 2400.0

    poss = np.full(n, 900.0)
    poss[high] = 12000.0

    sigma_epm = 5.0
    sigma_rapm = 3.0
    m0 = 150.0
    p0 = 1500.0

    player_ids = np.arange(10_000, 10_000 + n, dtype=int)

    epm_obs = {}
    rapm_obs = {}
    for i, pid in enumerate(player_ids):
        rows = []
        for t in (1, 2, 3):
            mean_t = alpha_true[i] + d_true[i] * (t - 1)
            sd_t = sigma_epm / np.sqrt(max(minutes[i], m0))
            y_t = mean_t + rng.normal(scale=sd_t)
            rows.append((t, float(y_t), float(minutes[i])))
        epm_obs[int(pid)] = rows

        mean_r = alpha_true[i] + d_true[i]
        sd_r = sigma_rapm / np.sqrt(max(poss[i], p0))
        r_i = mean_r + rng.normal(scale=sd_r)
        rapm_obs[int(pid)] = (float(r_i), float(poss[i]))

    cfg = ModelConfig(m0=m0, p0=p0, max_iter=30, tol=1e-4)
    post_mean, post_cov, *_ = fit_empirical_bayes(
        player_ids=player_ids,
        X=X,
        epm_obs=epm_obs,
        rapm_obs=rapm_obs,
        cfg=cfg,
    )

    d_hat = post_mean[:, 1]
    d_sd = np.sqrt(np.clip(post_cov[:, 1, 1], 1e-12, None))

    corr = np.corrcoef(d_true, d_hat)[0, 1]
    assert corr > 0.70, f"expected strong slope recovery; got corr={corr:.3f}"

    low_sd = float(d_sd[low].mean())
    high_sd = float(d_sd[high].mean())
    assert high_sd < low_sd, f"expected high exposure sd < low exposure sd; got {high_sd:.4f} vs {low_sd:.4f}"


def main() -> None:
    test_dev_rate_posterior_recovery_and_uncertainty_ordering()
    print("OK")


if __name__ == "__main__":
    main()
