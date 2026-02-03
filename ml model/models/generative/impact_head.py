"""
Impact Head: Latent Traits → RAPM
=================================
Structured prediction with:
- Main effects (each trait independently)
- ALL pairwise interactions with horseshoe shrinkage
- Player-specific residual (unmeasured value)
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from typing import Tuple, Optional

from .core import horseshoe_prior, compute_interaction_matrix, get_interaction_pair_names


def impact_head_model(
    z: jnp.ndarray,
    y: Optional[jnp.ndarray] = None,
    exposure: Optional[jnp.ndarray] = None,
    tau_main: float = 1.0,
    tau_interaction: float = 0.5,
    sigma_residual: float = 0.1,
    include_residual: bool = True,
) -> jnp.ndarray:
    """
    Predict RAPM from latent traits with structured interactions.
    
    y = intercept + Σ_k β_k·z_k + Σ_{a<b} ρ_ab·z_a·z_b + h_i + ε
    
    Where:
    - β_k ~ Horseshoe (main effects, moderately sparse)
    - ρ_ab ~ Horseshoe (interactions, very sparse)
    - h_i ~ Normal(0, σ_h²) (residual, strongly shrunk)
    
    Args:
        z: Latent traits [batch, K]
        y: Observed RAPM targets (None for prior predictive)
        exposure: Minutes/possessions for heteroscedastic noise [batch]
        tau_main: Global shrinkage for main effects
        tau_interaction: Global shrinkage for interactions (smaller = sparser)
        sigma_residual: Prior std for player residuals
        include_residual: Whether to include player-specific residual
    
    Returns:
        Predicted RAPM [batch]
    """
    batch_size, k = z.shape
    n_pairs = k * (k - 1) // 2
    
    # Intercept
    intercept = numpyro.sample("intercept", dist.Normal(0, 1.0))
    
    # === MAIN EFFECTS ===
    # β_k for each trait dimension
    beta = horseshoe_prior("beta_main", shape=(k,), tau_scale=tau_main)
    
    main_effect = jnp.dot(z, beta)  # [batch]
    numpyro.deterministic("main_effect", main_effect)
    
    # === PAIRWISE INTERACTIONS ===
    # ρ_ab for all pairs (a, b) where a < b
    # Total: K*(K-1)/2 pairs
    rho = horseshoe_prior("rho_interaction", shape=(n_pairs,), tau_scale=tau_interaction)
    
    # Compute interaction terms
    z_interactions = compute_interaction_matrix(z)  # [batch, n_pairs]
    interaction_effect = jnp.dot(z_interactions, rho)  # [batch]
    numpyro.deterministic("interaction_effect", interaction_effect)
    
    # === PLAYER RESIDUAL (unmeasured value) ===
    if include_residual:
        # Strongly shrunk toward 0
        h = numpyro.sample(
            "h_residual",
            dist.Normal(0, sigma_residual).expand([batch_size])
        )
    else:
        h = 0.0
    
    # === COMBINE ===
    mu = intercept + main_effect + interaction_effect + h
    numpyro.deterministic("mu_rapm", mu)
    
    # === OBSERVATION NOISE ===
    # Heteroscedastic: higher exposure = lower noise
    sigma_base = numpyro.sample("sigma_obs", dist.HalfNormal(1.0))
    
    if exposure is not None:
        # σ²_i = σ²_base / (exposure_i / ref + 1)
        ref_exposure = 5000.0  # Reference possessions
        noise_scale = sigma_base / jnp.sqrt(exposure / ref_exposure + 1.0)
    else:
        noise_scale = sigma_base
    
    # === LIKELIHOOD ===
    if y is not None:
        # Only include valid observations
        with numpyro.plate("obs", batch_size):
            numpyro.sample("y", dist.Normal(mu, noise_scale), obs=y)
    
    return mu


def decompose_prediction(
    z: jnp.ndarray,
    beta_samples: jnp.ndarray,
    rho_samples: jnp.ndarray,
    intercept_samples: jnp.ndarray,
    h_samples: Optional[jnp.ndarray] = None,
) -> dict:
    """
    Decompose RAPM prediction into trait contributions.
    
    Args:
        z: Latent traits for one player [K]
        beta_samples: Posterior samples of main effects [n_samples, K]
        rho_samples: Posterior samples of interactions [n_samples, n_pairs]
        intercept_samples: Posterior samples of intercept [n_samples]
        h_samples: Posterior samples of residual [n_samples] (optional)
    
    Returns:
        Dictionary with decomposition
    """
    k = z.shape[0]
    n_samples = beta_samples.shape[0]
    
    # Main effect contributions per trait
    # β_k * z_k for each k
    main_contributions = beta_samples * z  # [n_samples, K]
    main_mean = main_contributions.mean(axis=0)  # [K]
    main_std = main_contributions.std(axis=0)  # [K]
    
    # Interaction contributions per pair
    pairs = get_interaction_pair_names(k)
    z_pairs = jnp.array([z[a] * z[b] for a, b in pairs])  # [n_pairs]
    interaction_contributions = rho_samples * z_pairs  # [n_samples, n_pairs]
    interaction_mean = interaction_contributions.mean(axis=0)  # [n_pairs]
    interaction_std = interaction_contributions.std(axis=0)  # [n_pairs]
    
    # Total prediction
    total_main = main_contributions.sum(axis=1)  # [n_samples]
    total_interaction = interaction_contributions.sum(axis=1)  # [n_samples]
    
    if h_samples is not None:
        total = intercept_samples + total_main + total_interaction + h_samples
        residual_mean = h_samples.mean()
    else:
        total = intercept_samples + total_main + total_interaction
        residual_mean = 0.0
    
    return {
        'total_mean': float(total.mean()),
        'total_std': float(total.std()),
        'intercept_mean': float(intercept_samples.mean()),
        'main_contributions': {
            'mean': main_mean,
            'std': main_std,
            'total_mean': float(total_main.mean()),
        },
        'interaction_contributions': {
            'pairs': pairs,
            'mean': interaction_mean,
            'std': interaction_std,
            'total_mean': float(total_interaction.mean()),
        },
        'residual_mean': residual_mean,
    }


def get_active_interactions(
    rho_samples: jnp.ndarray,
    k: int,
    threshold: float = 0.1,
    credible_mass: float = 0.9,
) -> list:
    """
    Identify interactions that are meaningfully non-zero.
    
    An interaction is "active" if its posterior credible interval
    excludes zero OR its posterior mean exceeds threshold.
    
    Args:
        rho_samples: Posterior samples [n_samples, n_pairs]
        k: Number of trait dimensions
        threshold: Minimum |mean| to be considered active
        credible_mass: Mass for credible interval
    
    Returns:
        List of (pair_idx, pair_tuple, mean, ci_low, ci_high)
    """
    pairs = get_interaction_pair_names(k)
    n_pairs = len(pairs)
    
    active = []
    alpha = (1 - credible_mass) / 2
    
    for i in range(n_pairs):
        samples = rho_samples[:, i]
        mean = float(samples.mean())
        ci_low = float(jnp.percentile(samples, alpha * 100))
        ci_high = float(jnp.percentile(samples, (1 - alpha) * 100))
        
        # Check if credible interval excludes zero OR mean is large
        excludes_zero = (ci_low > 0) or (ci_high < 0)
        large_mean = abs(mean) > threshold
        
        if excludes_zero or large_mean:
            active.append({
                'pair_idx': i,
                'pair': pairs[i],
                'mean': mean,
                'ci_low': ci_low,
                'ci_high': ci_high,
            })
    
    # Sort by absolute mean
    active.sort(key=lambda x: abs(x['mean']), reverse=True)
    
    return active


def get_top_traits(
    z: jnp.ndarray,
    beta_samples: jnp.ndarray,
    cumulative_threshold: float = 0.8,
) -> list:
    """
    Get traits contributing to top X% of cumulative absolute contribution.
    
    Args:
        z: Latent traits for one player [K]
        beta_samples: Posterior samples of main effects [n_samples, K]
        cumulative_threshold: Include traits until this fraction of total
    
    Returns:
        List of (trait_idx, z_value, contribution_mean, contribution_std)
    """
    k = z.shape[0]
    
    # Compute contributions
    contributions = beta_samples * z  # [n_samples, K]
    contrib_mean = contributions.mean(axis=0)  # [K]
    contrib_std = contributions.std(axis=0)  # [K]
    
    # Sort by absolute contribution
    abs_contrib = jnp.abs(contrib_mean)
    sorted_idx = jnp.argsort(-abs_contrib)  # Descending
    
    # Cumulative sum
    total_abs = abs_contrib.sum()
    cumsum = 0.0
    top_traits = []
    
    for idx in sorted_idx:
        idx = int(idx)
        cumsum += abs_contrib[idx]
        
        top_traits.append({
            'trait_idx': idx,
            'z_value': float(z[idx]),
            'contribution_mean': float(contrib_mean[idx]),
            'contribution_std': float(contrib_std[idx]),
            'cumulative_fraction': float(cumsum / total_abs) if total_abs > 0 else 1.0,
        })
        
        if cumsum / total_abs >= cumulative_threshold:
            break
    
    return top_traits
