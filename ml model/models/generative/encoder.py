"""
ARD Encoder: College Features → Latent Traits
==============================================
Linear encoder with Automatic Relevance Determination (ARD).
Unused trait dimensions automatically shrink to zero.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from typing import Tuple

from .core import ard_prior


def encoder_model(
    x: jnp.ndarray,
    k_max: int = 32,
    ard_scale: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Encode college features into latent traits with ARD.
    
    z = W @ x + b
    
    where W has ARD prior (columns can shrink to 0).
    
    Args:
        x: College features [batch, D]
        k_max: Maximum number of latent dimensions
        ard_scale: Prior scale for ARD (smaller = more shrinkage)
    
    Returns:
        Tuple of (z [batch, K], ard_scales [K])
    """
    batch_size, d = x.shape
    
    # Encoder weights with ARD prior on columns (trait dimensions)
    # Each column k of W has its own scale α_k
    # If α_k → 0, trait k is not needed
    
    # Per-trait scales (ARD)
    alpha = numpyro.sample(
        "encoder_alpha",
        dist.HalfCauchy(ard_scale).expand([k_max])
    )
    
    # Encoder weights: W[d, k] ~ Normal(0, α_k)
    W = numpyro.sample(
        "encoder_W",
        dist.Normal(0, alpha).expand([d, k_max])
    )
    
    # Bias
    b = numpyro.sample(
        "encoder_b",
        dist.Normal(0, 1.0).expand([k_max])
    )
    
    # Compute latent traits
    z = jnp.dot(x, W) + b  # [batch, K]
    
    # Register z as a deterministic site for later extraction
    numpyro.deterministic("z", z)
    
    return z, alpha


def encoder_with_feature_groups(
    x_shooting: jnp.ndarray,
    x_passing: jnp.ndarray,
    x_rebounding: jnp.ndarray,
    x_defense: jnp.ndarray,
    x_context: jnp.ndarray,
    k_max: int = 32,
    ard_scale: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Encoder with separate feature groups for better interpretability.
    
    Each feature group contributes to traits through its own weight matrix,
    encouraging the model to learn that shooting stats → shooting-related traits.
    
    Args:
        x_shooting: Shooting features [batch, D1]
        x_passing: Passing features [batch, D2]
        x_rebounding: Rebounding features [batch, D3]
        x_defense: Defensive features [batch, D4]
        x_context: Context features [batch, D5]
        k_max: Maximum latent dimensions
        ard_scale: ARD prior scale
    
    Returns:
        Tuple of (z [batch, K], ard_scales [K])
    """
    # Per-trait scales (shared across all feature groups)
    alpha = numpyro.sample(
        "encoder_alpha",
        dist.HalfCauchy(ard_scale).expand([k_max])
    )
    
    # Separate weight matrices per feature group
    groups = [
        ("shooting", x_shooting),
        ("passing", x_passing),
        ("rebounding", x_rebounding),
        ("defense", x_defense),
        ("context", x_context),
    ]
    
    z_contributions = []
    
    for name, x_group in groups:
        if x_group is None or x_group.shape[1] == 0:
            continue
            
        d_group = x_group.shape[1]
        
        # Weight matrix for this group
        W_group = numpyro.sample(
            f"encoder_W_{name}",
            dist.Normal(0, alpha).expand([d_group, k_max])
        )
        
        # Contribution from this group
        z_group = jnp.dot(x_group, W_group)
        z_contributions.append(z_group)
    
    # Sum contributions
    z = sum(z_contributions)
    
    # Bias
    b = numpyro.sample(
        "encoder_b",
        dist.Normal(0, 1.0).expand([k_max])
    )
    
    z = z + b
    
    numpyro.deterministic("z", z)
    
    return z, alpha


def get_effective_traits(
    alpha_samples: jnp.ndarray,
    threshold: float = 0.1,
) -> Tuple[jnp.ndarray, int]:
    """
    Determine which traits are "active" based on ARD scales.
    
    Args:
        alpha_samples: Posterior samples of ARD scales [n_samples, K]
        threshold: Minimum posterior mean scale to be active
    
    Returns:
        Tuple of (active_mask [K], k_effective)
    """
    # Use posterior mean of scales
    alpha_mean = alpha_samples.mean(axis=0)
    
    active = alpha_mean > threshold
    k_eff = int(active.sum())
    
    return active, k_eff
