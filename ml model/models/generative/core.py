"""
Core utilities for Bayesian generative model.

Includes:
- Horseshoe prior for sparse shrinkage
- ARD (Automatic Relevance Determination) prior
- SVI training utilities
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide
from numpyro.optim import Adam
from typing import Optional, Dict, Any, Tuple
import numpy as np


def horseshoe_prior(
    name: str,
    shape: Tuple[int, ...],
    tau_scale: float = 1.0,
    regularized: bool = True,
) -> jnp.ndarray:
    """
    Horseshoe prior for sparse shrinkage.
    
    The horseshoe is a global-local shrinkage prior:
    - Global scale τ controls overall sparsity
    - Local scales λ_i allow individual coefficients to escape shrinkage
    
    Most coefficients shrink to ~0, but a few can be large.
    
    Args:
        name: Base name for the random variables
        shape: Shape of the coefficient array
        tau_scale: Prior scale for global shrinkage (smaller = more sparse)
        regularized: Use regularized horseshoe (more stable)
    
    Returns:
        Coefficient array with horseshoe prior
    """
    # Global shrinkage parameter
    tau = numpyro.sample(f"{name}_tau", dist.HalfCauchy(tau_scale))
    
    # Local shrinkage parameters (one per coefficient)
    lambdas = numpyro.sample(f"{name}_lambda", dist.HalfCauchy(1.0).expand(shape))
    
    if regularized:
        # Regularized horseshoe: c² controls the slab width
        c_sq = numpyro.sample(f"{name}_c_sq", dist.InverseGamma(2.0, 1.0))
        # Regularized local scale
        lambda_tilde = jnp.sqrt(c_sq * lambdas**2 / (c_sq + tau**2 * lambdas**2))
        scale = tau * lambda_tilde
    else:
        scale = tau * lambdas
    
    # Sample coefficients
    beta = numpyro.sample(f"{name}", dist.Normal(0, scale))
    
    return beta


def ard_prior(
    name: str,
    shape: Tuple[int, ...],
    scale_prior_scale: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Automatic Relevance Determination (ARD) prior.
    
    Each dimension has its own scale parameter that can shrink to zero
    if that dimension is not relevant.
    
    Args:
        name: Base name for random variables
        shape: Shape of the coefficient array (last dim is the one with ARD)
        scale_prior_scale: Prior scale for the ARD scales
    
    Returns:
        Tuple of (coefficients, scales)
    """
    n_dims = shape[-1] if len(shape) > 0 else 1
    
    # Per-dimension scales (can shrink to 0)
    scales = numpyro.sample(
        f"{name}_scales",
        dist.HalfCauchy(scale_prior_scale).expand([n_dims])
    )
    
    # Coefficients with ARD prior
    if len(shape) > 1:
        # Matrix case: each column has its own scale
        beta = numpyro.sample(
            f"{name}",
            dist.Normal(0, scales).expand(shape)
        )
    else:
        beta = numpyro.sample(
            f"{name}",
            dist.Normal(0, scales)
        )
    
    return beta, scales


def run_svi(
    model,
    guide,
    data: Dict[str, jnp.ndarray],
    num_steps: int = 5000,
    lr: float = 0.01,
    num_particles: int = 4,
    progress_bar: bool = True,
    seed: int = 42,
) -> Tuple[Any, Dict[str, jnp.ndarray]]:
    """
    Run Stochastic Variational Inference.
    
    Args:
        model: NumPyro model function
        guide: NumPyro guide function (or use autoguide)
        data: Dictionary of data arrays
        num_steps: Number of optimization steps
        lr: Learning rate
        num_particles: Number of particles for ELBO estimation
        progress_bar: Show progress bar
        seed: Random seed
    
    Returns:
        Tuple of (SVI result, posterior samples)
    """
    rng_key = jax.random.PRNGKey(seed)
    
    optimizer = Adam(lr)
    svi = SVI(
        model,
        guide,
        optimizer,
        loss=Trace_ELBO(num_particles=num_particles),
    )
    
    # Run SVI
    svi_result = svi.run(
        rng_key,
        num_steps,
        progress_bar=progress_bar,
        **data,
    )
    
    # Get posterior samples
    rng_key, sample_key = jax.random.split(rng_key)
    posterior_samples = guide.sample_posterior(
        sample_key,
        svi_result.params,
        sample_shape=(1000,),
    )
    
    return svi_result, posterior_samples


def get_effective_dimensions(
    scales: jnp.ndarray,
    threshold: float = 0.1,
) -> Tuple[jnp.ndarray, int]:
    """
    Determine which dimensions are "active" based on ARD scales.
    
    Args:
        scales: ARD scale parameters [K]
        threshold: Minimum scale to be considered active
    
    Returns:
        Tuple of (active mask, number of effective dimensions)
    """
    active = scales > threshold
    k_eff = int(active.sum())
    return active, k_eff


def compute_interaction_matrix(z: jnp.ndarray) -> jnp.ndarray:
    """
    Compute all pairwise interaction terms z_a * z_b for a < b.
    
    Args:
        z: Latent traits [batch, K]
    
    Returns:
        Interaction terms [batch, K*(K-1)/2]
    """
    batch_size, k = z.shape
    
    # Get indices for upper triangle (a < b)
    indices = jnp.triu_indices(k, k=1)
    
    # Compute z_a * z_b for all pairs
    z_a = z[:, indices[0]]  # [batch, n_pairs]
    z_b = z[:, indices[1]]  # [batch, n_pairs]
    
    return z_a * z_b


def get_interaction_pair_names(k: int) -> list:
    """
    Get names for all interaction pairs.
    
    Args:
        k: Number of dimensions
    
    Returns:
        List of (a, b) tuples for each interaction
    """
    pairs = []
    for a in range(k):
        for b in range(a + 1, k):
            pairs.append((a, b))
    return pairs


def group_shrinkage_prior(
    name: str,
    n_groups: int,
    dim: int,
    tau_global: float = 1.0,
    tau_local: float = 0.5,
) -> jnp.ndarray:
    """
    Group shrinkage prior for pathway deviations Δβ_k.
    
    Each group k has its own deviation vector Δβ_k ∈ R^dim.
    The prior encourages Δβ_k to be sparse (most deviations ~0) 
    but allows some groups to have meaningful adjustments.
    
    Structure:
    - Global shrinkage τ controls overall sparsity of deviations
    - Per-element local shrinkage λ_kd allows individual escapes
    
    Args:
        name: Base name for random variables
        n_groups: Number of groups (e.g., pathways K_p)
        dim: Dimension of each deviation vector (e.g., K_z)
        tau_global: Global shrinkage scale (smaller = sparser)
        tau_local: Per-element shrinkage scale
    
    Returns:
        Deviation matrix [n_groups, dim]
    """
    tau = numpyro.sample(f"{name}_tau", dist.HalfCauchy(tau_global))
    
    lambdas = numpyro.sample(
        f"{name}_lambda",
        dist.HalfCauchy(tau_local).expand([n_groups, dim])
    )
    
    scale = tau * lambdas
    
    delta_beta = numpyro.sample(
        f"{name}",
        dist.Normal(0, scale)
    )
    
    return delta_beta


class HeteroscedasticNormal(dist.Distribution):
    """
    Normal distribution with observation-specific variance.
    
    σ²_i = base_var / (exposure_i + ε)
    
    Higher exposure = lower variance = more confident observation.
    """
    
    def __init__(self, loc, base_var, exposure, eps=100.0):
        self.loc = loc
        self.base_var = base_var
        self.exposure = exposure
        self.eps = eps
        
        # Compute per-observation variance
        self.var = base_var / (exposure + eps)
        self.scale = jnp.sqrt(self.var)
        
        super().__init__(batch_shape=loc.shape, event_shape=())
    
    def sample(self, key, sample_shape=()):
        return self.loc + self.scale * jax.random.normal(key, sample_shape + self.batch_shape)
    
    def log_prob(self, value):
        return dist.Normal(self.loc, self.scale).log_prob(value)
