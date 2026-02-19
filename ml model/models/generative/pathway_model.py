"""
Hierarchical Pathway Translation Model
======================================
Phase 1 implementation: β_global + Δβ_k per pathway, no interactions.

Key design decisions (per proposal):
- z remains continuous (NO archetype priors upstream)
- Pathways only parameterize the z→impact translation via β deviations
- Gating on z, not raw x
- Selection head separate from aux stats head
- Gap targets masked if no minutes (conditional, not selection)
- Heteroscedastic observation noise by peak_poss
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import ClippedAdam
from typing import Dict, Optional, Tuple, Any, List
import numpy as np

from .core import horseshoe_prior, group_shrinkage_prior


class HierarchicalPathwayModel:
    """
    Hierarchical pathway model for prospect translation prediction.
    
    Architecture:
        College Features x → Encoder → z (continuous latent traits)
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
        Selection Head            Aux Stats Head              Pathway Gating
        P(made_nba|z)            year1_epm|min≥X              π_k = softmax(g(z))
        P(min≥500|made)          gap_ts|min≥X
              │                           │                           │
              └───────────────────────────┼───────────────────────────┘
                                          │
                                          ▼
                            Conditional Impact Head
                            μ_peakRAPM = Σ_k π_k · (intercept_k + β_k · z)
                            β_k = β_global + Δβ_k
    
    Key features:
    - β_global: weak ridge prior (preserves marginal traits)
    - Δβ_k: group shrinkage (pathway-specific adjustments)
    - Selection head: BCE for made_nba, P(min≥500|made)
    - Aux stats head: masked MSE if no minutes
    - Heteroscedastic noise: σ² / clip(peak_poss, 500, 10000)
    """
    
    def __init__(
        self,
        k_z: int = 24,
        k_p: int = 6,
        ard_scale: float = 1.0,
        beta_global_scale: float = 0.5,
        delta_beta_tau: float = 0.3,
        gating_scale: float = 0.5,
        entropy_weight: float = 0.01,
        min_poss: float = 500.0,
        max_poss: float = 10000.0,
    ):
        """
        Args:
            k_z: Latent dimension (number of traits)
            k_p: Number of pathways
            ard_scale: Prior scale for ARD on encoder
            beta_global_scale: Prior scale for global main effects (ridge-like)
            delta_beta_tau: Global shrinkage for pathway deviations
            gating_scale: Prior scale for gating weights
            entropy_weight: Regularization weight to prevent pathway collapse
            min_poss: Minimum possessions for clipping
            max_poss: Maximum possessions for clipping
        """
        self.k_z = k_z
        self.k_p = k_p
        self.ard_scale = ard_scale
        self.beta_global_scale = beta_global_scale
        self.delta_beta_tau = delta_beta_tau
        self.gating_scale = gating_scale
        self.entropy_weight = entropy_weight
        self.min_poss = min_poss
        self.max_poss = max_poss
        
        self.params = None
        self.guide = None
        self.losses = None
    
    def model(
        self,
        x: jnp.ndarray,
        y_peak_rapm: Optional[jnp.ndarray] = None,
        y_made_nba: Optional[jnp.ndarray] = None,
        y_min_threshold: Optional[jnp.ndarray] = None,
        y_aux: Optional[Dict[str, jnp.ndarray]] = None,
        peak_poss: Optional[jnp.ndarray] = None,
        mask_rapm: Optional[jnp.ndarray] = None,
        mask_aux: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        """
        Full hierarchical pathway model.
        
        Args:
            x: College features [N, D]
            y_peak_rapm: Peak RAPM targets [N]
            y_made_nba: Binary made NBA indicator [N]
            y_min_threshold: Binary min≥500 given made [N]
            y_aux: Dict of auxiliary targets (year1_epm, gap_ts, gap_usg)
            peak_poss: Possessions in peak window [N]
            mask_rapm: Boolean mask for valid RAPM obs [N]
            mask_aux: Dict of boolean masks for aux targets
        """
        n, d = x.shape
        k_z = self.k_z
        k_p = self.k_p
        
        # === ENCODER: x → z ===
        alpha = numpyro.sample(
            "alpha",
            dist.HalfCauchy(self.ard_scale).expand([k_z])
        )
        
        W_enc = numpyro.sample(
            "W_encoder",
            dist.Normal(0, 1).expand([d, k_z])
        )
        W_scaled = W_enc * alpha
        
        b_enc = numpyro.sample(
            "b_encoder",
            dist.Normal(0, 0.1).expand([k_z])
        )
        
        z = jnp.dot(x, W_scaled) + b_enc
        numpyro.deterministic("z", z)
        
        # === PATHWAY GATING: π_k(z) ===
        W_gate = numpyro.sample(
            "W_gate",
            dist.Normal(0, self.gating_scale).expand([k_z, k_p])
        )
        b_gate = numpyro.sample(
            "b_gate",
            dist.Normal(0, 0.1).expand([k_p])
        )
        
        logits = jnp.dot(z, W_gate) + b_gate
        pi = jax.nn.softmax(logits, axis=-1)
        numpyro.deterministic("pi", pi)
        
        pathway_entropy = -jnp.sum(pi * jnp.log(pi + 1e-10), axis=-1).mean()
        numpyro.deterministic("pathway_entropy", pathway_entropy)
        
        # Pathway collapse regularizer: penalize low entropy (collapse to one pathway)
        # Target entropy ≈ log(K_p) for uniform, we penalize when entropy < 0.5 * log(K_p)
        min_entropy = 0.5 * jnp.log(float(k_p))
        entropy_penalty = self.entropy_weight * jnp.maximum(0.0, min_entropy - pathway_entropy)
        numpyro.factor("pathway_entropy_penalty", -entropy_penalty)
        
        # Anti-collapse regularizer: encourage higher entropy (spread across pathways)
        # Positive factor = log-prob bonus for higher entropy
        numpyro.factor("entropy_regularizer", self.entropy_weight * pathway_entropy * n)
        
        # === SELECTION HEAD ===
        W_sel = numpyro.sample(
            "W_selection",
            dist.Normal(0, 0.5).expand([k_z])
        )
        b_sel = numpyro.sample(
            "b_selection",
            dist.Normal(0, 0.5)
        )
        
        logit_made = jnp.dot(z, W_sel) + b_sel
        p_made = jax.nn.sigmoid(logit_made)
        numpyro.deterministic("p_made_nba", p_made)
        
        if y_made_nba is not None:
            mask_made = ~jnp.isnan(y_made_nba)
            y_made_clean = jnp.nan_to_num(y_made_nba, nan=0.0)
            with numpyro.plate("selection_plate", n):
                with numpyro.handlers.mask(mask=mask_made):
                    numpyro.sample(
                        "obs_made_nba",
                        dist.Bernoulli(logits=logit_made),
                        obs=y_made_clean
                    )
        
        W_min = numpyro.sample(
            "W_minutes",
            dist.Normal(0, 0.5).expand([k_z])
        )
        b_min = numpyro.sample(
            "b_minutes",
            dist.Normal(0, 0.5)
        )
        
        logit_min = jnp.dot(z, W_min) + b_min
        p_min = jax.nn.sigmoid(logit_min)
        numpyro.deterministic("p_min_threshold", p_min)
        
        if y_min_threshold is not None:
            mask_min = ~jnp.isnan(y_min_threshold) & (y_made_nba == 1) if y_made_nba is not None else ~jnp.isnan(y_min_threshold)
            y_min_clean = jnp.nan_to_num(y_min_threshold, nan=0.0)
            with numpyro.plate("minutes_plate", n):
                with numpyro.handlers.mask(mask=mask_min):
                    numpyro.sample(
                        "obs_min_threshold",
                        dist.Bernoulli(logits=logit_min),
                        obs=y_min_clean
                    )
        
        # === AUX STATS HEAD (conditional on minutes) ===
        sigma_aux = numpyro.sample("sigma_aux", dist.HalfNormal(0.5))
        
        if y_aux is not None:
            for aux_name, aux_target in y_aux.items():
                w_aux = numpyro.sample(
                    f"w_aux_{aux_name}",
                    dist.Normal(0, 0.5).expand([k_z])
                )
                b_aux = numpyro.sample(
                    f"b_aux_{aux_name}",
                    dist.Normal(0, 0.5)
                )
                
                mu_aux = jnp.dot(z, w_aux) + b_aux
                numpyro.deterministic(f"mu_aux_{aux_name}", mu_aux)
                
                if mask_aux is not None and aux_name in mask_aux:
                    m = mask_aux[aux_name]
                else:
                    m = ~jnp.isnan(aux_target)
                
                with numpyro.handlers.mask(mask=m):
                    numpyro.sample(
                        f"obs_aux_{aux_name}",
                        dist.Normal(mu_aux, sigma_aux),
                        obs=jnp.where(m, aux_target, 0.0)
                    )
        
        # === CONDITIONAL IMPACT HEAD (Hierarchical β) ===
        beta_global = numpyro.sample(
            "beta_global",
            dist.Normal(0, self.beta_global_scale).expand([k_z])
        )
        
        delta_beta = group_shrinkage_prior(
            "delta_beta",
            n_groups=k_p,
            dim=k_z,
            tau_global=self.delta_beta_tau,
            tau_local=0.5,
        )
        
        beta_k = beta_global + delta_beta
        numpyro.deterministic("beta_k", beta_k)
        
        intercept_k = numpyro.sample(
            "intercept_k",
            dist.Normal(0, 0.5).expand([k_p])
        )
        
        pathway_contrib = jnp.einsum('nk,kd,nd->nk', pi, beta_k, z)
        pathway_contrib = pathway_contrib + pi * intercept_k
        
        mu_peak_rapm = pathway_contrib.sum(axis=-1)
        numpyro.deterministic("mu_peak_rapm", mu_peak_rapm)
        numpyro.deterministic("pathway_contributions", pathway_contrib)

        # Heteroscedastic observation noise by possessions:
        # variance ∝ 1 / clip(peak_poss, min_poss, max_poss)
        sigma_base = numpyro.sample("sigma_peak_base", dist.HalfNormal(1.0))
        if peak_poss is None:
            poss_clip = jnp.ones((n,), dtype=jnp.float32) * self.max_poss
        else:
            poss_clip = jnp.clip(peak_poss, self.min_poss, self.max_poss)
        sigma_scaled = sigma_base / jnp.sqrt(poss_clip / self.min_poss)

        if y_peak_rapm is not None:
            y_clean = jnp.nan_to_num(y_peak_rapm, nan=0.0)

            if mask_rapm is None:
                mask_rapm = ~jnp.isnan(y_peak_rapm)

            with numpyro.plate("impact_plate", n):
                with numpyro.handlers.mask(mask=mask_rapm):
                    numpyro.sample(
                        "obs_peak_rapm",
                        dist.Normal(mu_peak_rapm, sigma_scaled),
                        obs=y_clean
                    )
        
        return mu_peak_rapm
    
    def fit(
        self,
        x: np.ndarray,
        y_peak_rapm: np.ndarray,
        y_made_nba: Optional[np.ndarray] = None,
        y_min_threshold: Optional[np.ndarray] = None,
        y_aux: Optional[Dict[str, np.ndarray]] = None,
        peak_poss: Optional[np.ndarray] = None,
        num_steps: int = 5000,
        lr: float = 0.005,
        seed: int = 42,
        progress_bar: bool = True,
    ) -> Dict[str, Any]:
        """
        Fit the model using SVI.
        
        Args:
            x: College features [N, D]
            y_peak_rapm: Peak RAPM targets [N]
            y_made_nba: Binary made NBA [N]
            y_min_threshold: Binary min≥500 given made [N]
            y_aux: Dict of auxiliary targets
            peak_poss: Possessions in peak window [N]
            num_steps: SVI iterations
            lr: Learning rate
            seed: Random seed
            progress_bar: Show progress
        
        Returns:
            Dict with losses and diagnostics
        """
        rng_key = jax.random.PRNGKey(seed)
        
        x = jnp.array(x, dtype=jnp.float32)
        y_peak_rapm = jnp.array(y_peak_rapm, dtype=jnp.float32)
        
        if y_made_nba is not None:
            y_made_nba = jnp.array(y_made_nba, dtype=jnp.float32)
        if y_min_threshold is not None:
            y_min_threshold = jnp.array(y_min_threshold, dtype=jnp.float32)
        if y_aux is not None:
            y_aux = {k: jnp.array(v, dtype=jnp.float32) for k, v in y_aux.items()}
        if peak_poss is not None:
            peak_poss = jnp.array(peak_poss, dtype=jnp.float32)
        
        mask_rapm = ~jnp.isnan(y_peak_rapm)
        mask_aux = None
        if y_aux is not None:
            mask_aux = {k: ~jnp.isnan(v) for k, v in y_aux.items()}
        
        self.guide = AutoNormal(self.model, init_loc_fn=numpyro.infer.init_to_median)
        
        optimizer = ClippedAdam(lr, clip_norm=10.0)
        
        svi = SVI(
            self.model,
            self.guide,
            optimizer,
            loss=Trace_ELBO(num_particles=4),
        )
        
        svi_result = svi.run(
            rng_key,
            num_steps,
            progress_bar=progress_bar,
            x=x,
            y_peak_rapm=y_peak_rapm,
            y_made_nba=y_made_nba,
            y_min_threshold=y_min_threshold,
            y_aux=y_aux,
            peak_poss=peak_poss,
            mask_rapm=mask_rapm,
            mask_aux=mask_aux,
        )
        
        self.params = svi_result.params
        self.losses = svi_result.losses
        
        return {
            'final_loss': float(svi_result.losses[-1]),
            'losses': np.array(svi_result.losses),
        }
    
    def predict(
        self,
        x: np.ndarray,
        num_samples: int = 500,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Predict for new players.
        
        Returns dict with:
        - mean, std: Peak RAPM prediction
        - p_made_nba: Selection probability
        - p_min_threshold: Minutes threshold probability
        - pi: Pathway assignments [N, K_p]
        - z: Latent traits [N, K_z]
        - pathway_contributions: Per-pathway RAPM contributions [N, K_p]
        """
        if self.params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        rng_key = jax.random.PRNGKey(seed)
        x = jnp.array(x, dtype=jnp.float32)
        
        predictive = Predictive(
            self.model,
            guide=self.guide,
            params=self.params,
            num_samples=num_samples,
        )
        
        predictions = predictive(rng_key, x=x)
        
        mu = predictions['mu_peak_rapm']
        
        return {
            'mean': np.array(mu.mean(axis=0)),
            'std': np.array(mu.std(axis=0)),
            'samples': np.array(mu),
            'p_made_nba': np.array(predictions['p_made_nba'].mean(axis=0)),
            'p_min_threshold': np.array(predictions['p_min_threshold'].mean(axis=0)),
            'pi': np.array(predictions['pi'].mean(axis=0)),
            'z': np.array(predictions['z'].mean(axis=0)),
            'pathway_contributions': np.array(predictions['pathway_contributions'].mean(axis=0)),
        }
    
    def decompose(
        self,
        x: np.ndarray,
        player_idx: int = 0,
        num_samples: int = 500,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """
        Decompose prediction for a single player showing pathway contributions.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        predictions = self.predict(x, num_samples, seed)
        samples = self.get_posterior_samples(x, num_samples, seed)
        
        z_mean = predictions['z'][player_idx]
        pi_mean = predictions['pi'][player_idx]
        pathway_contribs = predictions['pathway_contributions'][player_idx]
        
        beta_global = np.array(samples['beta_global']).mean(axis=0)
        delta_beta = np.array(samples['delta_beta']).mean(axis=0)
        beta_k = beta_global + delta_beta
        intercept_k = np.array(samples['intercept_k']).mean(axis=0)
        
        decomposition = {
            'total_mean': float(predictions['mean'][player_idx]),
            'total_std': float(predictions['std'][player_idx]),
            'p_made_nba': float(predictions['p_made_nba'][player_idx]),
            'p_min_threshold': float(predictions['p_min_threshold'][player_idx]),
            'z_mean': z_mean.tolist(),
            'pathway_probs': pi_mean.tolist(),
            'pathways': [],
        }
        
        for k in range(self.k_p):
            pathway_info = {
                'pathway_idx': k,
                'probability': float(pi_mean[k]),
                'contribution': float(pathway_contribs[k]),
                'intercept': float(intercept_k[k]),
                'beta_global_dot_z': float(np.dot(beta_global, z_mean)),
                'delta_beta_dot_z': float(np.dot(delta_beta[k], z_mean)),
                'top_traits': [],
            }
            
            trait_contribs = beta_k[k] * z_mean
            sorted_idx = np.argsort(-np.abs(trait_contribs))[:5]
            for idx in sorted_idx:
                pathway_info['top_traits'].append({
                    'trait_idx': int(idx),
                    'z_value': float(z_mean[idx]),
                    'beta_k': float(beta_k[k, idx]),
                    'contribution': float(trait_contribs[idx]),
                })
            
            decomposition['pathways'].append(pathway_info)
        
        return decomposition
    
    def get_posterior_samples(
        self,
        x: np.ndarray,
        num_samples: int = 500,
        seed: int = 0,
    ) -> Dict[str, jnp.ndarray]:
        """Get posterior samples for parameters."""
        if self.params is None:
            raise RuntimeError("Model not fitted.")
        
        rng_key = jax.random.PRNGKey(seed)
        x = jnp.array(x, dtype=jnp.float32)
        
        predictive = Predictive(
            self.guide,
            params=self.params,
            num_samples=num_samples,
        )
        
        return predictive(rng_key, x=x)
    
    def get_pathway_summary(
        self,
        x: np.ndarray,
        num_samples: int = 500,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """
        Get summary of pathway usage and characteristics.
        """
        predictions = self.predict(x, num_samples, seed)
        samples = self.get_posterior_samples(x, num_samples, seed)
        
        pi = predictions['pi']
        pathway_usage = pi.mean(axis=0)
        
        beta_global = np.array(samples['beta_global']).mean(axis=0)
        delta_beta = np.array(samples['delta_beta']).mean(axis=0)
        
        pathway_chars = []
        for k in range(self.k_p):
            deviation_magnitude = np.abs(delta_beta[k]).mean()
            top_positive_idx = np.argsort(-delta_beta[k])[:3]
            top_negative_idx = np.argsort(delta_beta[k])[:3]
            
            pathway_chars.append({
                'pathway_idx': k,
                'usage_fraction': float(pathway_usage[k]),
                'deviation_magnitude': float(deviation_magnitude),
                'top_positive_traits': top_positive_idx.tolist(),
                'top_negative_traits': top_negative_idx.tolist(),
            })
        
        return {
            'n_players': len(x),
            'pathway_usage': pathway_usage.tolist(),
            'beta_global_magnitude': float(np.abs(beta_global).mean()),
            'pathways': pathway_chars,
        }
