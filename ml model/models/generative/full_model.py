"""
Full Generative Prospect Model
==============================
Complete model combining encoder + impact head + aux head.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal, AutoLowRankMultivariateNormal
from numpyro.optim import Adam, ClippedAdam
from typing import Dict, Optional, Tuple, Any
import numpy as np

from .core import horseshoe_prior, compute_interaction_matrix


class GenerativeProspectModel:
    """
    Full Bayesian generative model for prospect prediction.
    
    Architecture:
        College Features x → Encoder → Latent Traits z
                                            ↓
                              ┌─────────────┴─────────────┐
                              ↓                           ↓
                         Aux Head                    Impact Head
                         p(NBA_stats|z)              p(RAPM|z)
    
    Key features:
    - ARD on encoder: discovers effective trait dimensions
    - Horseshoe on interactions: discovers real synergies
    - Heteroscedastic noise: exposure-weighted uncertainty
    - Full decomposition: trait → RAPM contributions
    """
    
    def __init__(
        self,
        k_max: int = 32,
        ard_scale: float = 1.0,
        tau_main: float = 1.0,
        tau_interaction: float = 0.3,
        sigma_residual: float = 0.1,
        include_aux_head: bool = True,
    ):
        """
        Args:
            k_max: Maximum latent dimensions (ARD will shrink unused)
            ard_scale: Prior scale for ARD (smaller = more shrinkage)
            tau_main: Global shrinkage for main effects
            tau_interaction: Global shrinkage for interactions (smaller = sparser)
            sigma_residual: Prior std for player residuals
            include_aux_head: Whether to include auxiliary head
        """
        self.k_max = k_max
        self.ard_scale = ard_scale
        self.tau_main = tau_main
        self.tau_interaction = tau_interaction
        self.sigma_residual = sigma_residual
        self.include_aux_head = include_aux_head
        
        # Will be set after fitting
        self.params = None
        self.guide = None
        self.losses = None
    
    def model(
        self,
        x: jnp.ndarray,
        y_rapm: Optional[jnp.ndarray] = None,
        y_aux: Optional[Dict[str, jnp.ndarray]] = None,
        exposure_rapm: Optional[jnp.ndarray] = None,
        exposure_aux: Optional[jnp.ndarray] = None,
        mask_rapm: Optional[jnp.ndarray] = None,
        mask_aux: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        """
        Full generative model.
        
        Args:
            x: College features [N, D]
            y_rapm: Peak RAPM targets [N] (can have NaN)
            y_aux: Dict of auxiliary targets (year1 stats)
            exposure_rapm: Possessions for RAPM weighting [N]
            exposure_aux: Minutes for aux weighting [N]
            mask_rapm: Boolean mask for valid RAPM obs [N]
            mask_aux: Dict of boolean masks for aux targets
        """
        n, d = x.shape
        k = self.k_max
        n_pairs = k * (k - 1) // 2
        
        # === ENCODER: x → z ===
        # ARD scales for each latent dimension
        alpha = numpyro.sample(
            "alpha",
            dist.HalfCauchy(self.ard_scale).expand([k])
        )
        
        # Encoder weights with ARD
        W = numpyro.sample(
            "W_encoder",
            dist.Normal(0, 1).expand([d, k])
        )
        # Scale columns by alpha
        W_scaled = W * alpha
        
        # Bias
        b = numpyro.sample(
            "b_encoder",
            dist.Normal(0, 0.1).expand([k])
        )
        
        # Latent traits
        z = jnp.dot(x, W_scaled) + b  # [N, K]
        numpyro.deterministic("z", z)
        
        # === IMPACT HEAD: z → RAPM ===
        intercept = numpyro.sample("intercept", dist.Normal(0, 0.5))
        
        # Main effects with horseshoe
        beta = horseshoe_prior("beta", shape=(k,), tau_scale=self.tau_main)
        main_effect = jnp.dot(z, beta)  # [N]
        
        # All pairwise interactions with horseshoe
        rho = horseshoe_prior("rho", shape=(n_pairs,), tau_scale=self.tau_interaction)
        z_interact = compute_interaction_matrix(z)  # [N, n_pairs]
        interaction_effect = jnp.dot(z_interact, rho)  # [N]
        
        # Player residual (unmeasured value)
        h = numpyro.sample(
            "h",
            dist.Normal(0, self.sigma_residual).expand([n])
        )
        
        # Predicted RAPM
        mu_rapm = intercept + main_effect + interaction_effect + h
        numpyro.deterministic("mu_rapm", mu_rapm)
        numpyro.deterministic("main_effect", main_effect)
        numpyro.deterministic("interaction_effect", interaction_effect)
        
        # Observation noise
        sigma_rapm = numpyro.sample("sigma_rapm", dist.HalfNormal(1.0))
        
        # RAPM likelihood
        if y_rapm is not None:
            # Clean the target
            y_clean = jnp.nan_to_num(y_rapm, nan=0.0)
            
            # Create mask if needed
            if mask_rapm is None:
                mask_rapm = jnp.ones(n, dtype=bool)
            
            # Plate-based observation with masking
            with numpyro.plate("data", n):
                with numpyro.handlers.mask(mask=mask_rapm):
                    numpyro.sample("y_obs", dist.Normal(mu_rapm, sigma_rapm), obs=y_clean)
        
        # === AUX HEAD: z → Year-1 stats ===
        if self.include_aux_head and y_aux is not None:
            sigma_aux = numpyro.sample("sigma_aux", dist.HalfNormal(0.5))
            
            for aux_name, aux_target in y_aux.items():
                # Linear head for each aux target
                w_aux = numpyro.sample(
                    f"w_aux_{aux_name}",
                    dist.Normal(0, 0.5).expand([k])
                )
                b_aux = numpyro.sample(
                    f"b_aux_{aux_name}",
                    dist.Normal(0, 0.5)
                )
                
                mu_aux = jnp.dot(z, w_aux) + b_aux
                
                # Get mask
                if mask_aux is not None and aux_name in mask_aux:
                    m = mask_aux[aux_name]
                else:
                    m = ~jnp.isnan(aux_target)
                
                with numpyro.handlers.mask(mask=m):
                    numpyro.sample(
                        f"y_aux_{aux_name}",
                        dist.Normal(mu_aux, sigma_aux),
                        obs=jnp.where(m, aux_target, 0.0)
                    )
        
        return mu_rapm
    
    def fit(
        self,
        x: np.ndarray,
        y_rapm: np.ndarray,
        y_aux: Optional[Dict[str, np.ndarray]] = None,
        exposure_rapm: Optional[np.ndarray] = None,
        exposure_aux: Optional[np.ndarray] = None,
        num_steps: int = 10000,
        lr: float = 0.005,
        seed: int = 42,
        progress_bar: bool = True,
    ) -> Dict[str, Any]:
        """
        Fit the model using SVI.
        
        Args:
            x: College features [N, D]
            y_rapm: Peak RAPM targets [N]
            y_aux: Dict of auxiliary targets
            exposure_rapm: Possessions [N]
            exposure_aux: Minutes [N]
            num_steps: SVI iterations
            lr: Learning rate
            seed: Random seed
            progress_bar: Show progress
        
        Returns:
            Dict with losses and diagnostics
        """
        rng_key = jax.random.PRNGKey(seed)
        
        # Convert to JAX arrays
        x = jnp.array(x, dtype=jnp.float32)
        y_rapm = jnp.array(y_rapm, dtype=jnp.float32)
        
        if y_aux is not None:
            y_aux = {k: jnp.array(v, dtype=jnp.float32) for k, v in y_aux.items()}
        
        if exposure_rapm is not None:
            exposure_rapm = jnp.array(exposure_rapm, dtype=jnp.float32)
        
        # Create masks
        mask_rapm = ~jnp.isnan(y_rapm)
        mask_aux = None
        if y_aux is not None:
            mask_aux = {k: ~jnp.isnan(v) for k, v in y_aux.items()}
        
        # Guide (variational family)
        self.guide = AutoNormal(self.model, init_loc_fn=numpyro.infer.init_to_median)
        
        # Optimizer with gradient clipping
        optimizer = ClippedAdam(lr, clip_norm=10.0)
        
        # SVI
        svi = SVI(
            self.model,
            self.guide,
            optimizer,
            loss=Trace_ELBO(num_particles=4),
        )
        
        # Run
        svi_result = svi.run(
            rng_key,
            num_steps,
            progress_bar=progress_bar,
            x=x,
            y_rapm=y_rapm,
            y_aux=y_aux,
            exposure_rapm=exposure_rapm,
            mask_rapm=mask_rapm,
            mask_aux=mask_aux,
        )
        
        self.params = svi_result.params
        self.losses = svi_result.losses
        
        return {
            'final_loss': float(svi_result.losses[-1]),
            'losses': np.array(svi_result.losses),
        }
    
    def get_posterior_samples(
        self,
        x: np.ndarray,
        num_samples: int = 1000,
        seed: int = 0,
    ) -> Dict[str, jnp.ndarray]:
        """
        Get posterior samples for parameters and predictions.
        
        Args:
            x: College features [N, D]
            num_samples: Number of posterior samples
            seed: Random seed
        
        Returns:
            Dict of posterior samples
        """
        if self.params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        rng_key = jax.random.PRNGKey(seed)
        x = jnp.array(x, dtype=jnp.float32)
        
        # Sample from guide
        predictive = Predictive(
            self.guide,
            params=self.params,
            num_samples=num_samples,
        )
        
        samples = predictive(rng_key, x=x)
        
        return samples
    
    def predict(
        self,
        x: np.ndarray,
        num_samples: int = 500,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Predict RAPM for new players.
        
        Args:
            x: College features [N, D]
            num_samples: Posterior samples for uncertainty
            seed: Random seed
        
        Returns:
            Dict with predictions and uncertainty
        """
        if self.params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        rng_key = jax.random.PRNGKey(seed)
        x = jnp.array(x, dtype=jnp.float32)
        
        # Use Predictive for posterior predictions
        predictive = Predictive(
            self.model,
            guide=self.guide,
            params=self.params,
            num_samples=num_samples,
        )
        
        predictions = predictive(rng_key, x=x)
        
        mu_rapm = predictions['mu_rapm']  # [samples, N]
        
        return {
            'mean': np.array(mu_rapm.mean(axis=0)),
            'std': np.array(mu_rapm.std(axis=0)),
            'samples': np.array(mu_rapm),
            'z': np.array(predictions['z']),
            'main_effect': np.array(predictions['main_effect']),
            'interaction_effect': np.array(predictions['interaction_effect']),
        }
    
    def decompose(
        self,
        x: np.ndarray,
        player_idx: int = 0,
        num_samples: int = 500,
        cumulative_threshold: float = 0.8,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """
        Decompose prediction for a single player.
        
        Args:
            x: College features [N, D] or [D]
            player_idx: Which player to decompose (if x is [N, D])
            num_samples: Posterior samples
            cumulative_threshold: Threshold for "top traits"
            seed: Random seed
        
        Returns:
            Detailed decomposition dict
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        predictions = self.predict(x, num_samples, seed)
        samples = self.get_posterior_samples(x, num_samples, seed)
        
        # Extract for this player
        z = predictions['z'][:, player_idx, :]  # [samples, K]
        z_mean = z.mean(axis=0)  # [K]
        
        beta = samples['beta']  # [samples, K]
        rho = samples['rho']  # [samples, n_pairs]
        intercept = samples['intercept']  # [samples]
        
        k = self.k_max
        n_pairs = k * (k - 1) // 2
        
        # Main effect contributions: β_k * z_k
        main_contribs = beta * z_mean  # [samples, K]
        main_mean = np.array(main_contribs.mean(axis=0))
        main_std = np.array(main_contribs.std(axis=0))
        
        # Interaction contributions
        pairs = []
        for a in range(k):
            for b in range(a + 1, k):
                pairs.append((a, b))
        
        z_pairs = np.array([z_mean[a] * z_mean[b] for a, b in pairs])
        interact_contribs = np.array(rho) * z_pairs  # [samples, n_pairs]
        interact_mean = interact_contribs.mean(axis=0)
        interact_std = interact_contribs.std(axis=0)
        
        # Find top traits (cumulative threshold)
        abs_main = np.abs(main_mean)
        sorted_idx = np.argsort(-abs_main)
        total_abs = abs_main.sum()
        
        top_traits = []
        cumsum = 0.0
        for idx in sorted_idx:
            cumsum += abs_main[idx]
            top_traits.append({
                'trait_idx': int(idx),
                'z_value': float(z_mean[idx]),
                'contribution': float(main_mean[idx]),
                'contribution_std': float(main_std[idx]),
                'cumulative_frac': float(cumsum / total_abs) if total_abs > 0 else 1.0,
            })
            if cumsum / total_abs >= cumulative_threshold:
                break
        
        # Find active interactions
        active_interactions = []
        for i, (a, b) in enumerate(pairs):
            mean_i = float(interact_mean[i])
            std_i = float(interact_std[i])
            
            # Include if |mean| > 2*std (roughly significant)
            if abs(mean_i) > 0.05 or abs(mean_i) > 2 * std_i:
                active_interactions.append({
                    'pair': (int(a), int(b)),
                    'contribution': mean_i,
                    'contribution_std': std_i,
                })
        
        # Sort by absolute contribution
        active_interactions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        # Total prediction
        total_mean = float(predictions['mean'][player_idx])
        total_std = float(predictions['std'][player_idx])
        
        return {
            'total_mean': total_mean,
            'total_std': total_std,
            'intercept': float(np.array(intercept).mean()),
            'main_total': float(main_mean.sum()),
            'interaction_total': float(interact_mean.sum()),
            'top_traits': top_traits,
            'active_interactions': active_interactions[:10],  # Top 10
            'z_mean': z_mean.tolist(),
            'alpha_mean': np.array(samples['alpha']).mean(axis=0).tolist(),
        }
    
    def get_effective_dimensions(
        self,
        threshold: float = 0.1,
        num_samples: int = 500,
        seed: int = 0,
    ) -> Tuple[np.ndarray, int]:
        """
        Get which latent dimensions are "active" based on ARD.
        
        Returns:
            Tuple of (active_mask, k_effective)
        """
        if self.params is None:
            raise RuntimeError("Model not fitted.")
        
        # Dummy x for sampling
        x = jnp.zeros((1, 10))  # Adjust based on actual input dim
        samples = self.get_posterior_samples(x, num_samples, seed)
        
        alpha = np.array(samples['alpha'])  # [samples, K]
        alpha_mean = alpha.mean(axis=0)
        
        active = alpha_mean > threshold
        k_eff = int(active.sum())
        
        return active, k_eff
