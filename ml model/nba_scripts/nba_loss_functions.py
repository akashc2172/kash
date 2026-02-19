"""
NBA Loss Functions
==================
Implements loss functions with exposure-based weighting per the proposal.

Key specifications:
- Gaussian likelihood with heteroscedastic variance: σ² ∝ 1/(exposure+ε)
- Negative log predictive density (NLPD) for calibration
- Multi-task loss combining primary target and auxiliary supervision
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

EPSILON = 1e-6


class WeightedMSELoss(nn.Module):
    """
    MSE loss with sample weights.
    
    Per spec: "loss weighted by peak_poss" and "w = min(1, mp / mp_ref)"
    """
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions [batch, 1] or [batch, num_targets]
            target: Ground truth [batch, 1] or [batch, num_targets]
            weights: Sample weights [batch] for reliability weighting
            mask: Boolean mask [batch] for which samples to include
        """
        sq_error = (pred - target) ** 2
        
        # Build denominator for proper normalization
        denom = torch.ones_like(sq_error)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            sq_error = sq_error * mask_expanded
            denom = denom * mask_expanded
        
        if weights is not None:
            # Ensure weights are broadcastable
            if weights.dim() == 1:
                weights = weights.unsqueeze(-1)
            sq_error = sq_error * weights
            denom = denom * weights
        
        # Normalize by sum of valid weighted samples, not total batch size
        return sq_error.sum() / (denom.sum() + EPSILON)


class HeteroscedasticGaussianNLL(nn.Module):
    """
    Gaussian negative log-likelihood with observation-specific variance.
    
    Per spec: "σ² ∝ 1/(mp+ε)" and "σ² ∝ 1/(peak_poss+ε)"
    
    This allows the model to learn that low-exposure observations
    are noisier and should be downweighted in the loss.
    """
    def __init__(self, learn_scale: bool = False):
        super().__init__()
        self.learn_scale = learn_scale
        if learn_scale:
            # Learnable global scale factor
            self.log_scale = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        variance: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions [batch, num_targets]
            target: Ground truth [batch, num_targets]
            variance: Per-sample variance [batch, 1] based on exposure
            mask: Boolean mask [batch] for which samples to include
        """
        if self.learn_scale:
            variance = variance * torch.exp(self.log_scale)
        
        # Gaussian NLL: 0.5 * (log(2πσ²) + (y-μ)²/σ²)
        log_var = torch.log(variance + EPSILON)
        sq_error = (pred - target) ** 2
        nll = 0.5 * (log_var + sq_error / (variance + EPSILON))
        
        if mask is not None:
            nll = nll * mask.unsqueeze(-1).float()
            num_valid = mask.sum()
            return nll.sum() / (num_valid + EPSILON)
        
        return nll.mean()


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining primary target and auxiliary supervision.
    
    Per spec:
    - Primary target: Peak 3Y RAPM (y_peak_ovr/off/def)
    - Auxiliary supervision: Year-1 EPM (year1_epm_tot/off/def)
    - Auxiliary supervision: Development rate Y1->Y3 (dev_rate_y1_y3_mean)
    - Aux observations: Year-1 stats for p(a|z) likelihood
    
    The auxiliary heads help stabilize latent trait learning.
    """
    def __init__(
        self,
        primary_weight: float = 1.0,
        aux_target_weight: float = 0.3,
        aux_obs_weight: float = 0.1,
        dev_weight: float = 0.2,
    ):
        super().__init__()
        self.primary_weight = primary_weight
        self.aux_target_weight = aux_target_weight
        self.aux_obs_weight = aux_obs_weight
        self.dev_weight = dev_weight
        
        self.primary_loss = HeteroscedasticGaussianNLL()
        self.aux_target_loss = WeightedMSELoss()
        self.aux_obs_loss = WeightedMSELoss()
        self.dev_loss = WeightedMSELoss()
    
    def forward(
        self,
        pred_primary: torch.Tensor,
        target_primary: torch.Tensor,
        primary_variance: torch.Tensor,
        primary_mask: torch.Tensor,
        pred_aux_target: Optional[torch.Tensor] = None,
        target_aux_target: Optional[torch.Tensor] = None,
        aux_target_weights: Optional[torch.Tensor] = None,
        aux_target_mask: Optional[torch.Tensor] = None,
        pred_aux_obs: Optional[torch.Tensor] = None,
        target_aux_obs: Optional[torch.Tensor] = None,
        aux_obs_weights: Optional[torch.Tensor] = None,
        aux_obs_mask: Optional[torch.Tensor] = None,
        pred_dev_target: Optional[torch.Tensor] = None,
        target_dev_target: Optional[torch.Tensor] = None,
        dev_target_weights: Optional[torch.Tensor] = None,
        dev_target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute multi-task loss.
        
        Returns:
            total_loss: Combined weighted loss
            components: Dict of individual loss components for logging
        """
        components = {}
        
        # Primary loss (Peak RAPM)
        loss_primary = self.primary_loss(
            pred_primary, target_primary, primary_variance, primary_mask
        )
        components['loss_primary'] = loss_primary.item()
        
        total = self.primary_weight * loss_primary
        
        # Auxiliary target loss (Year-1 EPM)
        if pred_aux_target is not None and target_aux_target is not None:
            loss_aux_target = self.aux_target_loss(
                pred_aux_target, target_aux_target, aux_target_weights, aux_target_mask
            )
            components['loss_aux_target'] = loss_aux_target.item()
            total = total + self.aux_target_weight * loss_aux_target
        
        # Auxiliary observation loss (Year-1 stats for p(a|z))
        if pred_aux_obs is not None and target_aux_obs is not None:
            loss_aux_obs = self.aux_obs_loss(
                pred_aux_obs, target_aux_obs, aux_obs_weights, aux_obs_mask
            )
            components['loss_aux_obs'] = loss_aux_obs.item()
            total = total + self.aux_obs_weight * loss_aux_obs

        # Development-rate auxiliary (quality weighted)
        if pred_dev_target is not None and target_dev_target is not None:
            loss_dev = self.dev_loss(
                pred_dev_target, target_dev_target, dev_target_weights, dev_target_mask
            )
            components['loss_dev'] = loss_dev.item()
            total = total + self.dev_weight * loss_dev
        
        components['loss_total'] = total.item()
        return total, components


def compute_primary_variance(peak_poss: torch.Tensor) -> torch.Tensor:
    """
    Compute variance for primary target (Peak RAPM) based on peak possessions.
    
    Per spec: "σ² ∝ 1/(peak_poss+ε)"
    
    Only peak_poss is relevant for the peak RAPM reliability signal.
    """
    return 1.0 / (peak_poss + EPSILON)


def compute_year1_variance(year1_mp: torch.Tensor) -> torch.Tensor:
    """
    Compute variance for Year-1 targets/observations based on minutes played.
    
    Per spec: "σ² ∝ 1/(mp+ε)"
    
    Only year1_mp is relevant for the Year-1 stats reliability signal.
    """
    return 1.0 / (year1_mp + EPSILON)


def compute_exposure_variance(
    mp: torch.Tensor,
    poss: Optional[torch.Tensor] = None,
    mp_ref: float = 2000.0,
    poss_ref: float = 10000.0
) -> torch.Tensor:
    """
    DEPRECATED: Use compute_primary_variance or compute_year1_variance instead.
    
    This function incorrectly combined mp and poss.
    Kept for backwards compatibility but should not be used.
    """
    import warnings
    warnings.warn(
        "compute_exposure_variance is deprecated. Use compute_primary_variance or compute_year1_variance.",
        DeprecationWarning
    )
    return 1.0 / (mp + EPSILON)


if __name__ == '__main__':
    print("NBA Loss Functions module loaded.")
    print("Available losses: WeightedMSELoss, HeteroscedasticGaussianNLL, MultiTaskLoss")
