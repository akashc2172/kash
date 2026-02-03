"""
Prospect Model: Full Latent Space Model with Multi-Task Heads
=============================================================
Combines encoder with task-specific decoders for prediction and archetype discovery.

Architecture:
    College Features → Encoder → z (latent) → Multiple Decoders
                                    ↓
                         ┌──────────┼──────────┐
                         ▼          ▼          ▼
                    RAPM Head   Survival   Archetype
                                  Head       Head

Usage:
    model = ProspectModel(latent_dim=32, n_archetypes=8)
    outputs = model(tier1, tier2, career, within, tier2_mask, within_mask)
    # outputs contains: rapm_pred, survival_pred, archetype_logits, z
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np

from .player_encoder import PlayerEncoder, get_feature_dimensions


class RegressionHead(nn.Module):
    """Decoder head for regression targets (RAPM, gaps, EPM)."""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        n_targets: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_targets),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ArchetypeConditionedRegressionHead(nn.Module):
    """Regression head that conditions on learned archetype probabilities."""

    def __init__(
        self,
        latent_dim: int,
        n_archetypes: int,
        hidden_dim: int = 64,
        n_targets: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = latent_dim + n_archetypes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_targets),
        )

    def forward(self, z: torch.Tensor, archetype_probs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, archetype_probs], dim=-1)
        return self.net(x)


class SurvivalHead(nn.Module):
    """Decoder head for binary survival prediction (made_nba)."""
    
    def __init__(self, latent_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # Returns logits


class ArchetypeConditionedSurvivalHead(nn.Module):
    """Survival head that conditions on archetype probabilities."""

    def __init__(self, latent_dim: int, n_archetypes: int, hidden_dim: int = 32):
        super().__init__()
        in_dim = latent_dim + n_archetypes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, archetype_probs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, archetype_probs], dim=-1)
        return self.net(x)

class ArchetypeHead(nn.Module):
    """
    Archetype discovery via learned prototypes.
    
    Each archetype is a learnable prototype vector in latent space.
    Players are soft-assigned to archetypes based on distance.
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_archetypes: int = 8,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_archetypes = n_archetypes
        self.temperature = temperature
        
        # Learnable prototype vectors
        self.prototypes = nn.Parameter(torch.randn(n_archetypes, latent_dim) * 0.1)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [batch, latent_dim] Player embeddings
        
        Returns:
            assignments: [batch, n_archetypes] Soft archetype assignments (probabilities)
            distances: [batch, n_archetypes] Squared distances to each prototype
        """
        # Compute squared distances: ||z - p_k||^2
        # z: [B, D], prototypes: [K, D]
        # distances: [B, K]
        distances = torch.cdist(z, self.prototypes, p=2) ** 2
        
        # Soft assignments via softmax over negative distances
        assignments = F.softmax(-distances / self.temperature, dim=-1)
        
        return assignments, distances
    
    def get_archetype(self, z: torch.Tensor) -> torch.Tensor:
        """Hard assignment to nearest archetype."""
        _, distances = self.forward(z)
        return distances.argmin(dim=-1)
    
    def cluster_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Loss to encourage clustering around prototypes.
        
        Combines:
        1. Assignment entropy (encourage confident assignments)
        2. Prototype diversity (encourage prototypes to spread out)
        """
        assignments, distances = self.forward(z)
        
        # 1. Encourage confident assignments (low entropy)
        entropy = -(assignments * (assignments + 1e-8).log()).sum(dim=-1).mean()
        
        # 2. Encourage prototype diversity (prototypes should be spread out)
        proto_distances = torch.cdist(self.prototypes, self.prototypes, p=2)
        diversity_loss = -proto_distances.mean()  # Negative because we want to maximize
        
        return entropy + 0.1 * diversity_loss


class UncertaintyHead(nn.Module):
    """Predicts uncertainty (aleatoric) for regression targets."""
    
    def __init__(self, latent_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensure positive variance
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z) + 0.1  # Minimum variance floor


class ProspectModel(nn.Module):
    """
    Full prospect prediction model with latent space.
    
    Combines:
    - PlayerEncoder: Features → Latent embedding
    - RegressionHead: Latent → RAPM, gaps, EPM predictions
    - SurvivalHead: Latent → P(made_nba)
    - ArchetypeHead: Latent → Archetype assignments
    
    Args:
        latent_dim: Dimension of latent space (default: 32)
        n_archetypes: Number of archetype prototypes (default: 8)
        use_vae: Use VAE-style encoder with KL loss
        predict_uncertainty: Add uncertainty prediction heads
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        n_archetypes: int = 8,
        use_vae: bool = False,
        predict_uncertainty: bool = True,
        condition_on_archetypes: bool = False,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_archetypes = n_archetypes
        self.use_vae = use_vae
        self.predict_uncertainty = predict_uncertainty
        self.condition_on_archetypes = condition_on_archetypes
        
        # Get feature dimensions
        dims = get_feature_dimensions()
        
        # Encoder
        self.encoder = PlayerEncoder(
            tier1_dim=dims['tier1'],
            tier2_dim=dims['tier2'],
            career_dim=dims['career'],
            within_dim=dims['within'],
            latent_dim=latent_dim,
            use_vae=use_vae,
        )
        
        # Archetype head (always)
        self.archetype_head = ArchetypeHead(latent_dim, n_archetypes)

        # Regression + survival heads
        if condition_on_archetypes:
            self.rapm_head = ArchetypeConditionedRegressionHead(latent_dim, n_archetypes, n_targets=3)
            self.gap_head = ArchetypeConditionedRegressionHead(latent_dim, n_archetypes, n_targets=2)
            self.epm_head = ArchetypeConditionedRegressionHead(latent_dim, n_archetypes, n_targets=3)
            self.survival_head = ArchetypeConditionedSurvivalHead(latent_dim, n_archetypes)
        else:
            self.rapm_head = RegressionHead(latent_dim, n_targets=3)  # peak_ovr, off, def
            self.gap_head = RegressionHead(latent_dim, n_targets=2)   # gap_ts, gap_usg
            self.epm_head = RegressionHead(latent_dim, n_targets=3)   # year1 tot, off, def
            self.survival_head = SurvivalHead(latent_dim)
        
        # Uncertainty heads (optional)
        if predict_uncertainty:
            self.rapm_uncertainty = UncertaintyHead(latent_dim)
            self.gap_uncertainty = UncertaintyHead(latent_dim)
    
    def forward(
        self,
        tier1: torch.Tensor,
        tier2: torch.Tensor,
        career: torch.Tensor,
        within: torch.Tensor,
        tier2_mask: Optional[torch.Tensor] = None,
        within_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Returns dict with all predictions and latent embedding.
        """
        # Encode
        z, mu, logvar = self.encoder(tier1, tier2, career, within, tier2_mask, within_mask)
        
        archetype_assignments, archetype_distances = self.archetype_head(z)

        # Decode to predictions (optionally conditioned on archetype probs)
        if self.condition_on_archetypes:
            rapm_pred = self.rapm_head(z, archetype_assignments)
            gap_pred = self.gap_head(z, archetype_assignments)
            epm_pred = self.epm_head(z, archetype_assignments)
            survival_logits = self.survival_head(z, archetype_assignments)
        else:
            rapm_pred = self.rapm_head(z)
            gap_pred = self.gap_head(z)
            epm_pred = self.epm_head(z)
            survival_logits = self.survival_head(z)
        
        outputs = {
            'z': z,
            'rapm_pred': rapm_pred,           # [B, 3] - ovr, off, def
            'gap_pred': gap_pred,             # [B, 2] - ts, usg
            'epm_pred': epm_pred,             # [B, 3] - tot, off, def
            'survival_logits': survival_logits,  # [B, 1]
            'archetype_probs': archetype_assignments,  # [B, K]
            'archetype_distances': archetype_distances,  # [B, K]
        }
        
        # VAE outputs
        if self.use_vae and mu is not None:
            outputs['mu'] = mu
            outputs['logvar'] = logvar
        
        # Uncertainty outputs
        if self.predict_uncertainty:
            outputs['rapm_var'] = self.rapm_uncertainty(z)
            outputs['gap_var'] = self.gap_uncertainty(z)
        
        return outputs
    
    def encode(
        self,
        tier1: torch.Tensor,
        tier2: torch.Tensor,
        career: torch.Tensor,
        within: torch.Tensor,
        tier2_mask: Optional[torch.Tensor] = None,
        within_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get latent embedding only."""
        return self.encoder.encode(tier1, tier2, career, within, tier2_mask, within_mask)
    
    def predict_rapm(
        self,
        tier1: torch.Tensor,
        tier2: torch.Tensor,
        career: torch.Tensor,
        within: torch.Tensor,
        tier2_mask: Optional[torch.Tensor] = None,
        within_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict RAPM with optional uncertainty."""
        z = self.encode(tier1, tier2, career, within, tier2_mask, within_mask)
        pred = self.rapm_head(z)
        var = self.rapm_uncertainty(z) if self.predict_uncertainty else None
        return pred, var
    
    def get_archetype(
        self,
        tier1: torch.Tensor,
        tier2: torch.Tensor,
        career: torch.Tensor,
        within: torch.Tensor,
        tier2_mask: Optional[torch.Tensor] = None,
        within_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get hard archetype assignment."""
        z = self.encode(tier1, tier2, career, within, tier2_mask, within_mask)
        return self.archetype_head.get_archetype(z)


class ProspectLoss(nn.Module):
    """
    Combined loss function for ProspectModel.
    
    L = λ_rapm * MSE(rapm) 
      + λ_gap * MSE(gap)
      + λ_epm * MSE(epm)
      + λ_surv * BCE(survival)
      + λ_arch * ArchetypeLoss
      + λ_kl * KL(z)  [if VAE]
    """
    
    def __init__(
        self,
        lambda_rapm: float = 1.0,
        lambda_gap: float = 0.5,
        lambda_epm: float = 0.5,
        lambda_surv: float = 0.3,
        lambda_arch: float = 0.1,
        lambda_kl: float = 0.01,
        use_heteroscedastic: bool = True,
    ):
        super().__init__()
        self.lambda_rapm = lambda_rapm
        self.lambda_gap = lambda_gap
        self.lambda_epm = lambda_epm
        self.lambda_surv = lambda_surv
        self.lambda_arch = lambda_arch
        self.lambda_kl = lambda_kl
        self.use_heteroscedastic = use_heteroscedastic
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.
        
        Args:
            outputs: Model outputs dict
            targets: Dict with keys: rapm, gap, epm, survival (each with masks)
            weights: Optional sample weights
        
        Returns:
            Dict with individual losses and total
        """
        losses = {}
        
        # RAPM loss (primary)
        if 'rapm' in targets:
            rapm_target, rapm_mask = targets['rapm']
            rapm_pred = outputs['rapm_pred']
            
            if self.use_heteroscedastic and 'rapm_var' in outputs:
                # Heteroscedastic loss: -log p(y|μ,σ²)
                var = outputs['rapm_var']
                losses['rapm'] = self._heteroscedastic_loss(
                    rapm_pred[:, 0], rapm_target, var.squeeze(), rapm_mask
                )
            else:
                losses['rapm'] = self._masked_mse(rapm_pred[:, 0], rapm_target, rapm_mask)
        
        # Gap loss
        if 'gap' in targets:
            gap_target, gap_mask = targets['gap']
            gap_pred = outputs['gap_pred']
            losses['gap'] = self._masked_mse(gap_pred[:, 0], gap_target, gap_mask)
        
        # EPM loss
        if 'epm' in targets:
            epm_target, epm_mask = targets['epm']
            epm_pred = outputs['epm_pred']
            losses['epm'] = self._masked_mse(epm_pred[:, 0], epm_target, epm_mask)
        
        # Survival loss
        if 'survival' in targets:
            surv_target, surv_mask = targets['survival']
            surv_logits = outputs['survival_logits']
            losses['survival'] = self._masked_bce(surv_logits.squeeze(), surv_target, surv_mask)
        
        # Archetype clustering loss
        if 'archetype_probs' in outputs:
            losses['archetype'] = self._archetype_loss(outputs)
        
        # KL divergence (VAE)
        if 'mu' in outputs and 'logvar' in outputs:
            losses['kl'] = self._kl_loss(outputs['mu'], outputs['logvar'])
        
        # Total loss
        total = torch.tensor(0.0, device=outputs['z'].device)
        if 'rapm' in losses:
            total = total + self.lambda_rapm * losses['rapm']
        if 'gap' in losses:
            total = total + self.lambda_gap * losses['gap']
        if 'epm' in losses:
            total = total + self.lambda_epm * losses['epm']
        if 'survival' in losses:
            total = total + self.lambda_surv * losses['survival']
        if 'archetype' in losses:
            total = total + self.lambda_arch * losses['archetype']
        if 'kl' in losses:
            total = total + self.lambda_kl * losses['kl']
        
        losses['total'] = total
        return losses
    
    def _masked_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss with mask for missing targets."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return F.mse_loss(pred[mask], target[mask])
    
    def _masked_bce(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy with mask."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        return F.binary_cross_entropy_with_logits(logits[mask], target[mask].float())
    
    def _heteroscedastic_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        var: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian negative log-likelihood with learned variance."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        pred_m = pred[mask]
        target_m = target[mask]
        var_m = var[mask]
        
        # NLL: 0.5 * (log(σ²) + (y - μ)² / σ²)
        nll = 0.5 * (var_m.log() + (target_m - pred_m) ** 2 / var_m)
        return nll.mean()
    
    def _kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence from N(0,1) prior."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    def _archetype_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Archetype clustering loss."""
        probs = outputs['archetype_probs']
        # Encourage confident assignments (low entropy)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
        return entropy


def create_model(
    latent_dim: int = 32,
    n_archetypes: int = 8,
    use_vae: bool = False,
) -> ProspectModel:
    """Factory function to create model with default settings."""
    return ProspectModel(
        latent_dim=latent_dim,
        n_archetypes=n_archetypes,
        use_vae=use_vae,
        predict_uncertainty=True,
    )
