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
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        n_targets: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = input_dim if input_dim is not None else latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
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
    
    def __init__(self, latent_dim: int, hidden_dim: int = 32, input_dim: Optional[int] = None):
        super().__init__()
        in_dim = input_dim if input_dim is not None else latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
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
        year1_feature_dim: int = 5,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_archetypes = n_archetypes
        self.use_vae = use_vae
        self.predict_uncertainty = predict_uncertainty
        self.condition_on_archetypes = condition_on_archetypes
        self.year1_feature_dim = year1_feature_dim
        
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

        # Development-rate head with gated Year-1 interaction pathway.
        self.year1_hidden_dim = max(8, latent_dim // 2)
        self.year1_branch = nn.Sequential(
            nn.Linear(year1_feature_dim, self.year1_hidden_dim),
            nn.LayerNorm(self.year1_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.year1_to_latent = nn.Linear(self.year1_hidden_dim, latent_dim)
        dev_input_dim = latent_dim + self.year1_hidden_dim + latent_dim
        self.dev_head = RegressionHead(latent_dim, input_dim=dev_input_dim, n_targets=1)
        
        # Uncertainty heads (optional)
        if predict_uncertainty:
            self.rapm_uncertainty = UncertaintyHead(latent_dim)
            self.gap_uncertainty = UncertaintyHead(latent_dim)
            self.dev_uncertainty = UncertaintyHead(latent_dim, input_dim=dev_input_dim)
    
    def forward(
        self,
        tier1: torch.Tensor,
        tier2: torch.Tensor,
        career: torch.Tensor,
        within: torch.Tensor,
        tier2_mask: Optional[torch.Tensor] = None,
        within_mask: Optional[torch.Tensor] = None,
        year1: Optional[torch.Tensor] = None,
        year1_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Returns dict with all predictions and latent embedding.
        """
        batch_size = tier1.shape[0]

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

        # Year-1 interaction features are optional. If absent, this is prospect-only mode.
        if year1 is None:
            year1 = torch.zeros((batch_size, self.year1_feature_dim), device=tier1.device, dtype=tier1.dtype)
        if year1_mask is None:
            year1_mask = torch.zeros_like(year1)
        elif year1_mask.shape[-1] == 1:
            year1_mask = year1_mask.repeat(1, self.year1_feature_dim)

        year1_clean = torch.nan_to_num(year1, nan=0.0)
        year1_mask = torch.nan_to_num(year1_mask, nan=0.0).clamp(0.0, 1.0)
        year1_coverage = (year1_mask.sum(dim=1, keepdim=True) > 0).float()
        year1_hidden = self.year1_branch(year1_clean * year1_mask) * year1_coverage
        interaction_latent = z * torch.tanh(self.year1_to_latent(year1_hidden))
        dev_input = torch.cat([z, year1_hidden, interaction_latent], dim=-1)
        dev_pred = self.dev_head(dev_input)
        
        outputs = {
            'z': z,
            'rapm_pred': rapm_pred,           # [B, 3] - ovr, off, def
            'gap_pred': gap_pred,             # [B, 2] - ts, usg
            'epm_pred': epm_pred,             # [B, 3] - tot, off, def
            'dev_pred': dev_pred,             # [B, 1] - Y1->Y3 dev-rate
            'survival_logits': survival_logits,  # [B, 1]
            'archetype_probs': archetype_assignments,  # [B, K]
            'archetype_distances': archetype_distances,  # [B, K]
            'year1_coverage': year1_coverage,  # [B, 1]
        }
        
        # VAE outputs
        if self.use_vae and mu is not None:
            outputs['mu'] = mu
            outputs['logvar'] = logvar
        
        # Uncertainty outputs
        if self.predict_uncertainty:
            outputs['rapm_var'] = self.rapm_uncertainty(z)
            outputs['gap_var'] = self.gap_uncertainty(z)
            outputs['dev_var'] = self.dev_uncertainty(dev_input)
        
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
      + λ_dev * MSE(dev_rate)
      + λ_surv * BCE(survival)
      + λ_arch * ArchetypeLoss
      + λ_kl * KL(z)  [if VAE]
    """
    
    def __init__(
        self,
        lambda_rapm: float = 1.0,
        lambda_gap: float = 0.5,
        lambda_epm: float = 0.5,
        lambda_dev: float = 0.5,
        lambda_surv: float = 0.3,
        lambda_arch: float = 0.1,
        lambda_kl: float = 0.01,
        lambda_rapm_var: float = 0.15,
        use_heteroscedastic: bool = True,
    ):
        super().__init__()
        self.lambda_rapm = lambda_rapm
        self.lambda_gap = lambda_gap
        self.lambda_epm = lambda_epm
        self.lambda_dev = lambda_dev
        self.lambda_surv = lambda_surv
        self.lambda_arch = lambda_arch
        self.lambda_kl = lambda_kl
        self.lambda_rapm_var = lambda_rapm_var
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

        def _unpack_target(target_spec):
            """Accept (value, mask) or (value, mask, weight) or dict-style payload."""
            if isinstance(target_spec, tuple):
                if len(target_spec) == 3:
                    return target_spec[0], target_spec[1], target_spec[2]
                return target_spec[0], target_spec[1], None
            return (
                target_spec["value"],
                target_spec["mask"],
                target_spec.get("weight"),
            )
        
        # RAPM loss (primary)
        if 'rapm' in targets:
            rapm_target, rapm_mask, rapm_w = _unpack_target(targets['rapm'])
            rapm_pred = outputs['rapm_pred']
            # Support both legacy scalar target and 3-head RAPM supervision.
            if rapm_target.dim() == 1:
                losses['rapm'] = self._masked_mse(rapm_pred[:, 0], rapm_target, rapm_mask, sample_weights=rapm_w)
                if rapm_mask.sum() >= 32:
                    p = rapm_pred[:, 0][rapm_mask]
                    t = rapm_target[rapm_mask]
                    losses['rapm_var'] = (torch.std(p) - torch.std(t)) ** 2
            else:
                # Weighted multi-component RAPM loss: overall + off + def.
                rapm_mask = rapm_mask.bool()
                l_ovr = self._masked_mse(rapm_pred[:, 0], rapm_target[:, 0], rapm_mask[:, 0], sample_weights=rapm_w)
                l_off = self._masked_mse(rapm_pred[:, 1], rapm_target[:, 1], rapm_mask[:, 1], sample_weights=rapm_w)
                l_def = self._masked_mse(rapm_pred[:, 2], rapm_target[:, 2], rapm_mask[:, 2], sample_weights=rapm_w)
                losses['rapm'] = l_ovr + 0.40 * l_off + 0.40 * l_def
                if rapm_mask[:, 0].sum() >= 32:
                    p = rapm_pred[:, 0][rapm_mask[:, 0]]
                    t = rapm_target[:, 0][rapm_mask[:, 0]]
                    losses['rapm_var'] = (torch.std(p) - torch.std(t)) ** 2
        
        # Gap loss
        if 'gap' in targets:
            gap_target, gap_mask, gap_w = _unpack_target(targets['gap'])
            gap_pred = outputs['gap_pred']
            losses['gap'] = self._masked_mse(gap_pred[:, 0], gap_target, gap_mask, sample_weights=gap_w)
        
        # EPM loss
        if 'epm' in targets:
            epm_target, epm_mask, epm_w = _unpack_target(targets['epm'])
            epm_pred = outputs['epm_pred']
            losses['epm'] = self._masked_mse(epm_pred[:, 0], epm_target, epm_mask, sample_weights=epm_w)

        # Development-rate loss (quality-weighted, mask-aware)
        if 'dev' in targets and 'dev_pred' in outputs:
            dev_target_vals, dev_mask, dev_weights = _unpack_target(targets['dev'])

            dev_pred = outputs['dev_pred'][:, 0]
            if self.use_heteroscedastic and 'dev_var' in outputs:
                losses['dev'] = self._heteroscedastic_loss(
                    dev_pred,
                    dev_target_vals,
                    outputs['dev_var'][:, 0],
                    dev_mask,
                    sample_weights=dev_weights,
                )
            else:
                losses['dev'] = self._masked_mse(
                    dev_pred,
                    dev_target_vals,
                    dev_mask,
                    sample_weights=dev_weights,
                )
        
        # Survival loss
        if 'survival' in targets:
            surv_target, surv_mask, surv_w = _unpack_target(targets['survival'])
            surv_logits = outputs['survival_logits']
            losses['survival'] = self._masked_bce(surv_logits.squeeze(), surv_target, surv_mask, sample_weights=surv_w)
        
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
        if 'rapm_var' in losses:
            total = total + self.lambda_rapm_var * losses['rapm_var']
        if 'gap' in losses:
            total = total + self.lambda_gap * losses['gap']
        if 'epm' in losses:
            total = total + self.lambda_epm * losses['epm']
        if 'dev' in losses:
            total = total + self.lambda_dev * losses['dev']
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
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MSE loss with mask for missing targets."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        err = (pred[mask] - target[mask]) ** 2
        if sample_weights is not None:
            w = sample_weights[mask].clamp(min=0.0)
            return (err * w).sum() / (w.sum() + 1e-8)
        return err.mean()
    
    def _masked_bce(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Binary cross-entropy with mask."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits[mask], target[mask].float(), reduction='none')
        if sample_weights is not None:
            w = sample_weights[mask].clamp(min=0.0)
            return (bce * w).sum() / (w.sum() + 1e-8)
        return bce.mean()
    
    def _heteroscedastic_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        var: torch.Tensor,
        mask: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gaussian negative log-likelihood with learned variance."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        pred_m = pred[mask]
        target_m = target[mask]
        var_m = var[mask]
        
        # NLL: 0.5 * (log(σ²) + (y - μ)² / σ²)
        nll = 0.5 * (var_m.log() + (target_m - pred_m) ** 2 / var_m)
        if sample_weights is not None:
            w = sample_weights[mask].clamp(min=0.0)
            return (nll * w).sum() / (w.sum() + 1e-8)
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
