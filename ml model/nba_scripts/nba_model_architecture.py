"""
NBA Model Architecture
======================
Implements the latent trait model architecture per the proposal.

Architecture overview:
- Encoder: Maps bio/anthro features to latent trait vector z_i
- Decoder heads:
  - Primary head: Predicts peak RAPM from z_i
  - Aux target head: Predicts Year-1 EPM from z_i (multi-task)
  - Aux observation head: Reconstructs Year-1 stats for p(a|z)

Per spec:
- "build a probabilistic 'world model' that infers a latent skill vector z_i"
- "inference for new prospects uses only amateur data—no NBA stats"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

EPSILON = 1e-6


class LatentEncoder(nn.Module):
    """
    Encoder network: Bio/Anthro features -> Latent traits z.
    
    Per spec: "infers a latent skill vector z_i from amateur evidence x_i"
    
    Note: At inference time for prospects, we use ONLY college features.
    For NBA historical data, we use bio features since college data
    is not available.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, ...] = (64, 64),
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # For VAE: output mean and log-variance
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu: Latent mean [batch, latent_dim]
            logvar: Latent log-variance [batch, latent_dim]
        """
        h = self.backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Reparameterization trick for VAE sampling.
        
        Per spec: "probabilistic 'world model'"
        """
        if deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class PrimaryImpactHead(nn.Module):
    """
    Decoder head for primary target: Peak 3Y RAPM.
    
    Per spec: "Primary target: y_i,peak = peak 3-year RAPM"
    """
    def __init__(
        self,
        latent_dim: int,
        output_dim: int = 3,  # ovr, off, def
        hidden_dim: int = 32
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict peak RAPM from latent traits."""
        return self.net(z)


class AuxTargetHead(nn.Module):
    """
    Decoder head for auxiliary supervision: Year-1 EPM.
    
    Per spec: "Auxiliary supervision: m_i,1 = year-1 EPM"
    "train impact head to predict (year1_epm_tot/off/def) in addition to peak RAPM"
    """
    def __init__(
        self,
        latent_dim: int,
        output_dim: int = 4,  # tot, off, def, ewins
        hidden_dim: int = 32
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict Year-1 EPM from latent traits."""
        return self.net(z)


class AuxObservationHead(nn.Module):
    """
    Decoder head for auxiliary observations: Year-1 box stats.
    
    Per spec: "Year-1 per-season NBA stat used in the 'aux head' likelihood p(a | z)"
    "Teaches which latent traits manifest as which NBA behaviors"
    """
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,  # Number of aux observation columns
        hidden_dim: int = 64
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct Year-1 stats from latent traits."""
        return self.net(z)


class NBAProspectModel(nn.Module):
    """
    Full model combining encoder and all decoder heads.
    
    Per spec architecture:
    - Encoder: Bio/Anthro -> z
    - Primary head: z -> Peak RAPM
    - Aux target head: z -> Year-1 EPM
    - Aux observation head: z -> Year-1 stats reconstruction
    """
    def __init__(
        self,
        input_dim: int,
        num_aux_obs: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, ...] = (64, 64),
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = LatentEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        self.primary_head = PrimaryImpactHead(latent_dim=latent_dim)
        self.aux_target_head = AuxTargetHead(latent_dim=latent_dim)
        self.aux_obs_head = AuxObservationHead(
            latent_dim=latent_dim,
            output_dim=num_aux_obs
        )
        
        self.latent_dim = latent_dim
    
    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through full model.
        
        Args:
            x: Input features [batch, input_dim]
            deterministic: If True, use mean instead of sampling
        
        Returns:
            Dict with:
                z: Latent traits [batch, latent_dim]
                mu, logvar: Encoder outputs
                pred_primary: Peak RAPM predictions [batch, 3]
                pred_aux_target: Year-1 EPM predictions [batch, 4]
                pred_aux_obs: Year-1 stats reconstruction [batch, num_aux_obs]
        """
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar, deterministic)
        
        pred_primary = self.primary_head(z)
        pred_aux_target = self.aux_target_head(z)
        pred_aux_obs = self.aux_obs_head(z)
        
        return {
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'pred_primary': pred_primary,
            'pred_aux_target': pred_aux_target,
            'pred_aux_obs': pred_aux_obs,
        }
    
    def predict_peak(
        self,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict peak RAPM with uncertainty via Monte Carlo sampling.
        
        Per spec: "calibrated uncertainty"
        
        Returns:
            mean: Mean prediction [batch, 3]
            std: Standard deviation [batch, 3]
        """
        samples = []
        for _ in range(num_samples):
            out = self.forward(x, deterministic=False)
            samples.append(out['pred_primary'])
        
        stacked = torch.stack(samples, dim=0)  # [num_samples, batch, 3]
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)
        
        return mean, std
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get deterministic latent representation.
        
        Per spec: "Similarity/role emerges as posterior neighborhoods in z-space"
        """
        mu, _ = self.encoder(x)
        return mu


def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence from standard normal prior.
    
    For VAE training regularization.
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


if __name__ == '__main__':
    print("NBA Model Architecture module loaded.")
    print("Main model class: NBAProspectModel")
