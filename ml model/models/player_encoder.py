"""
Player Encoder: College Features → Latent Embedding
====================================================
Maps college basketball features to a latent space where similar players cluster.

Architecture:
- Separate MLP branches for:
  - Tier 1 (final-season snapshot + context anchors)
  - Tier 2 (spatial; optional/masked)
  - Career (multi-season progression)
  - Within-season windows ("star run"; optional/masked)
- Exposure/mask-aware gated fusion (prevents tiny-sample windows from dominating)
- Optional VAE-style μ/σ for uncertainty

Usage:
    encoder = PlayerEncoder(tier1_dim=17, tier2_dim=9, career_dim=12, within_dim=8, latent_dim=32)
    z = encoder(tier1_features, tier2_features, career_features, within_features, tier2_mask, within_mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FeatureBranchMLP(nn.Module):
    """MLP for a single feature branch (Tier 1, Tier 2, or Career)."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlayerEncoder(nn.Module):
    """
    Encodes college features into a latent player embedding.
    
    The encoder has four branches:
    - Tier 1 (Universal): Always available, 2010-2025
    - Tier 2 (Spatial): Only 2019+, masked when unavailable
    - Career: Aggregated multi-season features
    - Within-season: Late-season windows / "star run" features, masked when unavailable
    
    Args:
        tier1_dim: Number of Tier 1 features
        tier2_dim: Number of Tier 2 features
        career_dim: Number of career features
        latent_dim: Dimension of latent space (default: 32)
        hidden_dim: Hidden layer dimension (default: 64)
        dropout: Dropout probability (default: 0.2)
        use_vae: If True, output μ and log(σ²) for VAE training
    """
    
    def __init__(
        self,
        tier1_dim: int,
        tier2_dim: int,
        career_dim: int,
        within_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        use_vae: bool = False,
    ):
        super().__init__()
        
        self.tier1_dim = tier1_dim
        self.tier2_dim = tier2_dim
        self.career_dim = career_dim
        self.within_dim = within_dim
        self.latent_dim = latent_dim
        self.use_vae = use_vae
        
        # Branch-specific encoders
        self.tier1_branch = FeatureBranchMLP(
            tier1_dim, hidden_dim, hidden_dim // 2, dropout
        )
        self.tier2_branch = FeatureBranchMLP(
            tier2_dim, hidden_dim // 2, hidden_dim // 4, dropout
        )
        self.career_branch = FeatureBranchMLP(
            career_dim, hidden_dim, hidden_dim // 2, dropout
        )
        self.within_branch = (
            FeatureBranchMLP(within_dim, hidden_dim // 2, hidden_dim // 4, dropout)
            if within_dim > 0
            else None
        )
        
        # Fusion layer
        # We fuse: tier1(32) + tier2(16) + career(32) + within(16) => hidden_dim
        fusion_input_dim = hidden_dim // 2 + hidden_dim // 4 + hidden_dim // 2 + hidden_dim // 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Output layer(s)
        if use_vae:
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        else:
            self.fc_out = nn.Linear(hidden_dim, latent_dim)
        
        # Learnable default for missing Tier 2
        self.tier2_default = nn.Parameter(torch.zeros(1, hidden_dim // 4))
        # Learnable default for missing within-season windows
        self.within_default = nn.Parameter(torch.zeros(1, hidden_dim // 4))

        # Gate nets (scalars in [0,1]) that modulate optional branches.
        # We keep these intentionally tiny so "gating" doesn't become a second model.
        # Inputs are simple reliability proxies derived from the same branch inputs.
        self.gate_tier2 = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
        self.gate_within = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
        self.gate_career = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
        # Freshman/early-career modulation for within-season learning signals.
        # Lets the model learn that late-season surges can carry different weight
        # for Year-1/Year-2 college players versus older players.
        self.gate_within_early_career = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
    
    def forward(
        self,
        tier1: torch.Tensor,
        tier2: torch.Tensor,
        career: torch.Tensor,
        within: torch.Tensor,
        tier2_mask: Optional[torch.Tensor] = None,
        within_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            tier1: [batch, tier1_dim] Tier 1 features
            tier2: [batch, tier2_dim] Tier 2 features (can be zeros if masked)
            career: [batch, career_dim] Career features
            tier2_mask: [batch, 1] Binary mask (1 = has spatial data, 0 = missing)
            within_mask: [batch, 1] Binary mask (1 = has within-season windows, 0 = missing)
        
        Returns:
            z: [batch, latent_dim] Latent embedding
            mu: [batch, latent_dim] Mean (if VAE)
            logvar: [batch, latent_dim] Log variance (if VAE)
        """
        batch_size = tier1.shape[0]
        
        # Encode each branch
        h_tier1 = self.tier1_branch(tier1)
        h_tier2 = self.tier2_branch(tier2)
        h_career = self.career_branch(career)
        if self.within_branch is not None and self.within_dim > 0:
            h_within = self.within_branch(within)
        else:
            h_within = self.within_default.expand(batch_size, -1)
        
        # --- Tier 2 mask: use learned default when data is missing
        if tier2_mask is not None:
            tier2_mask = tier2_mask.view(batch_size, 1)
            h_tier2 = tier2_mask * h_tier2 + (1 - tier2_mask) * self.tier2_default

        # --- Within-season mask: use learned default when data is missing
        if within_mask is not None and self.within_dim > 0:
            within_mask = within_mask.view(batch_size, 1)
            h_within = within_mask * h_within + (1 - within_mask) * self.within_default

        # --- Exposure / reliability-aware gates (prevents tiny-sample "star runs" from dominating)
        # Tier 2 gate: essentially a soft version of the mask.
        g_tier2 = torch.ones((batch_size, 1), device=tier1.device)
        if tier2_mask is not None:
            # Force exact 0 when missing (mask=0) to avoid any accidental influence.
            g_tier2 = tier2_mask * self.gate_tier2(tier2_mask)

        # Career gate: based on career_years (first feature in CAREER_BASE_COLUMNS by convention).
        # We map career_years -> (0,1) so 0/1-year careers don't overpower.
        # If career_years is missing, loaders should have imputed 0 and this gate will downweight.
        career_years = career[:, 0:1].clamp(min=0.0)
        g_career = self.gate_career(torch.log1p(career_years))

        # Within gate: depends on mask + within exposure (minutes in last10 if present).
        # Convention: WITHIN_COLUMNS includes `final_ws_minutes_last10` as the 2nd element.
        g_within = torch.zeros((batch_size, 1), device=tier1.device)
        if within_mask is not None and self.within_dim > 0:
            # within[:, 1] should be minutes_last10; if absent it will be 0.
            ws_minutes_last10 = within[:, 1:2].clamp(min=0.0)
            gate_in = torch.cat([within_mask, torch.log1p(ws_minutes_last10)], dim=-1)
            # Force exact 0 when missing (mask=0) to avoid any accidental influence.
            g_within = within_mask * self.gate_within(gate_in)
            # Modulate within-season gate by early-career status (from career_years).
            # Higher factor for low career_years (e.g., freshmen) helps capture
            # in-season development effects the user cares about.
            early_factor = self.gate_within_early_career(-torch.log1p(career_years))
            g_within = g_within * (0.5 + early_factor)

        # Apply gates to optional embeddings (tier1 stays un-gated)
        h_tier2 = g_tier2 * h_tier2
        h_career = g_career * h_career
        h_within = g_within * h_within
        
        # Concatenate and fuse
        h_concat = torch.cat([h_tier1, h_tier2, h_career, h_within], dim=-1)
        h_fused = self.fusion(h_concat)
        
        # Output
        if self.use_vae:
            mu = self.fc_mu(h_fused)
            logvar = self.fc_logvar(h_fused)
            # Reparameterization trick
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
            else:
                z = mu  # Use mean at inference
            return z, mu, logvar
        else:
            z = self.fc_out(h_fused)
            return z, None, None
    
    def encode(
        self,
        tier1: torch.Tensor,
        tier2: torch.Tensor,
        career: torch.Tensor,
        within: torch.Tensor,
        tier2_mask: Optional[torch.Tensor] = None,
        within_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience method that returns only the latent embedding."""
        z, _, _ = self.forward(tier1, tier2, career, within, tier2_mask, within_mask)
        return z


class FeatureNormalizer(nn.Module):
    """
    Learnable feature normalization layer.
    
    Can be initialized with precomputed mean/std, then fine-tuned.
    """
    
    def __init__(self, num_features: int):
        super().__init__()
        self.register_buffer('mean', torch.zeros(num_features))
        self.register_buffer('std', torch.ones(num_features))
        self.scale = nn.Parameter(torch.ones(num_features))
        self.shift = nn.Parameter(torch.zeros(num_features))
    
    def set_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Set normalization statistics from data."""
        self.mean.copy_(mean)
        self.std.copy_(std.clamp(min=1e-6))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.mean) / self.std
        return x_norm * self.scale + self.shift


# =============================================================================
# FEATURE COLUMN DEFINITIONS
# =============================================================================

TIER1_COLUMNS = [
    # Final-season shot profile (counts -> rates)
    'college_rim_fg_pct', 'college_mid_fg_pct', 'college_three_fg_pct', 'college_ft_pct',
    'college_rim_share', 'college_mid_share', 'college_three_share',

    # Volume / role / context
    'college_shots_total', 'college_fga_total', 'college_ft_att',
    'college_games_played', 'college_poss_proxy',
    'college_minutes_total',
    'college_team_pace', 'college_is_power_conf',
    'college_team_srs', 'team_strength_srs',
    'college_team_rank',

    # Activity / “impact-adjacent” box rates
    'college_ast_total_per100poss', 'college_tov_total_per100poss',
    'college_stl_total_per100poss', 'college_blk_total_per100poss',
    'college_orb_total_per100poss', 'college_drb_total_per100poss', 'college_trb_total_per100poss',
    'college_dunk_rate', 'college_dunk_freq', 'college_putback_rate',
    'college_rim_pressure_index', 'college_contest_proxy',
    'college_transition_freq', 'college_deflection_proxy', 'college_pressure_handle_proxy',
    'college_assisted_share_rim', 'college_assisted_share_mid', 'college_assisted_share_three',
    'college_rapm_standard', 'college_o_rapm', 'college_d_rapm',
    'college_on_net_rating', 'college_on_ortg', 'college_on_drtg',
    'high_lev_att_rate', 'garbage_att_rate', 'leverage_poss_share',

    # Era-normalized anchors (helps mitigate drift)
    'college_three_fg_pct_z',
    'final_trueShootingPct_z', 'final_usage_z',
    'college_rim_fg_pct_z', 'college_mid_fg_pct_z', 'college_ft_pct_z',

    # Team context adjustments (team-season residuals)
    'final_trueShootingPct_team_resid', 'final_usage_team_resid',
    'college_three_fg_pct_team_resid',

    # Recruiting priors (nullable; model learns when present vs missing)
    'college_recruiting_rank', 'college_recruiting_stars', 'college_recruiting_rating',
]

TIER2_COLUMNS = [
    'college_avg_shot_dist', 'college_shot_dist_var',
    'college_corner_3_rate', 'college_corner_3_pct',
    'college_deep_3_rate', 'college_rim_purity',
    'college_xy_shots', 'college_xy_3_shots', 'college_xy_rim_shots',
]

CAREER_BASE_COLUMNS = [
    'career_years',
    # Explicit year/era context (draft-time safe) so the model can learn inflation/deflation by season.
    'college_final_season', 'draft_year_proxy',
    'season_index', 'class_year', 'age_at_season', 'has_age_at_season',
    # Physical profile (draft-time safe where available).
    'college_height_in', 'college_weight_lbs', 'has_college_height', 'has_college_weight',
    # Final season anchors
    'final_trueShootingPct', 'final_usage', 'final_poss_total',
    'final_rim_fg_pct', 'final_three_fg_pct', 'final_ft_pct',

    # Progression
    'slope_trueShootingPct', 'slope_usage',
    'career_wt_trueShootingPct', 'career_wt_usage',
    'delta_trueShootingPct', 'delta_usage',
    'slope_rim_fg_pct', 'slope_three_fg_pct', 'slope_ft_pct',
    'career_wt_rim_fg_pct', 'career_wt_three_fg_pct', 'career_wt_ft_pct',
    'delta_rim_fg_pct', 'delta_three_fg_pct', 'delta_ft_pct',

    # Career-stage breakout timing (continuous)
    'breakout_timing_avg', 'breakout_timing_volume', 'breakout_timing_usage', 'breakout_timing_eff',
    # Baseline breakout rank features (high-coverage, cross-season comparable).
    'breakout_rank_eff', 'breakout_rank_volume', 'breakout_rank_usage',

    # College dev-rate summaries
    'college_dev_p10', 'college_dev_p50', 'college_dev_p90', 'college_dev_quality_weight',

    # Transfer context — summary level only (individual deltas have <15% coverage)
    'transfer_mean_shock', 'has_transfer_context',
    'transfer_event_count', 'transfer_max_shock',
    'transfer_conf_delta_mean', 'transfer_pace_delta_mean', 'transfer_role_delta_mean',
    # Explicit within-availability context for gating/readiness.
    'has_within_window_data',

    # Wingspan is currently schema-only (all-null upstream), so excluded from active encoder.
]

WITHIN_COLUMNS = [
    # Within-season trajectory signals (final season only), mask-gated in encoder.
    'final_has_ws_last10',
    'final_ws_minutes_last10',
    'final_ws_pps_last10',
    'final_ws_delta_pps_last5_minus_prev5',
    'final_has_ws_breakout_timing_eff',
    'final_ws_breakout_timing_eff',
]

# Backward-compatible alias: older code referred to a single CAREER_COLUMNS list.
# We keep it as the concatenation so imports don't break, but the encoder treats
# career vs within-season as separate branches.
CAREER_COLUMNS = CAREER_BASE_COLUMNS + WITHIN_COLUMNS


def get_feature_dimensions():
    """Return dimensions for each feature tier."""
    return {
        'tier1': len(TIER1_COLUMNS),
        'tier2': len(TIER2_COLUMNS),
        'career': len(CAREER_BASE_COLUMNS),
        'within': len(WITHIN_COLUMNS),
    }


def create_default_encoder(latent_dim: int = 32, use_vae: bool = False) -> PlayerEncoder:
    """Create encoder with default feature dimensions."""
    dims = get_feature_dimensions()
    return PlayerEncoder(
        tier1_dim=dims['tier1'],
        tier2_dim=dims['tier2'],
        career_dim=dims['career'],
        within_dim=dims['within'],
        latent_dim=latent_dim,
        use_vae=use_vae,
    )
