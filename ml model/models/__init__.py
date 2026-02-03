"""
Models Package: Latent Space Architecture for Prospect Prediction
=================================================================

Key Components:
- PlayerEncoder: College features → Latent embedding (z)
- ProspectModel: Full model with multi-task heads
- ArchetypeAnalyzer: Post-training archetype discovery

Usage:
    from models import ProspectModel, ArchetypeAnalyzer
    
    model = ProspectModel(latent_dim=32, n_archetypes=8)
    # ... train model ...
    
    analyzer = ArchetypeAnalyzer(model, tier1_cols, tier2_cols, career_cols)
    analyzer.fit(features, targets)
    
    analysis = analyzer.analyze_player(prospect_features)
    print(analysis.narrative)
"""

from .player_encoder import (
    PlayerEncoder,
    FeatureBranchMLP,
    FeatureNormalizer,
    create_default_encoder,
    get_feature_dimensions,
    TIER1_COLUMNS,
    TIER2_COLUMNS,
    CAREER_BASE_COLUMNS,
    WITHIN_COLUMNS,
    CAREER_COLUMNS,  # compatibility alias = CAREER_BASE_COLUMNS + WITHIN_COLUMNS
)

from .prospect_model import (
    ProspectModel,
    ProspectLoss,
    RegressionHead,
    SurvivalHead,
    ArchetypeHead,
    UncertaintyHead,
    create_model,
)

from .archetype_analyzer import (
    ArchetypeAnalyzer,
    ArchetypeProfile,
    PlayerAnalysis,
)

__all__ = [
    # Encoder
    'PlayerEncoder',
    'FeatureBranchMLP',
    'FeatureNormalizer',
    'create_default_encoder',
    'get_feature_dimensions',
    'TIER1_COLUMNS',
    'TIER2_COLUMNS',
    'CAREER_BASE_COLUMNS',
    'WITHIN_COLUMNS',
    'CAREER_COLUMNS',
    # Model
    'ProspectModel',
    'ProspectLoss',
    'RegressionHead',
    'SurvivalHead',
    'ArchetypeHead',
    'UncertaintyHead',
    'create_model',
    # Analysis
    'ArchetypeAnalyzer',
    'ArchetypeProfile',
    'PlayerAnalysis',
]
