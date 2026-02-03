"""
Generative Latent Trait Model
=============================
Proper Bayesian model with ARD-discovered traits and horseshoe-shrunk interactions.

Key Components:
- core.py: Horseshoe prior, ARD utilities
- encoder.py: College features → latent traits with ARD
- impact_head.py: Traits → RAPM with main effects + all-pairs interactions
- full_model.py: Complete generative model
- decomposition.py: Trait contribution extraction
"""

from .core import horseshoe_prior, ard_prior, run_svi, group_shrinkage_prior
from .full_model import GenerativeProspectModel
from .pathway_model import HierarchicalPathwayModel

__all__ = [
    'horseshoe_prior',
    'ard_prior', 
    'run_svi',
    'group_shrinkage_prior',
    'GenerativeProspectModel',
    'HierarchicalPathwayModel',
]
