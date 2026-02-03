"""
Archetype Analyzer: Interpret and Visualize Player Archetypes
==============================================================
Post-training analysis of the latent space to discover and name archetypes.

Features:
- Cluster analysis of latent embeddings
- Archetype naming via feature profiles
- Player similarity search
- Narrative generation for prospects

Usage:
    analyzer = ArchetypeAnalyzer(model, feature_names)
    analyzer.fit(training_data)
    
    # Get archetype for a player
    archetype = analyzer.get_archetype(player_features)
    
    # Find similar players
    similar = analyzer.find_similar(player_features, k=5)
    
    # Generate narrative
    narrative = analyzer.generate_narrative(player_features)
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

try:
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class ArchetypeProfile:
    """Profile of a discovered archetype."""
    id: int
    name: str
    n_players: int
    avg_rapm: float
    survival_rate: float
    defining_features: Dict[str, float]  # Feature name -> z-score vs population
    example_players: List[str]
    description: str


@dataclass
class PlayerAnalysis:
    """Analysis result for a single player."""
    archetype_id: int
    archetype_name: str
    archetype_confidence: float
    rapm_prediction: float
    rapm_uncertainty: float
    survival_probability: float
    similar_players: List[Tuple[str, float]]  # (name, similarity)
    strengths: List[str]
    weaknesses: List[str]
    narrative: str


class ArchetypeAnalyzer:
    """
    Analyzes the latent space to discover and interpret archetypes.
    
    Works with a trained ProspectModel to:
    1. Extract latent embeddings for all players
    2. Cluster embeddings to discover archetypes
    3. Profile each archetype by feature importance
    4. Generate narratives for new prospects
    """
    
    # Default archetype name templates based on feature profiles
    ARCHETYPE_TEMPLATES = {
        'rim_runner': {
            'triggers': {'college_rim_fg_pct': 0.5, 'college_rim_share': 0.3, 'final_usage': -0.3},
            'name': 'Rim Runner',
            'description': 'High-efficiency finisher who scores primarily at the rim with limited shot creation',
        },
        'three_and_d': {
            'triggers': {'college_three_fg_pct': 0.3, 'final_usage': -0.2, 'college_on_net_rating': 0.2},
            'name': '3-and-D Wing',
            'description': 'Perimeter shooter with defensive impact and low usage role',
        },
        'shot_creator': {
            'triggers': {'final_usage': 0.5, 'college_assisted_share_three': -0.3},
            'name': 'Shot Creator',
            'description': 'High-usage player who creates their own offense',
        },
        'stretch_big': {
            'triggers': {'college_three_share': 0.3, 'college_rim_share': 0.2},
            'name': 'Stretch Big',
            'description': 'Inside-outside scorer who spaces the floor',
        },
        'point_of_attack': {
            'triggers': {'final_usage': 0.4, 'slope_usage': 0.2},
            'name': 'Point of Attack',
            'description': 'Ball-dominant playmaker with increasing responsibility',
        },
        'raw_athletic': {
            'triggers': {'college_rim_fg_pct': 0.3, 'college_three_fg_pct': -0.3, 'final_trueShootingPct': -0.2},
            'name': 'Raw Athletic',
            'description': 'Athletic player with underdeveloped shooting skills',
        },
        'floor_general': {
            'triggers': {'college_assisted_share_rim': -0.3, 'college_assisted_share_three': -0.3},
            'name': 'Floor General',
            'description': 'Playmaker who creates for others more than self',
        },
        'two_way_wing': {
            'triggers': {'college_on_net_rating': 0.4, 'final_trueShootingPct': 0.2},
            'name': 'Two-Way Wing',
            'description': 'Versatile wing with positive impact on both ends',
        },
    }
    
    def __init__(
        self,
        model,
        tier1_columns: List[str],
        tier2_columns: List[str],
        career_columns: List[str],
        within_columns: Optional[List[str]] = None,
        n_archetypes: int = 8,
    ):
        """
        Args:
            model: Trained ProspectModel
            tier1_columns: Names of Tier 1 features
            tier2_columns: Names of Tier 2 features
            career_columns: Names of career (multi-season) features
            within_columns: Names of within-season window features (optional)
            n_archetypes: Number of archetypes to discover
        """
        self.model = model
        self.tier1_columns = tier1_columns
        self.tier2_columns = tier2_columns
        self.career_columns = career_columns
        self.within_columns = within_columns or []
        self.all_columns = tier1_columns + tier2_columns + career_columns + self.within_columns
        self.n_archetypes = n_archetypes
        
        # Fitted state
        self.embeddings: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.archetype_profiles: Dict[int, ArchetypeProfile] = {}
        self.player_ids: Optional[List] = None
        self.player_names: Optional[List[str]] = None
        self.feature_stats: Optional[Dict] = None
    
    def fit(
        self,
        tier1_features: torch.Tensor,
        tier2_features: torch.Tensor,
        career_features: torch.Tensor,
        tier2_mask: torch.Tensor,
        within_features: Optional[torch.Tensor] = None,
        within_mask: Optional[torch.Tensor] = None,
        player_ids: Optional[List] = None,
        player_names: Optional[List[str]] = None,
        targets: Optional[Dict[str, np.ndarray]] = None,
    ) -> 'ArchetypeAnalyzer':
        """
        Fit the analyzer on training data.
        
        Extracts embeddings, clusters them, and profiles each archetype.
        """
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for archetype analysis")
        
        self.model.eval()
        
        # Extract embeddings
        with torch.no_grad():
            if within_features is None:
                # Backward compatibility (older checkpoints)
                within_features = torch.zeros((tier1_features.shape[0], 0), device=tier1_features.device)
            self.embeddings = self.model.encode(
                tier1_features, tier2_features, career_features, within_features, tier2_mask, within_mask
            ).cpu().numpy()
        
        # Store metadata
        self.player_ids = player_ids
        self.player_names = player_names or [f"Player_{i}" for i in range(len(self.embeddings))]
        
        # Compute feature statistics for profiling
        if within_features is None:
            all_features = torch.cat([tier1_features, tier2_features, career_features], dim=-1)
        else:
            all_features = torch.cat([tier1_features, tier2_features, career_features, within_features], dim=-1)
        self.feature_stats = {
            'mean': all_features.mean(dim=0).cpu().numpy(),
            'std': all_features.std(dim=0).cpu().numpy(),
        }
        self._all_features_np = all_features.cpu().numpy()
        
        # Cluster embeddings
        kmeans = KMeans(n_clusters=self.n_archetypes, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        self.cluster_centers = kmeans.cluster_centers_
        
        # Profile each archetype
        self._profile_archetypes(targets)
        
        return self
    
    def _profile_archetypes(self, targets: Optional[Dict[str, np.ndarray]] = None):
        """Create detailed profiles for each archetype."""
        for k in range(self.n_archetypes):
            mask = self.cluster_labels == k
            n_players = mask.sum()
            
            if n_players == 0:
                continue
            
            # Feature profile (z-scores vs population)
            cluster_features = self._all_features_np[mask]
            feature_zscores = {}
            for i, col in enumerate(self.all_columns):
                if self.feature_stats['std'][i] > 0:
                    cluster_mean = cluster_features[:, i].mean()
                    pop_mean = self.feature_stats['mean'][i]
                    pop_std = self.feature_stats['std'][i]
                    feature_zscores[col] = (cluster_mean - pop_mean) / pop_std
            
            # Sort by absolute z-score to find defining features
            sorted_features = sorted(
                feature_zscores.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            defining_features = dict(sorted_features[:5])
            
            # Outcomes
            avg_rapm = 0.0
            survival_rate = 0.0
            if targets is not None:
                if 'rapm' in targets:
                    rapm_vals = targets['rapm'][mask]
                    valid = ~np.isnan(rapm_vals)
                    if valid.any():
                        avg_rapm = rapm_vals[valid].mean()
                if 'survival' in targets:
                    surv_vals = targets['survival'][mask]
                    valid = ~np.isnan(surv_vals)
                    if valid.any():
                        survival_rate = surv_vals[valid].mean()
            
            # Example players
            cluster_player_names = [
                self.player_names[i] for i in range(len(self.player_names)) if mask[i]
            ][:5]
            
            # Auto-name archetype
            name, description = self._auto_name_archetype(defining_features)
            
            self.archetype_profiles[k] = ArchetypeProfile(
                id=k,
                name=name,
                n_players=int(n_players),
                avg_rapm=float(avg_rapm),
                survival_rate=float(survival_rate),
                defining_features=defining_features,
                example_players=cluster_player_names,
                description=description,
            )
    
    def _auto_name_archetype(
        self,
        defining_features: Dict[str, float],
    ) -> Tuple[str, str]:
        """Auto-generate archetype name based on feature profile."""
        best_match = None
        best_score = -float('inf')
        
        for template_key, template in self.ARCHETYPE_TEMPLATES.items():
            score = 0
            for feature, threshold in template['triggers'].items():
                if feature in defining_features:
                    if (threshold > 0 and defining_features[feature] > threshold) or \
                       (threshold < 0 and defining_features[feature] < threshold):
                        score += 1
            
            if score > best_score:
                best_score = score
                best_match = template_key
        
        if best_match and best_score >= 2:
            template = self.ARCHETYPE_TEMPLATES[best_match]
            return template['name'], template['description']
        
        # Fallback: name by top feature
        if defining_features:
            top_feature = list(defining_features.keys())[0]
            direction = "High" if list(defining_features.values())[0] > 0 else "Low"
            clean_name = top_feature.replace('college_', '').replace('final_', '').replace('_', ' ').title()
            return f"{direction} {clean_name} Type", f"Characterized by {direction.lower()} {clean_name}"
        
        return "Unknown Type", "Undefined archetype"
    
    def get_archetype(
        self,
        tier1: torch.Tensor,
        tier2: torch.Tensor,
        career: torch.Tensor,
        tier2_mask: Optional[torch.Tensor] = None,
        within: Optional[torch.Tensor] = None,
        within_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[int, str, float]:
        """
        Get archetype assignment for a player.
        
        Returns: (archetype_id, archetype_name, confidence)
        """
        self.model.eval()
        with torch.no_grad():
            if within is None:
                within = torch.zeros((tier1.shape[0], 0), device=tier1.device)
            z = self.model.encode(tier1, tier2, career, within, tier2_mask, within_mask).cpu().numpy()
        
        # Find nearest cluster center
        distances = np.linalg.norm(self.cluster_centers - z, axis=1)
        nearest = distances.argmin()
        
        # Confidence based on distance ratio
        sorted_dists = np.sort(distances)
        confidence = 1 - (sorted_dists[0] / (sorted_dists[1] + 1e-6))
        confidence = max(0, min(1, confidence))
        
        profile = self.archetype_profiles.get(nearest)
        name = profile.name if profile else f"Archetype {nearest}"
        
        return nearest, name, confidence
    
    def find_similar(
        self,
        tier1: torch.Tensor,
        tier2: torch.Tensor,
        career: torch.Tensor,
        tier2_mask: Optional[torch.Tensor] = None,
        within: Optional[torch.Tensor] = None,
        within_mask: Optional[torch.Tensor] = None,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Find k most similar players in the training set."""
        self.model.eval()
        with torch.no_grad():
            if within is None:
                within = torch.zeros((tier1.shape[0], 0), device=tier1.device)
            z = self.model.encode(tier1, tier2, career, within, tier2_mask, within_mask).cpu().numpy()
        
        # Compute distances to all training embeddings
        distances = np.linalg.norm(self.embeddings - z, axis=1)
        nearest_idx = np.argsort(distances)[:k]
        
        results = []
        for idx in nearest_idx:
            name = self.player_names[idx]
            similarity = 1 / (1 + distances[idx])  # Convert distance to similarity
            results.append((name, float(similarity)))
        
        return results
    
    def analyze_player(
        self,
        tier1: torch.Tensor,
        tier2: torch.Tensor,
        career: torch.Tensor,
        tier2_mask: Optional[torch.Tensor] = None,
        within: Optional[torch.Tensor] = None,
        within_mask: Optional[torch.Tensor] = None,
        player_name: str = "Prospect",
    ) -> PlayerAnalysis:
        """Full analysis for a single player."""
        self.model.eval()
        
        # Get archetype
        arch_id, arch_name, arch_conf = self.get_archetype(tier1, tier2, career, tier2_mask, within, within_mask)

        # Get full archetype distribution (soft assignments)
        arch_dist = self.get_archetype_distribution(
            tier1, tier2, career, tier2_mask, within, within_mask, cumulative=0.85, max_k=3
        )
        
        # Get predictions
        with torch.no_grad():
            if within is None:
                within = torch.zeros((tier1.shape[0], 0), device=tier1.device)
            outputs = self.model(tier1, tier2, career, within, tier2_mask, within_mask)
        
        rapm_pred = outputs['rapm_pred'][0, 0].item()
        rapm_var = outputs.get('rapm_var', torch.tensor([[0.5]]))[0, 0].item()
        survival_prob = torch.sigmoid(outputs['survival_logits'][0, 0]).item()
        
        # Find similar players
        similar = self.find_similar(tier1, tier2, career, tier2_mask, within, within_mask, k=5)
        
        # Analyze strengths/weaknesses
        if within is None:
            all_features = torch.cat([tier1, tier2, career], dim=-1).cpu().numpy()[0]
        else:
            all_features = torch.cat([tier1, tier2, career, within], dim=-1).cpu().numpy()[0]
        strengths, weaknesses = self._analyze_features(all_features)
        
        # Generate narrative
        narrative = self._generate_narrative(
            player_name, arch_name, arch_conf, rapm_pred, rapm_var,
            survival_prob, similar, strengths, weaknesses, arch_dist
        )
        
        return PlayerAnalysis(
            archetype_id=arch_id,
            archetype_name=arch_name,
            archetype_confidence=arch_conf,
            rapm_prediction=rapm_pred,
            rapm_uncertainty=np.sqrt(rapm_var),
            survival_probability=survival_prob,
            similar_players=similar,
            strengths=strengths,
            weaknesses=weaknesses,
            narrative=narrative,
        )

    @torch.no_grad()
    def get_archetype_distribution(
        self,
        tier1: torch.Tensor,
        tier2: torch.Tensor,
        career: torch.Tensor,
        tier2_mask: Optional[torch.Tensor] = None,
        within: Optional[torch.Tensor] = None,
        within_mask: Optional[torch.Tensor] = None,
        cumulative: float = 0.85,
        max_k: int = 3,
    ) -> List[Tuple[int, str, float]]:
        """
        Return the top archetype candidates up to a cumulative probability threshold.

        This prevents \"over-matching\" to a single cluster when the player is plausibly
        between multiple archetypes.
        """
        if within is None:
            within = torch.zeros((tier1.shape[0], 0), device=tier1.device)
        outputs = self.model(tier1, tier2, career, within, tier2_mask, within_mask)
        probs = outputs['archetype_probs'][0].detach().cpu().numpy().astype(float)

        order = np.argsort(-probs)
        picked = []
        total = 0.0
        for idx in order[: max_k * 2]:
            p = float(probs[idx])
            picked.append((int(idx), self.archetype_names[int(idx)], p))
            total += p
            if len(picked) >= max_k or total >= cumulative:
                break
        return picked
    
    def _analyze_features(
        self,
        features: np.ndarray,
    ) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses based on feature z-scores."""
        strengths = []
        weaknesses = []
        
        for i, col in enumerate(self.all_columns):
            if self.feature_stats['std'][i] < 1e-6:
                continue
            
            z = (features[i] - self.feature_stats['mean'][i]) / self.feature_stats['std'][i]
            clean_name = col.replace('college_', '').replace('final_', '').replace('_', ' ').title()
            
            if z > 1.0:
                strengths.append(f"Elite {clean_name}")
            elif z > 0.5:
                strengths.append(f"Strong {clean_name}")
            elif z < -1.0:
                weaknesses.append(f"Poor {clean_name}")
            elif z < -0.5:
                weaknesses.append(f"Below average {clean_name}")
        
        return strengths[:3], weaknesses[:3]
    
    def _generate_narrative(
        self,
        player_name: str,
        archetype: str,
        confidence: float,
        rapm: float,
        rapm_std: float,
        survival: float,
        similar: List[Tuple[str, float]],
        strengths: List[str],
        weaknesses: List[str],
        archetype_candidates: Optional[List[Tuple[int, str, float]]] = None,
    ) -> str:
        """Generate human-readable narrative."""
        comp_str = ", ".join([f"{name} ({sim:.0%})" for name, sim in similar[:3]])
        strength_str = ", ".join(strengths) if strengths else "No standout strengths"
        weakness_str = ", ".join(weaknesses) if weaknesses else "No major weaknesses"
        
        rapm_desc = "All-Star caliber" if rapm > 2 else "solid starter" if rapm > 0 else "role player" if rapm > -1 else "fringe NBA"

        cand_str = ""
        if archetype_candidates:
            # Show the top-2/3 archetypes and how close they are.
            parts = [f"{name} ({p:.0%})" for _, name, p in archetype_candidates]
            cand_str = "\n**Archetype Mix**: " + ", ".join(parts) + "\n"
        
        return f"""
**{player_name}** projects as a **{archetype}** ({confidence:.0%} confidence).

{cand_str}
**Projection**: {rapm:.2f} ± {rapm_std:.2f} peak RAPM ({rapm_desc})
**NBA Probability**: {survival:.0%}

**Comparisons**: {comp_str}

**Strengths**: {strength_str}
**Weaknesses**: {weakness_str}
""".strip()
    
    def get_archetype_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all archetypes."""
        rows = []
        for k, profile in self.archetype_profiles.items():
            rows.append({
                'archetype_id': profile.id,
                'name': profile.name,
                'n_players': profile.n_players,
                'avg_rapm': profile.avg_rapm,
                'survival_rate': profile.survival_rate,
                'description': profile.description,
                'examples': ', '.join(profile.example_players[:3]),
            })
        return pd.DataFrame(rows)
    
    def get_tsne_embeddings(self, perplexity: int = 30) -> np.ndarray:
        """Get 2D t-SNE projection of embeddings for visualization."""
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for t-SNE")
        
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        return tsne.fit_transform(self.embeddings)
