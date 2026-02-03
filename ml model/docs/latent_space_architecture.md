# Latent Space Architecture: Player Archetypes

**Version**: 1.0  
**Date**: 2026-02-01  
**Status**: Design + Implementation

---

## Why Latent Space?

XGBoost tells you **what** predicts success. Latent space tells you **why** by discovering:

| Approach | Answers | Limitation |
|----------|---------|------------|
| XGBoost | "3PT% matters" | No archetypes, no interactions |
| Latent Space | "This player is a rim-runner type, which succeeds via a different pathway than shot creators" | Requires more data |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         COLLEGE FEATURES (X)                             │
│   Tier 1: rim_fg_pct, three_share, usage, on_net_rating, ...            │
│   Tier 2: avg_shot_dist, corner_3_rate, rim_purity (masked)             │
│   Career: slope_ts, career_years, delta_usage                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FEATURE ENCODER                                  │
│                                                                          │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│   │  Tier1 MLP  │    │  Tier2 MLP  │    │ Career MLP  │                 │
│   │  (always)   │    │  (masked)   │    │  (always)   │                 │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                 │
│          │                  │                  │                         │
│          └──────────────────┼──────────────────┘                         │
│                             │                                            │
│                             ▼                                            │
│                    ┌─────────────────┐                                   │
│                    │  Fusion Layer   │                                   │
│                    │  (concat + MLP) │                                   │
│                    └────────┬────────┘                                   │
│                             │                                            │
│                             ▼                                            │
│                    ┌─────────────────┐                                   │
│                    │   μ, log(σ²)    │  ← VAE-style (optional)           │
│                    └────────┬────────┘                                   │
│                             │                                            │
│                             ▼                                            │
│                    ┌─────────────────┐                                   │
│                    │    z ∈ R^32     │  ← PLAYER EMBEDDING               │
│                    │  (latent space) │                                   │
│                    └─────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
┌───────────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐
│    RAPM DECODER       │ │ SURVIVAL DECODER│ │   ARCHETYPE HEAD        │
│                       │ │                 │ │                         │
│  z → peak_rapm        │ │  z → P(made_nba)│ │  z → archetype logits   │
│  z → gap_ts           │ │                 │ │                         │
│  z → year1_epm        │ │  Binary CE loss │ │  Soft clustering or     │
│                       │ │                 │ │  learned prototypes     │
│  Regression losses    │ │                 │ │                         │
└───────────────────────┘ └─────────────────┘ └─────────────────────────┘
```

---

## Key Design Decisions

### 1. Latent Dimension: 32

Why 32?
- **Too small (8)**: Can't capture enough variation
- **Too large (128)**: Overfits, archetypes become meaningless
- **32**: Sweet spot for ~1000 players, ~60 features

### 2. Tier 2 Masking Strategy

```python
# During training, randomly mask Tier 2 features (even when available)
# This teaches the model to work without spatial data
if has_spatial_data and random() < 0.3:
    tier2_features = zeros  # Dropout
    tier2_mask = 0
```

### 3. Archetype Discovery

Two approaches:

**A. Post-hoc Clustering (Simple)**
- Train encoder normally
- Apply K-Means to latent space
- Interpret clusters manually

**B. Prototype Learning (Better)**
- Learn K prototype vectors `p_1, ..., p_K` alongside encoder
- Soft assignment: `a_k = softmax(-||z - p_k||²)`
- Loss encourages players to be close to one prototype

### 4. Loss Function

```
L = λ_rapm * MSE(peak_rapm)           # Primary
  + λ_ts   * MSE(gap_ts)              # Translation signal
  + λ_epm  * MSE(year1_epm)           # Early signal
  + λ_surv * BCE(made_nba)            # Survival
  + λ_kl   * KL(q(z|x) || p(z))       # VAE regularization (optional)
  + λ_arch * ArchetypeLoss            # Prototype clustering
```

---

## Multi-Season Career Handling (Update)

**Problem**: College careers vary wildly (one-and-done vs 4-year vs transfers), so the latent input must treat season count as *context*, not as *quality*.

**Approach**: Encode each season, then pool into a career embedding with exposure-aware weighting. Fuse the final-season embedding with the pooled career embedding so late bloomers (e.g., 4th-year jumps) are not diluted by earlier seasons.

**Implementation Plan**: See `docs/latent_input_plan.md`.

**Key Ideas**:
- Per-season encoder → sequence pooling (attention or GRU/Transformer)
- Reliability weighting by minutes/games
- Gated fusion with final season embedding
- Transfer/role change flags as inputs
- Continuous breakout timing features (volume/usage/efficiency), learned in an archetype-conditioned way
- Optional wiring: condition outcome heads on `archetype_probs` (`condition_on_archetypes`) so sensitivity can differ by archetype

---

## Archetype Interpretations

Once trained, we can extract narratives:

### Discovering Archetypes

```python
# Get all player embeddings
Z = encoder(all_college_features)  # [N, 32]

# Cluster
kmeans = KMeans(n_clusters=8)
archetypes = kmeans.fit_predict(Z)

# Interpret by examining cluster centers
for k in range(8):
    players_in_k = players[archetypes == k]
    print(f"Archetype {k}: {players_in_k[:5]}")
    print(f"  Avg 3PT%: {players_in_k.three_fg_pct.mean():.3f}")
    print(f"  Avg Usage: {players_in_k.usage.mean():.3f}")
    print(f"  Avg RAPM: {players_in_k.peak_rapm.mean():.2f}")
```

### Example Archetypes (Hypothetical)

| Archetype | Profile | NBA Outcome | Examples |
|-----------|---------|-------------|----------|
| **Rim Runner** | High rim%, low usage, high ast'd% | +1.5 RAPM, 90% make it | Clint Capela, DeAndre Jordan |
| **3-and-D Wing** | High 3PT%, low usage, good +/- | +0.8 RAPM, 85% make it | Mikal Bridges, OG Anunoby |
| **Shot Creator** | High usage, mid TS%, ball handling | +0.5 RAPM, 70% make it | Varies widely |
| **Stretch Big** | High 3PT% for position, rim protection | +1.0 RAPM, 75% make it | Brook Lopez, Al Horford |
| **Raw Athletic** | High dunk%, low skill stats | -0.5 RAPM, 50% make it | High bust rate |

### Narrative Generation

```python
def describe_player(player_id):
    z = encoder(player_features)
    
    # Find nearest archetype
    distances = [||z - prototype_k|| for k in archetypes]
    nearest = argmin(distances)
    
    # Find similar NBA players
    similar = find_nearest_neighbors(z, nba_player_embeddings, k=3)
    
    return f"""
    Archetype: {archetype_names[nearest]} (confidence: {1 - distances[nearest]:.1%})
    Comp: {similar[0].name} ({similar[0].similarity:.1%} match)
    
    Projection:
    - Peak RAPM: {rapm_decoder(z):.2f} ± {rapm_uncertainty(z):.2f}
    - P(Make NBA): {survival_decoder(z):.1%}
    
    Key traits driving projection:
    - {explain_latent_dimension(z, dim=0)}
    - {explain_latent_dimension(z, dim=1)}
    """
```

---

## Implementation Plan

1. **`models/player_encoder.py`**: Feature → Latent encoder
2. **`models/prospect_model.py`**: Full model with all heads
3. **`models/archetype_module.py`**: Prototype learning + interpretation
4. **`train_latent_model.py`**: Training loop with compound loss
5. **`analyze_archetypes.py`**: Post-training archetype discovery

---

## Comparison: XGBoost vs Latent

| Aspect | XGBoost Baseline | Latent Space Model |
|--------|------------------|-------------------|
| **Prediction accuracy** | Baseline | Similar or better |
| **Feature importance** | ✅ Global + SHAP | ✅ Via decoder weights |
| **Archetypes** | ❌ | ✅ Learned clusters |
| **Player similarity** | ❌ | ✅ Distance in latent space |
| **"What-if" analysis** | ❌ | ✅ Interpolate in latent space |
| **Narrative generation** | ❌ | ✅ Decode + interpret |
| **Training complexity** | Low | Medium |
| **Data requirements** | ~500 players | ~800+ players |

---

## Next Steps

1. Implement `PlayerEncoder` with Tier 1/2/Career branches
2. Implement multi-head decoders
3. Train on mock data to validate architecture
4. Add archetype prototype learning
5. Build interpretation tools
