# Generative Prospect Model Report

**Generated**: 2026-02-18 13:39

## Model Configuration

| Parameter | Value |
|-----------|-------|
| K_max (latent dims) | 8 |
| ARD scale | 1.0 |
| Tau main | 1.0 |
| Tau interaction | 0.3 |
| SVI steps | 10 |

## Results

| Metric | Value |
|--------|-------|
| Effective dimensions | 8 / 8 |
| Active interactions | 15 / 28 |
| Final ELBO loss | 1900641.7500 |


## Prediction Performance

| Metric | Value |
|--------|-------|
| RMSE | 13.9476 |
| Correlation | -0.0860 |


## ARD Scales (Trait Importance)

| Trait | Scale (mean ± std) | Active |
|-------|-------------------|--------|
| 0 | 1.089 ± 0.113 | ✓ |
| 1 | 0.673 ± 0.068 | ✓ |
| 2 | 1.048 ± 0.104 | ✓ |
| 3 | 1.332 ± 0.132 | ✓ |
| 4 | 0.698 ± 0.065 | ✓ |
| 5 | 1.201 ± 0.119 | ✓ |
| 6 | 1.292 ± 0.137 | ✓ |
| 7 | 1.377 ± 0.143 | ✓ |

## Example Player Decompositions


### Player 0

**Predicted RAPM**: -5.87 ± 20.88

**Contributions**:
- Intercept: -0.260
- Main effects: 0.214
- Interactions: -5.239

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 2 | -6.02 | 2.233 |
| 3 | 9.48 | -1.972 |
| 7 | -12.12 | 1.459 |
| 5 | -2.65 | -0.736 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (6, 7) | 8.213 |
| (2, 7) | -7.385 |
| (5, 7) | -6.942 |
| (2, 3) | -4.102 |
| (2, 5) | 2.647 |

### Player 278

**Predicted RAPM**: 0.92 ± 1.61

**Contributions**:
- Intercept: -0.260
- Main effects: 1.196
- Interactions: 0.145

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 5 | 2.51 | 0.695 |
| 3 | -2.75 | 0.572 |
| 0 | 1.63 | -0.122 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (5, 7) | -0.464 |
| (3, 5) | 0.433 |
| (0, 5) | 0.375 |
| (4, 5) | -0.309 |
| (0, 4) | 0.255 |

### Player 557

**Predicted RAPM**: 1.44 ± 1.56

**Contributions**:
- Intercept: -0.260
- Main effects: -0.147
- Interactions: 1.829

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 2 | 2.42 | -0.898 |
| 5 | 1.11 | 0.309 |
| 7 | -2.46 | 0.296 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (2, 7) | 0.602 |
| (5, 7) | 0.591 |
| (6, 7) | 0.512 |
| (2, 5) | 0.447 |
| (2, 6) | -0.175 |
