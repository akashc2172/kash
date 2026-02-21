# Generative Prospect Model Report

**Generated**: 2026-02-18 13:40

## Model Configuration

| Parameter | Value |
|-----------|-------|
| K_max (latent dims) | 32 |
| ARD scale | 1.0 |
| Tau main | 1.0 |
| Tau interaction | 0.3 |
| SVI steps | 5 |

## Results

| Metric | Value |
|--------|-------|
| Effective dimensions | 32 / 32 |
| Active interactions | 231 / 496 |
| Final ELBO loss | 19082448.0000 |


## Prediction Performance

| Metric | Value |
|--------|-------|
| RMSE | 40.0106 |
| Correlation | -0.0505 |


## ARD Scales (Trait Importance)

| Trait | Scale (mean ± std) | Active |
|-------|-------------------|--------|
| 0 | 0.883 ± 0.089 | ✓ |
| 1 | 0.674 ± 0.069 | ✓ |
| 2 | 1.070 ± 0.107 | ✓ |
| 3 | 1.890 ± 0.192 | ✓ |
| 4 | 1.002 ± 0.095 | ✓ |
| 5 | 0.760 ± 0.077 | ✓ |
| 6 | 0.434 ± 0.044 | ✓ |
| 7 | 0.648 ± 0.066 | ✓ |
| 8 | 0.473 ± 0.048 | ✓ |
| 9 | 1.061 ± 0.105 | ✓ |
| 10 | 0.711 ± 0.071 | ✓ |
| 11 | 0.813 ± 0.077 | ✓ |
| 12 | 0.803 ± 0.081 | ✓ |
| 13 | 1.005 ± 0.104 | ✓ |
| 14 | 1.350 ± 0.133 | ✓ |
| 15 | 0.767 ± 0.078 | ✓ |
| 16 | 1.220 ± 0.118 | ✓ |
| 17 | 1.016 ± 0.100 | ✓ |
| 18 | 1.916 ± 0.192 | ✓ |
| 19 | 1.236 ± 0.119 | ✓ |
| 20 | 0.669 ± 0.069 | ✓ |
| 21 | 1.115 ± 0.113 | ✓ |
| 22 | 0.669 ± 0.061 | ✓ |
| 23 | 1.313 ± 0.134 | ✓ |
| 24 | 0.933 ± 0.094 | ✓ |
| 25 | 0.930 ± 0.094 | ✓ |
| 26 | 0.744 ± 0.074 | ✓ |
| 27 | 2.083 ± 0.201 | ✓ |
| 28 | 1.280 ± 0.124 | ✓ |
| 29 | 1.038 ± 0.105 | ✓ |
| 30 | 0.848 ± 0.089 | ✓ |
| 31 | 1.594 ± 0.158 | ✓ |

## Example Player Decompositions


### Player 0

**Predicted RAPM**: -3.01 ± 32.30

**Contributions**:
- Intercept: -0.309
- Main effects: 5.636
- Interactions: -9.897

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 31 | -6.76 | -2.470 |
| 4 | 4.84 | 1.919 |
| 9 | 4.13 | 1.780 |
| 13 | -4.56 | -1.426 |
| 18 | -10.27 | 1.175 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (18, 30) | -8.516 |
| (2, 31) | -5.658 |
| (18, 27) | -5.657 |
| (4, 18) | -5.626 |
| (9, 31) | 3.942 |

### Player 278

**Predicted RAPM**: 4.45 ± 8.71

**Contributions**:
- Intercept: -0.309
- Main effects: 0.245
- Interactions: 4.734

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 19 | -2.52 | -1.557 |
| 2 | 3.09 | -0.752 |
| 5 | 3.81 | 0.712 |
| 14 | -2.41 | 0.637 |
| 12 | 1.90 | 0.465 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (2, 5) | 1.338 |
| (5, 18) | 1.296 |
| (0, 1) | -1.294 |
| (3, 7) | 1.102 |
| (3, 21) | -1.048 |

### Player 557

**Predicted RAPM**: 2.41 ± 9.06

**Contributions**:
- Intercept: -0.309
- Main effects: 1.667
- Interactions: 0.582

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 19 | 3.08 | 1.903 |
| 18 | 8.02 | -0.917 |
| 31 | -2.35 | -0.860 |
| 4 | 1.76 | 0.697 |
| 14 | -2.10 | 0.555 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (18, 21) | 2.247 |
| (18, 30) | 2.131 |
| (18, 19) | -1.953 |
| (19, 24) | -1.853 |
| (4, 18) | 1.594 |
