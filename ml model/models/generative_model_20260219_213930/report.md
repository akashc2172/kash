# Generative Prospect Model Report

**Generated**: 2026-02-19 21:39

## Model Configuration

| Parameter | Value |
|-----------|-------|
| K_max (latent dims) | 32 |
| ARD scale | 1.0 |
| Tau main | 1.0 |
| Tau interaction | 0.3 |
| SVI steps | 50 |

## Results

| Metric | Value |
|--------|-------|
| Effective dimensions | 32 / 32 |
| Active interactions | 199 / 496 |
| Final ELBO loss | 287253.6875 |


## Prediction Performance

| Metric | Value |
|--------|-------|
| RMSE | 6.2089 |
| Correlation | 0.0062 |


## ARD Scales (Trait Importance)

| Trait | Scale (mean ± std) | Active |
|-------|-------------------|--------|
| 0 | 0.760 ± 0.073 | ✓ |
| 1 | 0.599 ± 0.059 | ✓ |
| 2 | 0.882 ± 0.085 | ✓ |
| 3 | 1.469 ± 0.138 | ✓ |
| 4 | 0.796 ± 0.075 | ✓ |
| 5 | 0.629 ± 0.065 | ✓ |
| 6 | 0.359 ± 0.037 | ✓ |
| 7 | 0.546 ± 0.053 | ✓ |
| 8 | 0.416 ± 0.042 | ✓ |
| 9 | 0.863 ± 0.078 | ✓ |
| 10 | 0.562 ± 0.055 | ✓ |
| 11 | 0.630 ± 0.059 | ✓ |
| 12 | 0.674 ± 0.063 | ✓ |
| 13 | 0.811 ± 0.080 | ✓ |
| 14 | 1.067 ± 0.107 | ✓ |
| 15 | 0.598 ± 0.057 | ✓ |
| 16 | 0.992 ± 0.086 | ✓ |
| 17 | 0.804 ± 0.079 | ✓ |
| 18 | 1.528 ± 0.140 | ✓ |
| 19 | 0.985 ± 0.087 | ✓ |
| 20 | 0.566 ± 0.056 | ✓ |
| 21 | 0.909 ± 0.089 | ✓ |
| 22 | 0.588 ± 0.056 | ✓ |
| 23 | 1.064 ± 0.111 | ✓ |
| 24 | 0.727 ± 0.067 | ✓ |
| 25 | 0.714 ± 0.068 | ✓ |
| 26 | 0.657 ± 0.064 | ✓ |
| 27 | 1.633 ± 0.149 | ✓ |
| 28 | 1.023 ± 0.104 | ✓ |
| 29 | 0.809 ± 0.077 | ✓ |
| 30 | 0.716 ± 0.074 | ✓ |
| 31 | 1.274 ± 0.125 | ✓ |

## Example Player Decompositions


### Player 0

**Predicted RAPM**: 4.69 ± 45.66

**Contributions**:
- Intercept: -0.349
- Main effects: -2.333
- Interactions: 3.731

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 19 | 7.77 | 3.973 |
| 9 | -8.01 | -3.052 |
| 4 | -6.38 | -2.599 |
| 14 | -6.66 | 1.790 |
| 31 | -5.99 | -1.558 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (9, 19) | -7.656 |
| (4, 14) | 7.275 |
| (19, 22) | 6.804 |
| (19, 31) | -5.597 |
| (9, 31) | -5.148 |

### Player 266

**Predicted RAPM**: 4.40 ± 11.83

**Contributions**:
- Intercept: -0.349
- Main effects: 1.265
- Interactions: 5.276

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 19 | 2.64 | 1.348 |
| 31 | -4.04 | -1.050 |
| 9 | 1.70 | 0.648 |
| 11 | -2.00 | 0.464 |
| 2 | 2.09 | -0.462 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (0, 1) | -1.772 |
| (0, 24) | -1.768 |
| (1, 31) | 1.735 |
| (1, 29) | -1.646 |
| (19, 24) | 1.605 |

### Player 532

**Predicted RAPM**: 0.31 ± 8.38

**Contributions**:
- Intercept: -0.349
- Main effects: 2.283
- Interactions: -2.396

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 9 | 2.69 | 1.027 |
| 14 | -3.32 | 0.892 |
| 10 | -1.12 | 0.582 |
| 4 | 1.04 | 0.425 |
| 27 | -4.51 | 0.416 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (27, 29) | 1.145 |
| (21, 27) | -1.020 |
| (22, 29) | -0.828 |
| (5, 27) | 0.601 |
| (4, 14) | -0.593 |
