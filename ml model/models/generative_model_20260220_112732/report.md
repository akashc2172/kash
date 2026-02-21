# Generative Prospect Model Report

**Generated**: 2026-02-20 11:27

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
| Active interactions | 204 / 496 |
| Final ELBO loss | 292551.7188 |


## Prediction Performance

| Metric | Value |
|--------|-------|
| RMSE | 6.1983 |
| Correlation | 0.0115 |


## ARD Scales (Trait Importance)

| Trait | Scale (mean ± std) | Active |
|-------|-------------------|--------|
| 0 | 0.750 ± 0.072 | ✓ |
| 1 | 0.599 ± 0.059 | ✓ |
| 2 | 0.882 ± 0.086 | ✓ |
| 3 | 1.469 ± 0.139 | ✓ |
| 4 | 0.796 ± 0.075 | ✓ |
| 5 | 0.624 ± 0.064 | ✓ |
| 6 | 0.361 ± 0.037 | ✓ |
| 7 | 0.543 ± 0.053 | ✓ |
| 8 | 0.416 ± 0.041 | ✓ |
| 9 | 0.863 ± 0.079 | ✓ |
| 10 | 0.562 ± 0.055 | ✓ |
| 11 | 0.630 ± 0.059 | ✓ |
| 12 | 0.674 ± 0.063 | ✓ |
| 13 | 0.811 ± 0.079 | ✓ |
| 14 | 1.067 ± 0.105 | ✓ |
| 15 | 0.598 ± 0.057 | ✓ |
| 16 | 0.992 ± 0.086 | ✓ |
| 17 | 0.804 ± 0.079 | ✓ |
| 18 | 1.528 ± 0.140 | ✓ |
| 19 | 0.985 ± 0.087 | ✓ |
| 20 | 0.566 ± 0.056 | ✓ |
| 21 | 0.909 ± 0.089 | ✓ |
| 22 | 0.588 ± 0.055 | ✓ |
| 23 | 1.064 ± 0.111 | ✓ |
| 24 | 0.727 ± 0.067 | ✓ |
| 25 | 0.714 ± 0.068 | ✓ |
| 26 | 0.652 ± 0.064 | ✓ |
| 27 | 1.633 ± 0.149 | ✓ |
| 28 | 1.023 ± 0.105 | ✓ |
| 29 | 0.819 ± 0.078 | ✓ |
| 30 | 0.716 ± 0.074 | ✓ |
| 31 | 1.274 ± 0.125 | ✓ |

## Example Player Decompositions


### Player 0

**Predicted RAPM**: 5.83 ± 46.29

**Contributions**:
- Intercept: -0.358
- Main effects: -2.197
- Interactions: 4.601

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 19 | 7.80 | 3.992 |
| 9 | -8.16 | -3.118 |
| 4 | -6.40 | -2.606 |
| 14 | -6.49 | 1.794 |
| 31 | -6.08 | -1.520 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (9, 19) | -7.216 |
| (4, 14) | 7.024 |
| (19, 22) | 6.613 |
| (19, 31) | -5.382 |
| (9, 31) | -5.327 |

### Player 270

**Predicted RAPM**: 4.06 ± 5.01

**Contributions**:
- Intercept: -0.358
- Main effects: 3.015
- Interactions: 1.538

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 4 | 2.76 | 1.123 |
| 19 | 1.56 | 0.796 |
| 9 | 1.86 | 0.712 |
| 12 | 1.87 | 0.535 |
| 20 | -1.80 | -0.468 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (4, 14) | 0.657 |
| (1, 20) | 0.628 |
| (1, 29) | -0.571 |
| (0, 1) | -0.556 |
| (4, 17) | -0.465 |

### Player 541

**Predicted RAPM**: 3.82 ± 5.39

**Contributions**:
- Intercept: -0.358
- Main effects: 0.164
- Interactions: 4.590

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 20 | 3.97 | 1.031 |
| 14 | -1.79 | 0.495 |
| 31 | -1.85 | -0.461 |
| 21 | 1.71 | 0.443 |
| 4 | -1.09 | -0.442 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (20, 21) | 0.904 |
| (20, 22) | 0.760 |
| (1, 20) | 0.707 |
| (2, 23) | -0.604 |
| (8, 20) | 0.556 |
