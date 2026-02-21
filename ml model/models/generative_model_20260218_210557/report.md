# Generative Prospect Model Report

**Generated**: 2026-02-18 21:06

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
| Active interactions | 227 / 496 |
| Final ELBO loss | 150130.3750 |


## Prediction Performance

| Metric | Value |
|--------|-------|
| RMSE | 4.7012 |
| Correlation | -0.0052 |


## ARD Scales (Trait Importance)

| Trait | Scale (mean ± std) | Active |
|-------|-------------------|--------|
| 0 | 0.732 ± 0.075 | ✓ |
| 1 | 0.581 ± 0.057 | ✓ |
| 2 | 0.897 ± 0.086 | ✓ |
| 3 | 1.483 ± 0.139 | ✓ |
| 4 | 0.799 ± 0.070 | ✓ |
| 5 | 0.606 ± 0.058 | ✓ |
| 6 | 0.393 ± 0.040 | ✓ |
| 7 | 0.577 ± 0.059 | ✓ |
| 8 | 0.392 ± 0.039 | ✓ |
| 9 | 0.900 ± 0.086 | ✓ |
| 10 | 0.612 ± 0.064 | ✓ |
| 11 | 0.674 ± 0.065 | ✓ |
| 12 | 0.652 ± 0.062 | ✓ |
| 13 | 0.824 ± 0.080 | ✓ |
| 14 | 1.075 ± 0.101 | ✓ |
| 15 | 0.645 ± 0.063 | ✓ |
| 16 | 0.995 ± 0.091 | ✓ |
| 17 | 0.830 ± 0.084 | ✓ |
| 18 | 1.550 ± 0.144 | ✓ |
| 19 | 0.986 ± 0.089 | ✓ |
| 20 | 0.567 ± 0.058 | ✓ |
| 21 | 0.943 ± 0.090 | ✓ |
| 22 | 0.546 ± 0.051 | ✓ |
| 23 | 1.119 ± 0.120 | ✓ |
| 24 | 0.785 ± 0.077 | ✓ |
| 25 | 0.739 ± 0.071 | ✓ |
| 26 | 0.674 ± 0.069 | ✓ |
| 27 | 1.653 ± 0.149 | ✓ |
| 28 | 1.057 ± 0.105 | ✓ |
| 29 | 0.835 ± 0.079 | ✓ |
| 30 | 0.719 ± 0.074 | ✓ |
| 31 | 1.284 ± 0.125 | ✓ |

## Example Player Decompositions


### Player 0

**Predicted RAPM**: 1.10 ± 32.39

**Contributions**:
- Intercept: -0.205
- Main effects: 7.087
- Interactions: -6.444

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 19 | 6.42 | 3.169 |
| 10 | -4.92 | 2.061 |
| 13 | 5.93 | 1.594 |
| 9 | -4.17 | -1.521 |
| 31 | 3.00 | 0.986 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (18, 30) | 4.818 |
| (3, 10) | -3.141 |
| (13, 17) | 3.133 |
| (8, 26) | -3.076 |
| (13, 25) | 2.939 |

### Player 266

**Predicted RAPM**: -1.72 ± 10.83

**Contributions**:
- Intercept: -0.205
- Main effects: -2.741
- Interactions: 0.890

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 19 | -3.74 | -1.845 |
| 2 | 3.56 | -0.837 |
| 9 | -1.99 | -0.726 |
| 10 | -1.23 | 0.514 |
| 26 | -2.68 | 0.457 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (18, 30) | -2.537 |
| (2, 30) | 2.063 |
| (27, 29) | -1.302 |
| (19, 30) | -1.081 |
| (21, 27) | -0.945 |

### Player 532

**Predicted RAPM**: -1.76 ± 7.69

**Contributions**:
- Intercept: -0.205
- Main effects: 1.532
- Interactions: -3.369

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 9 | 2.71 | 0.987 |
| 4 | 2.67 | 0.903 |
| 13 | -2.25 | -0.605 |
| 12 | 2.19 | 0.538 |
| 0 | -2.61 | -0.441 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (18, 29) | -0.995 |
| (4, 18) | 0.973 |
| (27, 29) | 0.817 |
| (0, 24) | -0.726 |
| (5, 9) | 0.691 |
