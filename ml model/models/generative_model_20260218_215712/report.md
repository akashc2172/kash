# Generative Prospect Model Report

**Generated**: 2026-02-18 21:57

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
| Active interactions | 220 / 496 |
| Final ELBO loss | 234589.9375 |


## Prediction Performance

| Metric | Value |
|--------|-------|
| RMSE | 6.0221 |
| Correlation | 0.0311 |


## ARD Scales (Trait Importance)

| Trait | Scale (mean ± std) | Active |
|-------|-------------------|--------|
| 0 | 0.817 ± 0.082 | ✓ |
| 1 | 0.581 ± 0.057 | ✓ |
| 2 | 0.918 ± 0.087 | ✓ |
| 3 | 1.486 ± 0.138 | ✓ |
| 4 | 0.898 ± 0.081 | ✓ |
| 5 | 0.619 ± 0.057 | ✓ |
| 6 | 0.411 ± 0.039 | ✓ |
| 7 | 0.558 ± 0.056 | ✓ |
| 8 | 0.451 ± 0.047 | ✓ |
| 9 | 0.893 ± 0.083 | ✓ |
| 10 | 0.672 ± 0.069 | ✓ |
| 11 | 0.672 ± 0.062 | ✓ |
| 12 | 0.671 ± 0.068 | ✓ |
| 13 | 0.832 ± 0.088 | ✓ |
| 14 | 1.114 ± 0.110 | ✓ |
| 15 | 0.658 ± 0.063 | ✓ |
| 16 | 1.007 ± 0.096 | ✓ |
| 17 | 0.872 ± 0.082 | ✓ |
| 18 | 1.582 ± 0.153 | ✓ |
| 19 | 1.047 ± 0.093 | ✓ |
| 20 | 0.579 ± 0.056 | ✓ |
| 21 | 0.972 ± 0.091 | ✓ |
| 22 | 0.593 ± 0.054 | ✓ |
| 23 | 1.118 ± 0.110 | ✓ |
| 24 | 0.797 ± 0.076 | ✓ |
| 25 | 0.881 ± 0.085 | ✓ |
| 26 | 0.627 ± 0.062 | ✓ |
| 27 | 1.660 ± 0.150 | ✓ |
| 28 | 1.097 ± 0.106 | ✓ |
| 29 | 0.861 ± 0.086 | ✓ |
| 30 | 0.719 ± 0.070 | ✓ |
| 31 | 1.290 ± 0.120 | ✓ |

## Example Player Decompositions


### Player 0

**Predicted RAPM**: 8.13 ± 47.37

**Contributions**:
- Intercept: -0.329
- Main effects: -2.766
- Interactions: 13.424

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 13 | -7.86 | -2.792 |
| 12 | 8.18 | 2.121 |
| 14 | 7.08 | -1.795 |
| 4 | 3.76 | 1.514 |
| 15 | -5.17 | -1.035 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (12, 13) | 6.603 |
| (6, 13) | -5.892 |
| (4, 14) | 5.096 |
| (4, 28) | 4.969 |
| (13, 15) | 4.478 |

### Player 266

**Predicted RAPM**: 4.94 ± 15.26

**Contributions**:
- Intercept: -0.329
- Main effects: -3.504
- Interactions: 8.635

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 19 | -4.59 | -2.579 |
| 21 | -4.94 | -1.119 |
| 10 | -2.06 | 0.930 |
| 18 | 5.78 | -0.843 |
| 13 | -2.10 | -0.745 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (19, 24) | 3.425 |
| (3, 21) | -2.537 |
| (19, 21) | 2.303 |
| (18, 21) | 2.162 |
| (18, 19) | 2.008 |

### Player 532

**Predicted RAPM**: -0.51 ± 8.64

**Contributions**:
- Intercept: -0.329
- Main effects: 0.148
- Interactions: 0.577

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 4 | 3.37 | 1.358 |
| 31 | 2.16 | 0.738 |
| 9 | -1.47 | -0.570 |
| 11 | 4.08 | -0.545 |
| 25 | 2.80 | 0.454 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (4, 28) | -1.720 |
| (1, 29) | -1.221 |
| (11, 28) | 1.123 |
| (11, 29) | -1.061 |
| (11, 16) | -0.959 |
