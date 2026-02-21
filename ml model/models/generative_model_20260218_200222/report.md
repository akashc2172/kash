# Generative Prospect Model Report

**Generated**: 2026-02-18 20:02

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
| Active interactions | 231 / 496 |
| Final ELBO loss | 704751.3750 |


## Prediction Performance

| Metric | Value |
|--------|-------|
| RMSE | 9.7070 |
| Correlation | 0.0620 |


## ARD Scales (Trait Importance)

| Trait | Scale (mean ± std) | Active |
|-------|-------------------|--------|
| 0 | 0.780 ± 0.078 | ✓ |
| 1 | 0.587 ± 0.059 | ✓ |
| 2 | 0.989 ± 0.099 | ✓ |
| 3 | 1.487 ± 0.139 | ✓ |
| 4 | 0.865 ± 0.080 | ✓ |
| 5 | 0.662 ± 0.063 | ✓ |
| 6 | 0.422 ± 0.040 | ✓ |
| 7 | 0.528 ± 0.053 | ✓ |
| 8 | 0.453 ± 0.048 | ✓ |
| 9 | 0.915 ± 0.089 | ✓ |
| 10 | 0.615 ± 0.066 | ✓ |
| 11 | 0.707 ± 0.066 | ✓ |
| 12 | 0.717 ± 0.072 | ✓ |
| 13 | 0.868 ± 0.091 | ✓ |
| 14 | 1.122 ± 0.109 | ✓ |
| 15 | 0.646 ± 0.061 | ✓ |
| 16 | 1.043 ± 0.094 | ✓ |
| 17 | 0.871 ± 0.085 | ✓ |
| 18 | 1.621 ± 0.155 | ✓ |
| 19 | 1.038 ± 0.098 | ✓ |
| 20 | 0.575 ± 0.062 | ✓ |
| 21 | 0.881 ± 0.084 | ✓ |
| 22 | 0.598 ± 0.052 | ✓ |
| 23 | 1.081 ± 0.109 | ✓ |
| 24 | 0.764 ± 0.072 | ✓ |
| 25 | 0.792 ± 0.080 | ✓ |
| 26 | 0.650 ± 0.065 | ✓ |
| 27 | 1.693 ± 0.159 | ✓ |
| 28 | 1.128 ± 0.110 | ✓ |
| 29 | 0.872 ± 0.085 | ✓ |
| 30 | 0.703 ± 0.077 | ✓ |
| 31 | 1.325 ± 0.130 | ✓ |

## Example Player Decompositions


### Player 0

**Predicted RAPM**: -9.21 ± 17.94

**Contributions**:
- Intercept: -0.203
- Main effects: -1.599
- Interactions: -8.152

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 23 | -7.44 | -1.224 |
| 28 | 6.12 | 0.610 |
| 9 | -1.50 | -0.582 |
| 20 | 1.91 | 0.559 |
| 31 | 1.63 | 0.556 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (24, 28) | 4.176 |
| (23, 28) | -2.724 |
| (2, 23) | 1.995 |
| (15, 24) | -1.972 |
| (23, 26) | -1.870 |

### Player 263

**Predicted RAPM**: 6.81 ± 11.89

**Contributions**:
- Intercept: -0.203
- Main effects: -1.494
- Interactions: 9.420

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 4 | -3.00 | -1.045 |
| 10 | -2.06 | 0.965 |
| 21 | -3.00 | -0.807 |
| 12 | -2.36 | -0.640 |
| 13 | 2.00 | 0.601 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (3, 21) | 2.653 |
| (3, 10) | 1.945 |
| (13, 17) | 1.786 |
| (21, 27) | -1.751 |
| (3, 6) | -1.392 |

### Player 527

**Predicted RAPM**: 5.36 ± 5.32

**Contributions**:
- Intercept: -0.203
- Main effects: 0.584
- Interactions: 5.372

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 31 | 1.66 | 0.566 |
| 9 | -0.91 | -0.352 |
| 14 | -1.23 | 0.343 |
| 18 | -3.73 | 0.338 |
| 12 | 1.21 | 0.327 |

**Active Interactions**:

| Pair | Contribution |
|------|--------------|
| (18, 23) | 1.538 |
| (22, 29) | 1.095 |
| (18, 28) | 0.710 |
| (18, 30) | 0.697 |
| (28, 31) | 0.645 |
