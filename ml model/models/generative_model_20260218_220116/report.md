# Generative Prospect Model Report

**Generated**: 2026-02-18 22:01

## Model Configuration

| Parameter | Value |
|-----------|-------|
| K_max (latent dims) | 32 |
| ARD scale | 1.0 |
| Tau main | 1.0 |
| Tau interaction | 0.3 |
| SVI steps | 5000 |

## Results

| Metric | Value |
|--------|-------|
| Effective dimensions | 0 / 32 |
| Active interactions | 8 / 496 |
| Final ELBO loss | 5843.2808 |


## Prediction Performance

| Metric | Value |
|--------|-------|
| RMSE | 1.7891 |
| Correlation | 0.2518 |


## ARD Scales (Trait Importance)

| Trait | Scale (mean ± std) | Active |
|-------|-------------------|--------|
| 0 | 0.003 ± 0.002 |  |
| 1 | 0.003 ± 0.002 |  |
| 2 | 0.004 ± 0.003 |  |
| 3 | 0.003 ± 0.002 |  |
| 4 | 0.003 ± 0.003 |  |
| 5 | 0.003 ± 0.002 |  |
| 6 | 0.004 ± 0.003 |  |
| 7 | 0.003 ± 0.003 |  |
| 8 | 0.003 ± 0.003 |  |
| 9 | 0.003 ± 0.003 |  |
| 10 | 0.003 ± 0.003 |  |
| 11 | 0.004 ± 0.003 |  |
| 12 | 0.003 ± 0.003 |  |
| 13 | 0.003 ± 0.003 |  |
| 14 | 0.003 ± 0.003 |  |
| 15 | 0.003 ± 0.003 |  |
| 16 | 0.003 ± 0.003 |  |
| 17 | 0.003 ± 0.003 |  |
| 18 | 0.042 ± 0.003 |  |
| 19 | 0.003 ± 0.002 |  |
| 20 | 0.003 ± 0.003 |  |
| 21 | 0.004 ± 0.003 |  |
| 22 | 0.004 ± 0.002 |  |
| 23 | 0.004 ± 0.003 |  |
| 24 | 0.003 ± 0.003 |  |
| 25 | 0.003 ± 0.002 |  |
| 26 | 0.003 ± 0.002 |  |
| 27 | 0.004 ± 0.003 |  |
| 28 | 0.003 ± 0.002 |  |
| 29 | 0.003 ± 0.003 |  |
| 30 | 0.003 ± 0.003 |  |
| 31 | 0.004 ± 0.003 |  |

## Example Player Decompositions


### Player 0

**Predicted RAPM**: 0.05 ± 0.46

**Contributions**:
- Intercept: 0.023
- Main effects: 0.014
- Interactions: -0.000

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 18 | -0.03 | 0.016 |
| 12 | -0.02 | -0.001 |
| 22 | 0.03 | 0.001 |
| 21 | 0.01 | -0.001 |
| 15 | -0.02 | 0.001 |

### Player 266

**Predicted RAPM**: -0.36 ± 0.29

**Contributions**:
- Intercept: 0.023
- Main effects: -0.374
- Interactions: 0.001

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 18 | 0.66 | -0.372 |

### Player 532

**Predicted RAPM**: 0.11 ± 0.25

**Contributions**:
- Intercept: 0.023
- Main effects: 0.087
- Interactions: 0.000

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 18 | -0.16 | 0.089 |
