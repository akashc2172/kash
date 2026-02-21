# Generative Prospect Model Report

**Generated**: 2026-02-19 12:18

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
| Active interactions | 10 / 496 |
| Final ELBO loss | 5861.8208 |


## Prediction Performance

| Metric | Value |
|--------|-------|
| RMSE | 1.7598 |
| Correlation | 0.3322 |


## ARD Scales (Trait Importance)

| Trait | Scale (mean ± std) | Active |
|-------|-------------------|--------|
| 0 | 0.003 ± 0.002 |  |
| 1 | 0.003 ± 0.002 |  |
| 2 | 0.003 ± 0.003 |  |
| 3 | 0.003 ± 0.002 |  |
| 4 | 0.003 ± 0.003 |  |
| 5 | 0.022 ± 0.002 |  |
| 6 | 0.003 ± 0.002 |  |
| 7 | 0.003 ± 0.003 |  |
| 8 | 0.003 ± 0.002 |  |
| 9 | 0.003 ± 0.002 |  |
| 10 | 0.003 ± 0.003 |  |
| 11 | 0.003 ± 0.003 |  |
| 12 | 0.003 ± 0.003 |  |
| 13 | 0.003 ± 0.002 |  |
| 14 | 0.003 ± 0.003 |  |
| 15 | 0.003 ± 0.002 |  |
| 16 | 0.003 ± 0.002 |  |
| 17 | 0.003 ± 0.002 |  |
| 18 | 0.030 ± 0.003 |  |
| 19 | 0.003 ± 0.002 |  |
| 20 | 0.003 ± 0.003 |  |
| 21 | 0.003 ± 0.003 |  |
| 22 | 0.003 ± 0.002 |  |
| 23 | 0.003 ± 0.003 |  |
| 24 | 0.003 ± 0.003 |  |
| 25 | 0.003 ± 0.002 |  |
| 26 | 0.003 ± 0.002 |  |
| 27 | 0.003 ± 0.002 |  |
| 28 | 0.003 ± 0.002 |  |
| 29 | 0.003 ± 0.003 |  |
| 30 | 0.003 ± 0.003 |  |
| 31 | 0.004 ± 0.003 |  |

## Example Player Decompositions


### Player 0

**Predicted RAPM**: -0.27 ± 0.51

**Contributions**:
- Intercept: 0.009
- Main effects: -0.312
- Interactions: -0.004

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 18 | 0.40 | -0.310 |

### Player 266

**Predicted RAPM**: -0.39 ± 0.32

**Contributions**:
- Intercept: 0.009
- Main effects: -0.374
- Interactions: -0.013

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 18 | 0.49 | -0.374 |

### Player 532

**Predicted RAPM**: 0.27 ± 0.28

**Contributions**:
- Intercept: 0.009
- Main effects: 0.260
- Interactions: 0.004

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 18 | -0.34 | 0.262 |
