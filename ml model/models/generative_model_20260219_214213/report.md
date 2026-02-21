# Generative Prospect Model Report

**Generated**: 2026-02-19 21:42

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
| Active interactions | 9 / 496 |
| Final ELBO loss | 5841.3574 |


## Prediction Performance

| Metric | Value |
|--------|-------|
| RMSE | 1.7621 |
| Correlation | 0.3265 |


## ARD Scales (Trait Importance)

| Trait | Scale (mean ± std) | Active |
|-------|-------------------|--------|
| 0 | 0.003 ± 0.002 |  |
| 1 | 0.003 ± 0.002 |  |
| 2 | 0.004 ± 0.003 |  |
| 3 | 0.003 ± 0.002 |  |
| 4 | 0.003 ± 0.002 |  |
| 5 | 0.014 ± 0.003 |  |
| 6 | 0.003 ± 0.002 |  |
| 7 | 0.003 ± 0.003 |  |
| 8 | 0.003 ± 0.002 |  |
| 9 | 0.003 ± 0.002 |  |
| 10 | 0.003 ± 0.002 |  |
| 11 | 0.003 ± 0.003 |  |
| 12 | 0.003 ± 0.002 |  |
| 13 | 0.003 ± 0.002 |  |
| 14 | 0.003 ± 0.002 |  |
| 15 | 0.003 ± 0.002 |  |
| 16 | 0.003 ± 0.002 |  |
| 17 | 0.003 ± 0.003 |  |
| 18 | 0.035 ± 0.002 |  |
| 19 | 0.003 ± 0.002 |  |
| 20 | 0.003 ± 0.003 |  |
| 21 | 0.003 ± 0.002 |  |
| 22 | 0.003 ± 0.002 |  |
| 23 | 0.003 ± 0.003 |  |
| 24 | 0.003 ± 0.002 |  |
| 25 | 0.002 ± 0.002 |  |
| 26 | 0.003 ± 0.002 |  |
| 27 | 0.003 ± 0.002 |  |
| 28 | 0.003 ± 0.002 |  |
| 29 | 0.003 ± 0.002 |  |
| 30 | 0.003 ± 0.003 |  |
| 31 | 0.003 ± 0.003 |  |

## Example Player Decompositions


### Player 0

**Predicted RAPM**: -0.04 ± 0.47

**Contributions**:
- Intercept: 0.025
- Main effects: -0.105
- Interactions: 0.000

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 18 | 0.19 | -0.116 |

### Player 266

**Predicted RAPM**: -0.34 ± 0.30

**Contributions**:
- Intercept: 0.025
- Main effects: -0.369
- Interactions: 0.002

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 18 | 0.58 | -0.346 |

### Player 532

**Predicted RAPM**: 0.36 ± 0.29

**Contributions**:
- Intercept: 0.025
- Main effects: 0.357
- Interactions: 0.004

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 18 | -0.41 | 0.245 |
| 5 | 0.16 | 0.114 |
