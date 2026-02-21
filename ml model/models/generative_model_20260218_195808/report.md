# Generative Prospect Model Report

**Generated**: 2026-02-18 19:58

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
| Active interactions | 7 / 496 |
| Final ELBO loss | 5826.1353 |


## Prediction Performance

| Metric | Value |
|--------|-------|
| RMSE | 1.7919 |
| Correlation | 0.2537 |


## ARD Scales (Trait Importance)

| Trait | Scale (mean ± std) | Active |
|-------|-------------------|--------|
| 0 | 0.003 ± 0.002 |  |
| 1 | 0.003 ± 0.002 |  |
| 2 | 0.003 ± 0.003 |  |
| 3 | 0.003 ± 0.002 |  |
| 4 | 0.003 ± 0.003 |  |
| 5 | 0.003 ± 0.002 |  |
| 6 | 0.003 ± 0.003 |  |
| 7 | 0.003 ± 0.003 |  |
| 8 | 0.003 ± 0.002 |  |
| 9 | 0.003 ± 0.003 |  |
| 10 | 0.003 ± 0.003 |  |
| 11 | 0.003 ± 0.003 |  |
| 12 | 0.003 ± 0.003 |  |
| 13 | 0.003 ± 0.002 |  |
| 14 | 0.003 ± 0.002 |  |
| 15 | 0.003 ± 0.003 |  |
| 16 | 0.003 ± 0.002 |  |
| 17 | 0.003 ± 0.003 |  |
| 18 | 0.003 ± 0.003 |  |
| 19 | 0.003 ± 0.002 |  |
| 20 | 0.003 ± 0.003 |  |
| 21 | 0.003 ± 0.003 |  |
| 22 | 0.003 ± 0.002 |  |
| 23 | 0.029 ± 0.003 |  |
| 24 | 0.003 ± 0.003 |  |
| 25 | 0.003 ± 0.002 |  |
| 26 | 0.003 ± 0.002 |  |
| 27 | 0.003 ± 0.003 |  |
| 28 | 0.003 ± 0.002 |  |
| 29 | 0.003 ± 0.003 |  |
| 30 | 0.003 ± 0.003 |  |
| 31 | 0.004 ± 0.003 |  |

## Example Player Decompositions


### Player 0

**Predicted RAPM**: 0.14 ± 0.30

**Contributions**:
- Intercept: 0.027
- Main effects: 0.113
- Interactions: 0.000

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 23 | 0.17 | 0.114 |

### Player 263

**Predicted RAPM**: -0.04 ± 0.22

**Contributions**:
- Intercept: 0.027
- Main effects: -0.087
- Interactions: -0.000

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 23 | -0.12 | -0.085 |

### Player 527

**Predicted RAPM**: 0.51 ± 0.25

**Contributions**:
- Intercept: 0.027
- Main effects: 0.464
- Interactions: 0.002

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 23 | 0.68 | 0.464 |
