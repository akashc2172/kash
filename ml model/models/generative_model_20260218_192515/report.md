# Generative Prospect Model Report

**Generated**: 2026-02-18 19:25

## Model Configuration

| Parameter | Value |
|-----------|-------|
| K_max (latent dims) | 16 |
| ARD scale | 1.0 |
| Tau main | 1.0 |
| Tau interaction | 0.3 |
| SVI steps | 300 |

## Results

| Metric | Value |
|--------|-------|
| Effective dimensions | 16 / 16 |
| Active interactions | 4 / 120 |
| Final ELBO loss | 11038.0020 |


## Prediction Performance

| Metric | Value |
|--------|-------|
| RMSE | 1.7598 |
| Correlation | 0.3342 |


## ARD Scales (Trait Importance)

| Trait | Scale (mean ± std) | Active |
|-------|-------------------|--------|
| 0 | 0.226 ± 0.022 | ✓ |
| 1 | 0.215 ± 0.020 | ✓ |
| 2 | 0.482 ± 0.045 | ✓ |
| 3 | 0.350 ± 0.031 | ✓ |
| 4 | 0.245 ± 0.021 | ✓ |
| 5 | 0.337 ± 0.030 | ✓ |
| 6 | 0.224 ± 0.022 | ✓ |
| 7 | 0.179 ± 0.018 | ✓ |
| 8 | 0.294 ± 0.025 | ✓ |
| 9 | 0.188 ± 0.017 | ✓ |
| 10 | 0.253 ± 0.023 | ✓ |
| 11 | 0.297 ± 0.026 | ✓ |
| 12 | 0.208 ± 0.019 | ✓ |
| 13 | 0.273 ± 0.027 | ✓ |
| 14 | 0.160 ± 0.017 | ✓ |
| 15 | 0.248 ± 0.023 | ✓ |

## Example Player Decompositions


### Player 0

**Predicted RAPM**: 0.42 ± 0.41

**Contributions**:
- Intercept: 0.069
- Main effects: 0.304
- Interactions: 0.076

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 10 | -0.52 | 0.201 |
| 9 | 0.54 | 0.105 |
| 2 | 0.50 | -0.084 |
| 12 | 0.31 | 0.033 |
| 0 | 0.37 | 0.029 |

### Player 278

**Predicted RAPM**: 0.02 ± 0.33

**Contributions**:
- Intercept: 0.069
- Main effects: -0.023
- Interactions: -0.015

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 10 | 0.13 | -0.051 |
| 2 | -0.21 | 0.035 |
| 9 | -0.16 | -0.031 |
| 14 | -0.22 | -0.029 |
| 0 | 0.34 | 0.027 |

### Player 557

**Predicted RAPM**: 0.05 ± 0.25

**Contributions**:
- Intercept: 0.069
- Main effects: 0.024
- Interactions: -0.031

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
| 9 | -0.28 | -0.054 |
| 10 | -0.14 | 0.054 |
| 14 | -0.38 | -0.049 |
| 4 | 0.57 | 0.036 |
| 12 | 0.22 | 0.024 |
