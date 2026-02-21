# Latent Space Model Analysis Report

**Generated**: 2026-02-18 19:03
**Model**: latent_model_20260218_190326

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 5.6556
- **Test RMSE**: 1.5771288871765137
- **Test Correlation**: 0.07809240377260877

## Discovered Archetypes

### Low Dev Quality Weight Type (Archetype 0)

- **Players**: 83
- **Avg RAPM**: 0.36
- **Survival Rate**: 81.9%
- **Description**: Characterized by low Dev Quality Weight
- **Examples**: P0, P3, P4

### High Dev Quality Weight Type (Archetype 1)

- **Players**: 188
- **Avg RAPM**: 0.23
- **Survival Rate**: 75.5%
- **Description**: Characterized by high Dev Quality Weight
- **Examples**: P1, P70, P71

### High Career Wt Trueshootingpct Type (Archetype 2)

- **Players**: 2
- **Avg RAPM**: -2.15
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Career Wt Trueshootingpct
- **Examples**: P6, P411

### High Career Years Type (Archetype 3)

- **Players**: 178
- **Avg RAPM**: 0.12
- **Survival Rate**: 63.5%
- **Description**: Characterized by high Career Years
- **Examples**: P63, P76, P85

### Low Dev Quality Weight Type (Archetype 4)

- **Players**: 77
- **Avg RAPM**: 0.69
- **Survival Rate**: 67.5%
- **Description**: Characterized by low Dev Quality Weight
- **Examples**: P8, P9, P11

### Low Dev P50 Type (Archetype 5)

- **Players**: 1
- **Avg RAPM**: 0.50
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Dev P50
- **Examples**: P27

### Low Ft Pct Type (Archetype 6)

- **Players**: 3
- **Avg RAPM**: 1.31
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Ft Pct
- **Examples**: P299, P531, P536

### Low Ft Pct Type (Archetype 7)

- **Players**: 6
- **Avg RAPM**: 0.20
- **Survival Rate**: 83.3%
- **Description**: Characterized by low Ft Pct
- **Examples**: P2, P164, P166

## Usage

```python
from models import ProspectModel, ArchetypeAnalyzer

# Load model
model = ProspectModel(latent_dim=32, n_archetypes=8)
model.load_state_dict(torch.load('model.pt'))

# Analyze a prospect
analysis = analyzer.analyze_player(tier1, tier2, career, tier2_mask)
print(analysis.narrative)
```
