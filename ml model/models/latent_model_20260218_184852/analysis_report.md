# Latent Space Model Analysis Report

**Generated**: 2026-02-18 18:48
**Model**: latent_model_20260218_184852

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 5.0701
- **Test RMSE**: 1.6493635177612305
- **Test Correlation**: -0.14944281278392488

## Discovered Archetypes

### High Career Years Type (Archetype 0)

- **Players**: 195
- **Avg RAPM**: 0.19
- **Survival Rate**: 69.2%
- **Description**: Characterized by high Career Years
- **Examples**: P14, P16, P18

### High Career Wt Trueshootingpct Type (Archetype 1)

- **Players**: 3
- **Avg RAPM**: -0.44
- **Survival Rate**: 66.7%
- **Description**: Characterized by high Career Wt Trueshootingpct
- **Examples**: P2, P345, P504

### Low Three Share Type (Archetype 2)

- **Players**: 72
- **Avg RAPM**: 0.26
- **Survival Rate**: 66.7%
- **Description**: Characterized by low Three Share
- **Examples**: P1, P5, P8

### Low Career Wt Trueshootingpct Type (Archetype 3)

- **Players**: 1
- **Avg RAPM**: 1.26
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Career Wt Trueshootingpct
- **Examples**: P495

### Low Career Years Type (Archetype 4)

- **Players**: 86
- **Avg RAPM**: 0.64
- **Survival Rate**: 77.9%
- **Description**: Characterized by low Career Years
- **Examples**: P4, P7, P9

### Low Career Years Type (Archetype 5)

- **Players**: 95
- **Avg RAPM**: 0.51
- **Survival Rate**: 77.9%
- **Description**: Characterized by low Career Years
- **Examples**: P0, P3, P11

### Low Ft Pct Type (Archetype 6)

- **Players**: 6
- **Avg RAPM**: 0.39
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Ft Pct
- **Examples**: P100, P153, P178

### High Three Share Type (Archetype 7)

- **Players**: 103
- **Avg RAPM**: 0.04
- **Survival Rate**: 63.1%
- **Description**: Characterized by high Three Share
- **Examples**: P6, P10, P17

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
