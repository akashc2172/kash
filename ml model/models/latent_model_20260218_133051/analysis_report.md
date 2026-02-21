# Latent Space Model Analysis Report

**Generated**: 2026-02-18 13:30
**Model**: latent_model_20260218_133051

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 5.0813
- **Test RMSE**: 1.5885413885116577
- **Test Correlation**: 0.1128651185998999

## Discovered Archetypes

### High Three Share Z Type (Archetype 0)

- **Players**: 162
- **Avg RAPM**: 0.12
- **Survival Rate**: 67.3%
- **Description**: Characterized by high Three Share Z
- **Examples**: P0, P4, P5

### High Rim Share Type (Archetype 1)

- **Players**: 1
- **Avg RAPM**: -1.84
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Rim Share
- **Examples**: P3

### Low Career Years Type (Archetype 2)

- **Players**: 179
- **Avg RAPM**: 0.47
- **Survival Rate**: 76.0%
- **Description**: Characterized by low Career Years
- **Examples**: P7, P8, P10

### High Career Wt Trueshootingpct Type (Archetype 3)

- **Players**: 1
- **Avg RAPM**: -2.47
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Career Wt Trueshootingpct
- **Examples**: P377

### Low Ft Pct Type (Archetype 4)

- **Players**: 3
- **Avg RAPM**: 1.31
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Ft Pct
- **Examples**: P268, P521, P533

### Low Poss Total Type (Archetype 5)

- **Players**: 29
- **Avg RAPM**: 0.73
- **Survival Rate**: 75.9%
- **Description**: Characterized by low Poss Total
- **Examples**: P41, P44, P87

### Low Ft Pct Type (Archetype 6)

- **Players**: 5
- **Avg RAPM**: -0.87
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Ft Pct
- **Examples**: P2, P138, P200

### High Career Years Type (Archetype 7)

- **Players**: 159
- **Avg RAPM**: 0.17
- **Survival Rate**: 69.2%
- **Description**: Characterized by high Career Years
- **Examples**: P1, P80, P127

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
