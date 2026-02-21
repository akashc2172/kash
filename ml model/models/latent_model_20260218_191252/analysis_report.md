# Latent Space Model Analysis Report

**Generated**: 2026-02-18 19:12
**Model**: latent_model_20260218_191252

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 6.1234
- **Test RMSE**: 1.5653669834136963
- **Test Correlation**: 0.08505589137684212

## Discovered Archetypes

### High Three Share Type (Archetype 0)

- **Players**: 121
- **Avg RAPM**: 0.00
- **Survival Rate**: 61.2%
- **Description**: Characterized by high Three Share
- **Examples**: P0, P5, P12

### High Career Wt Trueshootingpct Type (Archetype 1)

- **Players**: 2
- **Avg RAPM**: -2.15
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Career Wt Trueshootingpct
- **Examples**: P6, P411

### Low Three Share Z Type (Archetype 2)

- **Players**: 120
- **Avg RAPM**: 0.31
- **Survival Rate**: 66.7%
- **Description**: Characterized by low Three Share Z
- **Examples**: P4, P8, P16

### Low Career Wt Trueshootingpct Type (Archetype 3)

- **Players**: 2
- **Avg RAPM**: 1.12
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Career Wt Trueshootingpct
- **Examples**: P27, P536

### Low Ft Pct Type (Archetype 4)

- **Players**: 3
- **Avg RAPM**: -0.07
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Ft Pct
- **Examples**: P2, P219, P244

### High Poss Total Type (Archetype 5)

- **Players**: 264
- **Avg RAPM**: 0.42
- **Survival Rate**: 78.0%
- **Description**: Characterized by high Poss Total
- **Examples**: P3, P9, P10

### Low Shots Total Type (Archetype 6)

- **Players**: 24
- **Avg RAPM**: -0.01
- **Survival Rate**: 70.8%
- **Description**: Characterized by low Shots Total
- **Examples**: P1, P7, P63

### Low Ft Pct Type (Archetype 7)

- **Players**: 2
- **Avg RAPM**: 1.09
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Ft Pct
- **Examples**: P299, P531

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
