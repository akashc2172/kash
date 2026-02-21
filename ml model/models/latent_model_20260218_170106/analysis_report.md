# Latent Space Model Analysis Report

**Generated**: 2026-02-18 17:01
**Model**: latent_model_20260218_170106

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 5.3328
- **Test RMSE**: 1.5461995601654053
- **Test Correlation**: -0.0994706260691012

## Discovered Archetypes

### Low Poss Total Type (Archetype 0)

- **Players**: 11
- **Avg RAPM**: 0.05
- **Survival Rate**: 81.8%
- **Description**: Characterized by low Poss Total
- **Examples**: P1, P135, P151

### High Three Share Z Type (Archetype 1)

- **Players**: 158
- **Avg RAPM**: 0.07
- **Survival Rate**: 62.7%
- **Description**: Characterized by high Three Share Z
- **Examples**: P0, P3, P5

### High Career Wt Trueshootingpct Type (Archetype 2)

- **Players**: 2
- **Avg RAPM**: -2.15
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Career Wt Trueshootingpct
- **Examples**: P6, P411

### Low Three Share Type (Archetype 3)

- **Players**: 79
- **Avg RAPM**: 0.51
- **Survival Rate**: 64.6%
- **Description**: Characterized by low Three Share
- **Examples**: P4, P16, P17

### Low Dev P50 Type (Archetype 4)

- **Players**: 1
- **Avg RAPM**: 0.50
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Dev P50
- **Examples**: P27

### Low Ft Pct Type (Archetype 5)

- **Players**: 3
- **Avg RAPM**: 0.61
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Ft Pct
- **Examples**: P2, P299, P531

### High Ft Att Type (Archetype 6)

- **Players**: 271
- **Avg RAPM**: 0.33
- **Survival Rate**: 77.5%
- **Description**: Characterized by high Ft Att
- **Examples**: P8, P9, P10

### High Delta Trueshootingpct Type (Archetype 7)

- **Players**: 13
- **Avg RAPM**: 0.64
- **Survival Rate**: 84.6%
- **Description**: Characterized by high Delta Trueshootingpct
- **Examples**: P63, P140, P154

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
