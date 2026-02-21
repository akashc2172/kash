# Latent Space Model Analysis Report

**Generated**: 2026-02-18 18:48
**Model**: latent_model_20260218_184832

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 6.5346
- **Test RMSE**: 1.562717318534851
- **Test Correlation**: -0.11661739459908707

## Discovered Archetypes

### Low Ft Pct Type (Archetype 0)

- **Players**: 4
- **Avg RAPM**: 0.90
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Ft Pct
- **Examples**: P2, P299, P531

### High Three Share Z Type (Archetype 1)

- **Players**: 285
- **Avg RAPM**: 0.31
- **Survival Rate**: 71.9%
- **Description**: Characterized by high Three Share Z
- **Examples**: P0, P3, P4

### Low Poss Total Type (Archetype 2)

- **Players**: 17
- **Avg RAPM**: 0.64
- **Survival Rate**: 88.2%
- **Description**: Characterized by low Poss Total
- **Examples**: P63, P135, P140

### High Career Wt Trueshootingpct Type (Archetype 3)

- **Players**: 2
- **Avg RAPM**: -2.15
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Career Wt Trueshootingpct
- **Examples**: P6, P411

### Low Poss Total Type (Archetype 4)

- **Players**: 31
- **Avg RAPM**: -0.10
- **Survival Rate**: 67.7%
- **Description**: Characterized by low Poss Total
- **Examples**: P1, P8, P85

### High Ft Att Type (Archetype 5)

- **Players**: 196
- **Avg RAPM**: 0.27
- **Survival Rate**: 69.4%
- **Description**: Characterized by high Ft Att
- **Examples**: P9, P11, P13

### High Mid Fg Pct Type (Archetype 6)

- **Players**: 2
- **Avg RAPM**: 0.06
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Mid Fg Pct
- **Examples**: P219, P244

### Low Dev P50 Type (Archetype 7)

- **Players**: 1
- **Avg RAPM**: 0.50
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Dev P50
- **Examples**: P27

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
