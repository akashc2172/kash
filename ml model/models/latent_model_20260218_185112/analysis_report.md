# Latent Space Model Analysis Report

**Generated**: 2026-02-18 18:51
**Model**: latent_model_20260218_185112

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 5.0485
- **Test RMSE**: 1.5709785223007202
- **Test Correlation**: -0.20544827653843628

## Discovered Archetypes

### Low Poss Total Type (Archetype 0)

- **Players**: 33
- **Avg RAPM**: -0.07
- **Survival Rate**: 69.7%
- **Description**: Characterized by low Poss Total
- **Examples**: P85, P131, P133

### High Three Share Z Type (Archetype 1)

- **Players**: 110
- **Avg RAPM**: -0.12
- **Survival Rate**: 63.6%
- **Description**: Characterized by high Three Share Z
- **Examples**: P0, P4, P5

### High Fga Total Type (Archetype 2)

- **Players**: 264
- **Avg RAPM**: 0.44
- **Survival Rate**: 77.3%
- **Description**: Characterized by high Fga Total
- **Examples**: P3, P8, P9

### High Career Wt Trueshootingpct Type (Archetype 3)

- **Players**: 2
- **Avg RAPM**: -2.15
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Career Wt Trueshootingpct
- **Examples**: P6, P411

### High Ft Att Type (Archetype 4)

- **Players**: 105
- **Avg RAPM**: 0.36
- **Survival Rate**: 63.8%
- **Description**: Characterized by high Ft Att
- **Examples**: P13, P16, P17

### Low Trueshootingpct Z Type (Archetype 5)

- **Players**: 6
- **Avg RAPM**: 0.27
- **Survival Rate**: 83.3%
- **Description**: Characterized by low Trueshootingpct Z
- **Examples**: P2, P244, P299

### Low Poss Total Type (Archetype 6)

- **Players**: 17
- **Avg RAPM**: 0.73
- **Survival Rate**: 82.4%
- **Description**: Characterized by low Poss Total
- **Examples**: P1, P63, P135

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
