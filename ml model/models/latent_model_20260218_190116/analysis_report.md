# Latent Space Model Analysis Report

**Generated**: 2026-02-18 19:01
**Model**: latent_model_20260218_190116

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 4.9596
- **Test RMSE**: 1.5541203022003174
- **Test Correlation**: 0.06049734148412839

## Discovered Archetypes

### High Fga Total Type (Archetype 0)

- **Players**: 234
- **Avg RAPM**: 0.42
- **Survival Rate**: 78.2%
- **Description**: Characterized by high Fga Total
- **Examples**: P0, P3, P10

### High Career Wt Trueshootingpct Type (Archetype 1)

- **Players**: 2
- **Avg RAPM**: -2.15
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Career Wt Trueshootingpct
- **Examples**: P6, P411

### High Three Share Type (Archetype 2)

- **Players**: 132
- **Avg RAPM**: 0.06
- **Survival Rate**: 63.6%
- **Description**: Characterized by high Three Share
- **Examples**: P5, P26, P34

### Low Rim Fg Pct Type (Archetype 3)

- **Players**: 4
- **Avg RAPM**: 0.07
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Rim Fg Pct
- **Examples**: P2, P27, P219

### Low Three Fg Pct Type (Archetype 4)

- **Players**: 28
- **Avg RAPM**: 0.75
- **Survival Rate**: 60.7%
- **Description**: Characterized by low Three Fg Pct
- **Examples**: P45, P47, P59

### Low Three Share Type (Archetype 5)

- **Players**: 126
- **Avg RAPM**: 0.12
- **Survival Rate**: 68.3%
- **Description**: Characterized by low Three Share
- **Examples**: P1, P4, P7

### High Delta Trueshootingpct Type (Archetype 6)

- **Players**: 10
- **Avg RAPM**: 0.85
- **Survival Rate**: 80.0%
- **Description**: Characterized by high Delta Trueshootingpct
- **Examples**: P63, P140, P154

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
