# Latent Space Model Analysis Report

**Generated**: 2026-02-18 20:04
**Model**: latent_model_20260218_200401

## Training Results

- **Best Epoch**: 19
- **Best Val Loss**: 2.3489
- **Test RMSE**: 1.627843976020813
- **Test Correlation**: -0.12284420034356826

## Discovered Archetypes

### Low Dev Quality Weight Type (Archetype 0)

- **Players**: 105
- **Avg RAPM**: 0.58
- **Survival Rate**: 74.3%
- **Description**: Characterized by low Dev Quality Weight
- **Examples**: Ivan Johnson, Dennis Horner, Kyrie Irving

### High Dev Quality Weight Type (Archetype 1)

- **Players**: 77
- **Avg RAPM**: 0.02
- **Survival Rate**: 70.1%
- **Description**: Characterized by high Dev Quality Weight
- **Examples**: John Jenkins, Jeffery Taylor, Tyler Zeller

### Low Team Pace Type (Archetype 2)

- **Players**: 16
- **Avg RAPM**: 0.42
- **Survival Rate**: 75.0%
- **Description**: Characterized by low Team Pace
- **Examples**: Kim English, Festus Ezeli, Terrence Jones

### High Career Years Type (Archetype 3)

- **Players**: 123
- **Avg RAPM**: 0.18
- **Survival Rate**: 63.4%
- **Description**: Characterized by high Career Years
- **Examples**: Greg Smith, Khris Middleton, Chris Babb

### High Breakout Timing Avg Type (Archetype 4)

- **Players**: 129
- **Avg RAPM**: 0.27
- **Survival Rate**: 75.2%
- **Description**: Characterized by high Breakout Timing Avg
- **Examples**: E'Twaun Moore, JaJuan Johnson, MarShon Brooks

### Low Team Pace Type (Archetype 5)

- **Players**: 23
- **Avg RAPM**: 0.08
- **Survival Rate**: 65.2%
- **Description**: Characterized by low Team Pace
- **Examples**: Jordan Hamilton, Fab Melo, Kris Joseph

### Low Rim Fg Pct Type (Archetype 6)

- **Players**: 4
- **Avg RAPM**: 0.66
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Rim Fg Pct
- **Examples**: Reggie Bullock Jr., Ryan Kelly, Jeff Withey

### Low Ft Pct Type (Archetype 7)

- **Players**: 2
- **Avg RAPM**: 1.09
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Ft Pct
- **Examples**: Tyler Johnson, Mitchell Robinson

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
