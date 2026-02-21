# Latent Space Model Analysis Report

**Generated**: 2026-02-18 19:55
**Model**: latent_model_20260218_195548

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 5.8443
- **Test RMSE**: 1.602468490600586
- **Test Correlation**: 0.0731617261366817

## Discovered Archetypes

### Low Ft Att Type (Archetype 0)

- **Players**: 128
- **Avg RAPM**: 0.03
- **Survival Rate**: 62.5%
- **Description**: Characterized by low Ft Att
- **Examples**: Ivan Johnson, E'Twaun Moore, Mychel Thompson

### Low Team Pace Type (Archetype 1)

- **Players**: 19
- **Avg RAPM**: 0.59
- **Survival Rate**: 78.9%
- **Description**: Characterized by low Team Pace
- **Examples**: Jordan Hamilton, Fab Melo, Jae Crowder

### Low Three Share Z Type (Archetype 2)

- **Players**: 79
- **Avg RAPM**: 0.51
- **Survival Rate**: 63.3%
- **Description**: Characterized by low Three Share Z
- **Examples**: Jimmy Butler III, Kyrie Irving, Tristan Thompson

### Low Ft Pct Type (Archetype 3)

- **Players**: 3
- **Avg RAPM**: 1.31
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Ft Pct
- **Examples**: Tyler Johnson, Mitchell Robinson, Cameron Johnson

### Low Ft Pct Type (Archetype 4)

- **Players**: 2
- **Avg RAPM**: 0.06
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Ft Pct
- **Examples**: Reggie Bullock Jr., PJ Hairston

### High Transfer Mean Shock Type (Archetype 5)

- **Players**: 17
- **Avg RAPM**: -0.29
- **Survival Rate**: 64.7%
- **Description**: Characterized by high Transfer Mean Shock
- **Examples**: Larry Drew II, Rodney Hood, Bryce Dejean-Jones

### High Career Wt Trueshootingpct Type (Archetype 6)

- **Players**: 1
- **Avg RAPM**: -2.47
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Career Wt Trueshootingpct
- **Examples**: Troy Williams

### High Shots Total Type (Archetype 7)

- **Players**: 230
- **Avg RAPM**: 0.35
- **Survival Rate**: 77.4%
- **Description**: Characterized by high Shots Total
- **Examples**: JaJuan Johnson, MarShon Brooks, Dennis Horner

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
