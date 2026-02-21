# Latent Space Model Analysis Report

**Generated**: 2026-02-18 20:02
**Model**: latent_model_20260218_200209

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 5.4234
- **Test RMSE**: 1.6288245916366577
- **Test Correlation**: 0.12871246457333615

## Discovered Archetypes

### High Ft Att Type (Archetype 0)

- **Players**: 146
- **Avg RAPM**: 0.21
- **Survival Rate**: 76.0%
- **Description**: Characterized by high Ft Att
- **Examples**: Dennis Horner, Cory Higgins, Donald Sloan

### Low Ft Att Type (Archetype 1)

- **Players**: 71
- **Avg RAPM**: -0.04
- **Survival Rate**: 62.0%
- **Description**: Characterized by low Ft Att
- **Examples**: E'Twaun Moore, Mychel Thompson, Brandon Knight

### Low Three Share Z Type (Archetype 2)

- **Players**: 69
- **Avg RAPM**: 0.52
- **Survival Rate**: 62.3%
- **Description**: Characterized by low Three Share Z
- **Examples**: Jordan Williams, Jimmy Butler III, Kyrie Irving

### High Career Wt Trueshootingpct Type (Archetype 3)

- **Players**: 1
- **Avg RAPM**: -2.47
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Career Wt Trueshootingpct
- **Examples**: Troy Williams

### High Fga Total Type (Archetype 4)

- **Players**: 144
- **Avg RAPM**: 0.33
- **Survival Rate**: 70.8%
- **Description**: Characterized by high Fga Total
- **Examples**: Ivan Johnson, JaJuan Johnson, MarShon Brooks

### Low Fga Total Type (Archetype 5)

- **Players**: 13
- **Avg RAPM**: 0.48
- **Survival Rate**: 84.6%
- **Description**: Characterized by low Fga Total
- **Examples**: Derrick Williams, Michael Kidd-Gilchrist, Dion Waiters

### Low Ft Pct Type (Archetype 6)

- **Players**: 4
- **Avg RAPM**: 1.13
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Ft Pct
- **Examples**: Reggie Bullock Jr., PJ Hairston, Mitchell Robinson

### Low Team Pace Type (Archetype 7)

- **Players**: 31
- **Avg RAPM**: 0.44
- **Survival Rate**: 77.4%
- **Description**: Characterized by low Team Pace
- **Examples**: Jordan Hamilton, Fab Melo, Kris Joseph

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
