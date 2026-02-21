# Latent Space Model Analysis Report

**Generated**: 2026-02-18 19:46
**Model**: latent_model_20260218_194603

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 6.3547
- **Test RMSE**: 1.5586228370666504
- **Test Correlation**: 0.03992657040729814

## Discovered Archetypes

### High Fga Total Type (Archetype 0)

- **Players**: 214
- **Avg RAPM**: 0.35
- **Survival Rate**: 72.0%
- **Description**: Characterized by high Fga Total
- **Examples**: Ivan Johnson, Lance Thomas, Kemba Walker

### Low Ft Pct Type (Archetype 1)

- **Players**: 2
- **Avg RAPM**: 1.09
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Ft Pct
- **Examples**: Tyler Johnson, Mitchell Robinson

### High Career Wt Trueshootingpct Type (Archetype 2)

- **Players**: 1
- **Avg RAPM**: -2.47
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Career Wt Trueshootingpct
- **Examples**: Troy Williams

### Low Shots Total Type (Archetype 3)

- **Players**: 13
- **Avg RAPM**: 0.80
- **Survival Rate**: 76.9%
- **Description**: Characterized by low Shots Total
- **Examples**: Festus Ezeli, Kim English, Darius Miller

### Low Ft Pct Type (Archetype 4)

- **Players**: 2
- **Avg RAPM**: 0.06
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Ft Pct
- **Examples**: Reggie Bullock Jr., PJ Hairston

### High Ft Att Type (Archetype 5)

- **Players**: 146
- **Avg RAPM**: 0.28
- **Survival Rate**: 71.9%
- **Description**: Characterized by high Ft Att
- **Examples**: Terrel Harris, Donald Sloan, Jerome Dyson

### Low Ft Att Type (Archetype 6)

- **Players**: 69
- **Avg RAPM**: 0.05
- **Survival Rate**: 62.3%
- **Description**: Characterized by low Ft Att
- **Examples**: Courtney Fortson, Brandon Knight, Kawhi Leonard

### Low Team Pace Type (Archetype 7)

- **Players**: 32
- **Avg RAPM**: 0.14
- **Survival Rate**: 71.9%
- **Description**: Characterized by low Team Pace
- **Examples**: Andre Ingram, Jordan Hamilton, Kendall Marshall

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
