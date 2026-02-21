# Latent Space Model Analysis Report

**Generated**: 2026-02-18 19:46
**Model**: latent_model_20260218_194613

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 5.3213
- **Test RMSE**: 1.599914789199829
- **Test Correlation**: -0.03495184777195839

## Discovered Archetypes

### High Ft Att Type (Archetype 0)

- **Players**: 207
- **Avg RAPM**: 0.29
- **Survival Rate**: 76.8%
- **Description**: Characterized by high Ft Att
- **Examples**: Terrel Harris, Donald Sloan, Jerome Dyson

### High Transfer Event Count Type (Archetype 1)

- **Players**: 47
- **Avg RAPM**: 0.20
- **Survival Rate**: 68.1%
- **Description**: Characterized by high Transfer Event Count
- **Examples**: Jordan Hamilton, Julyan Stone, Terrence Ross

### Low Ft Pct Type (Archetype 2)

- **Players**: 2
- **Avg RAPM**: 1.09
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Ft Pct
- **Examples**: Tyler Johnson, Mitchell Robinson

### High Three Share Z Type (Archetype 3)

- **Players**: 132
- **Avg RAPM**: 0.07
- **Survival Rate**: 63.6%
- **Description**: Characterized by high Three Share Z
- **Examples**: Ivan Johnson, Courtney Fortson, Brandon Knight

### Low Ft Pct Type (Archetype 4)

- **Players**: 6
- **Avg RAPM**: 0.23
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Ft Pct
- **Examples**: Festus Ezeli, Doron Lamb, Reggie Bullock Jr.

### Low Three Share Z Type (Archetype 5)

- **Players**: 46
- **Avg RAPM**: 0.86
- **Survival Rate**: 63.0%
- **Description**: Characterized by low Three Share Z
- **Examples**: Kyrie Irving, Derrick Williams, Tristan Thompson

### High Career Wt Trueshootingpct Type (Archetype 6)

- **Players**: 1
- **Avg RAPM**: -2.47
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Career Wt Trueshootingpct
- **Examples**: Troy Williams

### Low Fga Total Type (Archetype 7)

- **Players**: 38
- **Avg RAPM**: 0.32
- **Survival Rate**: 71.1%
- **Description**: Characterized by low Fga Total
- **Examples**: Andre Ingram, Kendall Marshall, Terrence Jones

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
