# Latent Space Model Analysis Report

**Generated**: 2026-02-20 19:17
**Model**: latent_model_20260220_191735

## Training Results

- **Best Epoch**: 6
- **Best Val Loss**: 1.7189
- **Test RAPM RMSE**: 1.5980772972106934
- **Test RAPM Correlation**: 0.018862578135096657
- **Test EPM RMSE**: 2.8972315788269043
- **Test EPM Correlation**: -0.134493919431357

## Discovered Archetypes

### High Team Rank Type (Archetype 0)

- **Players**: 48
- **Avg RAPM**: -0.22
- **Survival Rate**: 62.5%
- **Description**: Characterized by high Team Rank
- **Examples**: Tyler Cavanaugh, Rodney Purvis, David Nwaba

### Low Team Srs Type (Archetype 1)

- **Players**: 8
- **Avg RAPM**: -1.35
- **Survival Rate**: 62.5%
- **Description**: Characterized by low Team Srs
- **Examples**: Ivan Johnson, Jaylen Morris, Willie Reed

### High Recruiting Rank Type (Archetype 2)

- **Players**: 29
- **Avg RAPM**: 0.77
- **Survival Rate**: 79.3%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Scott Machado, Kenneth Faried, CJ McCollum

### High Team Rank Type (Archetype 3)

- **Players**: 6
- **Avg RAPM**: 1.88
- **Survival Rate**: 50.0%
- **Description**: Characterized by high Team Rank
- **Examples**: Kentavious Caldwell-Pope, James Ennis III, Reggie Bullock Jr.

### Low Team Rank Type (Archetype 4)

- **Players**: 131
- **Avg RAPM**: 0.39
- **Survival Rate**: 75.6%
- **Description**: Characterized by low Team Rank
- **Examples**: Kay Felder, Justin Jackson, Devyn Marble

### Low Dunk Rate Type (Archetype 5)

- **Players**: 2
- **Avg RAPM**: -1.58
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Dunk Rate
- **Examples**: Festus Ezeli, DeQuan Jones

### High Recruiting Rank Type (Archetype 6)

- **Players**: 60
- **Avg RAPM**: 0.19
- **Survival Rate**: 71.7%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Antonius Cleveland, Mike Muscala, Isaiah Taylor

### High Fga Total Type (Archetype 7)

- **Players**: 212
- **Avg RAPM**: 0.22
- **Survival Rate**: 68.4%
- **Description**: Characterized by high Fga Total
- **Examples**: Alec Peters, Joe Young, Buddy Hield

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
