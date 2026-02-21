# Latent Space Model Analysis Report

**Generated**: 2026-02-20 20:15
**Model**: latent_model_20260220_201547

## Training Results

- **Best Epoch**: 27
- **Best Val Loss**: 1.8419
- **Test RAPM RMSE**: 1.851879596710205
- **Test RAPM Correlation**: -0.04634068926473902
- **Test EPM RMSE**: 3.329404592514038
- **Test EPM Correlation**: 0.0757419427069383

## Discovered Archetypes

### Low Recruiting Rating Type (Archetype 0)

- **Players**: 95
- **Avg RAPM**: 0.60
- **Survival Rate**: 77.9%
- **Description**: Characterized by low Recruiting Rating
- **Examples**: Milton Doyle, Justin Jackson, Yogi Ferrell

### Low Games Played Type (Archetype 1)

- **Players**: 55
- **Avg RAPM**: 1.14
- **Survival Rate**: 69.1%
- **Description**: Characterized by low Games Played
- **Examples**: Isaiah Hicks, Scott Machado, Kenneth Faried

### Low Dunk Rate Type (Archetype 2)

- **Players**: 17
- **Avg RAPM**: 0.28
- **Survival Rate**: 70.6%
- **Description**: Characterized by low Dunk Rate
- **Examples**: Chris Johnson, Darius Johnson-Odom, Festus Ezeli

### Low Age At Season Type (Archetype 3)

- **Players**: 67
- **Avg RAPM**: 0.90
- **Survival Rate**: 68.7%
- **Description**: Characterized by low Age At Season
- **Examples**: David Nwaba, Amile Jefferson, Luke Kornet

### High Recruiting Rating Type (Archetype 4)

- **Players**: 90
- **Avg RAPM**: 0.48
- **Survival Rate**: 75.6%
- **Description**: Characterized by high Recruiting Rating
- **Examples**: Alec Peters, Kay Felder, Frank Mason III

### High Fga Total Type (Archetype 5)

- **Players**: 121
- **Avg RAPM**: 0.28
- **Survival Rate**: 62.8%
- **Description**: Characterized by high Fga Total
- **Examples**: Joe Young, Buddy Hield, Shawn Long

### Low Usage Z Type (Archetype 6)

- **Players**: 36
- **Avg RAPM**: 0.88
- **Survival Rate**: 63.9%
- **Description**: Characterized by low Usage Z
- **Examples**: Mike Muscala, Isaiah Canaan, Elias Harris

### Low Games Played Type (Archetype 7)

- **Players**: 15
- **Avg RAPM**: 0.14
- **Survival Rate**: 86.7%
- **Description**: Characterized by low Games Played
- **Examples**: Josh Richardson, Derrick Williams, Solomon Hill

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
