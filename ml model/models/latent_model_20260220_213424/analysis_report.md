# Latent Space Model Analysis Report

**Generated**: 2026-02-20 21:34
**Model**: latent_model_20260220_213424

## Training Results

- **Best Epoch**: 22
- **Best Val Loss**: 1.8653
- **Test RAPM RMSE**: 1.8693963289260864
- **Test RAPM Correlation**: -0.0034139623552818152
- **Test EPM RMSE**: 3.3279857635498047
- **Test EPM Correlation**: -0.18678562417945763

## Discovered Archetypes

### High Ft Att Type (Archetype 0)

- **Players**: 66
- **Avg RAPM**: 0.57
- **Survival Rate**: 66.7%
- **Description**: Characterized by high Ft Att
- **Examples**: Milton Doyle, Tyrone Wallace, Charles Cooke

### High Recruiting Rating Type (Archetype 1)

- **Players**: 76
- **Avg RAPM**: 0.43
- **Survival Rate**: 67.1%
- **Description**: Characterized by high Recruiting Rating
- **Examples**: Justin Jackson, Dillon Brooks, Marcus Paige

### Low Fga Total Type (Archetype 2)

- **Players**: 46
- **Avg RAPM**: 0.95
- **Survival Rate**: 67.4%
- **Description**: Characterized by low Fga Total
- **Examples**: David Nwaba, Keith Benson, Kent Bazemore

### High Recruiting Rank Type (Archetype 3)

- **Players**: 92
- **Avg RAPM**: 0.45
- **Survival Rate**: 81.5%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Alec Peters, Kay Felder, Frank Mason III

### High Fga Total Type (Archetype 4)

- **Players**: 85
- **Avg RAPM**: 0.51
- **Survival Rate**: 72.9%
- **Description**: Characterized by high Fga Total
- **Examples**: Joe Young, Buddy Hield, Sindarius Thornwell

### High Rim Pressure Index Type (Archetype 5)

- **Players**: 38
- **Avg RAPM**: 0.93
- **Survival Rate**: 73.7%
- **Description**: Characterized by high Rim Pressure Index
- **Examples**: Shawn Long, Tyler Cavanaugh, Elfrid Payton

### Low Poss Proxy Type (Archetype 6)

- **Players**: 47
- **Avg RAPM**: 0.76
- **Survival Rate**: 59.6%
- **Description**: Characterized by low Poss Proxy
- **Examples**: Elias Harris, Robert Covington, Quincy Acy

### Low Recruiting Rating Type (Archetype 7)

- **Players**: 46
- **Avg RAPM**: 0.58
- **Survival Rate**: 67.4%
- **Description**: Characterized by low Recruiting Rating
- **Examples**: Rodney Purvis, Charles Jenkins, Scott Machado

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
