# Latent Space Model Analysis Report

**Generated**: 2026-02-20 20:29
**Model**: latent_model_20260220_202949

## Training Results

- **Best Epoch**: 40
- **Best Val Loss**: 1.8517
- **Test RAPM RMSE**: 1.9038087129592896
- **Test RAPM Correlation**: 0.007504667407818189
- **Test EPM RMSE**: 3.18770432472229
- **Test EPM Correlation**: 0.1265136171702848

## Discovered Archetypes

### Low Recruiting Rating Type (Archetype 0)

- **Players**: 69
- **Avg RAPM**: 0.39
- **Survival Rate**: 66.7%
- **Description**: Characterized by low Recruiting Rating
- **Examples**: Shawn Long, Milton Doyle, Tyrone Wallace

### High Recruiting Rating Type (Archetype 1)

- **Players**: 54
- **Avg RAPM**: 0.31
- **Survival Rate**: 70.4%
- **Description**: Characterized by high Recruiting Rating
- **Examples**: Josh Hart, Dillon Brooks, Antonius Cleveland

### Low Fga Total Type (Archetype 2)

- **Players**: 87
- **Avg RAPM**: 0.99
- **Survival Rate**: 60.9%
- **Description**: Characterized by low Fga Total
- **Examples**: David Nwaba, Scott Machado, Elias Harris

### High Team Pace Type (Archetype 3)

- **Players**: 60
- **Avg RAPM**: 0.47
- **Survival Rate**: 71.7%
- **Description**: Characterized by high Team Pace
- **Examples**: Sindarius Thornwell, Yogi Ferrell, Marcus Paige

### High Recruiting Rank Type (Archetype 4)

- **Players**: 88
- **Avg RAPM**: 0.46
- **Survival Rate**: 78.4%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Alec Peters, Kay Felder, Frank Mason III

### High Ast Total Per100Poss Type (Archetype 5)

- **Players**: 6
- **Avg RAPM**: 0.05
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Ast Total Per100Poss
- **Examples**: Peyton Siva, Ryan Kelly, Trey Burke

### High Fga Total Type (Archetype 6)

- **Players**: 58
- **Avg RAPM**: 0.51
- **Survival Rate**: 72.4%
- **Description**: Characterized by high Fga Total
- **Examples**: Joe Young, Buddy Hield, Georges Niang

### High Rim Pressure Index Type (Archetype 7)

- **Players**: 74
- **Avg RAPM**: 0.91
- **Survival Rate**: 71.6%
- **Description**: Characterized by high Rim Pressure Index
- **Examples**: Fred VanVleet, Larry Nance Jr., Jake Layman

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
