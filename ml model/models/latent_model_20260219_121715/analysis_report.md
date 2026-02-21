# Latent Space Model Analysis Report

**Generated**: 2026-02-19 12:17
**Model**: latent_model_20260219_121715

## Training Results

- **Best Epoch**: 1
- **Best Val Loss**: 5.7150
- **Test RMSE**: 1.7214654684066772
- **Test Correlation**: -0.011478925688595893

## Discovered Archetypes

### High Ast Total Per100Poss Type (Archetype 0)

- **Players**: 10
- **Avg RAPM**: 0.87
- **Survival Rate**: 70.0%
- **Description**: Characterized by high Ast Total Per100Poss
- **Examples**: Anthony Davis, Darius Johnson-Odom, Kendall Marshall

### Low Fga Total Type (Archetype 1)

- **Players**: 122
- **Avg RAPM**: 0.20
- **Survival Rate**: 68.9%
- **Description**: Characterized by low Fga Total
- **Examples**: Edmond Sumner, London Perrantes, Zach Collins

### Low Team Srs Type (Archetype 2)

- **Players**: 8
- **Avg RAPM**: -1.29
- **Survival Rate**: 50.0%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Ivan Johnson, Aaron Jackson

### High Recruiting Rank Type (Archetype 3)

- **Players**: 49
- **Avg RAPM**: 0.30
- **Survival Rate**: 77.6%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Luke Kornet, Ben Moore, Antonius Cleveland

### High Minutes Total Type (Archetype 4)

- **Players**: 4
- **Avg RAPM**: -1.43
- **Survival Rate**: 75.0%
- **Description**: Characterized by high Minutes Total
- **Examples**: Jalen Jones, Carldell Johnson, Henry Sims

### High Fga Total Type (Archetype 5)

- **Players**: 252
- **Avg RAPM**: 0.30
- **Survival Rate**: 71.0%
- **Description**: Characterized by high Fga Total
- **Examples**: Erik McCree, PJ Dozier, Josh Hart

### High Recruiting Rank Type (Archetype 6)

- **Players**: 21
- **Avg RAPM**: 0.47
- **Survival Rate**: 81.0%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Briante Weber, Eric Moreland, CJ McCollum

### Low Shots Total Type (Archetype 7)

- **Players**: 17
- **Avg RAPM**: 0.44
- **Survival Rate**: 64.7%
- **Description**: Characterized by low Shots Total
- **Examples**: Maurice Harkless, Andre Drummond, Festus Ezeli

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
