# Latent Space Model Analysis Report

**Generated**: 2026-02-18 21:56
**Model**: latent_model_20260218_215658

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 3.0563
- **Test RMSE**: 1.5772227048873901
- **Test Correlation**: 0.006772453100631356

## Discovered Archetypes

### High Ast Total Per100Poss Type (Archetype 0)

- **Players**: 8
- **Avg RAPM**: 0.48
- **Survival Rate**: 75.0%
- **Description**: Characterized by high Ast Total Per100Poss
- **Examples**: Darius Johnson-Odom, Kendall Marshall, Reggie Bullock Jr.

### High Games Played Type (Archetype 1)

- **Players**: 250
- **Avg RAPM**: 0.36
- **Survival Rate**: 73.6%
- **Description**: Characterized by high Games Played
- **Examples**: Erik McCree, PJ Dozier, Josh Hart

### High Recruiting Rank Type (Archetype 2)

- **Players**: 34
- **Avg RAPM**: -0.11
- **Survival Rate**: 82.4%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Luke Kornet, Ben Moore, Antonius Cleveland

### Low Team Srs Type (Archetype 3)

- **Players**: 13
- **Avg RAPM**: -0.66
- **Survival Rate**: 61.5%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, DeQuan Jones, Dion Waiters

### High Recruiting Rating Type (Archetype 4)

- **Players**: 66
- **Avg RAPM**: 0.26
- **Survival Rate**: 66.7%
- **Description**: Characterized by high Recruiting Rating
- **Examples**: Edmond Sumner, London Perrantes, Ike Anigbogu

### Low Games Played Type (Archetype 5)

- **Players**: 27
- **Avg RAPM**: 0.20
- **Survival Rate**: 66.7%
- **Description**: Characterized by low Games Played
- **Examples**: Harry Giles III, Pierre Jackson, Solomon Hill

### High Team Rank Type (Archetype 6)

- **Players**: 61
- **Avg RAPM**: 0.13
- **Survival Rate**: 62.3%
- **Description**: Characterized by high Team Rank
- **Examples**: Rodney Purvis, Tyler Cavanaugh, Jarrett Allen

### High Recruiting Rank Type (Archetype 7)

- **Players**: 24
- **Avg RAPM**: 0.53
- **Survival Rate**: 70.8%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Eric Moreland, CJ McCollum, Darius Miller

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
