# Latent Space Model Analysis Report

**Generated**: 2026-02-19 09:05
**Model**: latent_model_20260219_090509

## Training Results

- **Best Epoch**: 2
- **Best Val Loss**: 5.6659
- **Test RMSE**: 1.6752578020095825
- **Test Correlation**: -0.06015248349773766

## Discovered Archetypes

### Low Recruiting Rank Type (Archetype 0)

- **Players**: 314
- **Avg RAPM**: 0.29
- **Survival Rate**: 70.7%
- **Description**: Characterized by low Recruiting Rank
- **Examples**: Erik McCree, PJ Dozier, Josh Hart

### High Tov Total Per100Poss Type (Archetype 1)

- **Players**: 8
- **Avg RAPM**: 0.48
- **Survival Rate**: 75.0%
- **Description**: Characterized by high Tov Total Per100Poss
- **Examples**: DeQuan Jones, Chris Johnson, Dion Waiters

### High Recruiting Rank Type (Archetype 2)

- **Players**: 24
- **Avg RAPM**: 0.79
- **Survival Rate**: 70.8%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Shayne Whittington, Eric Moreland, Maurice Harkless

### Low Team Srs Type (Archetype 3)

- **Players**: 8
- **Avg RAPM**: -1.29
- **Survival Rate**: 50.0%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Ivan Johnson, Aaron Jackson

### High Recruiting Rank Type (Archetype 4)

- **Players**: 58
- **Avg RAPM**: 0.31
- **Survival Rate**: 75.9%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Edmond Sumner, London Perrantes, Ike Anigbogu

### Low Games Played Type (Archetype 5)

- **Players**: 46
- **Avg RAPM**: 0.14
- **Survival Rate**: 63.0%
- **Description**: Characterized by low Games Played
- **Examples**: Jarrett Allen, David Nwaba, Andre Drummond

### High Minutes Total Type (Archetype 6)

- **Players**: 4
- **Avg RAPM**: -1.75
- **Survival Rate**: 75.0%
- **Description**: Characterized by high Minutes Total
- **Examples**: Jalen Jones, Derrick Williams, Henry Sims

### High Recruiting Rank Type (Archetype 7)

- **Players**: 21
- **Avg RAPM**: 0.18
- **Survival Rate**: 85.7%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Luke Kornet, Antonius Cleveland, OG Anunoby

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
