# Latent Space Model Analysis Report

**Generated**: 2026-02-20 11:27
**Model**: latent_model_20260220_112718

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 6.7426
- **Test RMSE**: 1.6826329231262207
- **Test Correlation**: 0.06702711191024174

## Discovered Archetypes

### High Team Rank Type (Archetype 0)

- **Players**: 81
- **Avg RAPM**: 0.25
- **Survival Rate**: 67.9%
- **Description**: Characterized by high Team Rank
- **Examples**: Rodney Purvis, Tyler Cavanaugh, Jarrett Allen

### High Games Played Type (Archetype 1)

- **Players**: 239
- **Avg RAPM**: 0.35
- **Survival Rate**: 69.9%
- **Description**: Characterized by high Games Played
- **Examples**: Erik McCree, PJ Dozier, Josh Hart

### High Recruiting Rank Type (Archetype 2)

- **Players**: 48
- **Avg RAPM**: 0.13
- **Survival Rate**: 75.0%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Luke Kornet, Ben Moore, Antonius Cleveland

### Low Team Srs Type (Archetype 3)

- **Players**: 11
- **Avg RAPM**: -0.99
- **Survival Rate**: 63.6%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Dion Waiters, Ivan Johnson

### High Recruiting Rating Type (Archetype 4)

- **Players**: 77
- **Avg RAPM**: 0.08
- **Survival Rate**: 70.1%
- **Description**: Characterized by high Recruiting Rating
- **Examples**: Edmond Sumner, London Perrantes, Ike Anigbogu

### High Recruiting Rank Type (Archetype 5)

- **Players**: 20
- **Avg RAPM**: 0.69
- **Survival Rate**: 75.0%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Eric Moreland, CJ McCollum, Darius Miller

### High On Drtg Type (Archetype 6)

- **Players**: 8
- **Avg RAPM**: -0.17
- **Survival Rate**: 87.5%
- **Description**: Characterized by high On Drtg
- **Examples**: Donovan Mitchell, Isaiah Hicks, Jalen Jones

### High Blk Total Per100Poss Type (Archetype 7)

- **Players**: 12
- **Avg RAPM**: 0.17
- **Survival Rate**: 66.7%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Anthony Davis, Terrence Jones, Andre Drummond

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
