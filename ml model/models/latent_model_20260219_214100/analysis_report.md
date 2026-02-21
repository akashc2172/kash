# Latent Space Model Analysis Report

**Generated**: 2026-02-19 21:41
**Model**: latent_model_20260219_214100

## Training Results

- **Best Epoch**: 3
- **Best Val Loss**: 5.6149
- **Test RMSE**: 1.6414940357208252
- **Test Correlation**: -0.07144271176019583

## Discovered Archetypes

### High Recruiting Rating Type (Archetype 0)

- **Players**: 116
- **Avg RAPM**: 0.09
- **Survival Rate**: 71.6%
- **Description**: Characterized by high Recruiting Rating
- **Examples**: Edmond Sumner, London Perrantes, Donovan Mitchell

### Low Team Srs Type (Archetype 1)

- **Players**: 7
- **Avg RAPM**: -1.42
- **Survival Rate**: 42.9%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Aaron Jackson, Jordan Loyd

### High Games Played Type (Archetype 2)

- **Players**: 270
- **Avg RAPM**: 0.33
- **Survival Rate**: 70.4%
- **Description**: Characterized by high Games Played
- **Examples**: Erik McCree, PJ Dozier, Josh Hart

### High Tov Total Per100Poss Type (Archetype 3)

- **Players**: 4
- **Avg RAPM**: -0.18
- **Survival Rate**: 75.0%
- **Description**: Characterized by high Tov Total Per100Poss
- **Examples**: Dion Waiters, Carldell Johnson, Ivan Johnson

### Low Games Played Type (Archetype 4)

- **Players**: 7
- **Avg RAPM**: 1.42
- **Survival Rate**: 71.4%
- **Description**: Characterized by low Games Played
- **Examples**: Maurice Harkless, DeQuan Jones, Reggie Bullock Jr.

### Low Shots Total Type (Archetype 5)

- **Players**: 33
- **Avg RAPM**: 0.33
- **Survival Rate**: 72.7%
- **Description**: Characterized by low Shots Total
- **Examples**: Harry Giles III, Cheick Diallo, Anthony Davis

### High Recruiting Rank Type (Archetype 6)

- **Players**: 44
- **Avg RAPM**: 0.30
- **Survival Rate**: 75.0%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Luke Kornet, Antonius Cleveland, OG Anunoby

### High Blk Total Per100Poss Type (Archetype 7)

- **Players**: 2
- **Avg RAPM**: 0.26
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Festus Ezeli, Jeff Withey

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
