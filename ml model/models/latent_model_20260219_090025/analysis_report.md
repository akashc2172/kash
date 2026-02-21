# Latent Space Model Analysis Report

**Generated**: 2026-02-19 09:00
**Model**: latent_model_20260219_090025

## Training Results

- **Best Epoch**: 8
- **Best Val Loss**: 5.5709
- **Test RMSE**: 1.6521964073181152
- **Test Correlation**: -0.04512399773690663

## Discovered Archetypes

### Low Season Type (Archetype 0)

- **Players**: 108
- **Avg RAPM**: 0.03
- **Survival Rate**: 74.1%
- **Description**: Characterized by low Season
- **Examples**: Edmond Sumner, Ike Anigbogu, Rodney Purvis

### Low Team Srs Type (Archetype 1)

- **Players**: 8
- **Avg RAPM**: -1.31
- **Survival Rate**: 50.0%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Carldell Johnson, Aaron Jackson

### Low Dev P10 Type (Archetype 2)

- **Players**: 4
- **Avg RAPM**: 0.60
- **Survival Rate**: 75.0%
- **Description**: Characterized by low Dev P10
- **Examples**: Festus Ezeli, DeQuan Jones, Otto Porter Jr.

### High Games Played Type (Archetype 3)

- **Players**: 226
- **Avg RAPM**: 0.40
- **Survival Rate**: 70.4%
- **Description**: Characterized by high Games Played
- **Examples**: Erik McCree, PJ Dozier, Josh Hart

### High Recruiting Rank Type (Archetype 4)

- **Players**: 50
- **Avg RAPM**: 0.40
- **Survival Rate**: 76.0%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Luke Kornet, Antonius Cleveland, OG Anunoby

### High High Lev Att Rate Type (Archetype 5)

- **Players**: 3
- **Avg RAPM**: -0.07
- **Survival Rate**: 66.7%
- **Description**: Characterized by high High Lev Att Rate
- **Examples**: Dion Waiters, Ivan Johnson, Willie Reed

### High Team Rank Type (Archetype 6)

- **Players**: 8
- **Avg RAPM**: 0.52
- **Survival Rate**: 75.0%
- **Description**: Characterized by high Team Rank
- **Examples**: Maurice Harkless, Reggie Bullock Jr., Chris Johnson

### Low Games Played Type (Archetype 7)

- **Players**: 76
- **Avg RAPM**: 0.17
- **Survival Rate**: 67.1%
- **Description**: Characterized by low Games Played
- **Examples**: London Perrantes, Jamel Artis, John Collins

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
