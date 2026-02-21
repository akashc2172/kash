# Latent Space Model Analysis Report

**Generated**: 2026-02-20 11:57
**Model**: latent_model_20260220_115737

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 7.0010
- **Test RAPM RMSE**: N/A
- **Test RAPM Correlation**: N/A
- **Test EPM RMSE**: 2.5238993167877197
- **Test EPM Correlation**: -0.051885273741899274

## Discovered Archetypes

### Low Team Srs Type (Archetype 0)

- **Players**: 16
- **Avg RAPM**: -0.61
- **Survival Rate**: 62.5%
- **Description**: Characterized by low Team Srs
- **Examples**: Justin Robinson, Jaylen Morris, Dion Waiters

### High Recruiting Rank Type (Archetype 1)

- **Players**: 129
- **Avg RAPM**: 0.14
- **Survival Rate**: 71.3%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Rui Hachimura, Zach Norvell Jr., Grant Williams

### High Xy Rim Shots Type (Archetype 2)

- **Players**: 34
- **Avg RAPM**: -0.24
- **Survival Rate**: 52.9%
- **Description**: Characterized by high Xy Rim Shots
- **Examples**: Eric Paschall, Bol Bol, Louis King

### Low Games Played Type (Archetype 3)

- **Players**: 67
- **Avg RAPM**: 0.26
- **Survival Rate**: 74.6%
- **Description**: Characterized by low Games Played
- **Examples**: Cameron Johnson, Devin Cannady, Dean Wade

### Low Recruiting Rank Type (Archetype 4)

- **Players**: 284
- **Avg RAPM**: 0.28
- **Survival Rate**: 69.4%
- **Description**: Characterized by low Recruiting Rank
- **Examples**: Brandon Clarke, Nassir Little, John Konchar

### Low Games Played Type (Archetype 5)

- **Players**: 16
- **Avg RAPM**: 1.04
- **Survival Rate**: 62.5%
- **Description**: Characterized by low Games Played
- **Examples**: Maurice Harkless, Andre Drummond, Festus Ezeli

### Low Games Played Type (Archetype 6)

- **Players**: 43
- **Avg RAPM**: 0.09
- **Survival Rate**: 60.5%
- **Description**: Characterized by low Games Played
- **Examples**: Hassani Gravett, Zylan Cheatham, DaQuan Jeffries

### High Minutes Total Type (Archetype 7)

- **Players**: 91
- **Avg RAPM**: 0.31
- **Survival Rate**: 62.6%
- **Description**: Characterized by high Minutes Total
- **Examples**: Devontae Cacok, Marques Bolden, Kevin Porter Jr.

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
