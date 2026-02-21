# Latent Space Model Analysis Report

**Generated**: 2026-02-20 11:43
**Model**: latent_model_20260220_114321

## Training Results

- **Best Epoch**: 22
- **Best Val Loss**: 3.9085
- **Test RAPM RMSE**: 1.6318564414978027
- **Test RAPM Correlation**: 0.04799813693650293
- **Test EPM RMSE**: 1.6671779155731201
- **Test EPM Correlation**: 0.019272057495421227

## Discovered Archetypes

### High Games Played Type (Archetype 0)

- **Players**: 318
- **Avg RAPM**: 0.30
- **Survival Rate**: 71.4%
- **Description**: Characterized by high Games Played
- **Examples**: Erik McCree, PJ Dozier, Josh Hart

### High Recruiting Rank Type (Archetype 1)

- **Players**: 17
- **Avg RAPM**: -0.07
- **Survival Rate**: 70.6%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Briante Weber, Shayne Whittington, Eric Moreland

### Low Team Srs Type (Archetype 2)

- **Players**: 8
- **Avg RAPM**: -1.34
- **Survival Rate**: 62.5%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Aaron Jackson, C.J. Wilcox

### High Team Rank Type (Archetype 3)

- **Players**: 78
- **Avg RAPM**: 0.14
- **Survival Rate**: 67.9%
- **Description**: Characterized by high Team Rank
- **Examples**: Rodney Purvis, Tyler Cavanaugh, Jarrett Allen

### High Recruiting Rank Type (Archetype 4)

- **Players**: 11
- **Avg RAPM**: 0.69
- **Survival Rate**: 81.8%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: CJ McCollum, Tim Hardaway Jr., Kim English

### High Team Rank Type (Archetype 5)

- **Players**: 11
- **Avg RAPM**: 0.88
- **Survival Rate**: 63.6%
- **Description**: Characterized by high Team Rank
- **Examples**: Maurice Harkless, DeQuan Jones, Reggie Bullock Jr.

### High Blk Total Per100Poss Type (Archetype 6)

- **Players**: 9
- **Avg RAPM**: 1.10
- **Survival Rate**: 88.9%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Anthony Davis, Darius Miller, Trey Burke

### Low Team Pace Type (Archetype 7)

- **Players**: 44
- **Avg RAPM**: -0.07
- **Survival Rate**: 63.6%
- **Description**: Characterized by low Team Pace
- **Examples**: Ike Anigbogu, Harry Giles III, OG Anunoby

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
