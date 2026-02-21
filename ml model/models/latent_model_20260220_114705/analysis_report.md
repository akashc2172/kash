# Latent Space Model Analysis Report

**Generated**: 2026-02-20 11:47
**Model**: latent_model_20260220_114705

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 7.1860
- **Test RAPM RMSE**: 1.569963812828064
- **Test RAPM Correlation**: 0.18509215449003644
- **Test EPM RMSE**: 2.490018606185913
- **Test EPM Correlation**: -0.08855848907103744

## Discovered Archetypes

### High Team Rank Type (Archetype 0)

- **Players**: 64
- **Avg RAPM**: 0.16
- **Survival Rate**: 64.1%
- **Description**: Characterized by high Team Rank
- **Examples**: Rodney Purvis, Tyler Cavanaugh, Jarrett Allen

### High Recruiting Rank Type (Archetype 1)

- **Players**: 54
- **Avg RAPM**: 0.37
- **Survival Rate**: 74.1%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Luke Kornet, OG Anunoby, T.J. McConnell

### High Minutes Total Type (Archetype 2)

- **Players**: 93
- **Avg RAPM**: 0.18
- **Survival Rate**: 65.6%
- **Description**: Characterized by high Minutes Total
- **Examples**: Marvin Bagley III, Wendell Carter Jr., Gary Trent Jr.

### High Games Played Type (Archetype 3)

- **Players**: 275
- **Avg RAPM**: 0.32
- **Survival Rate**: 70.9%
- **Description**: Characterized by high Games Played
- **Examples**: Erik McCree, PJ Dozier, Josh Hart

### Low Games Played Type (Archetype 4)

- **Players**: 18
- **Avg RAPM**: 0.28
- **Survival Rate**: 61.1%
- **Description**: Characterized by low Games Played
- **Examples**: Harry Giles III, Anthony Davis, Michael Kidd-Gilchrist

### High Recruiting Rank Type (Archetype 5)

- **Players**: 56
- **Avg RAPM**: 0.12
- **Survival Rate**: 73.2%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Michael Porter Jr., Edmond Sumner, London Perrantes

### Low Team Srs Type (Archetype 6)

- **Players**: 18
- **Avg RAPM**: -0.25
- **Survival Rate**: 66.7%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Maurice Harkless, Festus Ezeli

### High Ast Total Per100Poss Type (Archetype 7)

- **Players**: 6
- **Avg RAPM**: 0.65
- **Survival Rate**: 66.7%
- **Description**: Characterized by high Ast Total Per100Poss
- **Examples**: Darius Johnson-Odom, Kendall Marshall, Jeff Withey

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
