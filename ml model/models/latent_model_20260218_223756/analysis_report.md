# Latent Space Model Analysis Report

**Generated**: 2026-02-18 22:37
**Model**: latent_model_20260218_223756

## Training Results

- **Best Epoch**: 32
- **Best Val Loss**: 3.4432
- **Test RMSE**: 1.6233816146850586
- **Test Correlation**: 0.020868845412439968

## Discovered Archetypes

### Low Games Played Type (Archetype 0)

- **Players**: 45
- **Avg RAPM**: -0.26
- **Survival Rate**: 64.4%
- **Description**: Characterized by low Games Played
- **Examples**: Rodney Purvis, Jarrett Allen, Antonio Blakeney

### High Shots Total Type (Archetype 1)

- **Players**: 93
- **Avg RAPM**: 0.14
- **Survival Rate**: 67.7%
- **Description**: Characterized by high Shots Total
- **Examples**: Erik McCree, Mangok Mathiang, Zach Collins

### High Recruiting Rank Type (Archetype 2)

- **Players**: 123
- **Avg RAPM**: 0.09
- **Survival Rate**: 71.5%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Edmond Sumner, London Perrantes, Ike Anigbogu

### Low Team Srs Type (Archetype 3)

- **Players**: 22
- **Avg RAPM**: 0.09
- **Survival Rate**: 63.6%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Maurice Harkless, Andre Drummond

### High Ft Att Type (Archetype 4)

- **Players**: 68
- **Avg RAPM**: 1.14
- **Survival Rate**: 76.5%
- **Description**: Characterized by high Ft Att
- **Examples**: Amile Jefferson, Jayson Tatum, Bam Adebayo

### High Blk Total Per100Poss Type (Archetype 5)

- **Players**: 10
- **Avg RAPM**: 1.45
- **Survival Rate**: 60.0%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Anthony Davis, Michael Kidd-Gilchrist, Terrence Jones

### High Fga Total Type (Archetype 6)

- **Players**: 107
- **Avg RAPM**: 0.06
- **Survival Rate**: 72.9%
- **Description**: Characterized by high Fga Total
- **Examples**: PJ Dozier, Josh Hart, Donovan Mitchell

### High Recruiting Rank Type (Archetype 7)

- **Players**: 15
- **Avg RAPM**: 0.69
- **Survival Rate**: 86.7%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: CJ McCollum, Darius Miller, Tim Hardaway Jr.

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
