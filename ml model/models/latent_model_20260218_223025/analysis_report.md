# Latent Space Model Analysis Report

**Generated**: 2026-02-18 22:30
**Model**: latent_model_20260218_223025

## Training Results

- **Best Epoch**: 28
- **Best Val Loss**: 1.6043
- **Test RMSE**: 1.619343876838684
- **Test Correlation**: -0.08945660140879624

## Discovered Archetypes

### Low Poss Proxy Type (Archetype 0)

- **Players**: 36
- **Avg RAPM**: 0.11
- **Survival Rate**: 72.2%
- **Description**: Characterized by low Poss Proxy
- **Examples**: Ike Anigbogu, Chris Boucher, Deyonta Davis

### Low Team Srs Type (Archetype 1)

- **Players**: 10
- **Avg RAPM**: -1.05
- **Survival Rate**: 60.0%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Dion Waiters, Carldell Johnson

### High Blk Total Per100Poss Type (Archetype 2)

- **Players**: 2
- **Avg RAPM**: 1.75
- **Survival Rate**: 50.0%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Anthony Davis, Nerlens Noel

### Low Games Played Type (Archetype 3)

- **Players**: 47
- **Avg RAPM**: 0.21
- **Survival Rate**: 61.7%
- **Description**: Characterized by low Games Played
- **Examples**: Jarrett Allen, David Nwaba, Pierre Jackson

### High Recruiting Rank Type (Archetype 4)

- **Players**: 54
- **Avg RAPM**: 0.16
- **Survival Rate**: 75.9%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Luke Kornet, Ben Moore, Antonius Cleveland

### High Blk Total Per100Poss Type (Archetype 5)

- **Players**: 1
- **Avg RAPM**: 1.14
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Jeff Withey

### Low Games Played Type (Archetype 6)

- **Players**: 12
- **Avg RAPM**: 0.85
- **Survival Rate**: 58.3%
- **Description**: Characterized by low Games Played
- **Examples**: Harry Giles III, Michael Kidd-Gilchrist, Terrence Jones

### High Games Played Type (Archetype 7)

- **Players**: 321
- **Avg RAPM**: 0.30
- **Survival Rate**: 72.3%
- **Description**: Characterized by high Games Played
- **Examples**: Erik McCree, PJ Dozier, Josh Hart

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
