# Latent Space Model Analysis Report

**Generated**: 2026-02-20 12:18
**Model**: latent_model_20260220_121821

## Training Results

- **Best Epoch**: 9
- **Best Val Loss**: 1.7008
- **Test RAPM RMSE**: 1.6157532930374146
- **Test RAPM Correlation**: -0.012369875266752779
- **Test EPM RMSE**: 1.6895054578781128
- **Test EPM Correlation**: -0.13264805930053508

## Discovered Archetypes

### Low Games Played Type (Archetype 0)

- **Players**: 29
- **Avg RAPM**: 0.57
- **Survival Rate**: 72.4%
- **Description**: Characterized by low Games Played
- **Examples**: Ike Anigbogu, Harry Giles III, Anthony Davis

### High Breakout Rank Volume Type (Archetype 1)

- **Players**: 130
- **Avg RAPM**: -0.05
- **Survival Rate**: 61.5%
- **Description**: Characterized by high Breakout Rank Volume
- **Examples**: Erik McCree, Josh Hart, Donovan Mitchell

### High Recruiting Rank Type (Archetype 2)

- **Players**: 57
- **Avg RAPM**: 0.28
- **Survival Rate**: 70.2%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Luke Kornet, Ben Moore, Antonius Cleveland

### High Dev Quality Weight Type (Archetype 3)

- **Players**: 100
- **Avg RAPM**: 0.17
- **Survival Rate**: 77.0%
- **Description**: Characterized by high Dev Quality Weight
- **Examples**: PJ Dozier, Mangok Mathiang, Caleb Swanigan

### High Age At Season Type (Archetype 4)

- **Players**: 60
- **Avg RAPM**: 0.17
- **Survival Rate**: 71.7%
- **Description**: Characterized by high Age At Season
- **Examples**: Edmond Sumner, London Perrantes, Rodney Purvis

### Low Team Srs Type (Archetype 5)

- **Players**: 20
- **Avg RAPM**: -0.36
- **Survival Rate**: 60.0%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Maurice Harkless, DeQuan Jones

### Low Breakout Timing Usage Type (Archetype 6)

- **Players**: 89
- **Avg RAPM**: 0.88
- **Survival Rate**: 74.2%
- **Description**: Characterized by low Breakout Timing Usage
- **Examples**: Zach Collins, T.J. Leaf, Lonzo Ball

### Low Team Pace Type (Archetype 7)

- **Players**: 11
- **Avg RAPM**: -0.56
- **Survival Rate**: 90.9%
- **Description**: Characterized by low Team Pace
- **Examples**: Jalen Jones, Darius Miller, Gorgui Dieng

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
