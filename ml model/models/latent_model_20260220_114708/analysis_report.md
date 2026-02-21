# Latent Space Model Analysis Report

**Generated**: 2026-02-20 11:47
**Model**: latent_model_20260220_114708

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 5.8660
- **Test RAPM RMSE**: 1.6442927122116089
- **Test RAPM Correlation**: 0.17137137663088914
- **Test EPM RMSE**: 2.1019954681396484
- **Test EPM Correlation**: -0.10740190335006347

## Discovered Archetypes

### High Team Rank Type (Archetype 0)

- **Players**: 13
- **Avg RAPM**: 0.78
- **Survival Rate**: 46.2%
- **Description**: Characterized by high Team Rank
- **Examples**: Keljin Blevins, Maurice Harkless, Andre Drummond

### High Shot Dist Var Type (Archetype 1)

- **Players**: 59
- **Avg RAPM**: 0.20
- **Survival Rate**: 59.3%
- **Description**: Characterized by high Shot Dist Var
- **Examples**: Zach Norvell Jr., Caleb Martin, Coby White

### High Minutes Total Type (Archetype 2)

- **Players**: 101
- **Avg RAPM**: 0.16
- **Survival Rate**: 64.4%
- **Description**: Characterized by high Minutes Total
- **Examples**: Cameron Johnson, Tacko Fall, Devontae Cacok

### High Recruiting Rank Type (Archetype 3)

- **Players**: 93
- **Avg RAPM**: 0.02
- **Survival Rate**: 71.0%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Grant Williams, Jordan Bone, Admiral Schofield

### High Recruiting Rank Type (Archetype 4)

- **Players**: 52
- **Avg RAPM**: 0.41
- **Survival Rate**: 71.2%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Kyle Alexander, Devin Cannady, Luke Kornet

### Low Team Srs Type (Archetype 5)

- **Players**: 14
- **Avg RAPM**: -0.75
- **Survival Rate**: 71.4%
- **Description**: Characterized by low Team Srs
- **Examples**: Justin Robinson, Jaylen Morris, Dion Waiters

### High Team Rank Type (Archetype 6)

- **Players**: 66
- **Avg RAPM**: -0.00
- **Survival Rate**: 66.7%
- **Description**: Characterized by high Team Rank
- **Examples**: Hassani Gravett, Darius Garland, Charlie Brown Jr.

### Low Minutes Total Type (Archetype 7)

- **Players**: 282
- **Avg RAPM**: 0.34
- **Survival Rate**: 69.9%
- **Description**: Characterized by low Minutes Total
- **Examples**: Rui Hachimura, Brandon Clarke, Cody Martin

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
