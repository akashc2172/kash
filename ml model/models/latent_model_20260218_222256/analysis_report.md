# Latent Space Model Analysis Report

**Generated**: 2026-02-18 22:22
**Model**: latent_model_20260218_222256

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 3.3834
- **Test RMSE**: 1.5936191082000732
- **Test Correlation**: 0.03057278624706825

## Discovered Archetypes

### Low Team Srs Type (Archetype 0)

- **Players**: 17
- **Avg RAPM**: -0.45
- **Survival Rate**: 58.8%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Darius Johnson-Odom, Shane Larkin

### High Career Wt Usage Type (Archetype 1)

- **Players**: 69
- **Avg RAPM**: 0.13
- **Survival Rate**: 75.4%
- **Description**: Characterized by high Career Wt Usage
- **Examples**: Tyler Cavanaugh, Dennis Smith Jr., Derrick White

### High Recruiting Rank Type (Archetype 2)

- **Players**: 49
- **Avg RAPM**: 0.35
- **Survival Rate**: 77.6%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Luke Kornet, Antonius Cleveland, Josh Richardson

### High Games Played Type (Archetype 3)

- **Players**: 198
- **Avg RAPM**: 0.39
- **Survival Rate**: 70.7%
- **Description**: Characterized by high Games Played
- **Examples**: Erik McCree, PJ Dozier, Mangok Mathiang

### High Team Rank Type (Archetype 4)

- **Players**: 39
- **Avg RAPM**: -0.03
- **Survival Rate**: 61.5%
- **Description**: Characterized by high Team Rank
- **Examples**: Rodney Purvis, Jarrett Allen, Antonio Blakeney

### High Recruiting Rank Type (Archetype 5)

- **Players**: 35
- **Avg RAPM**: 0.33
- **Survival Rate**: 65.7%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Jamel Artis, John Collins, Ben Moore

### High Breakout Rank Volume Type (Archetype 6)

- **Players**: 51
- **Avg RAPM**: 0.16
- **Survival Rate**: 74.5%
- **Description**: Characterized by high Breakout Rank Volume
- **Examples**: Josh Hart, Edmond Sumner, London Perrantes

### Low Team Pace Type (Archetype 7)

- **Players**: 25
- **Avg RAPM**: 0.37
- **Survival Rate**: 72.0%
- **Description**: Characterized by low Team Pace
- **Examples**: Ike Anigbogu, Harry Giles III, Cheick Diallo

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
