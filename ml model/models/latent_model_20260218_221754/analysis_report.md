# Latent Space Model Analysis Report

**Generated**: 2026-02-18 22:17
**Model**: latent_model_20260218_221754

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 3.4717
- **Test RMSE**: 1.5811423063278198
- **Test Correlation**: 0.05032225746287511

## Discovered Archetypes

### High Team Rank Type (Archetype 0)

- **Players**: 36
- **Avg RAPM**: 0.30
- **Survival Rate**: 63.9%
- **Description**: Characterized by high Team Rank
- **Examples**: Rodney Purvis, Jarrett Allen, David Nwaba

### High Games Played Type (Archetype 1)

- **Players**: 204
- **Avg RAPM**: 0.34
- **Survival Rate**: 69.6%
- **Description**: Characterized by high Games Played
- **Examples**: Erik McCree, PJ Dozier, Josh Hart

### Low Team Srs Type (Archetype 2)

- **Players**: 13
- **Avg RAPM**: -0.71
- **Survival Rate**: 61.5%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Festus Ezeli, Reggie Bullock Jr.

### Low Fga Total Type (Archetype 3)

- **Players**: 47
- **Avg RAPM**: 0.47
- **Survival Rate**: 66.0%
- **Description**: Characterized by low Fga Total
- **Examples**: Ike Anigbogu, Harry Giles III, Cheick Diallo

### High Team Rank Type (Archetype 4)

- **Players**: 53
- **Avg RAPM**: -0.22
- **Survival Rate**: 75.5%
- **Description**: Characterized by high Team Rank
- **Examples**: Tyler Cavanaugh, Derrick White, Jabari Bird

### High Recruiting Rank Type (Archetype 5)

- **Players**: 42
- **Avg RAPM**: 0.15
- **Survival Rate**: 78.6%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Luke Kornet, Antonius Cleveland, OG Anunoby

### Low Games Played Type (Archetype 6)

- **Players**: 12
- **Avg RAPM**: 1.50
- **Survival Rate**: 58.3%
- **Description**: Characterized by low Games Played
- **Examples**: Maurice Harkless, Darius Johnson-Odom, DeQuan Jones

### High Recruiting Rating Type (Archetype 7)

- **Players**: 76
- **Avg RAPM**: 0.23
- **Survival Rate**: 77.6%
- **Description**: Characterized by high Recruiting Rating
- **Examples**: Edmond Sumner, London Perrantes, Tyler Lydon

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
