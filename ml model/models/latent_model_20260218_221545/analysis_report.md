# Latent Space Model Analysis Report

**Generated**: 2026-02-18 22:15
**Model**: latent_model_20260218_221545

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 3.2731
- **Test RMSE**: 1.5758944749832153
- **Test Correlation**: 0.03095178432787091

## Discovered Archetypes

### High Recruiting Rank Type (Archetype 0)

- **Players**: 35
- **Avg RAPM**: -0.14
- **Survival Rate**: 68.6%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Luke Kornet, Antonius Cleveland, OG Anunoby

### High Team Rank Type (Archetype 1)

- **Players**: 101
- **Avg RAPM**: 0.14
- **Survival Rate**: 72.3%
- **Description**: Characterized by high Team Rank
- **Examples**: Tyler Lydon, Tyler Cavanaugh, Dennis Smith Jr.

### Low Games Played Type (Archetype 2)

- **Players**: 17
- **Avg RAPM**: 0.18
- **Survival Rate**: 64.7%
- **Description**: Characterized by low Games Played
- **Examples**: Harry Giles III, Anthony Davis, Michael Kidd-Gilchrist

### High Team Rank Type (Archetype 3)

- **Players**: 50
- **Avg RAPM**: 0.19
- **Survival Rate**: 60.0%
- **Description**: Characterized by high Team Rank
- **Examples**: Jaylen Morris, Rodney Purvis, Jarrett Allen

### High Games Played Type (Archetype 4)

- **Players**: 187
- **Avg RAPM**: 0.47
- **Survival Rate**: 71.1%
- **Description**: Characterized by high Games Played
- **Examples**: Erik McCree, PJ Dozier, Josh Hart

### Low Team Pace Type (Archetype 5)

- **Players**: 16
- **Avg RAPM**: -0.07
- **Survival Rate**: 81.2%
- **Description**: Characterized by low Team Pace
- **Examples**: Ike Anigbogu, Marshall Plumlee, Cheick Diallo

### High Recruiting Rank Type (Archetype 6)

- **Players**: 64
- **Avg RAPM**: 0.10
- **Survival Rate**: 75.0%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Edmond Sumner, London Perrantes, Justin Patton

### High Recruiting Rank Type (Archetype 7)

- **Players**: 13
- **Avg RAPM**: 0.72
- **Survival Rate**: 84.6%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: CJ McCollum, Tim Hardaway Jr., Trey Burke

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
