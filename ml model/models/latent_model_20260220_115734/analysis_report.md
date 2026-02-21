# Latent Space Model Analysis Report

**Generated**: 2026-02-20 11:57
**Model**: latent_model_20260220_115734

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 7.6534
- **Test RAPM RMSE**: N/A
- **Test RAPM Correlation**: N/A
- **Test EPM RMSE**: 2.532543182373047
- **Test EPM Correlation**: -0.040147680864863824

## Discovered Archetypes

### High Minutes Total Type (Archetype 0)

- **Players**: 88
- **Avg RAPM**: 0.27
- **Survival Rate**: 64.8%
- **Description**: Characterized by high Minutes Total
- **Examples**: Marvin Bagley III, Wendell Carter Jr., Gary Trent Jr.

### High Usage Z Type (Archetype 1)

- **Players**: 123
- **Avg RAPM**: 0.39
- **Survival Rate**: 72.4%
- **Description**: Characterized by high Usage Z
- **Examples**: Josh Hart, Donovan Mitchell, Tyler Lydon

### High Recruiting Rank Type (Archetype 2)

- **Players**: 59
- **Avg RAPM**: 0.13
- **Survival Rate**: 71.2%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Edmond Sumner, London Perrantes, Ike Anigbogu

### Low Team Rank Type (Archetype 3)

- **Players**: 167
- **Avg RAPM**: 0.33
- **Survival Rate**: 70.1%
- **Description**: Characterized by low Team Rank
- **Examples**: Erik McCree, PJ Dozier, Mangok Mathiang

### High Team Rank Type (Archetype 4)

- **Players**: 50
- **Avg RAPM**: -0.19
- **Survival Rate**: 62.0%
- **Description**: Characterized by high Team Rank
- **Examples**: Rodney Purvis, Jarrett Allen, Antonio Blakeney

### High Recruiting Rank Type (Archetype 5)

- **Players**: 40
- **Avg RAPM**: 0.27
- **Survival Rate**: 80.0%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Luke Kornet, Antonius Cleveland, OG Anunoby

### Low Team Srs Type (Archetype 6)

- **Players**: 29
- **Avg RAPM**: 0.24
- **Survival Rate**: 65.5%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Eric Moreland, Maurice Harkless

### Low Games Played Type (Archetype 7)

- **Players**: 28
- **Avg RAPM**: 0.14
- **Survival Rate**: 64.3%
- **Description**: Characterized by low Games Played
- **Examples**: Michael Porter Jr., Harry Giles III, Michael Kidd-Gilchrist

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
