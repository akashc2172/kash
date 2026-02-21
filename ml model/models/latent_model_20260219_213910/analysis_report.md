# Latent Space Model Analysis Report

**Generated**: 2026-02-19 21:39
**Model**: latent_model_20260219_213910

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 6.8257
- **Test RMSE**: 1.5770684480667114
- **Test Correlation**: -0.10496053732570446

## Discovered Archetypes

### High Team Rank Type (Archetype 0)

- **Players**: 67
- **Avg RAPM**: 0.11
- **Survival Rate**: 76.1%
- **Description**: Characterized by high Team Rank
- **Examples**: Tyler Cavanaugh, Derrick White, Jabari Bird

### High Recruiting Rank Type (Archetype 1)

- **Players**: 47
- **Avg RAPM**: 0.27
- **Survival Rate**: 72.3%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Luke Kornet, Antonius Cleveland, OG Anunoby

### Low Team Rank Type (Archetype 2)

- **Players**: 227
- **Avg RAPM**: 0.37
- **Survival Rate**: 69.6%
- **Description**: Characterized by low Team Rank
- **Examples**: Erik McCree, PJ Dozier, Josh Hart

### Low Team Srs Type (Archetype 3)

- **Players**: 9
- **Avg RAPM**: -1.21
- **Survival Rate**: 55.6%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Carldell Johnson, Ivan Johnson

### Low Games Played Type (Archetype 4)

- **Players**: 23
- **Avg RAPM**: 1.00
- **Survival Rate**: 65.2%
- **Description**: Characterized by low Games Played
- **Examples**: Maurice Harkless, Khris Middleton, Andre Drummond

### High Team Rank Type (Archetype 5)

- **Players**: 31
- **Avg RAPM**: 0.14
- **Survival Rate**: 64.5%
- **Description**: Characterized by high Team Rank
- **Examples**: Rodney Purvis, Jarrett Allen, Antonio Blakeney

### High Recruiting Rating Type (Archetype 6)

- **Players**: 67
- **Avg RAPM**: 0.10
- **Survival Rate**: 73.1%
- **Description**: Characterized by high Recruiting Rating
- **Examples**: Edmond Sumner, London Perrantes, Ike Anigbogu

### High On Ortg Type (Archetype 7)

- **Players**: 12
- **Avg RAPM**: -0.26
- **Survival Rate**: 91.7%
- **Description**: Characterized by high On Ortg
- **Examples**: Donovan Mitchell, Justin Jackson, Isaiah Hicks

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
