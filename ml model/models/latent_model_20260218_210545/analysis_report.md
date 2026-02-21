# Latent Space Model Analysis Report

**Generated**: 2026-02-18 21:05
**Model**: latent_model_20260218_210545

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 3.1886
- **Test RMSE**: 1.5660669803619385
- **Test Correlation**: 0.08430879821103293

## Discovered Archetypes

### High Recruiting Rating Type (Archetype 0)

- **Players**: 76
- **Avg RAPM**: 0.05
- **Survival Rate**: 75.0%
- **Description**: Characterized by high Recruiting Rating
- **Examples**: Josh Hart, Edmond Sumner, London Perrantes

### Low Team Pace Type (Archetype 1)

- **Players**: 27
- **Avg RAPM**: 0.33
- **Survival Rate**: 66.7%
- **Description**: Characterized by low Team Pace
- **Examples**: Jaylen Morris, Anthony Davis, Michael Kidd-Gilchrist

### High Shots Total Type (Archetype 2)

- **Players**: 212
- **Avg RAPM**: 0.40
- **Survival Rate**: 71.7%
- **Description**: Characterized by high Shots Total
- **Examples**: Erik McCree, PJ Dozier, Donovan Mitchell

### High Recruiting Rank Type (Archetype 3)

- **Players**: 25
- **Avg RAPM**: 0.23
- **Survival Rate**: 72.0%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Briante Weber, Shayne Whittington, Eric Moreland

### Low Team Pace Type (Archetype 4)

- **Players**: 36
- **Avg RAPM**: 0.02
- **Survival Rate**: 69.4%
- **Description**: Characterized by low Team Pace
- **Examples**: Ike Anigbogu, Harry Giles III, Cheick Diallo

### Low Usage Z Type (Archetype 5)

- **Players**: 54
- **Avg RAPM**: -0.04
- **Survival Rate**: 59.3%
- **Description**: Characterized by low Usage Z
- **Examples**: Mangok Mathiang, Zach Collins, Jarrett Allen

### High Recruiting Rank Type (Archetype 6)

- **Players**: 51
- **Avg RAPM**: 0.49
- **Survival Rate**: 76.5%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Luke Kornet, Jamel Artis, John Collins

### High Tov Total Per100Poss Type (Archetype 7)

- **Players**: 2
- **Avg RAPM**: -1.25
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Tov Total Per100Poss
- **Examples**: Carldell Johnson, Jeremy Pargo

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
