# Latent Space Model Analysis Report

**Generated**: 2026-02-20 11:32
**Model**: latent_model_20260220_113238

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 9.5699
- **Test RAPM RMSE**: 1.5576202869415283
- **Test RAPM Correlation**: -0.06274874000575036
- **Test EPM RMSE**: 2.880575180053711
- **Test EPM Correlation**: 0.05365090577744523

## Discovered Archetypes

### Low Team Srs Type (Archetype 0)

- **Players**: 18
- **Avg RAPM**: -0.13
- **Survival Rate**: 66.7%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Maurice Harkless, Festus Ezeli

### High Blk Total Per100Poss Type (Archetype 1)

- **Players**: 40
- **Avg RAPM**: -0.24
- **Survival Rate**: 67.5%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Edmond Sumner, Ike Anigbogu, Tyler Lydon

### High Recruiting Rank Type (Archetype 2)

- **Players**: 45
- **Avg RAPM**: 0.42
- **Survival Rate**: 73.3%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Luke Kornet, Antonius Cleveland, OG Anunoby

### High Recruiting Rank Type (Archetype 3)

- **Players**: 52
- **Avg RAPM**: 0.31
- **Survival Rate**: 69.2%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: London Perrantes, Jamel Artis, John Collins

### High On Ortg Type (Archetype 4)

- **Players**: 15
- **Avg RAPM**: 0.23
- **Survival Rate**: 93.3%
- **Description**: Characterized by high On Ortg
- **Examples**: Donovan Mitchell, Justin Jackson, Isaiah Hicks

### High Team Rank Type (Archetype 5)

- **Players**: 47
- **Avg RAPM**: -0.01
- **Survival Rate**: 68.1%
- **Description**: Characterized by high Team Rank
- **Examples**: Rodney Purvis, Jarrett Allen, Antonio Blakeney

### Low Recruiting Rank Type (Archetype 6)

- **Players**: 91
- **Avg RAPM**: 0.14
- **Survival Rate**: 68.1%
- **Description**: Characterized by low Recruiting Rank
- **Examples**: Tyler Cavanaugh, Dennis Smith Jr., Derrick White

### High Games Played Type (Archetype 7)

- **Players**: 188
- **Avg RAPM**: 0.42
- **Survival Rate**: 70.7%
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
