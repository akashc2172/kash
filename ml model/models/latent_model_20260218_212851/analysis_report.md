# Latent Space Model Analysis Report

**Generated**: 2026-02-18 21:29
**Model**: latent_model_20260218_212851

## Training Results

- **Best Epoch**: 28
- **Best Val Loss**: 1.6161
- **Test RMSE**: 1.5906000137329102
- **Test Correlation**: -0.098710481929903

## Discovered Archetypes

### High Shots Total Type (Archetype 0)

- **Players**: 76
- **Avg RAPM**: 0.45
- **Survival Rate**: 71.1%
- **Description**: Characterized by high Shots Total
- **Examples**: Erik McCree, Zach Collins, Tyler Cavanaugh

### Low Ast Total Per100Poss Type (Archetype 1)

- **Players**: 51
- **Avg RAPM**: 0.27
- **Survival Rate**: 74.5%
- **Description**: Characterized by low Ast Total Per100Poss
- **Examples**: Luke Kornet, Charles Cooke, OG Anunoby

### High Fga Total Type (Archetype 2)

- **Players**: 123
- **Avg RAPM**: 0.05
- **Survival Rate**: 72.4%
- **Description**: Characterized by high Fga Total
- **Examples**: PJ Dozier, Nigel Williams-Goss, Caleb Swanigan

### Low Usage Z Type (Archetype 3)

- **Players**: 32
- **Avg RAPM**: 0.70
- **Survival Rate**: 65.6%
- **Description**: Characterized by low Usage Z
- **Examples**: Amile Jefferson, Skal Labissiere, Alex Poythress

### Low Team Pace Type (Archetype 4)

- **Players**: 48
- **Avg RAPM**: -0.72
- **Survival Rate**: 68.8%
- **Description**: Characterized by low Team Pace
- **Examples**: Jaylen Morris, Harry Giles III, Cheick Diallo

### High Recruiting Rank Type (Archetype 5)

- **Players**: 41
- **Avg RAPM**: 1.13
- **Survival Rate**: 70.7%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Mangok Mathiang, Bam Adebayo, Chris Boucher

### High Blk Total Per100Poss Type (Archetype 6)

- **Players**: 13
- **Avg RAPM**: 1.08
- **Survival Rate**: 69.2%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Anthony Davis, Michael Kidd-Gilchrist, Andre Drummond

### High Ast Total Per100Poss Type (Archetype 7)

- **Players**: 99
- **Avg RAPM**: 0.22
- **Survival Rate**: 70.7%
- **Description**: Characterized by high Ast Total Per100Poss
- **Examples**: Josh Hart, Edmond Sumner, London Perrantes

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
