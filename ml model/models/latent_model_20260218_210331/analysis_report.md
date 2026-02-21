# Latent Space Model Analysis Report

**Generated**: 2026-02-18 21:03
**Model**: latent_model_20260218_210331

## Training Results

- **Best Epoch**: 24
- **Best Val Loss**: 1.5829
- **Test RMSE**: 1.6156105995178223
- **Test Correlation**: -0.0655116130112249

## Discovered Archetypes

### High Recruiting Rating Type (Archetype 0)

- **Players**: 131
- **Avg RAPM**: 0.10
- **Survival Rate**: 68.7%
- **Description**: Characterized by high Recruiting Rating
- **Examples**: Josh Hart, Edmond Sumner, London Perrantes

### Low Ast Total Per100Poss Type (Archetype 1)

- **Players**: 69
- **Avg RAPM**: 0.75
- **Survival Rate**: 72.5%
- **Description**: Characterized by low Ast Total Per100Poss
- **Examples**: Mangok Mathiang, Bam Adebayo, Chris Boucher

### High Shots Total Type (Archetype 2)

- **Players**: 79
- **Avg RAPM**: 0.17
- **Survival Rate**: 65.8%
- **Description**: Characterized by high Shots Total
- **Examples**: Erik McCree, Zach Collins, Tyler Cavanaugh

### High Blk Total Per100Poss Type (Archetype 3)

- **Players**: 11
- **Avg RAPM**: 1.21
- **Survival Rate**: 54.5%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Anthony Davis, Michael Kidd-Gilchrist, Andre Drummond

### High Blk Total Per100Poss Type (Archetype 4)

- **Players**: 3
- **Avg RAPM**: 0.57
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Festus Ezeli, Jeff Withey, Mason Plumlee

### High Fga Total Type (Archetype 5)

- **Players**: 111
- **Avg RAPM**: 0.22
- **Survival Rate**: 78.4%
- **Description**: Characterized by high Fga Total
- **Examples**: PJ Dozier, Donovan Mitchell, Nigel Williams-Goss

### Low Team Pace Type (Archetype 6)

- **Players**: 42
- **Avg RAPM**: -0.67
- **Survival Rate**: 71.4%
- **Description**: Characterized by low Team Pace
- **Examples**: Jaylen Morris, Ike Anigbogu, Harry Giles III

### High Recruiting Rank Type (Archetype 7)

- **Players**: 37
- **Avg RAPM**: 0.92
- **Survival Rate**: 67.6%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Amile Jefferson, Skal Labissiere, Rondae Hollis-Jefferson

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
